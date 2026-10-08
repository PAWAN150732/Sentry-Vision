"""
PPE Compliance Engine
======================
Wraps a YOLO model + Ultralytics tracker and turns raw per-frame
detections into per-person, identity-persistent PPE compliance
decisions, exactly matching the pipeline described in the project
deck:

    Camera Input -> YOLO Detection -> Tracking -> Association -> Rule Engine -> Log

This module is imported by both `detect_track.py` (CLI / OpenCV
window) and `streamlit_app.py` (dashboard), so the logic only lives
in one place.
"""

import os
import csv
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
from ultralytics import YOLO

import config


@dataclass
class PersonStatus:
    track_id: int
    bbox: tuple                 # (x1, y1, x2, y2)
    has_helmet: bool
    has_vest: bool
    missing: List[str] = field(default_factory=list)

    @property
    def compliant(self) -> bool:
        return self.has_helmet and self.has_vest

    @property
    def status_label(self) -> str:
        return "COMPLIANT" if self.compliant else "PPE VIOLATION"


class PPEComplianceEngine:
    def __init__(
        self,
        model_path: str = config.MODEL_PATH,
        tracker_config: str = config.TRACKER_CONFIG,
        conf_threshold: float = config.CONF_THRESHOLD,
        log_path: str = config.VIOLATION_LOG_CSV,
    ):
        self.model = YOLO(model_path)
        self.class_names: Dict[int, str] = self.model.names or config.DEFAULT_CLASS_NAMES
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold
        self.log_path = log_path
        self._ensure_log_file()

        # de-dupe: only write one log row per (track_id, status) change
        self._last_status: Dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _ensure_log_file(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "frame_idx", "track_id", "status", "missing_ppe"]
                )

    def _log_if_changed(self, frame_idx: int, person: PersonStatus):
        prev = self._last_status.get(person.track_id)
        if prev == person.status_label:
            return  # no change, skip duplicate row
        self._last_status[person.track_id] = person.status_label
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    frame_idx,
                    person.track_id,
                    person.status_label,
                    ";".join(person.missing) if person.missing else "",
                ]
            )

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @staticmethod
    def _iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _region_for(self, person_box, item_class: str):
        """Return the sub-region of a person's box relevant to a PPE item
        (head region for a helmet, torso region for a vest)."""
        x1, y1, x2, y2 = person_box
        height = y2 - y1
        if item_class == config.HELMET_CLASS:
            return (x1, y1, x2, y1 + height * config.HEAD_REGION_FRACTION)
        return (x1, y1 + height * config.HEAD_REGION_FRACTION, x2, y2)

    def _is_associated(self, person_box, item_box, item_class: str) -> bool:
        region = self._region_for(person_box, item_class)
        cx, cy = self._center(item_box)
        mx = config.CENTER_CONTAINMENT_MARGIN * (region[2] - region[0])
        my = config.CENTER_CONTAINMENT_MARGIN * (region[3] - region[1])
        inside = (region[0] - mx) <= cx <= (region[2] + mx) and (
            region[1] - my
        ) <= cy <= (region[3] + my)
        if inside:
            return True
        return self._iou(region, item_box) >= config.IOU_THRESHOLD

    # ------------------------------------------------------------------ #
    # Main per-frame entry point
    # ------------------------------------------------------------------ #
    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> List[PersonStatus]:
        """Run detection + tracking on one frame and return PPE compliance
        status for every tracked person."""
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            verbose=False,
        )[0]

        people, ppe_items = [], []
        self.last_raw_detections = []
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.class_names.get(cls_id, str(cls_id))
                xyxy = tuple(box.xyxy[0].tolist())
                track_id = int(box.id[0]) if box.id is not None else -1
                conf = float(box.conf[0])
                
                self.last_raw_detections.append({
                    "class": cls_name,
                    "bbox": xyxy,
                    "conf": conf,
                    "track_id": track_id
                })

                if cls_name == config.PERSON_CLASS:
                    people.append((track_id, xyxy))
                else:
                    ppe_items.append((cls_name, xyxy))

        statuses = []
        for track_id, person_box in people:
            has_helmet = any(
                self._is_associated(person_box, item_box, config.HELMET_CLASS)
                for cls_name, item_box in ppe_items
                if cls_name == config.HELMET_CLASS
            )
            has_vest = any(
                self._is_associated(person_box, item_box, config.VEST_CLASS)
                for cls_name, item_box in ppe_items
                if cls_name == config.VEST_CLASS
            )
            missing = []
            if not has_helmet:
                missing.append(config.HELMET_CLASS)
            if not has_vest:
                missing.append(config.VEST_CLASS)

            status = PersonStatus(
                track_id=track_id,
                bbox=person_box,
                has_helmet=has_helmet,
                has_vest=has_vest,
                missing=missing,
            )
            statuses.append(status)
            self._log_if_changed(frame_idx, status)

        return statuses
