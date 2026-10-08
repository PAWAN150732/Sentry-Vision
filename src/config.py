"""
Central configuration for the PPE Compliance Monitoring system.
Edit these values to match your trained model / dataset.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Model ----
# Path to your trained YOLO weights. Until you train your own model,
# this can point at a base Ultralytics checkpoint (auto-downloaded)
# purely so the pipeline runs end-to-end; detections will be generic
# COCO classes (e.g. "person") until you swap in a PPE-trained model.
MODEL_PATH = os.path.join(BASE_DIR, "runs", "models", "ppe_yolo", "weights", "best.pt")

# Names your PPE dataset should use (this is the recommended labeling
# scheme referenced in the project deck: person + PPE items, with
# violations derived by the rule engine rather than labeled directly).
# If you train with a different class list, this is overwritten at
# runtime by the model's own `model.names`.
DEFAULT_CLASS_NAMES = {
    0: "helmet",
    6: "Person",
    2: "vest",
}

# Which class names count as "the worker" vs "PPE items to associate"
PERSON_CLASS = "Person"
HELMET_CLASS = "helmet"
VEST_CLASS = "vest"
PPE_CLASSES = {HELMET_CLASS, VEST_CLASS}

# Tracker config shipped with Ultralytics (ByteTrack). "botsort.yaml"
# is also available if you want appearance-based re-ID.
TRACKER_CONFIG = "bytetrack.yaml"

# ---- Association rule engine ----
# A PPE item is "associated" with a person if the item's box center
# falls inside the person's box, expanded by this margin (fraction of
# the person box's width/height), OR if IoU with the person box's
# relevant region exceeds IOU_THRESHOLD.
CENTER_CONTAINMENT_MARGIN = 0.15
IOU_THRESHOLD = 0.10

# Fraction of the person's bounding box (from the top) considered the
# "head region" for helmet association, and the rest for vest/torso.
HEAD_REGION_FRACTION = 0.35

# Detection confidence threshold
CONF_THRESHOLD = 0.35

# ---- Logging ----
VIOLATION_LOG_CSV = "outputs/violation_log.csv"

# ---- Display ----
COLOR_COMPLIANT = (0, 200, 0)      # BGR green
COLOR_VIOLATION = (0, 0, 255)      # BGR red
COLOR_PPE_ITEM = (255, 180, 0)     # BGR light blue
