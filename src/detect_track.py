"""
Command-line real-time PPE monitoring.

Usage:
    python detect_track.py --source 0                # webcam
    python detect_track.py --source path/to/video.mp4
    python detect_track.py --source rtsp://...        # CCTV / IP camera

Press 'q' to quit the display window.
"""

import argparse
import cv2

import config
from ppe_engine import PPEComplianceEngine


def draw_overlays(frame, raw_detections):
    # Add title "CONSTRUCTION SAFETY DETECTION"
    # Draw black text first for shadow/outline, then thin white text
    title = "CONSTRUCTION SAFETY DETECTION"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    # Center text roughly
    tx, ty = (frame.shape[1] - tw) // 2, 60
    cv2.putText(frame, title, (tx+2, ty+2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, title, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

    for det in raw_detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_name = det["class"]
        conf = det["conf"]
        
        display_name = cls_name
        if cls_name == "helmet":
            display_name = "Construction-Hat"
        elif cls_name == "vest":
            display_name = "Safety-Vest"
        
        # Format label like "0.85 Person"
        label = f"{conf:.2f} {display_name}"
        
        # Color: Blue (BGR format) for Person, Green for others
        color = (255, 0, 0) if cls_name == config.PERSON_CLASS else (0, 255, 0)
        
        # Box thickness
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Filled rectangle for text background
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        
        # Text color
        text_color = (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255)
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2,
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description="Real-time PPE compliance monitoring")
    parser.add_argument("--source", default="0", help="0 for webcam, or a video/RTSP path")
    parser.add_argument("--model", default=config.MODEL_PATH, help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    engine = PPEComplianceEngine(model_path=args.model, conf_threshold=args.conf)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frame_idx = 0
    print(f"Streaming from {source}. Violation log: {config.VIOLATION_LOG_CSV}. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        statuses = engine.process_frame(frame, frame_idx)
        frame = draw_overlays(frame, engine.last_raw_detections)

        compliant = sum(1 for s in statuses if s.compliant)
        cv2.putText(
            frame,
            f"Workers: {len(statuses)}  Compliant: {compliant}  Violations: {len(statuses) - compliant}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        cv2.imshow("SentryVision — PPE Compliance Monitoring", frame)
        frame_idx += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
