"""
Fine-tune a YOLO model on your PPE dataset.

Expected dataset layout (standard Ultralytics YOLO format), produced
automatically if you export a Roboflow project in "YOLOv8" format:

    data/
      data.yaml
      train/images, train/labels
      valid/images, valid/labels
      test/images,  test/labels

data.yaml should declare the classes, e.g.:

    names: ['person', 'helmet', 'vest']

Usage:
    python train.py --data ../data/data.yaml --epochs 100 --imgsz 640
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on a PPE dataset")
    parser.add_argument("--data", default="construction-ppe.yaml", help="Path to data.yaml")
    parser.add_argument("--base-model", default="yolov8n.pt", help="Base checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="../models", help="Where to save training runs")
    parser.add_argument("--name", default="ppe_yolo", help="Run name")
    args = parser.parse_args()

    model = YOLO(args.base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    metrics = model.val()
    print("Validation metrics:", metrics.results_dict)
    print(f"\nBest weights saved under {args.project}/{args.name}/weights/best.pt")
    print("Point MODEL_PATH in config.py (or --model on detect_track.py) at that file.")


if __name__ == "__main__":
    main()
