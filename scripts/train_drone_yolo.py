from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune a single-class YOLO detector for drone-as-object detection."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the Ultralytics dataset YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11s.pt",
        help="Base checkpoint. Recommended default balances speed and small-object accuracy.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience.")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/drone_train"),
        help="Directory where Ultralytics will save training outputs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11s_drone",
        help="Experiment name under the project directory.",
    )
    parser.add_argument("--device", type=str, default=None, help="Training device, for example 0, cpu, or mps.")
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Data loader workers. Lower defaults are safer in Colab.",
    )
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic augmentation near the end of training.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        device=args.device,
        workers=args.workers,
        close_mosaic=args.close_mosaic,
        pretrained=True,
        optimizer="auto",
        cache=False,
        plots=True,
    )
    save_dir = Path(results.save_dir)
    print(f"Training finished. Outputs saved under: {save_dir}")
    print(f"Use the checkpoint at: {save_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
