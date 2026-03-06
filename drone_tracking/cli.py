from __future__ import annotations

import argparse
from pathlib import Path

from drone_tracking.detector import DroneDetector
from drone_tracking.export import export_manifest_to_parquet
from drone_tracking.pipeline import VideoPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drone detection and Kalman tracking pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process = subparsers.add_parser("process", help="Run detection and tracking on a directory of .mp4 files.")
    process.add_argument("--input-dir", type=Path, required=True, help="Directory containing .mp4 videos.")
    process.add_argument("--weights", type=str, required=True, help="Path to detector weights.")
    process.add_argument("--target-label", type=str, default="drone", help='Model label to keep, or "*" for all classes.')
    process.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for tracking videos and rendered frames.")
    process.add_argument("--detections-dir", type=Path, default=Path("detections"), help="Directory for raw detection-positive frames.")
    process.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame.")
    process.add_argument("--device", type=str, default=None, help="Ultralytics device string, for example cpu, 0, or mps.")
    process.add_argument("--conf-threshold", type=float, default=0.25, help="Minimum detector confidence.")
    process.add_argument("--iou-threshold", type=float, default=0.45, help="Detector NMS IoU threshold.")
    process.add_argument("--max-distance", type=float, default=80.0, help="Maximum pixel distance for track-detection association.")
    process.add_argument("--max-missed-frames", type=int, default=8, help="How many consecutive misses a track can survive.")
    process.add_argument("--min-hits", type=int, default=2, help="Minimum updates before a track is considered confirmed.")
    process.add_argument("--process-noise", type=float, default=5.0, help="Kalman process noise scale.")
    process.add_argument("--measurement-noise", type=float, default=20.0, help="Kalman measurement noise scale.")

    export = subparsers.add_parser("export-hf", help="Export the detection manifest as a Parquet dataset.")
    export.add_argument("--detections-dir", type=Path, default=Path("detections"), help="Directory containing manifest.jsonl.")
    export.add_argument("--output-file", type=Path, default=Path("detections/detections.parquet"), help="Output Parquet file.")

    return parser


def run_process(args: argparse.Namespace) -> int:
    detector = DroneDetector(
        weights=args.weights,
        target_label=args.target_label,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device,
    )
    pipeline = VideoPipeline(
        detector=detector,
        output_dir=args.output_dir,
        detections_dir=args.detections_dir,
        frame_step=args.frame_step,
        max_distance=args.max_distance,
        max_missed_frames=args.max_missed_frames,
        min_hits=args.min_hits,
        process_noise=args.process_noise,
        measurement_noise=args.measurement_noise,
    )
    results = pipeline.process_directory(args.input_dir)
    for result in results:
        output_video = result.output_video.as_posix() if result.output_video else "none"
        print(
            f"{result.video_path.name}: processed_frames={result.processed_frames} "
            f"detection_frames={result.detection_frames} output_video={output_video}"
        )
    return 0


def run_export(args: argparse.Namespace) -> int:
    output_path = export_manifest_to_parquet(args.detections_dir, args.output_file)
    print(f"Parquet dataset written to {output_path.as_posix()}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "process":
        return run_process(args)
    if args.command == "export-hf":
        return run_export(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
