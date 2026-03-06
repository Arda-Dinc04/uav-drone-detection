from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, Image


def export_manifest_to_parquet(detections_dir: Path, output_path: Path) -> Path:
    manifest_path = detections_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}")

    rows: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = Path(record["image_path"])
            if not image_path.is_absolute():
                image_path = Path.cwd() / image_path
            rows.append(
                {
                    "image": str(image_path),
                    "video_name": record["video_name"],
                    "frame_index": record["frame_index"],
                    "width": record["width"],
                    "height": record["height"],
                    "detections": json.dumps(record["detections"]),
                }
            )

    dataset = Dataset.from_list(rows).cast_column("image", Image())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output_path))
    return output_path
