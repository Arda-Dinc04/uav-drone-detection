from __future__ import annotations

from typing import Any

from ultralytics import YOLO

from drone_tracking.models import Detection


def _resolve_label(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


class DroneDetector:
    def __init__(
        self,
        weights: str,
        target_label: str = "drone",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
    ) -> None:
        self.model = YOLO(weights)
        self.target_label = target_label.lower()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            label = _resolve_label(result.names, class_id).lower()
            if self.target_label != "*" and label != self.target_label:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(box.conf.item()),
                    label=label,
                )
            )
        detections.sort(key=lambda detection: detection.confidence, reverse=True)
        return detections
