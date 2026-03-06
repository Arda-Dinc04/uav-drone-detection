from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def size(self) -> tuple[float, float]:
        return (self.x2 - self.x1, self.y2 - self.y1)

    def to_dict(self) -> dict[str, float | str]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "confidence": round(self.confidence, 4),
            "label": self.label,
        }


@dataclass(slots=True)
class TrackState:
    track_id: int
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    trajectory: list[tuple[float, float]]
    predicted: bool
    time_since_update: int
    confirmed: bool
