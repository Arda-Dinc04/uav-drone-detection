from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from itertools import count
from math import hypot

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from drone_tracking.models import Detection, TrackState


def _constant_velocity_process_noise(dt: float, process_noise: float) -> np.ndarray:
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    return process_noise * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ],
        dtype=float,
    )


@dataclass
class KalmanTrack:
    track_id: int
    detection: Detection
    dt: float
    process_noise: float
    measurement_noise: float
    max_history: int
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    confidence: float = field(init=False)
    label: str = field(init=False)

    def __post_init__(self) -> None:
        cx, cy = self.detection.center
        width, height = self.detection.size
        self.width = max(width, 1.0)
        self.height = max(height, 1.0)
        self.confidence = self.detection.confidence
        self.label = self.detection.label
        self.trajectory: deque[tuple[float, float]] = deque(maxlen=self.max_history)
        self.last_predicted = False

        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.kf.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        self.kf.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self.kf.P = np.diag([50.0, 50.0, 250.0, 250.0]).astype(float)
        self.kf.R = np.diag([self.measurement_noise, self.measurement_noise]).astype(float)
        self.kf.Q = _constant_velocity_process_noise(self.dt, self.process_noise)
        self.record_state(predicted=False)

    @property
    def current_center(self) -> tuple[float, float]:
        return (float(self.kf.x[0]), float(self.kf.x[1]))

    @property
    def current_bbox(self) -> tuple[float, float, float, float]:
        cx, cy = self.current_center
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Detection) -> None:
        cx, cy = detection.center
        width, height = detection.size
        self.kf.update(np.array([cx, cy], dtype=float))
        self.width = max(0.6 * self.width + 0.4 * width, 1.0)
        self.height = max(0.6 * self.height + 0.4 * height, 1.0)
        self.confidence = detection.confidence
        self.label = detection.label
        self.hits += 1
        self.time_since_update = 0

    def record_state(self, predicted: bool) -> None:
        self.last_predicted = predicted
        self.trajectory.append(self.current_center)

    def is_confirmed(self, min_hits: int) -> bool:
        return self.hits >= min_hits or self.age <= min_hits

    def distance_to(self, detection: Detection) -> float:
        tx, ty = self.current_center
        dx, dy = detection.center
        return hypot(tx - dx, ty - dy)

    def snapshot(self, min_hits: int) -> TrackState:
        return TrackState(
            track_id=self.track_id,
            center=self.current_center,
            bbox=self.current_bbox,
            trajectory=list(self.trajectory),
            predicted=self.last_predicted,
            time_since_update=self.time_since_update,
            confirmed=self.is_confirmed(min_hits),
        )


class MultiObjectTracker:
    def __init__(
        self,
        dt: float,
        max_distance: float = 80.0,
        max_missed_frames: int = 8,
        min_hits: int = 2,
        process_noise: float = 5.0,
        measurement_noise: float = 20.0,
        max_history: int = 256,
    ) -> None:
        self.dt = dt
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.min_hits = min_hits
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_history = max_history
        self.tracks: list[KalmanTrack] = []
        self._id_source = count(1)

    def _spawn_track(self, detection: Detection) -> None:
        self.tracks.append(
            KalmanTrack(
                track_id=next(self._id_source),
                detection=detection,
                dt=self.dt,
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
                max_history=self.max_history,
            )
        )

    def _associate(self, detections: list[Detection]) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        if not self.tracks or not detections:
            return [], set(range(len(self.tracks))), set(range(len(detections)))

        cost_matrix = np.full((len(self.tracks), len(detections)), self.max_distance + 1.0, dtype=float)
        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                cost_matrix[track_index, detection_index] = track.distance_to(detection)

        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matches: list[tuple[int, int]] = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        for track_index, detection_index in zip(row_indices, col_indices):
            if cost_matrix[track_index, detection_index] > self.max_distance:
                continue
            matches.append((track_index, detection_index))
            unmatched_tracks.discard(track_index)
            unmatched_detections.discard(detection_index)

        return matches, unmatched_tracks, unmatched_detections

    def step(self, detections: list[Detection]) -> list[TrackState]:
        for track in self.tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_detections = self._associate(detections)

        for track_index, detection_index in matches:
            track = self.tracks[track_index]
            track.update(detections[detection_index])
            track.record_state(predicted=False)

        for track_index in unmatched_tracks:
            self.tracks[track_index].record_state(predicted=True)

        for detection_index in unmatched_detections:
            self._spawn_track(detections[detection_index])

        self.tracks = [
            track for track in self.tracks if track.time_since_update <= self.max_missed_frames
        ]
        return [track.snapshot(self.min_hits) for track in self.tracks if track.is_confirmed(self.min_hits)]
