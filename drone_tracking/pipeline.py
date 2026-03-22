from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from drone_tracking.detector import DroneDetector
from drone_tracking.models import Detection, TrackState
from drone_tracking.tracker import MultiObjectTracker


@dataclass(slots=True)
class VideoResult:
    video_path: Path
    processed_frames: int
    detection_frames: int
    output_video: Path | None


def _track_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    blue, green, red = rng.integers(80, 255, size=3)
    return (int(blue), int(green), int(red))


def _heading_endpoint(track: TrackState, arrow_length: float = 28.0) -> tuple[int, int] | None:
    recent_points = track.trajectory[-6:]
    dx = dy = 0.0
    if len(recent_points) >= 2:
        start_x, start_y = recent_points[0]
        end_x, end_y = recent_points[-1]
        dx = end_x - start_x
        dy = end_y - start_y

    if float(np.hypot(dx, dy)) < 1.5:
        vx, vy = track.velocity
        dx, dy = vx, vy

    norm = float(np.hypot(dx, dy))
    if norm < 0.75:
        return None

    scale = arrow_length / norm
    return (
        int(track.center[0] + dx * scale),
        int(track.center[1] + dy * scale),
    )


def _draw_detections(frame, detections: list[Detection]) -> None:
    for detection in detections:
        x1, y1, x2, y2 = (int(detection.x1), int(detection.y1), int(detection.x2), int(detection.y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 40), 2)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 220, 40),
            2,
            cv2.LINE_AA,
        )


def _draw_tracks(frame, tracks: list[TrackState]) -> None:
    for track in tracks:
        color = _track_color(track.track_id)
        if len(track.trajectory) >= 2:
            points = np.array([(int(x), int(y)) for x, y in track.trajectory], dtype=np.int32)
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

        cx, cy = (int(track.center[0]), int(track.center[1]))
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(
            frame,
            f"track {track.track_id}",
            (cx + 6, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

        end_point = _heading_endpoint(track)
        if end_point is not None:
            cv2.arrowedLine(
                frame,
                (cx, cy),
                end_point,
                color,
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )

        if track.predicted:
            x1, y1, x2, y2 = (int(value) for value in track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
            cv2.putText(
                frame,
                "predicted",
                (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )


def _render_frame(frame, detections: list[Detection], tracks: list[TrackState]):
    rendered = frame.copy()
    _draw_detections(rendered, detections)
    _draw_tracks(rendered, tracks)
    return rendered


def _compose_video_with_ffmpeg(rendered_frames_dir: Path, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        f"{fps:.4f}",
        "-i",
        str(rendered_frames_dir / "frame_%06d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to assemble the output video.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "ffmpeg failed while writing the output video.") from exc


def _relative_to_cwd(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


class VideoPipeline:
    def __init__(
        self,
        detector: DroneDetector,
        output_dir: Path,
        detections_dir: Path,
        frame_step: int = 1,
        max_distance: float = 80.0,
        max_missed_frames: int = 8,
        min_hits: int = 2,
        process_noise: float = 5.0,
        measurement_noise: float = 20.0,
    ) -> None:
        if frame_step < 1:
            raise ValueError("frame_step must be at least 1.")
        self.detector = detector
        self.output_dir = output_dir
        self.detections_dir = detections_dir
        self.frame_step = frame_step
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.min_hits = min_hits
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def process_directory(self, input_dir: Path) -> list[VideoResult]:
        videos = sorted(input_dir.glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No .mp4 files found in {input_dir}")

        self.detections_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows: list[dict[str, object]] = []
        results: list[VideoResult] = []
        for video_path in videos:
            result, video_manifest = self.process_video(video_path)
            results.append(result)
            manifest_rows.extend(video_manifest)

        manifest_path = self.detections_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row) + "\n")

        return results

    def process_video(self, video_path: Path) -> tuple[VideoResult, list[dict[str, object]]]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        tracker = MultiObjectTracker(
            dt=self.frame_step / fps,
            max_distance=self.max_distance,
            max_missed_frames=self.max_missed_frames,
            min_hits=self.min_hits,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )

        detection_frames_dir = self.detections_dir / video_path.stem
        rendered_frames_dir = self.output_dir / "rendered_frames" / video_path.stem
        detection_frames_dir.mkdir(parents=True, exist_ok=True)
        rendered_frames_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows: list[dict[str, object]] = []
        processed_frames = 0
        detection_frames = 0
        rendered_frames = 0
        frame_index = -1

        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % self.frame_step != 0:
                continue

            processed_frames += 1
            detections = self.detector.detect(frame)
            tracks = tracker.step(detections)

            if detections:
                detection_frames += 1
                frame_output_path = detection_frames_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(frame_output_path), frame)
                manifest_rows.append(
                    {
                        "image_path": _relative_to_cwd(frame_output_path),
                        "video_name": video_path.name,
                        "frame_index": frame_index,
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                        "detections": [detection.to_dict() for detection in detections],
                    }
                )

            if tracks:
                rendered = _render_frame(frame, detections, tracks)
                rendered_frames += 1
                rendered_frame_path = rendered_frames_dir / f"frame_{rendered_frames:06d}.jpg"
                cv2.imwrite(str(rendered_frame_path), rendered)

        capture.release()

        output_video = None
        if rendered_frames:
            output_video = self.output_dir / "videos" / f"{video_path.stem}_tracked.mp4"
            _compose_video_with_ffmpeg(rendered_frames_dir, output_video, fps / self.frame_step)

        return (
            VideoResult(
                video_path=video_path,
                processed_frames=processed_frames,
                detection_frames=detection_frames,
                output_video=output_video,
            ),
            manifest_rows,
        )
