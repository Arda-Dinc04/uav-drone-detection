# UAV Drone Detection and Tracking

This repository is a baseline implementation for the assignment: detect drones in every `.mp4` inside a directory, track detections with a Kalman filter, save detection-positive frames to `detections/`, and produce one tracking video per input.

The code is intentionally configurable around the detector weights because the only part you still need to supply is a model that actually recognizes the drone itself. The pipeline already handles the rest: frame iteration, detector inference, Kalman prediction and update, trajectory rendering, and Parquet export for the Hugging Face deliverable.

## Why this setup

The assignment target is the drone object itself, not ground objects viewed from a drone. Be careful here: the official VisDrone benchmark is primarily drone-captured imagery for detecting objects such as pedestrians and vehicles, so it is not a strong default for this assignment if your class of interest is `drone`.

The practical path is:

1. Pick or build a drone-as-object dataset in YOLO or COCO format.
2. Fine-tune a detector such as Ultralytics YOLO on that dataset.
3. Run this repository against the assignment videos.
4. Export the detection-positive frames into Parquet and upload the tracking outputs.

## Repository layout

```text
drone_tracking/
  cli.py
  detector.py
  export.py
  models.py
  pipeline.py
  tracker.py
configs/
  drone_dataset.example.yaml
README.md
pyproject.toml
```

## Environment setup

Create a virtual environment and install the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install `ffmpeg` if it is not already available:

```bash
brew install ffmpeg
```

## Download the test videos

Use `yt-dlp` exactly as required by the assignment:

```bash
brew install yt-dlp
mkdir -p videos
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
  -o "videos/drone_video_1.mp4" \
  "https://www.youtube.com/watch?v=DhmZ6W1UAv4"
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" \
  -o "videos/drone_video_2.mp4" \
  "https://www.youtube.com/watch?v=YrydHPwRelI"
```

## Detector recommendation

Use Ultralytics YOLO first. It is the fastest way to get a custom drone detector trained and integrated into the rest of the assignment. A good workflow is:

1. Export or convert your dataset to YOLO format.
2. Fill in [`configs/drone_dataset.example.yaml`](/Users/ardadinc/Documents/New project/configs/drone_dataset.example.yaml) with the real dataset paths.
3. Fine-tune from a pretrained checkpoint such as `yolo11n.pt`, `yolo11s.pt`, or your preferred YOLO variant.

Example:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=configs/drone_dataset.example.yaml \
  epochs=50 \
  imgsz=960
```

If you already have a trained checkpoint from Roboflow, Hugging Face, or a prior experiment, you can use it directly with this repository. The runtime assumes the model exposes a `drone` class label. If your label name differs, pass `--target-label`.

For the recommended Colab fine-tuning path, exact dataset layout, and balanced training defaults, use [TRAINING.md](/Users/ardadinc/Documents/New%20project/TRAINING.md).

## Run the assignment pipeline

Process every `.mp4` in `videos/`:

```bash
drone-pipeline process \
  --input-dir videos \
  --weights runs/detect/train/weights/best.pt \
  --target-label drone \
  --output-dir outputs \
  --detections-dir detections
```

Useful overrides:

```bash
drone-pipeline process \
  --input-dir videos \
  --weights path/to/best.pt \
  --device mps \
  --conf-threshold 0.20 \
  --frame-step 1 \
  --max-distance 100 \
  --max-missed-frames 10
```

What the pipeline writes:

1. `detections/<video_name>/frame_*.jpg`
2. `detections/manifest.jsonl`
3. `outputs/videos/<video_name>_tracked.mp4`
4. `outputs/rendered_frames/<video_name>/frame_*.jpg`

## Tracking design

The tracker is a simple Kalman-based multi-object tracker built with `filterpy`.

State vector:

```text
[x, y, vx, vy]
```

Where:

1. `x, y` are the bounding-box center coordinates in pixels.
2. `vx, vy` are center velocities in pixels per frame.

Per frame:

1. Every live track predicts its next center using a constant-velocity motion model.
2. New detections are associated to predicted tracks using Hungarian matching on Euclidean center distance.
3. Matched tracks receive a Kalman update.
4. Unmatched tracks continue as predictions for a limited number of missed frames.
5. Trajectories are rendered as 2D polylines over the output frames.

The output video keeps only frames where the detector or tracker says a drone is still present. Detection boxes are drawn in green. Predicted-only boxes are drawn in orange.

## Export the detection dataset to Parquet

After running the detector, export the saved sample frames:

```bash
drone-pipeline export-hf \
  --detections-dir detections \
  --output-file detections/detections.parquet
```

That Parquet file is the artifact you can upload as the Hugging Face dataset deliverable. The manifest includes the image plus the detection metadata for each saved frame.

## Suggested report points for `README.md`

Your final submission README should explicitly cover:

1. The dataset you used and why it matches drone-as-object detection.
2. The detector architecture, checkpoint, image size, and confidence threshold.
3. The Kalman state vector and why you used constant velocity.
4. The process and measurement noise values you chose.
5. Failure cases such as small targets, motion blur, missed detections, and false positives.
6. How many consecutive missed frames the tracker tolerates before dropping a track.

## Known limits

This baseline does not train the detector for you. You still need to choose a drone dataset and either fine-tune or download a checkpoint that recognizes the drone itself.

If your dataset contains only one drone per frame, this tracker still works. If your videos contain multiple drones, the current matching logic will maintain multiple Kalman tracks as long as the detector is consistent enough.
