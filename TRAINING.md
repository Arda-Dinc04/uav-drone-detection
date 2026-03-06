# Drone Fine-Tuning Guide

This is the shortest path to a drone-specific detector that is accurate enough for the assignment and still practical for personal projects.

## Recommended training target

Use a single-class detector:

```text
class 0 = drone
```

Recommended starting checkpoint:

```text
yolo11s.pt
```

Reasoning:

1. `yolo11n.pt` is fast but more likely to miss very small drones.
2. `yolo11s.pt` is still lightweight enough for Colab and usually gives a better tradeoff for small-object detection.
3. You do not need `m`, `l`, or `x` first. That is usually wasted training time for this assignment.

## Exact dataset structure

Use YOLO detection format. Your dataset should look like this in Colab:

```text
/content/drone_dataset/
  images/
    train/
      frame_000001.jpg
      ...
    val/
      frame_000101.jpg
      ...
    test/
      frame_000201.jpg
      ...
  labels/
    train/
      frame_000001.txt
      ...
    val/
      frame_000101.txt
      ...
    test/
      frame_000201.txt
      ...
```

Each label file must have one line per box:

```text
0 x_center y_center width height
```

Example:

```text
0 0.451562 0.188889 0.037500 0.050000
```

Rules:

1. Coordinates must be normalized to `[0, 1]`.
2. The image file and label file names must match.
3. Empty label files are allowed for negative examples.
4. Keep only the drone class if your goal is a clean assignment model.

## Colab setup

Create a new Colab notebook and run:

```bash
!nvidia-smi
!pip install ultralytics
```

Upload or unzip your dataset so it lands at `/content/drone_dataset`, then copy the repo dataset config or recreate it in Colab as `/content/drone_dataset.yaml`.

Use this YAML content:

```yaml
path: /content/drone_dataset
train: images/train
val: images/val
test: images/test

names:
  0: drone
```

That same template already exists locally at [configs/drone_dataset.colab.yaml](/Users/ardadinc/Documents/New%20project/configs/drone_dataset.colab.yaml).

## Fast training recipe

This is the recommended first run:

```bash
!yolo detect train \
  model=yolo11s.pt \
  data=/content/drone_dataset.yaml \
  epochs=40 \
  imgsz=960 \
  batch=16 \
  patience=12 \
  close_mosaic=10
```

Why these settings:

1. `imgsz=960` helps because the drone occupies very few pixels.
2. `epochs=40` is enough for a first serious pass without dragging training out.
3. `patience=12` keeps the run from wasting time if validation stalls.
4. `close_mosaic=10` reduces late-stage augmentation noise.

If Colab memory is tight:

1. Drop `batch` from `16` to `8`.
2. If needed, drop `imgsz` from `960` to `832`.

## Repo training script

If you want the same setup from this repo instead of typing the long command manually:

```bash
python scripts/train_drone_yolo.py \
  --data /content/drone_dataset.yaml \
  --model yolo11s.pt \
  --epochs 40 \
  --imgsz 960 \
  --batch 16
```

Script location: [scripts/train_drone_yolo.py](/Users/ardadinc/Documents/New%20project/scripts/train_drone_yolo.py)

## After training

Your best checkpoint will usually end up here:

```text
runs/detect/train/weights/best.pt
```

Or under the custom project/name path if you use the repo script.

Download `best.pt` and run your tracker locally:

```bash
drone-pipeline process \
  --input-dir videos \
  --weights path/to/best.pt \
  --target-label drone \
  --output-dir outputs \
  --detections-dir detections \
  --conf-threshold 0.20
```

## Dataset sourcing advice

Pick a dataset where the annotated object is the drone itself.

Good practical sources to inspect:

1. Roboflow Universe drone-detection projects.
2. Hugging Face datasets that expose YOLO or COCO annotations for UAV-as-object detection.

Avoid using a dataset just because it mentions drones. Many such datasets are actually for objects seen from onboard drone cameras.
