# UAV Drone Detection and Tracking

This repository contains my submission for the UAV drone detection and tracking assignment. The pipeline detects the drone in video, tracks it over time with a Kalman filter, saves detection-positive frames, and renders output videos with bounding boxes, a 2D trajectory polyline, and a smoothed heading vector.

## Submission Links

Add the final public links here before submitting:

1. Hugging Face dataset: `PASTE_HF_DATASET_LINK_HERE`
2. YouTube video for `drone_video_1_tracked.mp4`: `https://youtu.be/43nhXW6dSFE`
3. YouTube video for `drone_video_2_tracked.mp4`: `https://youtu.be/hZzecY-NcCc`

## Deliverables

Included in this repository:

1. Detection-positive sample frames in [detections](/Users/ardadinc/Documents/New%20project/detections) with a Parquet export at [detections.parquet](/Users/ardadinc/Documents/New%20project/detections/detections.parquet)
2. Output tracking videos for the two official course videos in [outputs/videos](/Users/ardadinc/Documents/New%20project/outputs/videos)
3. Additional benchmark tracking videos on extra test cases in [outputs_extra/videos](/Users/ardadinc/Documents/New%20project/outputs_extra/videos)
4. A consolidated set of presentation-ready tracked videos in [final_videos](/Users/ardadinc/Documents/New%20project/final_videos)
5. Source videos from the assignment in [videos](/Users/ardadinc/Documents/New%20project/videos) and extra benchmark clips in [videos_extra](/Users/ardadinc/Documents/New%20project/videos_extra)
6. The full code pipeline in [drone_tracking](/Users/ardadinc/Documents/New%20project/drone_tracking)

## Detector Choice

The final detector used for the submission run is a pretrained Hugging Face checkpoint:

1. Model: `AeroYOLO_best.pt`
2. Source: `QuincySorrentino/AeroYOLO`
3. Detector family: Ultralytics YOLO
4. Class map: `aircraft`, `drone`, `helicopter`

I used this model because the generic COCO checkpoint (`yolo11n.pt`) consistently labeled the flying target as `airplane`, `bird`, or `kite` instead of `drone`. AeroYOLO is not perfect, but it is a stronger drone-focused starting point and produced consistent `drone` detections on the course video and several additional test clips.

Final inference settings used for submission outputs:

1. Confidence threshold: `0.08`
2. Target label kept for the assignment output: `drone`
3. Input handling: every `.mp4` in a directory is processed automatically
4. For weak official clips, I also tested a lower threshold of `0.08` to recover additional drone detections

## Dataset Choice

For training exploration, I evaluated the `ChinnaSAMY1/drone-detection-dataset` Hugging Face dataset because it contains labeled drone bounding boxes and is distributed in Parquet format, which makes it straightforward to load and convert.

I explicitly avoided using VisDrone as the primary detector-training source because it is mainly a benchmark for objects seen from drone-mounted cameras, not the drone itself as the target object.

I did not complete a full custom fine-tune for the final submission because the full training run was too heavy for the available Colab budget and runtime. Instead, I switched to a pretrained drone-specific detector and focused on validating the end-to-end detection and tracking pipeline.

## Kalman Filter Design

The tracker is implemented with `filterpy` in [tracker.py](/Users/ardadinc/Documents/New%20project/drone_tracking/tracker.py).

State vector:

```text
[x, y, vx, vy]
```

Where:

1. `x, y` are the pixel coordinates of the bounding-box center
2. `vx, vy` are the center velocities in pixels per frame

Tracking loop:

1. Predict the next state with a constant-velocity motion model
2. Associate detections to predicted tracks with Hungarian matching on Euclidean distance
3. Update matched tracks with the current detector observation
4. Keep unmatched tracks alive briefly so the trajectory does not collapse on short missed detections

Tracker parameters used in the current pipeline:

1. `max_distance = 80`
2. `max_missed_frames = 8`
3. `min_hits = 2`
4. `process_noise = 5.0`
5. `measurement_noise = 20.0`

## Trajectory and Heading Visualization

Each output frame contains:

1. The detector bounding box
2. The Kalman-filtered track ID
3. A 2D trajectory polyline showing the history of the tracked center
4. A smoothed heading vector showing where the drone is moving next

The heading vector was stabilized so it no longer expands aggressively when the raw velocity estimate spikes. The current version uses recent trajectory motion first and falls back to Kalman velocity only when needed.

## Final Test Set

### Course videos

The two official course videos used for submission are:

1. [drone_video_1.mp4](/Users/ardadinc/Documents/New%20project/videos/drone_video_1.mp4)
2. [drone_video_2.mp4](/Users/ardadinc/Documents/New%20project/videos/drone_video_2.mp4)

### Extra benchmark videos

To go beyond the provided clip, I added extra drone-visible test cases in [videos_extra](/Users/ardadinc/Documents/New%20project/videos_extra):

1. `mixkit_close_outdoor_drone.mp4`
2. `mixkit_small_drone_sky.mp4`
3. `mixkit_drone_city_spinning.mp4`
4. `mixkit_drone_circles_abandoned_space.mp4`

These extra clips were chosen to test:

1. close-up drone appearance
2. small-object detection against the sky
3. urban clutter
4. circular and faster motion

## Results Summary

### Official video results

Final tracked outputs:

1. [drone_video_1_tracked.mp4](/Users/ardadinc/Documents/New%20project/outputs/videos/drone_video_1_tracked.mp4)
2. [drone_video_2_tracked.mp4](/Users/ardadinc/Documents/New%20project/outputs/videos/drone_video_2_tracked.mp4)

Course-video result summary from local runs:

1. `drone_video_1.mp4`: strong detection and stable tracking on the provided clip
2. `drone_video_2.mp4`: more challenging; additional segment-level testing was used to understand where the detector succeeded and failed
3. Output videos contain only frames where the detector/tracker kept the drone alive, matching the assignment requirement

### Extra benchmark result

Tracked outputs are saved in [outputs_extra/videos](/Users/ardadinc/Documents/New%20project/outputs_extra/videos), and a small curated set is also copied into [final_videos](/Users/ardadinc/Documents/New%20project/final_videos).

Observed behavior from earlier evaluation of the same AeroYOLO checkpoint:

1. `mixkit_drone_circles_abandoned_space.mp4`: strongest extra-case result, with stable drone detections during circular motion
2. `mixkit_drone_city_spinning.mp4`: moderate result, drone detected but not on every frame
3. `mixkit_small_drone_sky.mp4`: partial success, but class confusion appears when the drone becomes small
4. `mixkit_close_outdoor_drone.mp4`: the model often sees the target but may classify it as `helicopter` or `aircraft` instead of `drone`

These extra tests were useful because they exposed a real failure mode that the assignment clip alone would not show: the detector is reasonably good at finding flying objects, but not always semantically stable enough to label them as `drone`.

## Failure Cases

The main failure modes I observed are:

1. Small drones at long range can be confused with `aircraft` or `helicopter`
2. Some close-up consumer-drone footage is detected, but mislabeled
3. Fast motion and background clutter can reduce detector consistency
4. When detections drop out, the Kalman tracker can bridge short gaps, but it cannot recover indefinitely without new observations

How the tracker handles misses:

1. It continues predicting for up to `8` missed frames
2. The trajectory remains visible during these short gaps
3. Tracks are removed once the miss limit is exceeded

## Commands Used

Run the provided course videos:

```bash
drone-pipeline process \
  --input-dir videos \
  --weights pretrained_weights/AeroYOLO_best.pt \
  --target-label drone \
  --output-dir outputs \
  --detections-dir detections \
  --conf-threshold 0.08
```

Run the extra benchmark set:

```bash
drone-pipeline process \
  --input-dir videos_extra \
  --weights pretrained_weights/AeroYOLO_best.pt \
  --target-label drone \
  --output-dir outputs_extra \
  --detections-dir detections_extra \
  --conf-threshold 0.20
```

Export the detection-positive frames to Parquet:

```bash
drone-pipeline export-hf \
  --detections-dir detections \
  --output-file detections/detections.parquet
```

## Repository Structure

The final submission-relevant folders are:

```text
drone_tracking/
videos/
videos_extra/
detections/
outputs/
outputs_extra/
README.md
pyproject.toml
```

## Next Improvements

If I continued this project beyond the submission, the next changes would be:

1. Evaluate a stronger single-class drone model to reduce `drone` vs `helicopter` confusion
2. Fine-tune on a smaller but more targeted drone-as-object dataset
3. Add a class-remapping evaluation mode for exploratory testing
4. Add simple quantitative metrics over hand-labeled benchmark clips
