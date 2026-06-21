# AI Self-Driving Car Stack

This project is a cleaned-up, single-stack version of the Udacity self-driving car reference material.

It includes:
- lane detection with Canny/Sobel + HLS thresholding
- perspective transform + sliding-window lane fit
- PID steering control
- basic perception for traffic lights
- optional vehicle detection using the pretrained HOG + linear SVM assets from the reference repo
- image/video evaluation and annotated output export

## What it runs on

- images: `.jpg`, `.png`, `.bmp`, `.webp`
- videos: `.mp4`, `.mov`, `.avi`
- folders containing any of the above

## Quick start

```bash
pip install -r requirements.txt
python -m sdc_ai.cli --input path/to/image_or_video --output outputs
```

To enable vehicle detection with the pretrained reference model, point to the extracted model folder:

```bash
python -m sdc_ai.cli --input path/to/media --output outputs --vehicle-model-dir path/to/project_5_vehicle_detection/data
```

## Notes

- The lane detector is classical computer vision, which is well aligned with the Canny/Hough and perspective-transform requirements.
- The control block uses PID steering based on lane-center offset.
- The vehicle detector is optional; if the pretrained SVM files are not available, the rest of the stack still runs.

## Files

- `sdc_ai/lane.py` — lane detection and lane overlay
- `sdc_ai/perception.py` — traffic light and vehicle perception
- `sdc_ai/pid.py` — PID controller
- `sdc_ai/pipeline.py` — end-to-end driving stack
- `sdc_ai/cli.py` — command-line runner

## Input assets from the reference zip

This project was built from the structure and code patterns in the uploaded Udacity reference archive:
- `project_1_lane_finding_basic`
- `project_4_advanced_lane_finding`
- `project_5_vehicle_detection`
- `project_9_PID_control`

For best results, use road clips with clear lane markings.

For a quick clip test, add `--max-frames 100 --frame-step 2`.
