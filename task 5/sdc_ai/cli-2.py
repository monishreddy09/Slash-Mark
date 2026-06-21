from __future__ import annotations

import argparse
from pathlib import Path
import csv

import cv2

from .pipeline import AutonomousDrivingSystem


def iter_media(path: Path):
    if path.is_dir():
        for p in sorted(path.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".mp4", ".mov", ".avi"}:
                yield p
    else:
        yield path


def process_image(model: AutonomousDrivingSystem, src: Path, out_dir: Path):
    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"Could not read image: {src}")
    out, meta = model.process_frame(img)
    out_path = out_dir / f"{src.stem}_annotated.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return out_path, meta


def process_video(model: AutonomousDrivingSystem, src: Path, out_dir: Path, max_frames=None, frame_step=1):
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_dir / f"{src.stem}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    summary = []
    frame_idx = 0
    written_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, frame_step) != 0:
            frame_idx += 1
            continue
        out, meta = model.process_frame(frame)
        writer.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        summary.append(meta)
        written_idx += 1
        frame_idx += 1
        if max_frames is not None and written_idx >= max_frames:
            break

    cap.release()
    writer.release()

    return out_path, summary


def main():
    parser = argparse.ArgumentParser(description="Autonomous driving demo: lane detection + perception + PID control")
    parser.add_argument("--input", required=True, help="Image, video, or folder containing media")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--vehicle-model-dir", default=None, help="Directory with svm_trained.pickle, feature_scaler.pickle, feat_extraction_params.pickle")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit the number of video frames processed")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth video frame")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AutonomousDrivingSystem(Path(args.vehicle_model_dir) if args.vehicle_model_dir else None)

    results = []
    for media in iter_media(input_path):
        if media.suffix.lower() in {".mp4", ".mov", ".avi"}:
            out_path, summary = process_video(model, media, output_dir, max_frames=args.max_frames, frame_step=args.frame_step)
            avg_steer = sum(m["steering"] for m in summary) / max(1, len(summary))
            avg_throttle = sum(m["throttle"] for m in summary) / max(1, len(summary))
            results.append([media.name, str(out_path), len(summary), avg_steer, avg_throttle])
        else:
            out_path, meta = process_image(model, media, output_dir)
            results.append([media.name, str(out_path), 1, meta["steering"], meta["throttle"]])

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "output", "frames", "avg_steering", "avg_throttle"])
        writer.writerows(results)

    print(f"Saved results to {output_dir}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()
