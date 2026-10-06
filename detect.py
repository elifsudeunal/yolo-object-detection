"""Pretrained YOLOv5u object detection and analytics pipeline."""

import argparse
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SAMPLE_IMAGES = [
    "https://ultralytics.com/images/bus.jpg",
    "https://ultralytics.com/images/zidane.jpg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv5u detection and analytics pipeline")
    parser.add_argument("--input", type=Path, default=Path("data"), help="folder with input images")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="folder for results")
    parser.add_argument("--conf", type=float, default=0.40, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--model", type=str, default="yolov5su.pt", help="pretrained weights")
    return parser.parse_args()


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def download_samples(folder: Path) -> None:
    """Download the official Ultralytics example images into an empty input folder."""
    folder.mkdir(parents=True, exist_ok=True)
    for url in SAMPLE_IMAGES:
        target = folder / url.rsplit("/", 1)[-1]
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, target)


def sync(device: str) -> None:
    # CUDA kernels run asynchronously; synchronize so wall-clock time is accurate.
    if device == "cuda":
        torch.cuda.synchronize()


def detect(model: YOLO, image_path: Path, conf: float, imgsz: int, device: str):
    """Run inference on one image. Returns (result, wall-clock time in ms)."""
    sync(device)
    start = time.perf_counter()
    result = model.predict(str(image_path), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
    sync(device)
    return result, (time.perf_counter() - start) * 1000


def analyze(result, image_name: str, elapsed_ms: float) -> dict:
    """Turn a YOLO result into a structured record."""
    class_ids = result.boxes.cls.int().tolist()
    confidences = result.boxes.conf.tolist()
    counts = Counter(result.names[i] for i in class_ids)
    return {
        "image": image_name,
        "total_objects": len(class_ids),
        "counts": dict(counts.most_common()),
        "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "inference_time_ms": round(elapsed_ms, 2),  # preprocess + model + postprocess
        "model_forward_ms": round(result.speed["inference"], 2),  # network only (Ultralytics)
    }


def save_annotated(result, path: Path) -> None:
    annotated_bgr = result.plot()  # numpy array with boxes, labels, confidences
    Image.fromarray(annotated_bgr[..., ::-1]).save(path)  # BGR -> RGB


def print_summary(device: str, records: list[dict], skipped: list[str]) -> None:
    total_counts: Counter = Counter()
    for r in records:
        total_counts.update(r["counts"])
    avg_ms = sum(r["inference_time_ms"] for r in records) / len(records)

    print("\n" + "=" * 40)
    print("Detection Summary")
    print("=" * 40)
    print(f"Device: {device.upper()}")
    print(f"Images processed: {len(records)}")
    if skipped:
        print(f"Images skipped: {len(skipped)} ({', '.join(skipped)})")
    print(f"Total objects detected: {sum(total_counts.values())}")
    print("\nClass counts:")
    for name, n in total_counts.most_common():
        print(f"  {name}: {n}")
    print(f"\nAverage inference time: {avg_ms:.1f} ms")
    print("=" * 40)


def main() -> int:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Input handling: default folder is created and filled with samples if empty.
    if not args.input.exists():
        if args.input == Path("data"):
            args.input.mkdir()
        else:
            print(f"Input folder not found: {args.input}")
            return 1
    images = find_images(args.input)
    if not images:
        try:
            download_samples(args.input)
        except OSError as e:
            print(f"Could not download sample images: {e}")
            return 1
        images = find_images(args.input)

    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # Warm-up run (not measured): excludes CUDA initialization from timings.
    detect(model, images[0], args.conf, args.imgsz, device)

    records, skipped = [], []
    for path in images:
        try:
            result, elapsed_ms = detect(model, path, args.conf, args.imgsz, device)
            save_annotated(result, args.output / f"detected_{path.stem}.jpg")
            record = analyze(result, path.name, elapsed_ms)
        except Exception as e:  # keep the batch going if one image is unreadable
            print(f"Skipping {path.name}: {e}")
            skipped.append(path.name)
            continue
        records.append(record)
        print(f"{path.name}: {record['total_objects']} objects, {record['inference_time_ms']:.1f} ms")

    if not records:
        print("No images could be processed.")
        return 1

    report = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "model": args.model,
        "confidence_threshold": args.conf,
        "image_size": args.imgsz,
        "images_processed": len(records),
        "average_inference_time_ms": round(sum(r["inference_time_ms"] for r in records) / len(records), 2),
        "results": records,
    }
    (args.output / "detection_summary.json").write_text(json.dumps(report, indent=4))

    print_summary(device, records, skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())