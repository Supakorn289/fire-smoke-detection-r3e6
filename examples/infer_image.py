#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


IMGSZ = 768
CONF = 0.25
IOU = 0.70
MAX_DET = 300


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run R3-E6 Fire/Smoke inference."
    )

    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--model",
        default="fire_smoke_r3_e6_final.pt",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--project",
        default="runs/inference",
    )
    parser.add_argument(
        "--name",
        default="r3e6",
    )
    parser.add_argument(
        "--show",
        action="store_true",
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            "Download fire_smoke_r3_e6_final.pt "
            "from GitHub Release v1.0.0-r3e6."
        )

    model = YOLO(str(model_path))

    project_dir = Path(args.project).resolve()

    kwargs = {
        "source": args.source,
        "imgsz": IMGSZ,
        "conf": CONF,
        "iou": IOU,
        "max_det": MAX_DET,
        "save": True,
        "show": args.show,
        "project": str(project_dir),
        "name": args.name,
    }

    if args.device is not None:
        kwargs["device"] = args.device

    results = model.predict(**kwargs)

    total = sum(len(result.boxes) for result in results)

    print(f"Results: {len(results)}")
    print(f"Detections: {total}")

    if results:
        print(f"Saved to: {results[0].save_dir}")


if __name__ == "__main__":
    main()
