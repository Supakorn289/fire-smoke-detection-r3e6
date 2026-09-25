#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime
import json
import gc

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "datasets/bootstrap_v1/data.yaml"

PROJECT = ROOT / "runs/teachers_r0"
REPORT_DIR = ROOT / "reports/teachers_r0"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


TEACHERS = [
    ("teacher_n_r0", "yolov8n.pt"),
    ("teacher_s_r0", "yolov8s.pt"),
    ("teacher_m_r0", "yolov8m.pt"),
]


TRAIN_ARGS = {
    "data": str(DATA),

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    "epochs": 150,
    "patience": 35,

    # 768 ช่วย small-object มากกว่า 640
    "imgsz": 768,

    # ให้ Ultralytics หา batch ตาม VRAM
    "batch": -1,

    "device": 0,
    "workers": 8,

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    "optimizer": "AdamW",
    "lr0": 0.001,
    "weight_decay": 0.0005,

    "cos_lr": True,

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------
    "hsv_h": 0.015,
    "hsv_s": 0.65,
    "hsv_v": 0.40,

    "degrees": 3.0,
    "translate": 0.10,
    "scale": 0.50,
    "shear": 1.0,

    "fliplr": 0.5,
    "flipud": 0.0,

    # Mosaic มีประโยชน์กับ object ที่เล็ก
    "mosaic": 1.0,

    # ไม่ใช้ MixUp สูงเกินไปกับ smoke
    "mixup": 0.05,

    "close_mosaic": 15,

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------
    "amp": True,
    "cache": False,

    "plots": True,
    "save": True,

    "seed": 42,
    "deterministic": True,

    "project": str(PROJECT),
}


def environment():

    print("=" * 72)
    print("ROUND-0 TEACHER TRAINING")
    print("=" * 72)

    if not DATA.exists():
        raise FileNotFoundError(DATA)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    props = torch.cuda.get_device_properties(0)

    print("GPU       :", torch.cuda.get_device_name(0))
    print(
        "VRAM      :",
        f"{props.total_memory / 1024**3:.2f} GB"
    )
    print("Dataset   :", DATA)
    print("Image size:", TRAIN_ARGS["imgsz"])
    print("Epochs    :", TRAIN_ARGS["epochs"])

    print("=" * 72)


def train_one(name, weights):

    run_dir = PROJECT / name
    best = run_dir / "weights/best.pt"

    if best.exists():
        print(f"\nSKIP {name}")
        print(f"Existing: {best}")

        return {
            "name": name,
            "status": "already_complete",
            "best": str(best),
        }

    print()
    print("=" * 72)
    print("START :", name)
    print("BASE  :", weights)
    print("=" * 72)

    model = YOLO(weights)

    args = TRAIN_ARGS.copy()

    args["name"] = name
    args["exist_ok"] = True

    start = datetime.now().isoformat()

    model.train(**args)

    end = datetime.now().isoformat()

    status = (
        "complete"
        if best.exists()
        else "finished_without_best"
    )

    result = {
        "name": name,
        "base_weights": weights,
        "status": status,
        "started": start,
        "finished": end,
        "best": str(best),
    }

    del model

    gc.collect()
    torch.cuda.empty_cache()

    return result


def main():

    environment()

    summary = []

    for name, weights in TEACHERS:

        try:
            result = train_one(name, weights)
            summary.append(result)

        except KeyboardInterrupt:

            print("\nInterrupted by user.")
            print("Existing checkpoints are preserved.")
            break

        except Exception as exc:

            print()
            print(f"ERROR: {name}")
            print(repr(exc))

            summary.append({
                "name": name,
                "status": "error",
                "error": repr(exc),
            })

            gc.collect()
            torch.cuda.empty_cache()

    output = REPORT_DIR / "training_summary.json"

    output.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("ROUND-0 TRAINING SUMMARY")
    print("=" * 72)

    for row in summary:
        print(
            f"{row['name']:15}: "
            f"{row['status']}"
        )

    print()
    print("Report:", output)
    print("=" * 72)


if __name__ == "__main__":
    main()
