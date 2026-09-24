#!/usr/bin/env python3

from pathlib import Path
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT
    / "runs/teachers_r21/"
      "teacher_s_r21_preserve_v1/"
      "weights/best.pt"
)

DATA = (
    ROOT
    / "datasets/fasdd_r3_proxy_v1/"
      "data.yaml"
)

PROJECT = (
    ROOT
    / "runs/teachers_r3"
)

NAME = (
    "teacher_s_r3_proxy_v1"
)

REPORT = (
    ROOT
    / "reports/fasdd/r3_proxy_v1"
)

REPORT.mkdir(
    parents=True,
    exist_ok=True,
)


if not MODEL.exists():
    raise FileNotFoundError(MODEL)

if not DATA.exists():
    raise FileNotFoundError(DATA)


print("=" * 100)
print(
    "YOLOv8s — R3 FINAL PROXY "
    "HARD-NEGATIVE FINE-TUNE"
)
print("=" * 100)

print(
    "Base:",
    MODEL
)

print(
    "Data:",
    DATA
)

print(
    "LR0       : 0.00005"
)

print(
    "Max epochs: 12"
)

print(
    "Patience  : 4"
)

print(
    "save_period: 1"
)

print("=" * 100)


model = YOLO(
    str(MODEL)
)


model.train(
    data=str(DATA),

    epochs=12,
    patience=4,

    batch=-1,
    imgsz=768,

    cache=False,

    device=0,
    workers=6,

    optimizer="AdamW",

    seed=42,
    deterministic=True,

    rect=False,
    cos_lr=True,

    amp=True,
    multi_scale=0.0,

    lr0=0.00005,
    lrf=0.2,

    momentum=0.937,
    weight_decay=0.0005,

    warmup_epochs=1.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    box=7.5,
    cls=0.5,
    dfl=1.5,

    hsv_h=0.015,
    hsv_s=0.65,
    hsv_v=0.4,

    degrees=3.0,
    translate=0.10,
    scale=0.50,
    shear=1.0,
    perspective=0.0,

    flipud=0.0,
    fliplr=0.5,

    # Preservation-oriented augmentation
    mosaic=0.25,
    close_mosaic=4,
    mixup=0.0,
    copy_paste=0.0,

    project=str(PROJECT),
    name=NAME,

    exist_ok=False,

    # Keep every epoch.
    # If best.pt is not the best
    # positive/negative Pareto point,
    # we can select another epoch
    # WITHOUT retraining.
    save_period=1,

    plots=True,
    verbose=True,
)
