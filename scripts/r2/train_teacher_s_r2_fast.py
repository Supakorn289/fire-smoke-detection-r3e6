#!/usr/bin/env python3

from pathlib import Path
import json

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT
    / "runs/teachers_r1/"
    "teacher_s_r1/weights/best.pt"
)

DATA = (
    ROOT
    / "datasets/fasdd_r2_fast_v1/"
    "data.yaml"
)

PROJECT = (
    ROOT
    / "runs/teachers_r2"
)

NAME = (
    "teacher_s_r2_fast_v1"
)

REPORT = (
    ROOT
    / "reports/fasdd/r2_fast_v1"
)

REPORT.mkdir(
    parents=True,
    exist_ok=True,
)


if not MODEL.exists():
    raise FileNotFoundError(
        MODEL
    )

if not DATA.exists():
    raise FileNotFoundError(
        DATA
    )


config = {
    "base_model":
        str(MODEL),

    "data":
        str(DATA),

    "model":
        "YOLOv8s / R1 best.pt",

    "stage":
        "R2 hard-negative fine-tuning",

    "epochs":
        60,

    "patience":
        15,

    "imgsz":
        768,

    "batch":
        -1,

    "device":
        0,

    "workers":
        6,

    "optimizer":
        "AdamW",

    "lr0":
        0.0003,

    "lrf":
        0.01,

    "seed":
        42,
}


(
    REPORT
    / "train_config_s_r2_fast_v1.json"
).write_text(
    json.dumps(
        config,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 100)
print(
    "YOLOv8s — R2 FAST "
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
    "Epochs: 60"
)

print(
    "LR0: 0.0003"
)

print(
    "Purpose: preserve positive "
    "recall while reducing false alarms"
)

print("=" * 100)


model = YOLO(
    str(MODEL)
)


model.train(
    data=str(DATA),

    epochs=60,
    patience=15,

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
    close_mosaic=10,

    amp=True,
    multi_scale=0.0,

    lr0=0.0003,
    lrf=0.01,

    momentum=0.937,
    weight_decay=0.0005,

    warmup_epochs=2.0,
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

    mosaic=1.0,
    mixup=0.05,
    copy_paste=0.0,

    project=str(PROJECT),
    name=NAME,

    exist_ok=False,

    plots=True,
    verbose=True,
)
