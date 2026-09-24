#!/usr/bin/env python3

from pathlib import Path
import json

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT
    / "runs/teachers_r2/"
      "teacher_s_r2_fast_v1/"
      "weights/best.pt"
)

DATA = (
    ROOT
    / "datasets/fasdd_r21_preserve_v1/"
      "data.yaml"
)

PROJECT = (
    ROOT
    / "runs/teachers_r21"
)

NAME = (
    "teacher_s_r21_preserve_v1"
)

REPORT = (
    ROOT
    / "reports/fasdd/r21_preserve_v1"
)

REPORT.mkdir(
    parents=True,
    exist_ok=True,
)

if not MODEL.exists():
    raise FileNotFoundError(MODEL)

if not DATA.exists():
    raise FileNotFoundError(DATA)


config = {
    "base_model":
        str(MODEL),

    "data":
        str(DATA),

    "stage":
        "R2.1 preserve-positive",

    "epochs":
        20,

    "patience":
        7,

    "imgsz":
        768,

    "batch":
        -1,

    "lr0":
        0.0001,

    "lrf":
        0.1,

    "seed":
        42,
}


(
    REPORT
    / "train_config_s_r21_preserve.json"
).write_text(
    json.dumps(
        config,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 100)
print(
    "YOLOv8s — R2.1 "
    "PRESERVE-POSITIVE"
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
    "Max epochs: 20"
)

print(
    "Patience  : 7"
)

print(
    "LR0       : 0.0001"
)

print(
    "Purpose   : recover positive "
    "sensitivity while retaining "
    "R2 hard-negative learning"
)

print("=" * 100)


model = YOLO(
    str(MODEL)
)


model.train(
    data=str(DATA),

    epochs=20,
    patience=7,

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

    # Less destructive augmentation
    # during preservation fine-tuning.
    close_mosaic=5,

    amp=True,
    multi_scale=0.0,

    lr0=0.0001,
    lrf=0.1,

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

    mosaic=0.5,
    mixup=0.0,
    copy_paste=0.0,

    project=str(PROJECT),
    name=NAME,

    exist_ok=False,

    plots=True,
    verbose=True,
)
