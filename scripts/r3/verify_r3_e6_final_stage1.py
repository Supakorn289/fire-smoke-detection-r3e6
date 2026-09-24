#!/usr/bin/env python3

from pathlib import Path
import csv
import gc

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT
    / "runs/teachers_r3/"
      "teacher_s_r3_proxy_v1/"
      "weights/epoch6.pt"
)

SMALL_SMOKE_DATA = (
    ROOT
    / "datasets/fasdd_r1_v1/"
      "data_small_smoke.yaml"
)

HARDNEG_CSV = (
    ROOT
    / "reports/negative_v1/"
      "final_v1/hardneg_val.csv"
)

OUT = (
    ROOT
    / "reports/final_selection_v1/"
      "r3_e6_stage1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

THRESHOLDS = [
    0.10,
    0.25,
    0.50,
]

EXPECTED_HARDNEG = 94


for path in [
    MODEL,
    SMALL_SMOKE_DATA,
    HARDNEG_CSV,
]:
    if not path.exists():
        raise FileNotFoundError(path)


print("=" * 100)
print("R3-E6 FINAL VERIFICATION — STAGE 1")
print("=" * 100)

print(
    "Model:",
    MODEL
)

print(
    "Operating confidence candidate: 0.25"
)

print("=" * 100)


model = YOLO(
    str(MODEL)
)


# ============================================================
# A) SMALL-SMOKE SCENE 222
# ============================================================

print()
print("=" * 100)
print(
    "A) SMALL-SMOKE SCENE DIAGNOSTIC"
)
print("=" * 100)


metrics = model.val(
    data=str(
        SMALL_SMOKE_DATA
    ),

    imgsz=768,
    batch=2,

    device=0,
    workers=2,

    plots=False,
    verbose=True,

    project=str(
        OUT
        / "small_smoke_scene"
    ),

    name="r3_e6_small_smoke",
    exist_ok=True,
)


print()
print("-" * 100)
print("SMALL-SMOKE SUMMARY")
print("-" * 100)

print(
    f"ALL "
    f"P={metrics.box.mp:.4f} "
    f"R={metrics.box.mr:.4f} "
    f"mAP50={metrics.box.map50:.4f} "
    f"mAP50-95={metrics.box.map:.4f}"
)

print(
    "mAP50-95/class:",
    [
        round(
            float(x),
            4,
        )
        for x in metrics.box.maps
    ],
)

print(
    "\nUse the per-class table above "
    "for FIRE/SMOKE P, R and mAP."
)


del metrics

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# B) LEGACY HARD NEGATIVE 94
# ============================================================

print()
print("=" * 100)
print(
    "B) LEGACY HUMAN-VERIFIED "
    "HARD NEGATIVE — 94 IMAGES"
)
print("=" * 100)


with HARDNEG_CSV.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    rows = list(
        csv.DictReader(f)
    )


if len(rows) != EXPECTED_HARDNEG:

    raise RuntimeError(
        f"Expected {EXPECTED_HARDNEG} images, "
        f"got {len(rows)}"
    )


images = []

for row in rows:

    image = Path(
        row["image"]
    )

    if not image.exists():
        raise FileNotFoundError(
            image
        )

    images.append(
        image
    )


stats = {
    threshold: {
        "alerts": 0,
        "fire_alerts": 0,
        "smoke_alerts": 0,
        "boxes": 0,
    }
    for threshold in THRESHOLDS
}


for index, image in enumerate(
    images,
    start=1,
):

    with torch.inference_mode():

        result = model.predict(
            source=str(image),

            imgsz=768,

            conf=0.05,
            iou=0.70,
            max_det=100,

            device=0,
            batch=1,

            verbose=False,
        )[0]


    detections = []


    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        classes = (
            result.boxes.cls
            .detach()
            .cpu()
            .tolist()
        )

        confs = (
            result.boxes.conf
            .detach()
            .cpu()
            .tolist()
        )

        detections = [
            (
                int(cls),
                float(conf),
            )
            for cls, conf
            in zip(
                classes,
                confs,
            )
        ]


    for threshold in THRESHOLDS:

        selected = [
            (
                cls,
                conf,
            )
            for cls, conf
            in detections
            if conf >= threshold
        ]


        if selected:

            stats[
                threshold
            ]["alerts"] += 1


        if any(
            cls == 0
            for cls, conf
            in selected
        ):

            stats[
                threshold
            ]["fire_alerts"] += 1


        if any(
            cls == 1
            for cls, conf
            in selected
        ):

            stats[
                threshold
            ]["smoke_alerts"] += 1


        stats[
            threshold
        ]["boxes"] += len(
            selected
        )


    del result


    if (
        index % 10 == 0
        or index == EXPECTED_HARDNEG
    ):

        print(
            f"\rHardNeg "
            f"{index}/{EXPECTED_HARDNEG}",
            end="",
            flush=True,
        )


print("\n")

print("-" * 100)
print(
    "LEGACY HARD-NEGATIVE FPR"
)
print("-" * 100)

print(
    f"{'CONF':<8}"
    f"{'ALERTS':>12}"
    f"{'FPR':>12}"
    f"{'FIRE':>12}"
    f"{'SMOKE':>12}"
    f"{'BOXES':>12}"
)

print("-" * 100)


for threshold in THRESHOLDS:

    s = stats[
        threshold
    ]

    fpr = (
        s["alerts"]
        / EXPECTED_HARDNEG
    )

    print(
        f"{threshold:<8.2f}"
        f"{s['alerts']:>8}/94"
        f"{fpr:>12.4f}"
        f"{s['fire_alerts']:>12}"
        f"{s['smoke_alerts']:>12}"
        f"{s['boxes']:>12}"
    )


print()
print("=" * 100)
print(
    "REFERENCE — R2.1 LEGACY HARDNEG"
)
print("=" * 100)

print(
    "conf=.10  FPR=0.7021"
)

print(
    "conf=.25  FPR=0.4043"
)

print(
    "conf=.50  FPR=0.1596"
)

print()
print(
    "RESULT: R3-E6 FINAL "
    "STAGE-1 VERIFICATION PASS"
)

print("=" * 100)
