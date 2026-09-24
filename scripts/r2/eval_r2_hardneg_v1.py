#!/usr/bin/env python3

from pathlib import Path
import csv
import json

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT
    / "runs/teachers_r2/"
    "teacher_s_r2_fast_v1/"
    "weights/best.pt"
)

VAL_CSV = (
    ROOT
    / "reports/negative_v1/"
    "final_v1/hardneg_val.csv"
)

BASELINE_CSV = (
    ROOT
    / "reports/negative_v1/"
    "final_v1/"
    "r1_hardneg_val_baseline.csv"
)

OUT = (
    ROOT
    / "reports/fasdd/"
    "r2_fast_v1"
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

EXPECTED = 94


def load_images():

    if not VAL_CSV.exists():
        raise FileNotFoundError(
            VAL_CSV
        )

    paths = []

    with VAL_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            p = Path(
                row["image"]
            )

            if not p.exists():
                raise FileNotFoundError(
                    p
                )

            paths.append(p)

    if len(paths) != EXPECTED:
        raise RuntimeError(
            f"Expected {EXPECTED}, "
            f"got {len(paths)}"
        )

    return paths


def load_baseline():

    rows = {}

    if not BASELINE_CSV.exists():
        return rows

    with BASELINE_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        for row in csv.DictReader(f):

            if row["model"] != "s":
                continue

            threshold = float(
                row["threshold"]
            )

            rows[
                threshold
            ] = {
                "false_alert_images":
                    int(
                        row[
                            "false_alert_images"
                        ]
                    ),

                "image_fpr":
                    float(
                        row[
                            "image_fpr"
                        ]
                    ),
            }

    return rows


if not MODEL.exists():
    raise FileNotFoundError(
        MODEL
    )


images = load_images()
baseline = load_baseline()

print("=" * 92)
print(
    "R2-S HELD-OUT HARD-NEGATIVE "
    "EVALUATION"
)
print("=" * 92)

print(
    "Model :",
    MODEL
)

print(
    "Images:",
    len(images)
)

print()


model = YOLO(
    str(MODEL)
)


stats = {
    threshold: {
        "images":
            len(images),

        "false_alert_images":
            0,

        "fire_false_alert_images":
            0,

        "smoke_false_alert_images":
            0,

        "prediction_boxes":
            0,

        "fire_boxes":
            0,

        "smoke_boxes":
            0,
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

        confs = (
            result.boxes.conf
            .detach()
            .cpu()
            .tolist()
        )

        classes = (
            result.boxes.cls
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
            (cls, conf)
            for cls, conf
            in detections
            if conf >= threshold
        ]

        fire = [
            x
            for x in selected
            if x[0] == 0
        ]

        smoke = [
            x
            for x in selected
            if x[0] == 1
        ]


        s = stats[
            threshold
        ]

        if selected:
            s[
                "false_alert_images"
            ] += 1

        if fire:
            s[
                "fire_false_alert_images"
            ] += 1

        if smoke:
            s[
                "smoke_false_alert_images"
            ] += 1


        s[
            "prediction_boxes"
        ] += len(
            selected
        )

        s[
            "fire_boxes"
        ] += len(
            fire
        )

        s[
            "smoke_boxes"
        ] += len(
            smoke
        )


    if (
        index % 10 == 0
        or index == len(images)
    ):

        print(
            f"\rEvaluated "
            f"{index}/{len(images)}",
            end="",
            flush=True,
        )


print()
print()


rows = []

print("=" * 92)
print(
    "R1-S → R2-S HARD-NEGATIVE "
    "COMPARISON"
)
print("=" * 92)

print(
    f"{'CONF':<8}"
    f"{'R1 FPR':>12}"
    f"{'R2 FPR':>12}"
    f"{'Δ FPR':>12}"
    f"{'R2 ALERT':>12}"
    f"{'BOXES':>10}"
)

print("-" * 92)


for threshold in THRESHOLDS:

    s = stats[
        threshold
    ]

    total = s[
        "images"
    ]

    r2_fpr = (
        s[
            "false_alert_images"
        ]
        / total
    )

    fire_fpr = (
        s[
            "fire_false_alert_images"
        ]
        / total
    )

    smoke_fpr = (
        s[
            "smoke_false_alert_images"
        ]
        / total
    )


    r1_fpr = None
    delta = None

    if threshold in baseline:

        r1_fpr = baseline[
            threshold
        ][
            "image_fpr"
        ]

        delta = (
            r2_fpr
            - r1_fpr
        )


    print(
        f"{threshold:<8.2f}"
        f"{r1_fpr if r1_fpr is not None else float('nan'):>12.4f}"
        f"{r2_fpr:>12.4f}"
        f"{delta if delta is not None else float('nan'):>12.4f}"
        f"{s['false_alert_images']:>8}/{total:<3}"
        f"{s['prediction_boxes']:>10}"
    )


    rows.append({
        "model":
            "s_r2_fast_v1",

        "threshold":
            threshold,

        "images":
            total,

        "false_alert_images":
            s[
                "false_alert_images"
            ],

        "image_fpr":
            r2_fpr,

        "fire_false_alert_images":
            s[
                "fire_false_alert_images"
            ],

        "fire_image_fpr":
            fire_fpr,

        "smoke_false_alert_images":
            s[
                "smoke_false_alert_images"
            ],

        "smoke_image_fpr":
            smoke_fpr,

        "prediction_boxes":
            s[
                "prediction_boxes"
            ],

        "fire_boxes":
            s[
                "fire_boxes"
            ],

        "smoke_boxes":
            s[
                "smoke_boxes"
            ],

        "r1_s_image_fpr":
            r1_fpr,

        "delta_fpr":
            delta,
    })


csv_path = (
    OUT
    / "r2_s_hardneg_val.csv"
)

with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        rows
    )


json_path = (
    OUT
    / "r2_s_hardneg_val.json"
)

json_path.write_text(
    json.dumps(
        rows,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print(
    "CSV :",
    csv_path
)

print(
    "JSON:",
    json_path
)

print()
print(
    "RESULT: R2-S HARDNEG "
    "EVALUATION PASS"
)

print("=" * 92)
