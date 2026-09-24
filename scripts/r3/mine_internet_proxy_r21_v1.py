#!/usr/bin/env python3

from pathlib import Path
import csv
import gc
import html
import json
import os
import shutil

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT
    / "runs/teachers_r21/"
      "teacher_s_r21_preserve_v1/"
      "weights/best.pt"
)

METADATA = (
    ROOT
    / "reports/internet_proxy_negative_v1/"
      "internet_proxy_v1.json"
)

OUT = (
    ROOT
    / "reports/internet_proxy_negative_v1/"
      "r21_mining"
)

REVIEW = (
    OUT
    / "review_conf025"
)

EXPECTED = 2000

PREDICTION_FLOOR = 0.05

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
]


def severity(conf):

    if conf >= 0.75:
        return "EXTREME"

    if conf >= 0.50:
        return "HARD"

    if conf >= 0.25:
        return "MODERATE"

    if conf >= 0.10:
        return "LOW"

    return "NONE"


if not MODEL.exists():
    raise FileNotFoundError(MODEL)

if not METADATA.exists():
    raise FileNotFoundError(METADATA)


rows_meta = json.loads(
    METADATA.read_text(
        encoding="utf-8"
    )
)


if len(rows_meta) != EXPECTED:

    raise RuntimeError(
        f"Expected {EXPECTED} proxy images, "
        f"got {len(rows_meta)}"
    )


# ============================================================
# Validate pool
# ============================================================

seen = set()

for row in rows_meta:

    image = Path(
        row["image"]
    )

    if not image.exists():
        raise FileNotFoundError(
            image
        )

    digest = row[
        "sha256"
    ]

    if digest in seen:
        raise RuntimeError(
            f"Duplicate SHA256: {digest}"
        )

    seen.add(
        digest
    )


if OUT.exists():

    shutil.rmtree(
        OUT
    )


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

REVIEW.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 100)
print(
    "R2.1-S INTERNET PROXY MINING V1"
)
print("=" * 100)

print(
    "Model :",
    MODEL
)

print(
    "Images:",
    len(rows_meta)
)

print(
    "Prediction floor:",
    PREDICTION_FLOOR
)

print(
    "Mode  : one image at a time"
)

print("=" * 100)


model = YOLO(
    str(MODEL)
)


results_rows = []


# ============================================================
# INFERENCE
# ============================================================

for index, meta in enumerate(
    rows_meta,
    start=1,
):

    image = Path(
        meta["image"]
    )


    with torch.inference_mode():

        result = model.predict(
            source=str(image),
            imgsz=768,
            conf=PREDICTION_FLOOR,
            iou=0.70,
            max_det=100,
            device=0,
            batch=1,
            stream=False,
            verbose=False,
        )[0]


    fire_conf = 0.0
    smoke_conf = 0.0

    fire_count = 0
    smoke_count = 0


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


        for conf, cls in zip(
            confs,
            classes,
        ):

            conf = float(
                conf
            )

            cls = int(
                cls
            )


            if cls == 0:

                fire_count += 1

                fire_conf = max(
                    fire_conf,
                    conf,
                )


            elif cls == 1:

                smoke_count += 1

                smoke_conf = max(
                    smoke_conf,
                    conf,
                )


    top_conf = max(
        fire_conf,
        smoke_conf,
    )


    if top_conf == 0:

        top_class = "none"

    elif fire_conf >= smoke_conf:

        top_class = "fire"

    else:

        top_class = "smoke"


    results_rows.append({
        "index":
            index,

        "camera_id":
            meta["camera_id"],

        "image":
            meta["image"],

        "proxy_path":
            meta["proxy_path"],

        "sha256":
            meta["sha256"],

        "fire_conf":
            fire_conf,

        "smoke_conf":
            smoke_conf,

        "top_conf":
            top_conf,

        "top_class":
            top_class,

        "fire_count":
            fire_count,

        "smoke_count":
            smoke_count,

        "severity":
            severity(
                top_conf
            ),

        "human_decision":
            "",

        "human_note":
            "",
    })


    del result


    if (
        index % 50 == 0
        and torch.cuda.is_available()
    ):

        gc.collect()
        torch.cuda.empty_cache()


    if (
        index % 25 == 0
        or index == EXPECTED
    ):

        print(
            f"\rEvaluated "
            f"{index}/{EXPECTED} "
            f"({index/EXPECTED*100:5.1f}%)",
            end="",
            flush=True,
        )


print()


# ============================================================
# Sort by strongest alert first
# ============================================================

results_rows.sort(
    key=lambda row:
        float(
            row["top_conf"]
        ),
    reverse=True,
)


# ============================================================
# CSV
# ============================================================

csv_path = (
    OUT
    / "internet_proxy_r21_predictions.csv"
)


with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            results_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        results_rows
    )


# ============================================================
# Threshold statistics
# ============================================================

threshold_rows = []


for threshold in THRESHOLDS:

    alert_images = 0

    fire_alert_images = 0
    smoke_alert_images = 0

    total_boxes = 0
    fire_boxes = 0
    smoke_boxes = 0


    for row in results_rows:

        fire_conf = float(
            row["fire_conf"]
        )

        smoke_conf = float(
            row["smoke_conf"]
        )


        any_alert = (
            max(
                fire_conf,
                smoke_conf,
            )
            >= threshold
        )


        if any_alert:

            alert_images += 1


        if fire_conf >= threshold:

            fire_alert_images += 1


        if smoke_conf >= threshold:

            smoke_alert_images += 1


        # Approximate box counts from detections
        # surviving the requested class threshold.
        if fire_conf >= threshold:

            fire_boxes += int(
                row["fire_count"]
            )


        if smoke_conf >= threshold:

            smoke_boxes += int(
                row["smoke_count"]
            )


        total_boxes = (
            fire_boxes
            + smoke_boxes
        )


    threshold_rows.append({
        "threshold":
            threshold,

        "images":
            EXPECTED,

        "alert_images":
            alert_images,

        "proxy_alert_rate":
            alert_images
            / EXPECTED,

        "fire_alert_images":
            fire_alert_images,

        "fire_proxy_alert_rate":
            fire_alert_images
            / EXPECTED,

        "smoke_alert_images":
            smoke_alert_images,

        "smoke_proxy_alert_rate":
            smoke_alert_images
            / EXPECTED,

        "approx_prediction_boxes":
            total_boxes,
    })


threshold_csv = (
    OUT
    / "proxy_alert_rates.csv"
)


with threshold_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            threshold_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        threshold_rows
    )


# ============================================================
# Severity summary
# ============================================================

severity_names = [
    "EXTREME",
    "HARD",
    "MODERATE",
    "LOW",
    "NONE",
]


severity_counts = {
    name:
        sum(
            1
            for row in results_rows
            if (
                row["severity"]
                == name
            )
        )
    for name in severity_names
}


for name in severity_names:

    selected = [
        row
        for row in results_rows
        if (
            row["severity"]
            == name
        )
    ]


    path = (
        OUT
        / f"{name.lower()}.txt"
    )


    if selected:

        text = (
            "\n".join(
                row["image"]
                for row in selected
            )
            + "\n"
        )

    else:

        text = ""


    path.write_text(
        text,
        encoding="utf-8",
    )


# ============================================================
# Build review gallery for CONF >= 0.25
# ============================================================

review_rows = [
    row
    for row in results_rows
    if float(
        row["top_conf"]
    ) >= 0.25
]


cards = []


for review_index, row in enumerate(
    review_rows,
    start=1,
):

    source = Path(
        row["proxy_path"]
    )


    if not source.exists():
        raise FileNotFoundError(
            source
        )


    suffix = source.suffix.lower()


    filename = (
        f"{review_index:04d}_"
        f"{row['severity']}_"
        f"{row['top_class']}_"
        f"{float(row['top_conf']):.3f}_"
        f"cam{row['camera_id']}"
        f"{suffix}"
    )


    destination = (
        REVIEW
        / filename
    )


    os.symlink(
        source.resolve(),
        destination,
    )


    card = f"""
    <div class="card">

        <div class="header">
            #{review_index:04d}
            |
            {html.escape(row['severity'])}
        </div>

        <img
            src="{html.escape(filename)}"
            loading="lazy"
        >

        <div class="meta">

            camera =
            {html.escape(str(row['camera_id']))}
            <br>

            top =
            <b>
            {html.escape(row['top_class'])}
            {float(row['top_conf']):.3f}
            </b>
            <br>

            fire =
            {float(row['fire_conf']):.3f}
            <br>

            smoke =
            {float(row['smoke_conf']):.3f}

        </div>

    </div>
    """

    cards.append(
        card
    )


html_page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">

<title>
R2.1 Internet Proxy Review
</title>

<style>

body {{
    background: #111;
    color: #eee;
    font-family: Arial, sans-serif;
    margin: 20px;
}}

.info {{
    margin-bottom: 20px;
    color: #bbb;
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(
            auto-fill,
            minmax(320px, 1fr)
        );
    gap: 16px;
}}

.card {{
    background: #1d1d1d;
    border: 1px solid #444;
    border-radius: 8px;
    overflow: hidden;
}}

.header {{
    padding: 8px;
    font-weight: bold;
    background: #292929;
}}

.card img {{
    width: 100%;
    height: 260px;
    object-fit: contain;
    background: #000;
}}

.meta {{
    padding: 10px;
    font-family: monospace;
    line-height: 1.5;
}}

</style>
</head>

<body>

<h1>
R2.1-S Internet Proxy Review
</h1>

<div class="info">
Only predictions with confidence >= 0.25 are shown.
Do not assume they are negatives until visually reviewed.
</div>

<div class="grid">
{''.join(cards)}
</div>

</body>
</html>
"""


gallery_path = (
    REVIEW
    / "index.html"
)


gallery_path.write_text(
    html_page,
    encoding="utf-8",
)


# ============================================================
# Summary JSON
# ============================================================

summary = {
    "model":
        str(MODEL),

    "images":
        EXPECTED,

    "source":
        "SkyFinder",

    "dataset_status":
        "UNVERIFIED_PROXY_NEGATIVE",

    "severity_counts":
        severity_counts,

    "review_conf_threshold":
        0.25,

    "review_images":
        len(review_rows),

    "thresholds":
        threshold_rows,
}


summary_path = (
    OUT
    / "summary.json"
)


summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 100)
print(
    "INTERNET PROXY MINING SUMMARY"
)
print("=" * 100)


for name in severity_names:

    print(
        f"{name:10}: "
        f"{severity_counts[name]}"
    )


print()
print(
    "PROXY ALERT RATE"
)
print("-" * 100)


for row in threshold_rows:

    print(
        f"conf={row['threshold']:.2f} "
        f"alerts="
        f"{row['alert_images']:4d}/"
        f"{EXPECTED} "
        f"rate="
        f"{row['proxy_alert_rate']:.4f} "
        f"fire="
        f"{row['fire_alert_images']:4d} "
        f"smoke="
        f"{row['smoke_alert_images']:4d}"
    )


print()
print(
    "Images requiring review "
    "(conf >= 0.25):",
    len(review_rows)
)

print()

print(
    "Predictions:",
    csv_path
)

print(
    "Alert rates:",
    threshold_csv
)

print(
    "Gallery    :",
    gallery_path
)

print(
    "Summary    :",
    summary_path
)

print()
print(
    "IMPORTANT: These values are "
    "Proxy Alert Rates, NOT FPR yet."
)

print()

print(
    "RESULT: INTERNET PROXY "
    "R2.1 MINING PASS"
)

print("=" * 100)
