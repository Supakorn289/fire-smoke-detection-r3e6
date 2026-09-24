#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import math
import shutil

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

PT_MODEL = (
    ROOT
    / "models/final/"
      "fire_smoke_r3_e6_final.pt"
)

ONNX_MODEL = (
    ROOT
    / "models/final/onnx_v1/"
      "fire_smoke_r3_e6_final_768_fp32.onnx"
)

EXPECTED_PT_SHA256 = (
    "49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee"
    "354d9d7a5ae1cb8c53a22183"
)

EXPECTED_ONNX_SHA256 = (
    "e887a759bcc65c89803c8ea0aa3d8891bcb0e60a"
    "5fd334433b2d6d912ba3bc79"
)

DATA = (
    ROOT
    / "datasets/fasdd_r21_preserve_v1/"
      "data.yaml"
)

VAL_IMAGES = (
    ROOT
    / "datasets/fasdd_r21_preserve_v1/"
      "images/val"
)

VAL_LABELS = (
    ROOT
    / "datasets/fasdd_r21_preserve_v1/"
      "labels/val"
)

OUT = (
    ROOT
    / "reports/deployment_v1/"
      "onnx_v1/equivalence"
)

PT_RUN = OUT / "pt_val"
ONNX_RUN = OUT / "onnx_val"

EXPECTED_IMAGES = 3000

FIXED_CONF = 0.25
MATCH_IOU = 0.50

# PT-vs-ONNX detection matching.
EQUIV_IOU = 0.95

# Practical equivalence tolerances.
MAX_STANDARD_METRIC_DELTA = 0.005
MAX_FIXED_RECALL_DELTA = 0.005
MIN_DETECTION_MATCH_RATE = 0.990
MIN_MEAN_MATCH_IOU = 0.990
MAX_MEAN_CONF_DELTA = 0.005

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def xywh_to_xyxy(x, y, w, h):

    return (
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2,
    )


def box_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(
        0.0,
        ix2 - ix1,
    )

    ih = max(
        0.0,
        iy2 - iy1,
    )

    inter = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def read_gt(path):

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []


    result = []


    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:

            raise RuntimeError(
                f"Invalid GT row: {path}\n{line}"
            )


        cls = int(
            float(parts[0])
        )

        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])


        result.append({
            "cls":
                cls,

            "box":
                xywh_to_xyxy(
                    x, y, w, h
                ),
        })


    return result


def read_predictions(path):

    if not path.exists():
        return []


    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []


    result = []


    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 6:

            raise RuntimeError(
                f"Invalid prediction row: "
                f"{path}\n{line}"
            )


        cls = int(
            float(parts[0])
        )

        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        conf = float(
            parts[5]
        )


        result.append({
            "cls":
                cls,

            "conf":
                conf,

            "box":
                xywh_to_xyxy(
                    x, y, w, h
                ),
        })


    return result


def extract_standard(metrics):

    p = [
        float(x)
        for x in metrics.box.p
    ]

    r = [
        float(x)
        for x in metrics.box.r
    ]

    ap50 = [
        float(x)
        for x in metrics.box.ap50
    ]

    maps = [
        float(x)
        for x in metrics.box.maps
    ]


    return {
        "all": {
            "p":
                float(metrics.box.mp),

            "r":
                float(metrics.box.mr),

            "map50":
                float(metrics.box.map50),

            "map95":
                float(metrics.box.map),
        },

        "fire": {
            "p": p[0],
            "r": r[0],
            "map50": ap50[0],
            "map95": maps[0],
        },

        "smoke": {
            "p": p[1],
            "r": r[1],
            "map50": ap50[1],
            "map95": maps[1],
        },
    }


def fixed_recall(
    images,
    prediction_dir,
):

    stats = {
        0: {
            "gt": 0,
            "tp": 0,
        },

        1: {
            "gt": 0,
            "tp": 0,
        },
    }


    for image in images:

        gt = read_gt(
            VAL_LABELS
            / f"{image.stem}.txt"
        )


        preds = [
            p
            for p in read_predictions(
                prediction_dir
                / f"{image.stem}.txt"
            )
            if (
                p["conf"]
                >= FIXED_CONF
            )
        ]


        for cls in [0, 1]:

            gt_cls = [
                g
                for g in gt
                if g["cls"] == cls
            ]


            pred_cls = sorted(
                [
                    p
                    for p in preds
                    if p["cls"] == cls
                ],

                key=lambda x:
                    x["conf"],

                reverse=True,
            )


            stats[
                cls
            ]["gt"] += len(
                gt_cls
            )


            matched_gt = set()


            for pred in pred_cls:

                best_index = None
                best_iou = 0.0


                for gi, target in enumerate(
                    gt_cls
                ):

                    if gi in matched_gt:
                        continue


                    iou = box_iou(
                        pred["box"],
                        target["box"],
                    )


                    if iou > best_iou:

                        best_iou = iou
                        best_index = gi


                if (
                    best_index is not None
                    and best_iou >= MATCH_IOU
                ):

                    matched_gt.add(
                        best_index
                    )


            stats[
                cls
            ]["tp"] += len(
                matched_gt
            )


    fire_r = (
        stats[0]["tp"]
        / stats[0]["gt"]
    )

    smoke_r = (
        stats[1]["tp"]
        / stats[1]["gt"]
    )


    return {
        "fire": {
            **stats[0],
            "recall": fire_r,
        },

        "smoke": {
            **stats[1],
            "recall": smoke_r,
        },

        "macro":
            (
                fire_r
                + smoke_r
            ) / 2,
    }


def compare_detections(
    images,
    pt_dir,
    onnx_dir,
):

    pt_total = 0
    onnx_total = 0

    matched = 0

    iou_values = []
    conf_deltas = []


    for image in images:

        pt = [
            p
            for p in read_predictions(
                pt_dir
                / f"{image.stem}.txt"
            )
            if (
                p["conf"]
                >= FIXED_CONF
            )
        ]


        ox = [
            p
            for p in read_predictions(
                onnx_dir
                / f"{image.stem}.txt"
            )
            if (
                p["conf"]
                >= FIXED_CONF
            )
        ]


        pt_total += len(pt)
        onnx_total += len(ox)


        used_onnx = set()


        pt_sorted = sorted(
            pt,
            key=lambda x:
                x["conf"],
            reverse=True,
        )


        for p in pt_sorted:

            best_index = None
            best_iou = 0.0


            for oi, q in enumerate(ox):

                if oi in used_onnx:
                    continue

                if p["cls"] != q["cls"]:
                    continue


                iou = box_iou(
                    p["box"],
                    q["box"],
                )


                if iou > best_iou:

                    best_iou = iou
                    best_index = oi


            if (
                best_index is not None
                and best_iou >= EQUIV_IOU
            ):

                q = ox[
                    best_index
                ]

                used_onnx.add(
                    best_index
                )

                matched += 1

                iou_values.append(
                    best_iou
                )

                conf_deltas.append(
                    abs(
                        p["conf"]
                        - q["conf"]
                    )
                )


    denominator = max(
        pt_total,
        onnx_total,
        1,
    )


    match_rate = (
        matched
        / denominator
    )


    mean_iou = (
        sum(iou_values)
        / len(iou_values)
        if iou_values
        else 0.0
    )


    mean_conf_delta = (
        sum(conf_deltas)
        / len(conf_deltas)
        if conf_deltas
        else math.inf
    )


    max_conf_delta = (
        max(conf_deltas)
        if conf_deltas
        else math.inf
    )


    return {
        "pt_detections":
            pt_total,

        "onnx_detections":
            onnx_total,

        "matched":
            matched,

        "match_rate":
            match_rate,

        "mean_iou":
            mean_iou,

        "mean_conf_delta":
            mean_conf_delta,

        "max_conf_delta":
            max_conf_delta,
    }


# ============================================================
# PRE-FLIGHT
# ============================================================

for path in [
    PT_MODEL,
    ONNX_MODEL,
    DATA,
    VAL_IMAGES,
    VAL_LABELS,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


pt_hash = sha256(
    PT_MODEL
)

onnx_hash = sha256(
    ONNX_MODEL
)


if pt_hash != EXPECTED_PT_SHA256:

    raise RuntimeError(
        "PT SHA256 mismatch"
    )


if onnx_hash != EXPECTED_ONNX_SHA256:

    raise RuntimeError(
        "ONNX SHA256 mismatch"
    )


images = sorted([
    p
    for p in VAL_IMAGES.iterdir()
    if (
        p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTS
    )
])


if len(images) != EXPECTED_IMAGES:

    raise RuntimeError(
        f"Expected {EXPECTED_IMAGES} VAL images, "
        f"got {len(images)}"
    )


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


for run in [
    PT_RUN,
    ONNX_RUN,
]:

    if run.exists():
        shutil.rmtree(run)


print("=" * 110)
print(
    "PT ↔ ONNX EQUIVALENCE GATE V2 — FIXED SHAPE"
)
print("=" * 110)

print(
    "VAL images:",
    len(images)
)

print(
    "PT SHA256  :",
    pt_hash
)

print(
    "ONNX SHA256:",
    onnx_hash
)

print(
    "Device     : CPU"
)

print(
    "Batch      : 1"
)

print(
    "imgsz      : 768"
)

print(
    "Fixed conf :",
    FIXED_CONF
)

print()

print(
    "FASDD TEST IS NOT USED."
)

print("=" * 110)


# ============================================================
# PT VAL
# ============================================================

print()
print(
    "A) PYTORCH VAL"
)

pt_model = YOLO(
    str(PT_MODEL)
)


pt_metrics = pt_model.val(
    data=str(DATA),

    split="val",

    imgsz=768,
    batch=1,

    # CRITICAL:
    # Force identical fixed 768x768 preprocessing for PT and
    # static ONNX. Without this, PT validation may use
    # rectangular padding while static ONNX uses square input.
    rect=False,

    conf=0.001,
    iou=0.70,
    max_det=300,

    device="cpu",
    workers=2,

    save_txt=True,
    save_conf=True,

    plots=False,
    verbose=True,

    project=str(OUT),
    name="pt_val",
    exist_ok=True,
)


pt_standard = extract_standard(
    pt_metrics
)


# ============================================================
# ONNX VAL
# ============================================================

print()
print(
    "B) ONNX RUNTIME CPU VAL"
)

onnx_model = YOLO(
    str(ONNX_MODEL)
)


onnx_metrics = onnx_model.val(
    data=str(DATA),

    split="val",

    imgsz=768,
    batch=1,

    # CRITICAL:
    # Force identical fixed 768x768 preprocessing for PT and
    # static ONNX. Without this, PT validation may use
    # rectangular padding while static ONNX uses square input.
    rect=False,

    conf=0.001,
    iou=0.70,
    max_det=300,

    device="cpu",
    workers=2,

    save_txt=True,
    save_conf=True,

    plots=False,
    verbose=True,

    project=str(OUT),
    name="onnx_val",
    exist_ok=True,
)


onnx_standard = extract_standard(
    onnx_metrics
)


# ============================================================
# FIXED CONF RECALL
# ============================================================

print()
print(
    "C) FIXED CONF=.25 RECALL"
)


pt_fixed = fixed_recall(
    images,
    PT_RUN / "labels",
)


onnx_fixed = fixed_recall(
    images,
    ONNX_RUN / "labels",
)


# ============================================================
# DETECTION AGREEMENT
# ============================================================

print()
print(
    "D) DETECTION/BBOX AGREEMENT"
)


agreement = compare_detections(
    images,
    PT_RUN / "labels",
    ONNX_RUN / "labels",
)


# ============================================================
# DELTAS
# ============================================================

standard_deltas = {}


max_standard_delta = 0.0


for cls in [
    "all",
    "fire",
    "smoke",
]:

    standard_deltas[
        cls
    ] = {}


    for metric in [
        "p",
        "r",
        "map50",
        "map95",
    ]:

        delta = abs(
            pt_standard[
                cls
            ][
                metric
            ]
            -
            onnx_standard[
                cls
            ][
                metric
            ]
        )


        standard_deltas[
            cls
        ][
            metric
        ] = delta


        max_standard_delta = max(
            max_standard_delta,
            delta,
        )


fixed_fire_delta = abs(
    pt_fixed["fire"]["recall"]
    -
    onnx_fixed["fire"]["recall"]
)


fixed_smoke_delta = abs(
    pt_fixed["smoke"]["recall"]
    -
    onnx_fixed["smoke"]["recall"]
)


max_fixed_delta = max(
    fixed_fire_delta,
    fixed_smoke_delta,
)


# ============================================================
# GATE
# ============================================================

checks = {
    "standard_metrics":
        (
            max_standard_delta
            <= MAX_STANDARD_METRIC_DELTA
        ),

    "fixed_recall":
        (
            max_fixed_delta
            <= MAX_FIXED_RECALL_DELTA
        ),

    "detection_match_rate":
        (
            agreement["match_rate"]
            >= MIN_DETECTION_MATCH_RATE
        ),

    "bbox_iou":
        (
            agreement["mean_iou"]
            >= MIN_MEAN_MATCH_IOU
        ),

    "confidence":
        (
            agreement[
                "mean_conf_delta"
            ]
            <= MAX_MEAN_CONF_DELTA
        ),
}


passed = all(
    checks.values()
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 120)
print(
    "STANDARD METRICS COMPARISON"
)
print("=" * 120)

print(
    f"{'CLASS':<10}"
    f"{'PT P':>10}"
    f"{'ONNX P':>10}"
    f"{'PT R':>10}"
    f"{'ONNX R':>10}"
    f"{'PT mAP50':>12}"
    f"{'ONNX mAP50':>12}"
    f"{'PT mAP95':>12}"
    f"{'ONNX mAP95':>12}"
)


for cls in [
    "all",
    "fire",
    "smoke",
]:

    p = pt_standard[
        cls
    ]

    o = onnx_standard[
        cls
    ]


    print(
        f"{cls:<10}"
        f"{p['p']:>10.4f}"
        f"{o['p']:>10.4f}"
        f"{p['r']:>10.4f}"
        f"{o['r']:>10.4f}"
        f"{p['map50']:>12.4f}"
        f"{o['map50']:>12.4f}"
        f"{p['map95']:>12.4f}"
        f"{o['map95']:>12.4f}"
    )


print()
print(
    "Max standard metric delta:",
    f"{max_standard_delta:.6f}"
)


print()
print("=" * 120)
print(
    "FIXED CONF=.25 RECALL"
)
print("=" * 120)

print(
    f"PT   Fire  : "
    f"{pt_fixed['fire']['tp']}/"
    f"{pt_fixed['fire']['gt']} "
    f"R={pt_fixed['fire']['recall']:.6f}"
)

print(
    f"ONNX Fire  : "
    f"{onnx_fixed['fire']['tp']}/"
    f"{onnx_fixed['fire']['gt']} "
    f"R={onnx_fixed['fire']['recall']:.6f}"
)

print(
    f"PT   Smoke : "
    f"{pt_fixed['smoke']['tp']}/"
    f"{pt_fixed['smoke']['gt']} "
    f"R={pt_fixed['smoke']['recall']:.6f}"
)

print(
    f"ONNX Smoke : "
    f"{onnx_fixed['smoke']['tp']}/"
    f"{onnx_fixed['smoke']['gt']} "
    f"R={onnx_fixed['smoke']['recall']:.6f}"
)

print(
    "Max fixed recall delta:",
    f"{max_fixed_delta:.6f}"
)


print()
print("=" * 120)
print(
    "DETECTION AGREEMENT @ CONF=.25"
)
print("=" * 120)

print(
    "PT detections   :",
    agreement["pt_detections"]
)

print(
    "ONNX detections :",
    agreement["onnx_detections"]
)

print(
    "Matched         :",
    agreement["matched"]
)

print(
    "Match rate      :",
    f"{agreement['match_rate']:.6f}"
)

print(
    "Mean bbox IoU   :",
    f"{agreement['mean_iou']:.6f}"
)

print(
    "Mean conf delta :",
    f"{agreement['mean_conf_delta']:.8f}"
)

print(
    "Max conf delta  :",
    f"{agreement['max_conf_delta']:.8f}"
)


print()
print("=" * 120)
print(
    "EQUIVALENCE GATE"
)
print("=" * 120)


for name, status in checks.items():

    print(
        f"{name:<28}: "
        f"{'PASS' if status else 'FAIL'}"
    )


print()


if passed:

    print(
        "RESULT: PT ↔ ONNX EQUIVALENCE PASS"
    )

else:

    print(
        "RESULT: PT ↔ ONNX EQUIVALENCE FAIL"
    )


print("=" * 120)


# ============================================================
# SAVE RECORD
# ============================================================

record = {
    "status":
        (
            "PASS"
            if passed
            else "FAIL"
        ),

    "pt_sha256":
        pt_hash,

    "onnx_sha256":
        onnx_hash,

    "dataset":
        "FASDD R1 Main VAL 3000",

    "test_used":
        False,

    "preprocessing":
        "rect=False for PT and ONNX",

    "settings": {
        "imgsz": 768,
        "batch": 1,
        "device": "CPU",
        "fixed_confidence": FIXED_CONF,
        "nms_iou": 0.70,
    },

    "pt_standard":
        pt_standard,

    "onnx_standard":
        onnx_standard,

    "standard_deltas":
        standard_deltas,

    "max_standard_delta":
        max_standard_delta,

    "pt_fixed":
        pt_fixed,

    "onnx_fixed":
        onnx_fixed,

    "max_fixed_recall_delta":
        max_fixed_delta,

    "agreement":
        agreement,

    "checks":
        checks,
}


record_path = (
    OUT
    / "PT_ONNX_EQUIVALENCE.json"
)


record_path.write_text(
    json.dumps(
        record,
        indent=2,
    ),
    encoding="utf-8",
)


print(
    "Record:",
    record_path
)
