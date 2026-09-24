#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT
    / "models/final/"
      "fire_smoke_r3_e6_final.pt"
)

EXPECTED_MODEL_SHA256 = (
    "49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee"
    "354d9d7a5ae1cb8c53a22183"
)

DATASET = (
    ROOT
    / "datasets/fasdd_final_test_locked_v1"
)

DATA_YAML = (
    DATASET
    / "data.yaml"
)

TEST_IMAGES = (
    DATASET
    / "images/test"
)

TEST_LABELS = (
    DATASET
    / "labels/test"
)

OUT = (
    ROOT
    / "reports/final_test_v1"
)

RUN_NAME = "fasdd_test_r3_e6_final"

RUN_DIR = (
    OUT
    / RUN_NAME
)

PRED_LABELS = (
    RUN_DIR
    / "labels"
)

COMPLETED = (
    OUT
    / "FINAL_TEST_COMPLETED.json"
)

SUMMARY_TXT = (
    OUT
    / "FINAL_TEST_SUMMARY.txt"
)


EXPECTED_IMAGES = 15863
EXPECTED_NEGATIVES = 6525
EXPECTED_FIRE_GT = 9005
EXPECTED_SMOKE_GT = 8208

OPERATING_CONF = 0.25

MATCH_IOU = 0.50

SMALL_AREA = 0.005
LARGE_AREA = 0.05

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# UTILITIES
# ============================================================

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


def xywh_to_xyxy(
    x,
    y,
    w,
    h,
):

    return (
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2,
    )


def box_iou(
    a,
    b,
):

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

    inter = (
        iw * ih
    )

    area_a = (
        max(
            0.0,
            ax2 - ax1,
        )
        *
        max(
            0.0,
            ay2 - ay1,
        )
    )

    area_b = (
        max(
            0.0,
            bx2 - bx1,
        )
        *
        max(
            0.0,
            by2 - by1,
        )
    )

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return (
        inter / union
    )


def size_name(area):

    if area < SMALL_AREA:
        return "small"

    if area < LARGE_AREA:
        return "medium"

    return "large"


def read_gt(path):

    objects = []

    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return objects


    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:

            raise RuntimeError(
                f"Invalid GT row:\n"
                f"{path}\n"
                f"{line}"
            )


        cls = int(
            float(parts[0])
        )

        x = float(parts[1])
        y = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])


        objects.append({
            "cls":
                cls,

            "box":
                xywh_to_xyxy(
                    x,
                    y,
                    w,
                    h,
                ),

            "size":
                size_name(
                    w * h
                ),
        })


    return objects


def read_predictions(path):

    if not path.exists():
        return []


    text = path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []


    objects = []


    for line in text.splitlines():

        parts = line.split()

        # Ultralytics save_txt=True + save_conf=True:
        # cls x_center y_center width height confidence
        if len(parts) != 6:

            raise RuntimeError(
                f"Invalid prediction row:\n"
                f"{path}\n"
                f"{line}"
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


        objects.append({
            "cls":
                cls,

            "conf":
                conf,

            "box":
                xywh_to_xyxy(
                    x,
                    y,
                    w,
                    h,
                ),
        })


    return objects


# ============================================================
# PRE-FLIGHT / FREEZE PROTECTION
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


if COMPLETED.exists():

    raise RuntimeError(
        "\n"
        "FINAL TEST HAS ALREADY BEEN COMPLETED.\n"
        f"Record: {COMPLETED}\n"
        "\n"
        "Do not execute TEST again."
    )


for path in [
    MODEL,
    DATASET,
    DATA_YAML,
    TEST_IMAGES,
    TEST_LABELS,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


model_hash = sha256(
    MODEL
)


if model_hash != EXPECTED_MODEL_SHA256:

    raise RuntimeError(
        "\nFINAL MODEL SHA256 MISMATCH\n"
        f"Expected: {EXPECTED_MODEL_SHA256}\n"
        f"Actual  : {model_hash}\n"
    )


images = sorted([
    p
    for p in TEST_IMAGES.iterdir()
    if (
        p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTS
    )
])


if len(images) != EXPECTED_IMAGES:

    raise RuntimeError(
        "FINAL TEST image count mismatch: "
        f"{len(images)} != {EXPECTED_IMAGES}"
    )


# Validate GT identity before inference.
pre_fire_gt = 0
pre_smoke_gt = 0
pre_negatives = 0


for image in images:

    label = (
        TEST_LABELS
        / f"{image.stem}.txt"
    )

    if not label.exists():

        raise FileNotFoundError(
            label
        )


    gt = read_gt(
        label
    )


    if not gt:
        pre_negatives += 1


    for obj in gt:

        if obj["cls"] == 0:
            pre_fire_gt += 1

        elif obj["cls"] == 1:
            pre_smoke_gt += 1

        else:

            raise RuntimeError(
                f"Unexpected class "
                f"{obj['cls']} in {label}"
            )


if pre_negatives != EXPECTED_NEGATIVES:

    raise RuntimeError(
        "Negative identity mismatch: "
        f"{pre_negatives} "
        f"!= {EXPECTED_NEGATIVES}"
    )


if pre_fire_gt != EXPECTED_FIRE_GT:

    raise RuntimeError(
        "Fire GT identity mismatch: "
        f"{pre_fire_gt} "
        f"!= {EXPECTED_FIRE_GT}"
    )


if pre_smoke_gt != EXPECTED_SMOKE_GT:

    raise RuntimeError(
        "Smoke GT identity mismatch: "
        f"{pre_smoke_gt} "
        f"!= {EXPECTED_SMOKE_GT}"
    )


# Remove only stale incomplete prediction directory.
# COMPLETED.json does not exist at this point.
if RUN_DIR.exists():

    shutil.rmtree(
        RUN_DIR
    )


print("=" * 110)
print(
    "FINAL FASDD CLEAN V1 TEST — R3-E6"
)
print("=" * 110)

print(
    "Model :",
    MODEL
)

print(
    "SHA256:",
    model_hash
)

print(
    "Dataset:",
    DATASET
)

print(
    "Images :",
    EXPECTED_IMAGES
)

print(
    "Fire GT:",
    pre_fire_gt
)

print(
    "Smoke GT:",
    pre_smoke_gt
)

print(
    "Negative images:",
    pre_negatives
)

print(
    "Frozen operating confidence:",
    OPERATING_CONF
)

print(
    "GT matching IoU:",
    MATCH_IOU
)

print()

print(
    "THIS IS THE FINAL LOCKED TEST EVALUATION."
)

print(
    "TEST results must not be used "
    "for model/checkpoint/threshold tuning."
)

print("=" * 110)


# ============================================================
# ONE MODEL INFERENCE PASS ON FINAL TEST
# ============================================================

model = YOLO(
    str(MODEL)
)


metrics = model.val(
    data=str(
        DATA_YAML
    ),

    split="test",

    imgsz=768,

    # Keep very low confidence for AP calculation
    # and later derive conf=.25 from saved predictions.
    conf=0.001,

    iou=0.70,
    max_det=300,

    batch=2,
    device=0,
    workers=2,

    save_txt=True,
    save_conf=True,

    plots=False,
    verbose=True,

    project=str(
        OUT
    ),

    name=RUN_NAME,
    exist_ok=True,
)


if not PRED_LABELS.exists():

    raise RuntimeError(
        "Saved prediction directory "
        f"not found: {PRED_LABELS}"
    )


# ============================================================
# STANDARD TEST METRICS
# ============================================================

p_class = [
    float(x)
    for x in metrics.box.p
]

r_class = [
    float(x)
    for x in metrics.box.r
]

ap50_class = [
    float(x)
    for x in metrics.box.ap50
]

map95_class = [
    float(x)
    for x in metrics.box.maps
]


if not (
    len(p_class) >= 2
    and len(r_class) >= 2
    and len(ap50_class) >= 2
    and len(map95_class) >= 2
):

    raise RuntimeError(
        "Expected metrics for "
        "fire and smoke classes"
    )


standard = {
    "all": {
        "precision":
            float(
                metrics.box.mp
            ),

        "recall":
            float(
                metrics.box.mr
            ),

        "map50":
            float(
                metrics.box.map50
            ),

        "map50_95":
            float(
                metrics.box.map
            ),
    },

    "fire": {
        "precision":
            p_class[0],

        "recall":
            r_class[0],

        "map50":
            ap50_class[0],

        "map50_95":
            map95_class[0],
    },

    "smoke": {
        "precision":
            p_class[1],

        "recall":
            r_class[1],

        "map50":
            ap50_class[1],

        "map50_95":
            map95_class[1],
    },
}


speed = {
    str(k):
        float(v)
    for k, v
    in metrics.speed.items()
}


# ============================================================
# DERIVE FIXED CONF=.25 METRICS FROM SAVED PREDICTIONS
# NO SECOND MODEL INFERENCE
# ============================================================

classes = {
    0: "fire",
    1: "smoke",
}

sizes = [
    "small",
    "medium",
    "large",
]


fixed_stats = {
    0: {
        "gt": 0,
        "tp": 0,
    },

    1: {
        "gt": 0,
        "tp": 0,
    },
}


size_stats = {
    cls: {
        size: {
            "gt": 0,
            "tp": 0,
        }
        for size in sizes
    }
    for cls in [0, 1]
}


negative_stats = {
    "images": 0,

    "alert_images": 0,

    "fire_alert_images": 0,

    "smoke_alert_images": 0,

    "boxes": 0,

    "fire_boxes": 0,

    "smoke_boxes": 0,
}


for index, image in enumerate(
    images,
    start=1,
):

    stem = image.stem


    gt_path = (
        TEST_LABELS
        / f"{stem}.txt"
    )

    pred_path = (
        PRED_LABELS
        / f"{stem}.txt"
    )


    gt = read_gt(
        gt_path
    )


    predictions = [
        p
        for p in read_predictions(
            pred_path
        )
        if (
            p["conf"]
            >= OPERATING_CONF
        )
    ]


    # --------------------------------------------------------
    # NEGATIVE IMAGE FALSE POSITIVES
    # --------------------------------------------------------

    if not gt:

        negative_stats[
            "images"
        ] += 1


        if predictions:

            negative_stats[
                "alert_images"
            ] += 1


        fire_predictions = [
            p
            for p in predictions
            if p["cls"] == 0
        ]

        smoke_predictions = [
            p
            for p in predictions
            if p["cls"] == 1
        ]


        if fire_predictions:

            negative_stats[
                "fire_alert_images"
            ] += 1


        if smoke_predictions:

            negative_stats[
                "smoke_alert_images"
            ] += 1


        negative_stats[
            "fire_boxes"
        ] += len(
            fire_predictions
        )

        negative_stats[
            "smoke_boxes"
        ] += len(
            smoke_predictions
        )

        negative_stats[
            "boxes"
        ] += len(
            predictions
        )


    # --------------------------------------------------------
    # CLASS-SPECIFIC GT MATCHING
    # --------------------------------------------------------

    for cls in [0, 1]:

        gt_cls = [
            target
            for target in gt
            if (
                target["cls"]
                == cls
            )
        ]


        pred_cls = sorted(
            [
                prediction
                for prediction
                in predictions
                if (
                    prediction["cls"]
                    == cls
                )
            ],

            key=lambda x:
                x["conf"],

            reverse=True,
        )


        fixed_stats[
            cls
        ]["gt"] += len(
            gt_cls
        )


        for target in gt_cls:

            size_stats[
                cls
            ][
                target["size"]
            ][
                "gt"
            ] += 1


        matched_gt = set()


        for prediction in pred_cls:

            best_index = None
            best_iou = 0.0


            for gt_index, target in enumerate(
                gt_cls
            ):

                if gt_index in matched_gt:
                    continue


                iou = box_iou(
                    prediction[
                        "box"
                    ],
                    target[
                        "box"
                    ],
                )


                if iou > best_iou:

                    best_iou = iou
                    best_index = gt_index


            if (
                best_index is not None
                and best_iou >= MATCH_IOU
            ):

                matched_gt.add(
                    best_index
                )


        fixed_stats[
            cls
        ]["tp"] += len(
            matched_gt
        )


        for gt_index in matched_gt:

            target = gt_cls[
                gt_index
            ]

            size_stats[
                cls
            ][
                target["size"]
            ][
                "tp"
            ] += 1


    if (
        index % 1000 == 0
        or index == EXPECTED_IMAGES
    ):

        print(
            f"\rPost-processing "
            f"{index}/{EXPECTED_IMAGES}",
            end="",
            flush=True,
        )


print("\n")


# ============================================================
# FINAL DERIVED METRICS
# ============================================================

if (
    negative_stats["images"]
    != EXPECTED_NEGATIVES
):

    raise RuntimeError(
        "Derived negative count mismatch"
    )


fixed_results = {}


for cls in [0, 1]:

    gt = fixed_stats[
        cls
    ]["gt"]

    tp = fixed_stats[
        cls
    ]["tp"]

    fn = (
        gt - tp
    )


    fixed_results[
        classes[cls]
    ] = {
        "gt":
            gt,

        "tp":
            tp,

        "fn":
            fn,

        "recall":
            (
                tp / gt
                if gt
                else 0.0
            ),
    }


fixed_results[
    "macro_recall"
] = (
    fixed_results[
        "fire"
    ][
        "recall"
    ]
    +
    fixed_results[
        "smoke"
    ][
        "recall"
    ]
) / 2


size_results = {}


for cls in [0, 1]:

    class_name = classes[
        cls
    ]

    size_results[
        class_name
    ] = {}


    for size in sizes:

        gt = size_stats[
            cls
        ][size]["gt"]

        tp = size_stats[
            cls
        ][size]["tp"]

        fn = (
            gt - tp
        )


        size_results[
            class_name
        ][size] = {
            "gt":
                gt,

            "tp":
                tp,

            "fn":
                fn,

            "recall":
                (
                    tp / gt
                    if gt
                    else 0.0
                ),
        }


negative_results = {
    **negative_stats,

    "fpr":
        (
            negative_stats[
                "alert_images"
            ]
            /
            negative_stats[
                "images"
            ]
        ),

    "fire_fpr":
        (
            negative_stats[
                "fire_alert_images"
            ]
            /
            negative_stats[
                "images"
            ]
        ),

    "smoke_fpr":
        (
            negative_stats[
                "smoke_alert_images"
            ]
            /
            negative_stats[
                "images"
            ]
        ),
}


# ============================================================
# SAVE FINAL LOCKED RESULT
# ============================================================

record = {
    "status":
        "FINAL_TEST_COMPLETED",

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "model": {
        "path":
            str(MODEL),

        "sha256":
            model_hash,

        "architecture":
            "YOLOv8s",

        "checkpoint":
            "R3 epoch6",
    },

    "dataset": {
        "name":
            "FASDD Clean V1",

        "images":
            EXPECTED_IMAGES,

        "fire_gt":
            EXPECTED_FIRE_GT,

        "smoke_gt":
            EXPECTED_SMOKE_GT,

        "negative_images":
            EXPECTED_NEGATIVES,
    },

    "evaluation": {
        "imgsz":
            768,

        "operating_confidence":
            OPERATING_CONF,

        "matching_iou":
            MATCH_IOU,
    },

    "standard_metrics":
        standard,

    "fixed_confidence_metrics":
        fixed_results,

    "size_aware":
        size_results,

    "negative_test":
        negative_results,

    "speed_ms_per_image":
        speed,

    "policy": (
        "Model, checkpoint and operating "
        "confidence were frozen before TEST. "
        "These TEST results must not be used "
        "for subsequent training, checkpoint "
        "selection, or confidence tuning."
    ),
}


COMPLETED.write_text(
    json.dumps(
        record,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# HUMAN READABLE SUMMARY
# ============================================================

lines = []


def add(value=""):

    lines.append(
        str(value)
    )


add("=" * 100)

add(
    "FINAL FASDD CLEAN V1 TEST — R3-E6"
)

add("=" * 100)

add(
    f"Model SHA256 : {model_hash}"
)

add(
    f"Images       : {EXPECTED_IMAGES}"
)

add(
    f"Confidence   : {OPERATING_CONF}"
)

add(
    f"Match IoU    : {MATCH_IOU}"
)

add()

add(
    "STANDARD METRICS"
)

add("-" * 100)


for name in [
    "all",
    "fire",
    "smoke",
]:

    m = standard[
        name
    ]

    add(
        f"{name.upper():<8} "
        f"P={m['precision']:.4f} "
        f"R={m['recall']:.4f} "
        f"mAP50={m['map50']:.4f} "
        f"mAP50-95={m['map50_95']:.4f}"
    )


add()

add(
    "FIXED CONF=.25 GT RECALL "
    "@ IoU=.50"
)

add("-" * 100)


for name in [
    "fire",
    "smoke",
]:

    m = fixed_results[
        name
    ]

    add(
        f"{name.upper():<8} "
        f"TP={m['tp']}/{m['gt']} "
        f"FN={m['fn']} "
        f"Recall={m['recall']:.4f}"
    )


add(
    f"MACRO R  "
    f"{fixed_results['macro_recall']:.4f}"
)


add()

add(
    "SIZE-AWARE @ CONF=.25 / IoU=.50"
)

add("-" * 100)


for name in [
    "fire",
    "smoke",
]:

    for size in sizes:

        m = size_results[
            name
        ][size]

        add(
            f"{name}_{size:<8} "
            f"TP={m['tp']:5d}/"
            f"{m['gt']:5d} "
            f"FN={m['fn']:5d} "
            f"Recall={m['recall']:.4f}"
        )


add()

add(
    "NEGATIVE TEST @ CONF=.25"
)

add("-" * 100)


add(
    "Negative images : "
    f"{negative_results['images']}"
)

add(
    "Alert images    : "
    f"{negative_results['alert_images']}"
)

add(
    "FPR             : "
    f"{negative_results['fpr']:.4f}"
)

add(
    "Fire FPR        : "
    f"{negative_results['fire_fpr']:.4f}"
)

add(
    "Smoke FPR       : "
    f"{negative_results['smoke_fpr']:.4f}"
)

add(
    "Prediction boxes: "
    f"{negative_results['boxes']}"
)


add()

add(
    "SPEED (ms/image)"
)

add("-" * 100)


for key, value in speed.items():

    add(
        f"{key:<16}: "
        f"{value:.4f}"
    )


add()

add("=" * 100)

add(
    "STATUS: FINAL_TEST_COMPLETED"
)

add(
    "DO NOT USE TEST RESULTS "
    "FOR FURTHER MODEL TUNING."
)

add("=" * 100)


SUMMARY_TXT.write_text(
    "\n".join(
        lines
    )
    + "\n",
    encoding="utf-8",
)


print(
    "\n".join(
        lines
    )
)

print()

print(
    "JSON:",
    COMPLETED
)

print(
    "TXT :",
    SUMMARY_TXT
)
