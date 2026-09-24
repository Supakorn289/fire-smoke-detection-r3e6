#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import math

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT
    / "datasets"
    / "fasdd_r1_v1"
)

IMAGES = DATASET / "images" / "val"
LABELS = DATASET / "labels" / "val"

OUT = (
    ROOT
    / "reports"
    / "fasdd"
    / "r1_size_aware"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


MODELS = {
    "n": (
        ROOT
        / "runs"
        / "teachers_r1"
        / "teacher_n_r1"
        / "weights"
        / "best.pt"
    ),

    "s": (
        ROOT
        / "runs"
        / "teachers_r1"
        / "teacher_s_r1"
        / "weights"
        / "best.pt"
    ),

    "m": (
        ROOT
        / "runs"
        / "teachers_r1"
        / "teacher_m_r1"
        / "weights"
        / "best.pt"
    ),
}


CLASS_NAMES = {
    0: "fire",
    1: "smoke",
}

THRESHOLDS = [
    0.10,
    0.25,
    0.50,
]

IOU_THRESHOLD = 0.50

SMALL_MAX = 0.005
MEDIUM_MAX = 0.05

PREDICTION_CONF_FLOOR = 0.01


def size_class(area):

    if area < SMALL_MAX:
        return "small"

    if area < MEDIUM_MAX:
        return "medium"

    return "large"


def box_iou(a, b):

    x1 = max(
        a[0],
        b[0],
    )

    y1 = max(
        a[1],
        b[1],
    )

    x2 = min(
        a[2],
        b[2],
    )

    y2 = min(
        a[3],
        b[3],
    )

    inter_w = max(
        0.0,
        x2 - x1,
    )

    inter_h = max(
        0.0,
        y2 - y1,
    )

    inter = (
        inter_w
        * inter_h
    )

    area_a = max(
        0.0,
        a[2] - a[0],
    ) * max(
        0.0,
        a[3] - a[1],
    )

    area_b = max(
        0.0,
        b[2] - b[0],
    ) * max(
        0.0,
        b[3] - b[1],
    )

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def load_gt(
    label_path,
    width,
    height,
):

    gt = []

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return gt

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            raise RuntimeError(
                f"Malformed label: {label_path}"
            )

        cls = int(
            parts[0]
        )

        xc, yc, bw, bh = map(
            float,
            parts[1:],
        )

        normalized_area = (
            bw * bh
        )

        size = size_class(
            normalized_area
        )

        x1 = (
            xc - bw / 2
        ) * width

        y1 = (
            yc - bh / 2
        ) * height

        x2 = (
            xc + bw / 2
        ) * width

        y2 = (
            yc + bh / 2
        ) * height

        gt.append({
            "cls": cls,
            "size": size,
            "box": (
                x1,
                y1,
                x2,
                y2,
            ),
        })

    return gt


def initialize_stats():

    stats = {}

    for threshold in THRESHOLDS:

        stats[threshold] = {
            "gt": defaultdict(int),
            "tp": defaultdict(int),
            "fn": defaultdict(int),

            "tp_conf_sum":
                defaultdict(float),

            "pred":
                defaultdict(int),

            "pred_tp":
                defaultdict(int),

            "fp":
                defaultdict(int),
        }

    return stats


def gt_key(
    cls,
    size,
):

    return (
        f"{CLASS_NAMES[cls]}"
        f"_{size}"
    )


def pred_size(
    box,
    width,
    height,
):

    bw = max(
        0.0,
        box[2] - box[0],
    )

    bh = max(
        0.0,
        box[3] - box[1],
    )

    area = (
        (bw / width)
        * (bh / height)
    )

    return size_class(
        area
    )


def evaluate_threshold(
    gt,
    predictions,
    width,
    height,
    threshold,
    stats,
):

    current = stats[
        threshold
    ]

    # --------------------------------------------------------
    # Register all GT
    # --------------------------------------------------------

    for item in gt:

        key = gt_key(
            item["cls"],
            item["size"],
        )

        current["gt"][
            key
        ] += 1

    # --------------------------------------------------------
    # Filter predictions
    # --------------------------------------------------------

    preds = [
        p
        for p in predictions
        if p["conf"] >= threshold
    ]

    preds.sort(
        key=lambda x:
            x["conf"],
        reverse=True,
    )

    matched_gt = set()

    # --------------------------------------------------------
    # Greedy confidence-ordered matching
    # --------------------------------------------------------

    for pred in preds:

        psize = pred_size(
            pred["box"],
            width,
            height,
        )

        pkey = gt_key(
            pred["cls"],
            psize,
        )

        current["pred"][
            pkey
        ] += 1

        best_iou = 0.0
        best_index = None

        for idx, target in enumerate(
            gt
        ):

            if idx in matched_gt:
                continue

            if (
                target["cls"]
                != pred["cls"]
            ):
                continue

            iou = box_iou(
                pred["box"],
                target["box"],
            )

            if iou > best_iou:

                best_iou = iou
                best_index = idx

        if (
            best_index is not None
            and
            best_iou >= IOU_THRESHOLD
        ):

            matched_gt.add(
                best_index
            )

            target = gt[
                best_index
            ]

            key = gt_key(
                target["cls"],
                target["size"],
            )

            current["tp"][
                key
            ] += 1

            current["tp_conf_sum"][
                key
            ] += pred["conf"]

            current["pred_tp"][
                pkey
            ] += 1

        else:

            current["fp"][
                pkey
            ] += 1

    # --------------------------------------------------------
    # False negatives
    # --------------------------------------------------------

    for idx, target in enumerate(
        gt
    ):

        if idx in matched_gt:
            continue

        key = gt_key(
            target["cls"],
            target["size"],
        )

        current["fn"][
            key
        ] += 1


def make_report(
    model_name,
    stats,
):

    rows = []

    ordered_keys = [
        "fire_small",
        "fire_medium",
        "fire_large",
        "smoke_small",
        "smoke_medium",
        "smoke_large",
    ]

    for threshold in THRESHOLDS:

        s = stats[
            threshold
        ]

        for key in ordered_keys:

            gt = s["gt"][
                key
            ]

            tp = s["tp"][
                key
            ]

            fn = s["fn"][
                key
            ]

            recall = (
                tp / gt
                if gt
                else 0.0
            )

            mean_conf = (
                s["tp_conf_sum"][key]
                / tp
                if tp
                else 0.0
            )

            pred = s["pred"][
                key
            ]

            pred_tp = s["pred_tp"][
                key
            ]

            fp = s["fp"][
                key
            ]

            pred_precision = (
                pred_tp / pred
                if pred
                else 0.0
            )

            rows.append({
                "model":
                    model_name,

                "conf_threshold":
                    threshold,

                "iou_threshold":
                    IOU_THRESHOLD,

                "class_size":
                    key,

                "gt":
                    gt,

                "tp":
                    tp,

                "fn":
                    fn,

                "gt_recall":
                    recall,

                "tp_mean_conf":
                    mean_conf,

                "predicted_boxes":
                    pred,

                "prediction_tp":
                    pred_tp,

                "prediction_fp":
                    fp,

                "predicted_size_precision":
                    pred_precision,
            })

    return rows


def print_report(
    model_name,
    rows,
):

    print()
    print("=" * 108)
    print(
        f"R1 SIZE-AWARE EVALUATION — "
        f"TEACHER {model_name.upper()}"
    )
    print("=" * 108)

    for threshold in THRESHOLDS:

        print()
        print(
            f"CONFIDENCE >= {threshold:.2f} "
            f"| IoU >= {IOU_THRESHOLD:.2f}"
        )

        print("-" * 108)

        print(
            f"{'CLASS/SIZE':20}"
            f"{'GT':>8}"
            f"{'TP':>8}"
            f"{'FN':>8}"
            f"{'RECALL':>12}"
            f"{'TP CONF':>12}"
            f"{'PRED':>8}"
            f"{'FP':>8}"
            f"{'PRED-P':>12}"
        )

        for row in rows:

            if (
                row[
                    "conf_threshold"
                ]
                != threshold
            ):
                continue

            print(
                f"{row['class_size']:20}"
                f"{row['gt']:8d}"
                f"{row['tp']:8d}"
                f"{row['fn']:8d}"
                f"{row['gt_recall']:12.4f}"
                f"{row['tp_mean_conf']:12.4f}"
                f"{row['predicted_boxes']:8d}"
                f"{row['prediction_fp']:8d}"
                f"{row['predicted_size_precision']:12.4f}"
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "n",
            "s",
            "m",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    image_paths = sorted(
        [
            p
            for p in IMAGES.iterdir()
            if p.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }
        ],
        key=lambda x:
            x.name,
    )

    if len(image_paths) != 3000:

        raise RuntimeError(
            f"Expected 3000 VAL images, "
            f"got {len(image_paths)}"
        )

    selected_models = (
        ["n", "s", "m"]
        if args.model == "all"
        else [args.model]
    )

    all_rows = []

    for model_name in (
        selected_models
    ):

        model_path = MODELS[
            model_name
        ]

        if not model_path.exists():

            raise FileNotFoundError(
                model_path
            )

        print()
        print("=" * 108)
        print(
            f"LOADING TEACHER "
            f"{model_name.upper()}"
        )
        print("=" * 108)

        print(
            model_path
        )

        model = YOLO(
            str(model_path)
        )

        stats = (
            initialize_stats()
        )

        results = model.predict(
            source=[
                str(p)
                for p in image_paths
            ],
            imgsz=768,
            conf=PREDICTION_CONF_FLOOR,
            iou=0.70,
            max_det=300,
            device=0,
            batch=args.batch,
            stream=True,
            verbose=False,
        )

        count = 0

        for result in results:

            count += 1

            path = Path(
                result.path
            )

            label_path = (
                LABELS
                / f"{path.stem}.txt"
            )

            if not label_path.exists():

                raise FileNotFoundError(
                    label_path
                )

            height, width = (
                result.orig_shape
            )

            gt = load_gt(
                label_path,
                width,
                height,
            )

            predictions = []

            if result.boxes is not None:

                xyxy = (
                    result.boxes.xyxy
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

                classes = (
                    result.boxes.cls
                    .detach()
                    .cpu()
                    .tolist()
                )

                for box, conf, cls in zip(
                    xyxy,
                    confs,
                    classes,
                ):

                    predictions.append({
                        "box":
                            tuple(
                                map(
                                    float,
                                    box,
                                )
                            ),

                        "conf":
                            float(
                                conf
                            ),

                        "cls":
                            int(
                                cls
                            ),
                    })

            for threshold in (
                THRESHOLDS
            ):

                evaluate_threshold(
                    gt,
                    predictions,
                    width,
                    height,
                    threshold,
                    stats,
                )

            if (
                count % 250 == 0
                or
                count == 3000
            ):

                print(
                    f"\r{model_name.upper()} "
                    f"{count}/3000 "
                    f"({count/3000*100:5.1f}%)",
                    end="",
                    flush=True,
                )

        print()

        if count != 3000:

            raise RuntimeError(
                f"Expected 3000 results, "
                f"got {count}"
            )

        rows = make_report(
            model_name,
            stats,
        )

        print_report(
            model_name,
            rows,
        )

        all_rows.extend(
            rows
        )

        model_json = (
            OUT
            / f"teacher_{model_name}_r1_size_aware.json"
        )

        model_json.write_text(
            json.dumps(
                rows,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ========================================================
    # COMBINED CSV
    # ========================================================

    csv_path = (
        OUT
        / "r1_size_aware_all_models.csv"
    )

    fields = [
        "model",
        "conf_threshold",
        "iou_threshold",
        "class_size",
        "gt",
        "tp",
        "fn",
        "gt_recall",
        "tp_mean_conf",
        "predicted_boxes",
        "prediction_tp",
        "prediction_fp",
        "predicted_size_precision",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            all_rows
        )

    print()
    print("=" * 108)
    print("SIZE-AWARE EVALUATION COMPLETE")
    print("=" * 108)

    print(
        "Models:",
        ", ".join(
            x.upper()
            for x in selected_models
        )
    )

    print(
        "Images:",
        len(image_paths)
    )

    print(
        "IoU:",
        IOU_THRESHOLD
    )

    print(
        "Confidence thresholds:",
        THRESHOLDS
    )

    print()
    print(
        "CSV:",
        csv_path
    )

    print(
        "Reports:",
        OUT
    )

    print()
    print(
        "RESULT: R1 SIZE-AWARE EVALUATION PASS"
    )

    print("=" * 108)


if __name__ == "__main__":
    main()
