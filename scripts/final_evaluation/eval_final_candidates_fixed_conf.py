#!/usr/bin/env python3

from pathlib import Path
import json
import gc

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

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

MODELS = {
    "R21": (
        ROOT
        / "runs/teachers_r21/"
          "teacher_s_r21_preserve_v1/"
          "weights/best.pt"
    ),

    "R3_E0": (
        ROOT
        / "runs/teachers_r3/"
          "teacher_s_r3_proxy_v1/"
          "weights/epoch0.pt"
    ),

    "R3_E5": (
        ROOT
        / "runs/teachers_r3/"
          "teacher_s_r3_proxy_v1/"
          "weights/epoch5.pt"
    ),

    "R3_E6": (
        ROOT
        / "runs/teachers_r3/"
          "teacher_s_r3_proxy_v1/"
          "weights/epoch6.pt"
    ),
}

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
]

IOU_THRESHOLD = 0.50
PRED_FLOOR = 0.05

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

OUT = (
    ROOT
    / "reports/final_selection_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


def box_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def load_gt(label_path):

    result = []

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return result

    for line in text.splitlines():

        cls, x, y, w, h = (
            line.split()
        )

        cls = int(
            float(cls)
        )

        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)

        result.append({
            "cls": cls,

            "box": (
                x - w / 2,
                y - h / 2,
                x + w / 2,
                y + h / 2,
            ),
        })

    return result


images = sorted([
    p
    for p in VAL_IMAGES.rglob("*")
    if (
        p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTS
    )
])


if len(images) != 3000:

    raise RuntimeError(
        f"Expected 3000 VAL images, "
        f"got {len(images)}"
    )


for model_path in MODELS.values():

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )


print("=" * 112)
print(
    "FINAL CANDIDATE "
    "FIXED-CONFIDENCE GT RECALL"
)
print("=" * 112)

print(
    "Images:",
    len(images)
)

print(
    "IoU:",
    IOU_THRESHOLD
)

print(
    "Thresholds:",
    THRESHOLDS
)

print("=" * 112)


all_results = {}


for model_name, model_path in MODELS.items():

    print()
    print("=" * 112)

    print(
        model_name,
        model_path
    )

    print("=" * 112)


    stats = {
        threshold: {
            0: {
                "gt": 0,
                "tp": 0,
            },

            1: {
                "gt": 0,
                "tp": 0,
            },
        }
        for threshold in THRESHOLDS
    }


    model = YOLO(
        str(model_path)
    )


    for index, image in enumerate(
        images,
        start=1,
    ):

        label = (
            VAL_LABELS
            / f"{image.stem}.txt"
        )

        if not label.exists():

            raise FileNotFoundError(
                label
            )


        gt = load_gt(
            label
        )


        with torch.inference_mode():

            result = model.predict(
                source=str(image),
                imgsz=768,
                conf=PRED_FLOOR,
                iou=0.70,
                max_det=300,
                device=0,
                batch=1,
                verbose=False,
            )[0]


        predictions = []


        if (
            result.boxes is not None
            and len(result.boxes)
        ):

            xyxyn = (
                result.boxes.xyxyn
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

            confs = (
                result.boxes.conf
                .detach()
                .cpu()
                .tolist()
            )


            predictions = [
                {
                    "cls":
                        int(cls),

                    "conf":
                        float(conf),

                    "box":
                        tuple(
                            float(x)
                            for x in box
                        ),
                }

                for box, cls, conf
                in zip(
                    xyxyn,
                    classes,
                    confs,
                )
            ]


        for threshold in THRESHOLDS:

            selected = [
                prediction
                for prediction
                in predictions
                if (
                    prediction["conf"]
                    >= threshold
                )
            ]


            for cls in [0, 1]:

                gt_cls = [
                    item
                    for item in gt
                    if item["cls"]
                    == cls
                ]

                pred_cls = [
                    item
                    for item in selected
                    if item["cls"]
                    == cls
                ]


                stats[
                    threshold
                ][cls]["gt"] += len(
                    gt_cls
                )


                used_predictions = set()


                # Match strongest predictions first.
                pred_order = sorted(
                    range(
                        len(pred_cls)
                    ),
                    key=lambda i:
                        pred_cls[i][
                            "conf"
                        ],
                    reverse=True,
                )


                matched_gt = set()


                for pi in pred_order:

                    best_gt = None
                    best_iou = 0.0


                    for gi, target in enumerate(
                        gt_cls
                    ):

                        if gi in matched_gt:
                            continue


                        iou = box_iou(
                            pred_cls[pi][
                                "box"
                            ],
                            target["box"],
                        )


                        if iou > best_iou:

                            best_iou = iou
                            best_gt = gi


                    if (
                        best_gt is not None
                        and best_iou
                        >= IOU_THRESHOLD
                    ):

                        matched_gt.add(
                            best_gt
                        )

                        used_predictions.add(
                            pi
                        )


                stats[
                    threshold
                ][cls]["tp"] += len(
                    matched_gt
                )


        del result


        if (
            index % 50 == 0
            and torch.cuda.is_available()
        ):

            gc.collect()
            torch.cuda.empty_cache()


        if (
            index % 100 == 0
            or index == len(images)
        ):

            print(
                f"\r{model_name} "
                f"{index}/{len(images)}",
                end="",
                flush=True,
            )


    print()


    model_result = {}


    for threshold in THRESHOLDS:

        fire_gt = stats[
            threshold
        ][0]["gt"]

        fire_tp = stats[
            threshold
        ][0]["tp"]

        smoke_gt = stats[
            threshold
        ][1]["gt"]

        smoke_tp = stats[
            threshold
        ][1]["tp"]


        fire_recall = (
            fire_tp
            / fire_gt
        )

        smoke_recall = (
            smoke_tp
            / smoke_gt
        )

        macro_recall = (
            fire_recall
            + smoke_recall
        ) / 2


        model_result[
            str(threshold)
        ] = {
            "fire_gt":
                fire_gt,

            "fire_tp":
                fire_tp,

            "fire_recall":
                fire_recall,

            "smoke_gt":
                smoke_gt,

            "smoke_tp":
                smoke_tp,

            "smoke_recall":
                smoke_recall,

            "macro_recall":
                macro_recall,
        }


        print(
            f"conf={threshold:.2f} "
            f"Fire={fire_tp}/{fire_gt} "
            f"R={fire_recall:.4f} | "
            f"Smoke={smoke_tp}/{smoke_gt} "
            f"R={smoke_recall:.4f} | "
            f"MacroR={macro_recall:.4f}"
        )


    all_results[
        model_name
    ] = model_result


    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


json_path = (
    OUT
    / "fixed_conf_positive_recall.json"
)


json_path.write_text(
    json.dumps(
        all_results,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 130)
print(
    "COMPACT FIXED-CONFIDENCE RECALL"
)
print("=" * 130)

print(
    f"{'MODEL':<10}"
    f"{'CONF':>8}"
    f"{'FIRE R':>12}"
    f"{'SMOKE R':>12}"
    f"{'MACRO R':>12}"
)

print("-" * 130)


for model_name, values in all_results.items():

    for threshold in THRESHOLDS:

        row = values[
            str(threshold)
        ]

        print(
            f"{model_name:<10}"
            f"{threshold:>8.2f}"
            f"{row['fire_recall']:>12.4f}"
            f"{row['smoke_recall']:>12.4f}"
            f"{row['macro_recall']:>12.4f}"
        )


print()
print(
    "JSON:",
    json_path
)

print()

print(
    "RESULT: FIXED-CONFIDENCE "
    "POSITIVE RECALL PASS"
)

print("=" * 130)
