#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import gc
import json

import torch
from ultralytics import YOLO

from eval_r1_size_aware import (
    IMAGES,
    LABELS,
    OUT,
    MODELS,
    THRESHOLDS,
    IOU_THRESHOLD,
    PREDICTION_CONF_FLOOR,
    initialize_stats,
    load_gt,
    evaluate_threshold,
    make_report,
    print_report,
)


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


CSV_FIELDS = [
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


def save_csv(path, rows):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["n", "s", "m"],
    )

    args = parser.parse_args()

    model_name = args.model

    image_paths = sorted(
        [
            p
            for p in IMAGES.iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda p: p.name,
    )

    if len(image_paths) != 3000:
        raise RuntimeError(
            f"Expected 3000 VAL images, got {len(image_paths)}"
        )

    model_path = MODELS[model_name]

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 100)
    print("R1 SIZE-AWARE LOW-MEMORY EVALUATION")
    print("=" * 100)
    print("Teacher :", model_name.upper())
    print("Model   :", model_path)
    print("Images  :", len(image_paths))
    print("Mode    : ONE IMAGE AT A TIME")
    print("imgsz   : 768")
    print("IoU     :", IOU_THRESHOLD)
    print("Conf    :", THRESHOLDS)
    print("=" * 100)

    model = YOLO(
        str(model_path)
    )

    stats = initialize_stats()

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        # ----------------------------------------------------
        # Predict ONE image only
        # ----------------------------------------------------

        with torch.inference_mode():
            results = model.predict(
                source=str(image_path),
                imgsz=768,
                conf=PREDICTION_CONF_FLOOR,
                iou=0.70,
                max_det=300,
                device=0,
                batch=1,
                stream=False,
                verbose=False,
            )

        if len(results) != 1:
            raise RuntimeError(
                f"Expected 1 prediction result, got {len(results)}"
            )

        result = results[0]

        label_path = (
            LABELS
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                label_path
            )

        height, width = result.orig_shape

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

                predictions.append(
                    {
                        "box": tuple(
                            float(x)
                            for x in box
                        ),
                        "conf": float(conf),
                        "cls": int(cls),
                    }
                )

        for threshold in THRESHOLDS:

            evaluate_threshold(
                gt,
                predictions,
                width,
                height,
                threshold,
                stats,
            )

        # ----------------------------------------------------
        # Explicitly release per-image objects
        # ----------------------------------------------------

        del predictions
        del gt
        del result
        del results

        # Python GC every 10 images
        if index % 10 == 0:
            gc.collect()

        # CUDA cache every 50 images
        if (
            index % 50 == 0
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()

        if (
            index % 25 == 0
            or index == len(image_paths)
        ):

            print(
                f"\rTeacher {model_name.upper()} "
                f"{index:4d}/3000 "
                f"({index / 3000 * 100:5.1f}%)",
                end="",
                flush=True,
            )

    print()

    rows = make_report(
        model_name,
        stats,
    )

    print_report(
        model_name,
        rows,
    )

    json_path = (
        OUT
        / f"teacher_{model_name}_r1_size_aware.json"
    )

    csv_path = (
        OUT
        / f"teacher_{model_name}_r1_size_aware.csv"
    )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )

    save_csv(
        csv_path,
        rows,
    )

    print()
    print("=" * 100)
    print(
        f"TEACHER {model_name.upper()} SIZE-AWARE COMPLETE"
    )
    print("=" * 100)
    print("JSON :", json_path)
    print("CSV  :", csv_path)
    print()
    print(
        f"RESULT: TEACHER {model_name.upper()} "
        "R1 SIZE-AWARE PASS"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
