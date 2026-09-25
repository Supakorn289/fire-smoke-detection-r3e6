#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import gc
import json

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

POOL = {
    "fire_only": ROOT / "dataset_raw/fire_only",
    "smoke_only": ROOT / "dataset_raw/smoke_only",
    "fire_smoke": ROOT / "dataset_raw/fire_smoke",
    "negative": ROOT / "dataset_raw/negative/dfire_background",
}

MODELS = {
    "teacher_n_r0":
        ROOT / "runs/teachers_r0/teacher_n_r0/weights/best.pt",

    "teacher_s_r0":
        ROOT / "runs/teachers_r0/teacher_s_r0/weights/best.pt",

    "teacher_m_r0":
        ROOT / "runs/teachers_r0/teacher_m_r0/weights/best.pt",
}

OUT = ROOT / "reports/auto_curator_r0"
CACHE_DIR = OUT / "predictions"

MANIFEST = OUT / "dfire_2000_manifest.csv"
SUMMARY = OUT / "prediction_cache_summary.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

IMGSZ = 768

# เก็บ prediction ต่ำไว้ด้วย
# Auto-Curator จะเป็นคนตัดสิน threshold ภายหลัง
CONF = 0.05

IOU = 0.60

CHUNK = 64
BATCH = 16


def get_split(image: Path):

    parts = image.stem.split("_", 2)

    if (
        len(parts) >= 3
        and parts[0] == "dfire"
        and parts[1] in {"train", "val", "test"}
    ):
        return parts[1]

    return "unknown"


def get_images(folder: Path):

    return sorted(
        p.resolve()
        for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTS
    )


def build_manifest():

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for category, folder in POOL.items():

        for image in get_images(folder):

            label = image.with_suffix(".txt")

            rows.append({
                "image": str(image),
                "label":
                    str(label.resolve())
                    if label.exists()
                    else "",
                "category": category,
                "source": "dfire",
                "source_split": get_split(image),
            })

    rows.sort(
        key=lambda x: (
            x["category"],
            x["image"],
        )
    )

    with MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "label",
                "category",
                "source",
                "source_split",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def prediction_record(result):

    predictions = []

    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return predictions

    xyxy = boxes.xyxyn.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()

    for box, cls, conf in zip(
        xyxy,
        classes,
        confs,
    ):

        predictions.append({
            "class_id": int(cls),
            "confidence": round(float(conf), 6),
            "xyxy": [
                round(float(v), 7)
                for v in box
            ],
        })

    return predictions


def run_teacher(name, weights, rows):

    if not weights.exists():
        raise FileNotFoundError(weights)

    cache_file = CACHE_DIR / f"{name}.jsonl"
    temp_file = CACHE_DIR / f"{name}.jsonl.tmp"

    print()
    print("=" * 72)
    print(f"TEACHER : {name}")
    print(f"WEIGHTS : {weights}")
    print("=" * 72)

    model = YOLO(str(weights))

    names = model.names

    print("Classes :", names)

    if (
        names.get(0) != "fire"
        or names.get(1) != "smoke"
    ):
        raise RuntimeError(
            f"Unexpected class mapping: {names}"
        )

    stats = Counter()

    with temp_file.open(
        "w",
        encoding="utf-8",
    ) as out:

        total = len(rows)

        for start in range(
            0,
            total,
            CHUNK,
        ):

            chunk = rows[
                start:start + CHUNK
            ]

            sources = [
                row["image"]
                for row in chunk
            ]

            results = model.predict(
                source=sources,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                batch=BATCH,
                device=0,
                verbose=False,
            )

            if len(results) != len(chunk):
                raise RuntimeError(
                    "Prediction count does not "
                    "match source count."
                )

            for row, result in zip(
                chunk,
                results,
            ):

                preds = prediction_record(
                    result
                )

                record = {
                    "image": row["image"],
                    "predictions": preds,
                }

                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                stats["images"] += 1
                stats["detections"] += len(preds)

                for pred in preds:

                    cls = pred["class_id"]

                    stats[
                        f"class_{cls}"
                    ] += 1

                    if pred["confidence"] >= 0.25:
                        stats[
                            f"class_{cls}_conf25"
                        ] += 1

            done = min(
                start + CHUNK,
                total,
            )

            print(
                f"\r{name}: "
                f"{done}/{total}",
                end="",
                flush=True,
            )

    print()

    temp_file.replace(cache_file)

    del model

    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": name,
        "cache": str(cache_file),
        "images": stats["images"],
        "detections": stats["detections"],
        "fire_detections": stats["class_0"],
        "smoke_detections": stats["class_1"],
        "fire_conf25": stats["class_0_conf25"],
        "smoke_conf25": stats["class_1_conf25"],
    }


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available"
        )

    print("=" * 72)
    print("ROUND-0 TEACHER PREDICTION CACHE")
    print("=" * 72)

    print(
        "GPU :",
        torch.cuda.get_device_name(0)
    )

    rows = build_manifest()

    print("Images :", len(rows))
    print("Manifest:", MANIFEST)

    if len(rows) != 2000:

        print()
        print(
            "WARNING: expected 2000 "
            f"candidate images, found {len(rows)}"
        )

    summaries = []

    for name, weights in MODELS.items():

        summaries.append(
            run_teacher(
                name,
                weights,
                rows,
            )
        )

    SUMMARY.write_text(
        json.dumps(
            summaries,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("PREDICTION CACHE COMPLETE")
    print("=" * 72)

    for row in summaries:

        print()
        print(row["model"])
        print(
            "  images          :",
            row["images"]
        )
        print(
            "  detections      :",
            row["detections"]
        )
        print(
            "  fire @ >=0.25   :",
            row["fire_conf25"]
        )
        print(
            "  smoke @ >=0.25  :",
            row["smoke_conf25"]
        )

    print()
    print("Summary :", SUMMARY)
    print("=" * 72)


if __name__ == "__main__":
    main()
