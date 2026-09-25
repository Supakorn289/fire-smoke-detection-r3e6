#!/usr/bin/env python3

from pathlib import Path
import csv
import json
import gc

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "datasets/bootstrap_v1/data.yaml"

RUNS = ROOT / "runs/teachers_r0"

REPORT_DIR = ROOT / "reports/teachers_r0"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_REPORT = REPORT_DIR / "test_metrics.csv"
JSON_REPORT = REPORT_DIR / "test_metrics.json"


TEACHERS = {
    "teacher_n_r0": (
        RUNS
        / "teacher_n_r0"
        / "weights"
        / "best.pt"
    ),

    "teacher_s_r0": (
        RUNS
        / "teacher_s_r0"
        / "weights"
        / "best.pt"
    ),

    "teacher_m_r0": (
        RUNS
        / "teacher_m_r0"
        / "weights"
        / "best.pt"
    ),
}


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return 0.0


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("=" * 80)
    print("ROUND-0 TEACHER TEST BENCHMARK")
    print("=" * 80)

    print(
        "GPU     :",
        torch.cuda.get_device_name(0)
    )

    print(
        "Dataset :",
        DATA
    )

    print("=" * 80)

    records = []

    for name, weights in TEACHERS.items():

        print()
        print("=" * 80)
        print(f"TESTING: {name}")
        print("=" * 80)

        if not weights.exists():

            print(
                f"ERROR: missing weights: {weights}"
            )

            continue

        model = YOLO(str(weights))

        metrics = model.val(
            data=str(DATA),
            split="test",

            imgsz=768,

            batch=16,
            device=0,

            conf=0.001,
            iou=0.60,

            plots=True,

            project=str(
                RUNS / "test"
            ),

            name=name,
            exist_ok=True,
        )

        # ----------------------------------------------------
        # Overall
        # ----------------------------------------------------

        overall = {
            "model": name,

            "precision":
                safe_float(metrics.box.mp),

            "recall":
                safe_float(metrics.box.mr),

            "map50":
                safe_float(metrics.box.map50),

            "map50_95":
                safe_float(metrics.box.map),
        }

        # ----------------------------------------------------
        # Per class
        # ----------------------------------------------------

        names = model.names

        p = metrics.box.p
        r = metrics.box.r
        ap50 = metrics.box.ap50
        ap = metrics.box.ap

        for cls_id, cls_name in names.items():

            if cls_id >= len(p):
                continue

            overall[
                f"{cls_name}_precision"
            ] = safe_float(p[cls_id])

            overall[
                f"{cls_name}_recall"
            ] = safe_float(r[cls_id])

            overall[
                f"{cls_name}_map50"
            ] = safe_float(ap50[cls_id])

            overall[
                f"{cls_name}_map50_95"
            ] = safe_float(ap[cls_id])

        records.append(overall)

        print()
        print(
            f"P          : "
            f"{overall['precision']:.4f}"
        )

        print(
            f"R          : "
            f"{overall['recall']:.4f}"
        )

        print(
            f"mAP50      : "
            f"{overall['map50']:.4f}"
        )

        print(
            f"mAP50-95   : "
            f"{overall['map50_95']:.4f}"
        )

        print()

        for cls in ("fire", "smoke"):

            if f"{cls}_precision" not in overall:
                continue

            print(
                f"{cls.upper():5} "
                f"P={overall[f'{cls}_precision']:.4f} "
                f"R={overall[f'{cls}_recall']:.4f} "
                f"mAP50={overall[f'{cls}_map50']:.4f} "
                f"mAP50-95="
                f"{overall[f'{cls}_map50_95']:.4f}"
            )

        del model
        del metrics

        gc.collect()
        torch.cuda.empty_cache()

    if not records:

        raise RuntimeError(
            "No teacher models were evaluated"
        )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    JSON_REPORT.write_text(
        json.dumps(
            records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    fieldnames = sorted({
        key
        for row in records
        for key in row.keys()
    })

    # model ไว้หน้าสุด
    fieldnames = [
        "model"
    ] + [
        x
        for x in fieldnames
        if x != "model"
    ]

    with CSV_REPORT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(records)

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranked = sorted(
        records,
        key=lambda x: x["map50_95"],
        reverse=True,
    )

    print()
    print("=" * 80)
    print("ROUND-0 RANKING — TEST SET")
    print("=" * 80)

    for i, row in enumerate(
        ranked,
        start=1
    ):

        print(
            f"{i}. "
            f"{row['model']:15} "
            f"P={row['precision']:.3f} "
            f"R={row['recall']:.3f} "
            f"mAP50={row['map50']:.3f} "
            f"mAP50-95={row['map50_95']:.3f}"
        )

    print()
    print(
        "CSV  :",
        CSV_REPORT
    )

    print(
        "JSON :",
        JSON_REPORT
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
