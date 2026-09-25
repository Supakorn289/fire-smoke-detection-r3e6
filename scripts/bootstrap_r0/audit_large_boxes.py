#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT
    / "datasets"
    / "bootstrap_v1"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "bootstrap_v1"
)

REPORT_FILE = (
    REPORT_DIR
    / "large_boxes.csv"
)

SPLITS = (
    "train",
    "val",
    "test",
)

CLASS_NAMES = {
    0: "fire",
    1: "smoke",
}

# bbox กินพื้นที่มากกว่า 80% ของภาพ
# จะถูกจัดเป็น warning
LARGE_BOX_THRESHOLD = 0.80


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    stats = Counter()

    print("=" * 72)
    print("BOOTSTRAP V1 — EXTREMELY LARGE BOX AUDIT")
    print("=" * 72)

    print(
        f"Dataset   : {DATASET}"
    )

    print(
        f"Threshold : "
        f"{LARGE_BOX_THRESHOLD:.0%} of image"
    )

    print("=" * 72)

    for split in SPLITS:

        label_dir = (
            DATASET
            / "labels"
            / split
        )

        if not label_dir.exists():

            print(
                f"WARNING: missing directory: "
                f"{label_dir}"
            )

            continue

        split_count = 0

        for label_path in sorted(
            label_dir.glob("*.txt")
        ):

            text = label_path.read_text(
                encoding="utf-8",
                errors="replace"
            ).strip()

            if not text:
                continue

            for line_no, line in enumerate(
                text.splitlines(),
                start=1
            ):

                parts = line.split()

                if len(parts) != 5:
                    continue

                try:

                    class_id = int(
                        parts[0]
                    )

                    x_center = float(
                        parts[1]
                    )

                    y_center = float(
                        parts[2]
                    )

                    width = float(
                        parts[3]
                    )

                    height = float(
                        parts[4]
                    )

                except ValueError:
                    continue

                area = width * height

                if area <= LARGE_BOX_THRESHOLD:
                    continue

                class_name = CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}"
                )

                image_stem = (
                    label_path.stem
                )

                rows.append({
                    "split": split,
                    "image_stem": image_stem,
                    "label": label_path.name,
                    "line": line_no,
                    "class_id": class_id,
                    "class_name": class_name,
                    "x_center": round(
                        x_center,
                        10
                    ),
                    "y_center": round(
                        y_center,
                        10
                    ),
                    "width": round(
                        width,
                        10
                    ),
                    "height": round(
                        height,
                        10
                    ),
                    "bbox_area": round(
                        area,
                        10
                    ),
                    "bbox_percent": round(
                        area * 100,
                        4
                    ),
                })

                split_count += 1

                stats[
                    f"split_{split}"
                ] += 1

                stats[
                    f"class_{class_id}"
                ] += 1

        print(
            f"{split.upper():8}: "
            f"{split_count} large boxes"
        )

    # --------------------------------------------------------
    # Sort largest first
    # --------------------------------------------------------

    rows.sort(
        key=lambda r: r["bbox_area"],
        reverse=True
    )

    # --------------------------------------------------------
    # CSV report
    # --------------------------------------------------------

    fieldnames = [
        "split",
        "image_stem",
        "label",
        "line",
        "class_id",
        "class_name",
        "x_center",
        "y_center",
        "width",
        "height",
        "bbox_area",
        "bbox_percent",
    ]

    with REPORT_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(rows)

    fire = stats["class_0"]
    smoke = stats["class_1"]

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"Total large boxes : "
        f"{total}"
    )

    print(
        f"Fire              : "
        f"{fire}"
    )

    print(
        f"Smoke             : "
        f"{smoke}"
    )

    if total:

        fire_pct = (
            fire / total * 100
        )

        smoke_pct = (
            smoke / total * 100
        )

        print(
            f"Fire %            : "
            f"{fire_pct:.1f}%"
        )

        print(
            f"Smoke %           : "
            f"{smoke_pct:.1f}%"
        )

    print()
    print(
        f"Train             : "
        f"{stats['split_train']}"
    )

    print(
        f"Val               : "
        f"{stats['split_val']}"
    )

    print(
        f"Test              : "
        f"{stats['split_test']}"
    )

    print()
    print(
        f"Report            : "
        f"{REPORT_FILE}"
    )

    # --------------------------------------------------------
    # Top largest boxes
    # --------------------------------------------------------

    if rows:

        print()
        print("=" * 72)
        print("TOP 10 LARGEST BOXES")
        print("=" * 72)

        for i, row in enumerate(
            rows[:10],
            start=1
        ):

            print(
                f"{i:2}. "
                f"{row['split']:5} "
                f"{row['class_name']:6} "
                f"{row['bbox_percent']:7.2f}% "
                f"{row['image_stem']}"
            )

    print()
    print("=" * 72)

    if total == 34:

        print(
            "RESULT: MATCH — "
            "found the expected 34 warning boxes."
        )

    else:

        print(
            f"RESULT: INFO — "
            f"found {total} boxes above "
            f"{LARGE_BOX_THRESHOLD:.0%} area."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
