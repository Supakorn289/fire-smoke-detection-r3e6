#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import json
import cv2


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "datasets" / "bootstrap_v1"

REPORT_DIR = ROOT / "reports" / "bootstrap_v1"
REPORT_FILE = REPORT_DIR / "validation_report.json"

SPLITS = ("train", "val", "test")

VALID_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

VALID_CLASSES = {
    0: "fire",
    1: "smoke",
}

# Floating-point tolerance
#
# เช่น:
# x2 = 1.0000005
#
# ถือว่าเกิดจากการปัดเศษ ไม่ใช่ bbox ผิดจริง
EPS = 1e-6

# Warning thresholds
EXTREMELY_TINY_AREA = 0.00005
EXTREMELY_LARGE_AREA = 0.80

# Image size warning
MIN_IMAGE_WIDTH = 160
MIN_IMAGE_HEIGHT = 120


# ============================================================
# HELPERS
# ============================================================

def get_images(folder: Path):

    if not folder.exists():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in VALID_EXTS
    )


def get_labels(folder: Path):

    if not folder.exists():
        return []

    return sorted(
        p for p in folder.glob("*.txt")
        if p.is_file()
    )


# ============================================================
# VALIDATE ONE SPLIT
# ============================================================

def validate_split(split: str):

    img_dir = DATA / "images" / split
    lbl_dir = DATA / "labels" / split

    stats = Counter()

    issues = []

    images = get_images(img_dir)
    labels = get_labels(lbl_dir)

    image_stems = {
        p.stem
        for p in images
    }

    label_stems = {
        p.stem
        for p in labels
    }

    # --------------------------------------------------------
    # Directory checks
    # --------------------------------------------------------

    if not img_dir.exists():
        stats["missing_image_directory"] += 1

        issues.append({
            "type": "missing_image_directory",
            "path": str(img_dir),
        })

        return stats, issues

    if not lbl_dir.exists():
        stats["missing_label_directory"] += 1

        issues.append({
            "type": "missing_label_directory",
            "path": str(lbl_dir),
        })

        return stats, issues

    # --------------------------------------------------------
    # Orphan labels
    # --------------------------------------------------------

    orphan_labels = label_stems - image_stems

    for stem in sorted(orphan_labels):

        stats["orphan_label"] += 1

        issues.append({
            "type": "orphan_label",
            "label": str(lbl_dir / f"{stem}.txt"),
        })

    # --------------------------------------------------------
    # Validate images + labels
    # --------------------------------------------------------

    for image_path in images:

        stats["images"] += 1

        image = cv2.imread(str(image_path))

        if image is None:

            stats["corrupt_image"] += 1

            issues.append({
                "type": "corrupt_image",
                "image": str(image_path),
            })

            continue

        height, width = image.shape[:2]

        if (
            width < MIN_IMAGE_WIDTH
            or height < MIN_IMAGE_HEIGHT
        ):
            stats["very_small_image"] += 1

        label_path = (
            lbl_dir
            / f"{image_path.stem}.txt"
        )

        # ----------------------------------------------------
        # Missing label
        # ----------------------------------------------------

        if not label_path.exists():

            stats["missing_label_file"] += 1

            issues.append({
                "type": "missing_label_file",
                "image": str(image_path),
            })

            continue

        text = label_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).strip()

        # Empty .txt = valid negative image
        if not text:

            stats["negative"] += 1
            continue

        # ----------------------------------------------------
        # Validate each bbox
        # ----------------------------------------------------

        for line_no, line in enumerate(
            text.splitlines(),
            start=1,
        ):

            stats["boxes"] += 1

            parts = line.split()

            # YOLO detection:
            # class x_center y_center width height
            if len(parts) != 5:

                stats["malformed"] += 1

                issues.append({
                    "type": "malformed",
                    "label": str(label_path),
                    "line": line_no,
                    "content": line,
                })

                continue

            try:

                cls = int(parts[0])

                x, y, bw, bh = map(
                    float,
                    parts[1:]
                )

            except ValueError:

                stats["parse_error"] += 1

                issues.append({
                    "type": "parse_error",
                    "label": str(label_path),
                    "line": line_no,
                    "content": line,
                })

                continue

            # ------------------------------------------------
            # Class ID
            # ------------------------------------------------

            if cls not in VALID_CLASSES:

                stats["invalid_class"] += 1

                issues.append({
                    "type": "invalid_class",
                    "label": str(label_path),
                    "line": line_no,
                    "class_id": cls,
                })

            else:

                stats[
                    f"class_{cls}_{VALID_CLASSES[cls]}"
                ] += 1

            # ------------------------------------------------
            # Normalized coordinate range
            # ------------------------------------------------

            if (
                x < -EPS
                or x > 1 + EPS
                or y < -EPS
                or y > 1 + EPS
                or bw < -EPS
                or bw > 1 + EPS
                or bh < -EPS
                or bh > 1 + EPS
            ):

                stats[
                    "outside_normalized_range"
                ] += 1

                issues.append({
                    "type":
                        "outside_normalized_range",
                    "label": str(label_path),
                    "line": line_no,
                    "values": [
                        x,
                        y,
                        bw,
                        bh,
                    ],
                })

            # ------------------------------------------------
            # Zero/negative area
            # ------------------------------------------------

            if bw <= 0 or bh <= 0:

                stats["zero_area"] += 1

                issues.append({
                    "type": "zero_area",
                    "label": str(label_path),
                    "line": line_no,
                    "width": bw,
                    "height": bh,
                })

                continue

            # ------------------------------------------------
            # Convert YOLO bbox -> corners
            # ------------------------------------------------

            x1 = x - bw / 2
            y1 = y - bh / 2
            x2 = x + bw / 2
            y2 = y + bh / 2

            # EPS tolerance is intentional.
            #
            # Example:
            # x2 = 1.0000005
            #
            # is accepted because it can result from
            # decimal rounding after clipping.
            if (
                x1 < -EPS
                or y1 < -EPS
                or x2 > 1 + EPS
                or y2 > 1 + EPS
            ):

                stats["box_outside_image"] += 1

                overflow = max(
                    max(0.0, -x1),
                    max(0.0, -y1),
                    max(0.0, x2 - 1.0),
                    max(0.0, y2 - 1.0),
                )

                issues.append({
                    "type": "box_outside_image",
                    "label": str(label_path),
                    "line": line_no,
                    "class_id": cls,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "overflow": overflow,
                })

            # ------------------------------------------------
            # Bounding-box area statistics
            # ------------------------------------------------

            area = bw * bh

            if area < EXTREMELY_TINY_AREA:

                stats[
                    "extremely_tiny_box"
                ] += 1

            if area > EXTREMELY_LARGE_AREA:

                stats[
                    "extremely_large_box"
                ] += 1

    return stats, issues


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("BOOTSTRAP V1 AUTOMATED VALIDATION")
    print("=" * 72)

    print(f"Dataset : {DATA}")
    print(f"Classes : {VALID_CLASSES}")
    print(f"EPS     : {EPS}")
    print("=" * 72)

    # These make dataset structurally unsafe for training
    CRITICAL_KEYS = {
        "missing_image_directory",
        "missing_label_directory",
        "corrupt_image",
        "missing_label_file",
        "orphan_label",
        "malformed",
        "parse_error",
        "invalid_class",
        "outside_normalized_range",
        "zero_area",
        "box_outside_image",
    }

    report = {
        "dataset": str(DATA),
        "eps": EPS,
        "classes": VALID_CLASSES,
        "splits": {},
    }

    total_stats = Counter()
    all_issues = []

    # --------------------------------------------------------
    # Each split
    # --------------------------------------------------------

    for split in SPLITS:

        stats, issues = validate_split(split)

        total_stats.update(stats)
        all_issues.extend(issues)

        report["splits"][split] = {
            "stats": dict(stats),
            "issues": issues,
        }

        print()
        print(f"[{split.upper()}]")
        print("-" * 72)

        display_order = [
            "images",
            "boxes",
            "class_0_fire",
            "class_1_smoke",
            "negative",
            "very_small_image",
            "extremely_tiny_box",
            "extremely_large_box",
            "missing_label_file",
            "orphan_label",
            "corrupt_image",
            "malformed",
            "parse_error",
            "invalid_class",
            "outside_normalized_range",
            "zero_area",
            "box_outside_image",
        ]

        shown = set()

        for key in display_order:

            value = stats[key]

            if value:

                print(
                    f"{key:30}: {value}"
                )

                shown.add(key)

        # Show any additional counters
        for key in sorted(stats):

            if key in shown:
                continue

            if stats[key]:

                print(
                    f"{key:30}: "
                    f"{stats[key]}"
                )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    critical_count = sum(
        total_stats[key]
        for key in CRITICAL_KEYS
    )

    warning_count = (
        total_stats["very_small_image"]
        + total_stats["extremely_tiny_box"]
        + total_stats["extremely_large_box"]
    )

    report["summary"] = {
        "total_images":
            total_stats["images"],

        "total_boxes":
            total_stats["boxes"],

        "fire_boxes":
            total_stats["class_0_fire"],

        "smoke_boxes":
            total_stats["class_1_smoke"],

        "negative_images":
            total_stats["negative"],

        "critical_issues":
            critical_count,

        "warnings":
            warning_count,
    }

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"Images             : "
        f"{total_stats['images']}"
    )

    print(
        f"Boxes              : "
        f"{total_stats['boxes']}"
    )

    print(
        f"Fire boxes         : "
        f"{total_stats['class_0_fire']}"
    )

    print(
        f"Smoke boxes        : "
        f"{total_stats['class_1_smoke']}"
    )

    print(
        f"Negative images    : "
        f"{total_stats['negative']}"
    )

    print(
        f"Critical issues    : "
        f"{critical_count}"
    )

    print(
        f"Warnings           : "
        f"{warning_count}"
    )

    print()
    print(f"Report             : {REPORT_FILE}")

    print("=" * 72)

    if critical_count == 0:

        print("RESULT: PASS")

        if warning_count:
            print(
                "NOTE  : Warnings remain, "
                "but they do not block training."
            )

    else:

        print(
            f"RESULT: FAIL "
            f"({critical_count} critical issues)"
        )

        print(
            "ACTION: Fix critical issues "
            "before training."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
