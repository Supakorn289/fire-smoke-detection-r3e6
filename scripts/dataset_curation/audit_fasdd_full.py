#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import sys

import cv2


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "sources" / "fasdd" / "FASDD_CV"

IMAGES_DIR = DATA / "images"

YOLO_DIR = (
    DATA
    / "annotations"
    / "YOLO_CV"
)

LABELS_DIR = YOLO_DIR / "labels"

REPORT_DIR = ROOT / "reports" / "fasdd"

REPORT_JSON = REPORT_DIR / "full_audit.json"

ISSUES_CSV = REPORT_DIR / "full_audit_issues.csv"


IMAGE_EXTS = {
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


# same size convention used in our previous dataset analysis
SMALL_AREA = 0.005
MEDIUM_AREA = 0.05

EXTREMELY_TINY_AREA = 0.00005
EXTREMELY_LARGE_AREA = 0.80

EPS = 1e-6


# ============================================================
# HELPERS
# ============================================================

def semantic_category(stem: str) -> str:

    s = stem.lower()

    if s.startswith("bothfireandsmoke"):
        return "fire_smoke"

    if s.startswith("fire"):
        return "fire_only"

    if s.startswith("smoke"):
        return "smoke_only"

    if (
        "neither" in s
        or "negative" in s
        or "nofire" in s
    ):
        return "negative"

    return "other"


def size_bucket(area: float) -> str:

    if area < SMALL_AREA:
        return "small"

    if area < MEDIUM_AREA:
        return "medium"

    return "large"


def read_split(path: Path):

    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    raw = [
        x.strip()
        for x in lines
        if x.strip()
    ]

    stems = [
        Path(x).stem
        for x in raw
    ]

    return raw, stems


def add_issue(
    issues,
    issue_type,
    stem,
    detail,
    split="",
):

    issues.append({
        "type": issue_type,
        "stem": stem,
        "split": split,
        "detail": detail,
    })


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 84)
    print("FASDD_CV FULL AUTOMATIC AUDIT")
    print("=" * 84)

    print("Dataset :", DATA.resolve())
    print("Images  :", IMAGES_DIR.resolve())
    print("Labels  :", LABELS_DIR.resolve())

    print("=" * 84)

    # --------------------------------------------------------
    # PATH CHECK
    # --------------------------------------------------------

    required = [
        IMAGES_DIR,
        LABELS_DIR,
        YOLO_DIR / "train.txt",
        YOLO_DIR / "val.txt",
        YOLO_DIR / "test.txt",
    ]

    for path in required:

        if not path.exists():

            print(
                f"ERROR: required path missing: {path}",
                file=sys.stderr,
            )

            sys.exit(1)

    # --------------------------------------------------------
    # DISCOVER FILES
    # --------------------------------------------------------

    image_paths = sorted(
        p
        for p in IMAGES_DIR.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
        )
    )

    label_paths = sorted(
        LABELS_DIR.glob("*.txt")
    )

    images_by_stem = {
        p.stem: p
        for p in image_paths
    }

    labels_by_stem = {
        p.stem: p
        for p in label_paths
    }

    image_stems = set(images_by_stem)
    label_stems = set(labels_by_stem)

    stats = Counter()

    categories = defaultdict(Counter)
    size_stats = defaultdict(Counter)
    split_stats = defaultdict(Counter)

    issues = []

    stats["images"] = len(image_paths)
    stats["label_files"] = len(label_paths)

    # ========================================================
    # IMAGE ↔ LABEL PAIRING
    # ========================================================

    missing_labels = sorted(
        image_stems - label_stems
    )

    orphan_labels = sorted(
        label_stems - image_stems
    )

    stats["missing_labels"] = len(
        missing_labels
    )

    stats["orphan_labels"] = len(
        orphan_labels
    )

    for stem in missing_labels:

        add_issue(
            issues,
            "missing_label",
            stem,
            str(images_by_stem[stem]),
        )

    for stem in orphan_labels:

        add_issue(
            issues,
            "orphan_label",
            stem,
            str(labels_by_stem[stem]),
        )

    # ========================================================
    # SPLIT AUDIT
    # ========================================================

    split_sets = {}
    split_lookup = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        raw, stems = read_split(
            YOLO_DIR / f"{split}.txt"
        )

        stem_set = set(stems)

        split_sets[split] = stem_set

        stats[f"split_{split}_lines"] = len(raw)
        stats[f"split_{split}_unique"] = len(stem_set)

        duplicate_count = (
            len(stems)
            - len(stem_set)
        )

        stats[
            f"split_{split}_duplicate_entries"
        ] = duplicate_count

        for stem in stem_set:

            if stem in split_lookup:

                stats["split_overlap"] += 1

                add_issue(
                    issues,
                    "split_overlap",
                    stem,
                    (
                        f"{split_lookup[stem]}"
                        f" + {split}"
                    ),
                )

            else:

                split_lookup[stem] = split

    split_union = (
        split_sets["train"]
        | split_sets["val"]
        | split_sets["test"]
    )

    stats["split_union"] = len(
        split_union
    )

    stats["train_val_overlap"] = len(
        split_sets["train"]
        & split_sets["val"]
    )

    stats["train_test_overlap"] = len(
        split_sets["train"]
        & split_sets["test"]
    )

    stats["val_test_overlap"] = len(
        split_sets["val"]
        & split_sets["test"]
    )

    images_without_split = sorted(
        image_stems - split_union
    )

    split_missing_images = sorted(
        split_union - image_stems
    )

    stats["images_without_split"] = len(
        images_without_split
    )

    stats["split_references_missing_image"] = len(
        split_missing_images
    )

    for stem in images_without_split:

        add_issue(
            issues,
            "image_without_split",
            stem,
            str(images_by_stem[stem]),
        )

    for stem in split_missing_images:

        add_issue(
            issues,
            "split_reference_missing_image",
            stem,
            "Referenced by split metadata",
        )

    # ========================================================
    # IMAGE + LABEL AUDIT
    # ========================================================

    total = len(image_paths)

    cv2.setNumThreads(0)

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        stem = image_path.stem

        category = semantic_category(
            stem
        )

        split = split_lookup.get(
            stem,
            "unknown",
        )

        categories[
            category
        ]["images"] += 1

        split_stats[
            split
        ]["images"] += 1

        # ----------------------------------------------------
        # IMAGE DECODE
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            stats["corrupt_images"] += 1

            add_issue(
                issues,
                "corrupt_image",
                stem,
                str(image_path),
                split,
            )

            continue

        height, width = image.shape[:2]

        if width <= 0 or height <= 0:

            stats["invalid_image_dimensions"] += 1

            add_issue(
                issues,
                "invalid_image_dimensions",
                stem,
                f"{width}x{height}",
                split,
            )

            continue

        if (
            width < 160
            or height < 120
        ):

            stats["very_small_images"] += 1

        # ----------------------------------------------------
        # LABEL EXISTENCE
        # ----------------------------------------------------

        label_path = labels_by_stem.get(
            stem
        )

        if label_path is None:
            continue

        text = label_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        # ----------------------------------------------------
        # EMPTY LABEL = NEGATIVE
        # ----------------------------------------------------

        if not text:

            stats["empty_labels"] += 1

            categories[
                category
            ]["empty"] += 1

            split_stats[
                split
            ]["negative_images"] += 1

            if category != "negative":

                stats[
                    "semantic_empty_mismatch"
                ] += 1

                add_issue(
                    issues,
                    "semantic_empty_mismatch",
                    stem,
                    f"filename_category={category}",
                    split,
                )

            continue

        # ----------------------------------------------------
        # POSITIVE LABEL
        # ----------------------------------------------------

        class_set = set()

        valid_boxes_this_image = 0

        for line_no, line in enumerate(
            text.splitlines(),
            start=1,
        ):

            stats["annotation_lines"] += 1

            parts = line.split()

            if len(parts) != 5:

                stats["malformed_lines"] += 1

                add_issue(
                    issues,
                    "malformed_line",
                    stem,
                    (
                        f"line={line_no}; "
                        f"value={line}"
                    ),
                    split,
                )

                continue

            try:

                cls = int(parts[0])

                xc, yc, bw, bh = map(
                    float,
                    parts[1:],
                )

            except ValueError:

                stats["parse_errors"] += 1

                add_issue(
                    issues,
                    "parse_error",
                    stem,
                    (
                        f"line={line_no}; "
                        f"value={line}"
                    ),
                    split,
                )

                continue

            if cls not in VALID_CLASSES:

                stats["invalid_classes"] += 1

                add_issue(
                    issues,
                    "invalid_class",
                    stem,
                    (
                        f"line={line_no}; "
                        f"class={cls}"
                    ),
                    split,
                )

                continue

            class_set.add(cls)

            # ------------------------------------------------
            # NORMALIZED COORDINATE CHECK
            # ------------------------------------------------

            if (
                xc < -EPS
                or xc > 1 + EPS
                or yc < -EPS
                or yc > 1 + EPS
                or bw < -EPS
                or bw > 1 + EPS
                or bh < -EPS
                or bh > 1 + EPS
            ):

                stats[
                    "invalid_normalized_coordinates"
                ] += 1

                add_issue(
                    issues,
                    "invalid_normalized_coordinates",
                    stem,
                    (
                        f"line={line_no}; "
                        f"value={line}"
                    ),
                    split,
                )

            if bw <= 0 or bh <= 0:

                stats["zero_area_boxes"] += 1

                add_issue(
                    issues,
                    "zero_area_box",
                    stem,
                    (
                        f"line={line_no}; "
                        f"value={line}"
                    ),
                    split,
                )

                continue

            stats["valid_boxes"] += 1

            valid_boxes_this_image += 1

            stats[f"class_{cls}"] += 1

            class_name = VALID_CLASSES[
                cls
            ]

            split_stats[
                split
            ][
                f"class_{cls}_boxes"
            ] += 1

            area = bw * bh

            bucket = size_bucket(
                area
            )

            size_stats[
                class_name
            ][bucket] += 1

            split_stats[
                split
            ][
                f"{class_name}_{bucket}"
            ] += 1

            # ------------------------------------------------
            # BOX FRAME BOUNDARY
            # ------------------------------------------------

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            if (
                x1 < -EPS
                or y1 < -EPS
                or x2 > 1 + EPS
                or y2 > 1 + EPS
            ):

                stats[
                    "boxes_outside_image"
                ] += 1

                add_issue(
                    issues,
                    "box_outside_image",
                    stem,
                    (
                        f"line={line_no}; "
                        f"x1={x1:.8f}; "
                        f"y1={y1:.8f}; "
                        f"x2={x2:.8f}; "
                        f"y2={y2:.8f}"
                    ),
                    split,
                )

            if area < EXTREMELY_TINY_AREA:

                stats[
                    "extremely_tiny_boxes"
                ] += 1

            if area > EXTREMELY_LARGE_AREA:

                stats[
                    "extremely_large_boxes"
                ] += 1

        if valid_boxes_this_image > 0:

            stats["positive_images"] += 1

            split_stats[
                split
            ]["positive_images"] += 1

        # ----------------------------------------------------
        # FILENAME SEMANTIC CHECK
        # ----------------------------------------------------

        expected = None

        if category == "fire_only":

            expected = {0}

        elif category == "smoke_only":

            expected = {1}

        elif category == "fire_smoke":

            expected = {0, 1}

        elif category == "negative":

            expected = set()

        if (
            expected is not None
            and class_set != expected
        ):

            stats[
                "semantic_class_mismatch"
            ] += 1

            categories[
                category
            ][
                "class_mismatch"
            ] += 1

            add_issue(
                issues,
                "semantic_class_mismatch",
                stem,
                (
                    f"expected={sorted(expected)}; "
                    f"actual={sorted(class_set)}"
                ),
                split,
            )

        else:

            categories[
                category
            ][
                "class_match"
            ] += 1

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            index % 5000 == 0
            or index == total
        ):

            pct = (
                index
                / total
                * 100
            )

            print(
                f"\rAuditing "
                f"{index:6}/{total} "
                f"({pct:5.1f}%)",
                end="",
                flush=True,
            )

    print()

    # ========================================================
    # FINAL INTEGRITY STATUS
    # ========================================================

    critical_keys = [
        "missing_labels",
        "orphan_labels",
        "corrupt_images",
        "invalid_image_dimensions",
        "malformed_lines",
        "parse_errors",
        "invalid_classes",
        "invalid_normalized_coordinates",
        "zero_area_boxes",
        "split_overlap",
        "images_without_split",
        "split_references_missing_image",
        "split_train_duplicate_entries",
        "split_val_duplicate_entries",
        "split_test_duplicate_entries",
    ]

    critical_issues = sum(
        stats[key]
        for key in critical_keys
    )

    warning_keys = [
        "boxes_outside_image",
        "extremely_tiny_boxes",
        "extremely_large_boxes",
        "semantic_class_mismatch",
        "semantic_empty_mismatch",
        "very_small_images",
    ]

    warnings = sum(
        stats[key]
        for key in warning_keys
    )

    # ========================================================
    # WRITE CSV
    # ========================================================

    with ISSUES_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "type",
                "stem",
                "split",
                "detail",
            ],
        )

        writer.writeheader()
        writer.writerows(issues)

    # ========================================================
    # WRITE JSON
    # ========================================================

    report = {
        "dataset":
            str(DATA.resolve()),

        "class_contract": {
            "0": "fire",
            "1": "smoke",
        },

        "stats":
            dict(stats),

        "categories": {
            key: dict(value)
            for key, value
            in categories.items()
        },

        "object_sizes": {
            key: dict(value)
            for key, value
            in size_stats.items()
        },

        "splits": {
            key: dict(value)
            for key, value
            in split_stats.items()
        },

        "critical_issues":
            critical_issues,

        "warnings":
            warnings,

        "result": (
            "STRUCTURAL_PASS"
            if critical_issues == 0
            else "STRUCTURAL_FAIL"
        ),
    }

    REPORT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 84)
    print("FASDD FULL AUDIT SUMMARY")
    print("=" * 84)

    summary_keys = [
        "images",
        "label_files",
        "annotation_lines",
        "valid_boxes",
        "class_0",
        "class_1",
        "positive_images",
        "empty_labels",
        "missing_labels",
        "orphan_labels",
        "corrupt_images",
        "invalid_image_dimensions",
        "malformed_lines",
        "parse_errors",
        "invalid_classes",
        "invalid_normalized_coordinates",
        "zero_area_boxes",
        "boxes_outside_image",
        "extremely_tiny_boxes",
        "extremely_large_boxes",
        "semantic_class_mismatch",
        "semantic_empty_mismatch",
        "very_small_images",
        "images_without_split",
        "split_references_missing_image",
        "split_overlap",
    ]

    for key in summary_keys:

        print(
            f"{key:36}: "
            f"{stats[key]}"
        )

    # ========================================================
    # SPLITS
    # ========================================================

    print()
    print("SOURCE SPLITS")
    print("-" * 84)

    for split in (
        "train",
        "val",
        "test",
    ):

        print(
            f"{split:6} "
            f"lines={stats[f'split_{split}_lines']:6} "
            f"unique={stats[f'split_{split}_unique']:6} "
            f"duplicates="
            f"{stats[f'split_{split}_duplicate_entries']:4}"
        )

    print()
    print(
        "Split union        :",
        stats["split_union"],
    )

    print(
        "train ∩ val        :",
        stats["train_val_overlap"],
    )

    print(
        "train ∩ test       :",
        stats["train_test_overlap"],
    )

    print(
        "val ∩ test         :",
        stats["val_test_overlap"],
    )

    # ========================================================
    # CATEGORIES
    # ========================================================

    print()
    print("FILENAME ↔ ANNOTATION CONSISTENCY")
    print("-" * 84)

    for category in (
        "fire_only",
        "smoke_only",
        "fire_smoke",
        "negative",
        "other",
    ):

        s = categories[
            category
        ]

        if not s["images"]:
            continue

        print(
            f"{category:14} "
            f"images={s['images']:6} "
            f"match={s['class_match']:6} "
            f"mismatch={s['class_mismatch']:6} "
            f"empty={s['empty']:6}"
        )

    # ========================================================
    # OBJECT SIZE
    # ========================================================

    print()
    print("OBJECT SIZE DISTRIBUTION")
    print("-" * 84)

    for class_name in (
        "fire",
        "smoke",
    ):

        s = size_stats[
            class_name
        ]

        total_objects = sum(
            s.values()
        )

        print()
        print(
            f"{class_name.upper()} "
            f"(total={total_objects})"
        )

        for bucket in (
            "small",
            "medium",
            "large",
        ):

            count = s[
                bucket
            ]

            pct = (
                count
                / total_objects
                * 100
                if total_objects
                else 0
            )

            print(
                f"  {bucket:8}: "
                f"{count:8} "
                f"({pct:6.2f}%)"
            )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 84)

    print(
        "Critical issues :",
        critical_issues,
    )

    print(
        "Warnings        :",
        warnings,
    )

    print()

    if critical_issues == 0:

        print(
            "RESULT: STRUCTURAL PASS"
        )

    else:

        print(
            "RESULT: STRUCTURAL FAIL"
        )

    print()
    print(
        "JSON report :",
        REPORT_JSON,
    )

    print(
        "Issues CSV  :",
        ISSUES_CSV,
    )

    print("=" * 84)


if __name__ == "__main__":
    main()
