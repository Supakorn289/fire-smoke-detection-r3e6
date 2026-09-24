#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import hashlib


ROOT = Path(__file__).resolve().parents[2]

ORIGINAL = (
    ROOT / "sources/fasdd/FASDD_CV"
)

CURATED = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)

ORIG_YOLO = ORIGINAL / "annotations/YOLO_CV"
CUR_YOLO = CURATED / "annotations/YOLO_CV"

ORIG_LABELS = ORIG_YOLO / "labels"
CUR_LABELS = CUR_YOLO / "labels"

EPS = 1e-6


def load_split(root, split):
    p = root / f"{split}.txt"

    return {
        Path(x.strip()).stem
        for x in p.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if x.strip()
    }


def sha256(path):
    return hashlib.sha256(
        path.read_bytes()
    ).digest()


def inspect_label(path, stats):

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not text:
        stats["empty"] += 1
        return

    for line in text.splitlines():

        stats["lines"] += 1

        parts = line.split()

        if len(parts) != 5:
            stats["malformed"] += 1
            continue

        try:
            cls = int(parts[0])
            xc, yc, w, h = map(
                float,
                parts[1:],
            )
        except ValueError:
            stats["parse_error"] += 1
            continue

        if cls not in {0, 1}:
            stats["invalid_class"] += 1

        if w <= 0 or h <= 0:
            stats["zero_area"] += 1
            continue

        if (
            xc < -EPS
            or xc > 1 + EPS
            or yc < -EPS
            or yc > 1 + EPS
            or w < -EPS
            or w > 1 + EPS
            or h < -EPS
            or h > 1 + EPS
        ):
            stats["invalid_normalized"] += 1

        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2

        if (
            x1 < -EPS
            or y1 < -EPS
            or x2 > 1 + EPS
            or y2 > 1 + EPS
        ):
            stats["outside"] += 1


def main():

    print("=" * 78)
    print("FASDD CURATED V1 FINAL VALIDATION")
    print("=" * 78)

    train = load_split(CUR_YOLO, "train")
    val = load_split(CUR_YOLO, "val")
    test = load_split(CUR_YOLO, "test")

    print("Train:", len(train))
    print("Val  :", len(val))
    print("Test :", len(test))
    print("Union:", len(train | val | test))

    print()
    print("Split overlaps")
    print("train ∩ val :", len(train & val))
    print("train ∩ test:", len(train & test))
    print("val ∩ test  :", len(val & test))

    label_count = len(
        list(CUR_LABELS.glob("*.txt"))
    )

    print()
    print("Curated labels:", label_count)

    # --------------------------------------------------------
    # TRAIN + VAL integrity
    # --------------------------------------------------------

    stats = Counter()

    train_val = train | val

    for i, stem in enumerate(
        sorted(train_val),
        1,
    ):
        path = CUR_LABELS / f"{stem}.txt"

        if not path.exists():
            stats["missing_label"] += 1
            continue

        inspect_label(path, stats)

    print()
    print("=" * 78)
    print("TRAIN + VAL SANITIZED INTEGRITY")
    print("=" * 78)

    keys = [
        "lines",
        "empty",
        "missing_label",
        "malformed",
        "parse_error",
        "invalid_class",
        "invalid_normalized",
        "zero_area",
        "outside",
    ]

    for key in keys:
        print(
            f"{key:20}: "
            f"{stats[key]}"
        )

    train_val_pass = all(
        stats[key] == 0
        for key in [
            "missing_label",
            "malformed",
            "parse_error",
            "invalid_class",
            "invalid_normalized",
            "zero_area",
            "outside",
        ]
    )

    # --------------------------------------------------------
    # TEST byte identity
    # --------------------------------------------------------

    different = []
    missing = []

    for stem in sorted(test):

        a = ORIG_LABELS / f"{stem}.txt"
        b = CUR_LABELS / f"{stem}.txt"

        if (
            not a.exists()
            or not b.exists()
        ):
            missing.append(stem)
            continue

        if sha256(a) != sha256(b):
            different.append(stem)

    print()
    print("=" * 78)
    print("TEST IMMUTABILITY")
    print("=" * 78)

    print(
        "Test labels checked:",
        len(test),
    )

    print(
        "Missing            :",
        len(missing),
    )

    print(
        "Different          :",
        len(different),
    )

    test_pass = (
        len(missing) == 0
        and len(different) == 0
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 78)

    print(
        "TRAIN/VAL SANITIZATION:",
        "PASS"
        if train_val_pass
        else "FAIL",
    )

    print(
        "TEST IMMUTABILITY     :",
        "PASS"
        if test_pass
        else "FAIL",
    )

    if (
        train_val_pass
        and test_pass
        and label_count == 95314
        and len(train | val | test) == 95314
    ):
        print()
        print(
            "RESULT: FASDD CURATED V1 PASS"
        )
    else:
        print()
        print(
            "RESULT: FASDD CURATED V1 FAIL"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()
