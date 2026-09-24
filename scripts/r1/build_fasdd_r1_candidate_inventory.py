#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT
    / "reports"
    / "fasdd"
    / "clean_v1"
    / "train.txt"
)

LABELS = (
    Path("/path/to/fire_datasets/fasdd/curated_v1")
    / "annotations"
    / "YOLO_CV"
    / "labels"
)

HASH_CSV = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "phash.csv"
)

OUT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
    / "r1_candidates"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_CSV = OUT_DIR / "train_inventory.csv"
OUT_JSON = OUT_DIR / "train_inventory_summary.json"


SMALL_MAX = 0.005
MEDIUM_MAX = 0.05


def object_size(area):

    if area < SMALL_MAX:
        return "small"

    if area < MEDIUM_MAX:
        return "medium"

    return "large"


def load_manifest():

    return [
        x.strip()
        for x in MANIFEST.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def load_hashes():

    result = {}

    if not HASH_CSV.exists():
        return result

    with HASH_CSV.open(
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            result[
                row["stem"]
            ] = {
                "phash":
                    row["phash"],

                "dhash":
                    row["dhash"],

                "width":
                    row["width"],

                "height":
                    row["height"],
            }

    return result


def analyze_label(stem):

    path = LABELS / f"{stem}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing label: {path}"
        )

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    counts = Counter()

    if not text:

        return {
            "category": "negative",
            "counts": counts,
            "box_count": 0,
        }

    classes = set()

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        cls = int(parts[0])

        if cls not in {0, 1}:
            continue

        _, _, w, h = map(
            float,
            parts[1:],
        )

        size = object_size(
            w * h
        )

        name = (
            "fire"
            if cls == 0
            else "smoke"
        )

        counts[
            f"{name}_{size}"
        ] += 1

        counts[
            f"{name}_total"
        ] += 1

        classes.add(cls)

    if not classes:
        category = "negative"

    elif classes == {0}:
        category = "fire_only"

    elif classes == {1}:
        category = "smoke_only"

    else:
        category = "fire_smoke"

    return {
        "category":
            category,

        "counts":
            counts,

        "box_count":
            counts["fire_total"]
            + counts["smoke_total"],
    }


def main():

    print("=" * 94)
    print("FASDD R1 TRAIN CANDIDATE INVENTORY")
    print("=" * 94)

    stems = load_manifest()

    if len(stems) != 45658:

        raise RuntimeError(
            f"Expected 45,658 clean train images, "
            f"found {len(stems)}"
        )

    hashes = load_hashes()

    print(
        "Clean train images:",
        f"{len(stems):,}"
    )

    print(
        "Hashes available  :",
        f"{len(hashes):,}"
    )

    rows = []

    categories = Counter()
    signal_images = Counter()
    object_totals = Counter()
    intersections = Counter()

    category_signal = defaultdict(
        Counter
    )

    for i, stem in enumerate(
        stems,
        1,
    ):

        result = analyze_label(
            stem
        )

        c = result[
            "counts"
        ]

        category = result[
            "category"
        ]

        categories[
            category
        ] += 1

        for key, value in c.items():
            object_totals[
                key
            ] += value

        has_small_fire = (
            c["fire_small"] > 0
        )

        has_medium_fire = (
            c["fire_medium"] > 0
        )

        has_large_fire = (
            c["fire_large"] > 0
        )

        has_small_smoke = (
            c["smoke_small"] > 0
        )

        has_medium_smoke = (
            c["smoke_medium"] > 0
        )

        has_large_smoke = (
            c["smoke_large"] > 0
        )

        has_fire = (
            c["fire_total"] > 0
        )

        has_smoke = (
            c["smoke_total"] > 0
        )

        flags = {
            "small_fire":
                has_small_fire,

            "medium_fire":
                has_medium_fire,

            "large_fire":
                has_large_fire,

            "small_smoke":
                has_small_smoke,

            "medium_smoke":
                has_medium_smoke,

            "large_smoke":
                has_large_smoke,

            "has_fire":
                has_fire,

            "has_smoke":
                has_smoke,

            "fire_smoke":
                has_fire
                and has_smoke,
        }

        for name, present in flags.items():

            if present:

                signal_images[
                    name
                ] += 1

                category_signal[
                    category
                ][name] += 1

        # ====================================================
        # Useful intersections for quota design
        # ====================================================

        if (
            has_small_smoke
            and has_fire
        ):
            intersections[
                "small_smoke_and_fire"
            ] += 1

        if (
            has_small_smoke
            and has_small_fire
        ):
            intersections[
                "small_smoke_and_small_fire"
            ] += 1

        if (
            has_small_fire
            and has_medium_smoke
        ):
            intersections[
                "small_fire_and_medium_smoke"
            ] += 1

        if (
            has_small_fire
            and has_smoke
        ):
            intersections[
                "small_fire_and_smoke"
            ] += 1

        if (
            category == "fire_smoke"
            and has_medium_smoke
        ):
            intersections[
                "fire_smoke_with_medium_smoke"
            ] += 1

        if (
            category == "fire_smoke"
            and has_small_smoke
        ):
            intersections[
                "fire_smoke_with_small_smoke"
            ] += 1

        hash_info = hashes.get(
            stem,
            {}
        )

        rows.append({
            "stem":
                stem,

            "category":
                category,

            "box_count":
                result["box_count"],

            "fire_total":
                c["fire_total"],

            "fire_small":
                c["fire_small"],

            "fire_medium":
                c["fire_medium"],

            "fire_large":
                c["fire_large"],

            "smoke_total":
                c["smoke_total"],

            "smoke_small":
                c["smoke_small"],

            "smoke_medium":
                c["smoke_medium"],

            "smoke_large":
                c["smoke_large"],

            "has_small_fire":
                int(
                    has_small_fire
                ),

            "has_medium_fire":
                int(
                    has_medium_fire
                ),

            "has_large_fire":
                int(
                    has_large_fire
                ),

            "has_small_smoke":
                int(
                    has_small_smoke
                ),

            "has_medium_smoke":
                int(
                    has_medium_smoke
                ),

            "has_large_smoke":
                int(
                    has_large_smoke
                ),

            "has_fire_smoke":
                int(
                    has_fire
                    and has_smoke
                ),

            "phash":
                hash_info.get(
                    "phash",
                    ""
                ),

            "dhash":
                hash_info.get(
                    "dhash",
                    ""
                ),

            "width":
                hash_info.get(
                    "width",
                    ""
                ),

            "height":
                hash_info.get(
                    "height",
                    ""
                ),
        })

        if (
            i % 5000 == 0
            or i == len(stems)
        ):

            print(
                f"\rAnalyzing "
                f"{i}/{len(stems)} "
                f"({i/len(stems)*100:5.1f}%)",
                end="",
                flush=True,
            )

    print()

    # ========================================================
    # WRITE CSV
    # ========================================================

    fields = [
        "stem",
        "category",
        "box_count",
        "fire_total",
        "fire_small",
        "fire_medium",
        "fire_large",
        "smoke_total",
        "smoke_small",
        "smoke_medium",
        "smoke_large",
        "has_small_fire",
        "has_medium_fire",
        "has_large_fire",
        "has_small_smoke",
        "has_medium_smoke",
        "has_large_smoke",
        "has_fire_smoke",
        "phash",
        "dhash",
        "width",
        "height",
    ]

    with OUT_CSV.open(
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
            rows
        )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "images":
            len(stems),

        "categories":
            dict(categories),

        "image_level_signals":
            dict(signal_images),

        "object_totals":
            dict(object_totals),

        "intersections":
            dict(intersections),

        "category_signal": {
            category:
                dict(values)

            for category, values
            in category_signal.items()
        },

        "notes": {
            "small_smoke_policy":
                (
                    "Do not set quota until "
                    "image-level availability "
                    "is known."
                ),

            "test_used":
                False,

            "val_used":
                False,
        },
    }

    OUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 94)
    print("R1 CANDIDATE INVENTORY SUMMARY")
    print("=" * 94)

    print()
    print("CATEGORY")
    print("-" * 94)

    for name in (
        "fire_only",
        "smoke_only",
        "fire_smoke",
        "negative",
    ):

        print(
            f"{name:18}: "
            f"{categories[name]:,}"
        )

    print()
    print("IMAGE-LEVEL SIGNALS")
    print("-" * 94)

    for name in (
        "small_fire",
        "medium_fire",
        "large_fire",
        "small_smoke",
        "medium_smoke",
        "large_smoke",
        "fire_smoke",
    ):

        print(
            f"{name:18}: "
            f"{signal_images[name]:,}"
        )

    print()
    print("KEY INTERSECTIONS")
    print("-" * 94)

    for name in (
        "small_smoke_and_fire",
        "small_smoke_and_small_fire",
        "small_fire_and_medium_smoke",
        "small_fire_and_smoke",
        "fire_smoke_with_medium_smoke",
        "fire_smoke_with_small_smoke",
    ):

        print(
            f"{name:34}: "
            f"{intersections[name]:,}"
        )

    print()
    print("CATEGORY × SIGNAL")
    print("-" * 94)

    for category in (
        "fire_only",
        "smoke_only",
        "fire_smoke",
        "negative",
    ):

        values = category_signal[
            category
        ]

        print()
        print(category)

        for signal in (
            "small_fire",
            "medium_fire",
            "small_smoke",
            "medium_smoke",
            "large_smoke",
        ):

            print(
                f"  {signal:16}: "
                f"{values[signal]:,}"
            )

    print()
    print(
        "CSV :",
        OUT_CSV
    )

    print(
        "JSON:",
        OUT_JSON
    )

    print()
    print(
        "No candidate has been selected yet."
    )

    print(
        "VAL and TEST were not used."
    )

    print("=" * 94)


if __name__ == "__main__":
    main()
