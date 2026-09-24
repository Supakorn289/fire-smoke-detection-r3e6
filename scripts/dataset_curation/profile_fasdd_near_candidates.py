#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "candidate_pairs.csv"
)

OUT_JSON = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "candidate_profile.json"
)


RELATIONS = (
    "same_train",
    "same_val",
    "same_test",
    "train_val",
    "train_test",
    "val_test",
)


# ============================================================
# THRESHOLDS TO TEST
# ============================================================

THRESHOLDS = [
    ("p0_d0", 0, 0),
    ("p1_d1", 1, 1),
    ("p2_d2", 2, 2),
    ("p3_d3", 3, 3),
    ("p4_d4", 4, 4),
    ("p4_d6", 4, 6),
    ("p5_d5", 5, 5),
    ("p6_d6", 6, 6),
    ("p6_d8", 6, 8),
    ("p8_d8", 8, 8),
    ("p8_d10", 8, 10),
    ("p10_d10", 10, 10),
]


ASPECT_LIMITS = [
    ("aspect_exact", 0.0),
    ("aspect_0.1pct", 0.001),
    ("aspect_0.5pct", 0.005),
    ("aspect_1pct", 0.01),
    ("aspect_2pct", 0.02),
    ("aspect_5pct", 0.05),
    ("aspect_8pct", 0.08),
]


def main():

    print("=" * 92)
    print("FASDD NEAR-DUPLICATE CANDIDATE PROFILER")
    print("=" * 92)

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    total = 0

    relation_counts = Counter()

    phash_hist = Counter()
    dhash_hist = Counter()
    joint_hist = Counter()

    same_dimensions = Counter()
    relation_same_dimensions = Counter()

    threshold_counts = Counter()
    threshold_cross = Counter()

    strict_combo = Counter()
    strict_combo_relation = defaultdict(Counter)

    aspect_counts = Counter()
    aspect_cross = Counter()

    # ========================================================
    # STREAM CSV
    # ========================================================

    with INPUT.open(
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            total += 1

            relation = row["relation"]

            pd = int(
                row["phash_distance"]
            )

            dd = int(
                row["dhash_distance"]
            )

            aspect = float(
                row["aspect_difference"]
            )

            wa = int(row["width_a"])
            ha = int(row["height_a"])

            wb = int(row["width_b"])
            hb = int(row["height_b"])

            exact_dims = (
                wa == wb
                and ha == hb
            )

            cross = (
                row["split_a"]
                != row["split_b"]
            )

            relation_counts[
                relation
            ] += 1

            phash_hist[pd] += 1
            dhash_hist[dd] += 1
            joint_hist[
                (pd, dd)
            ] += 1

            if exact_dims:

                same_dimensions[
                    "all"
                ] += 1

                relation_same_dimensions[
                    relation
                ] += 1

                if cross:
                    same_dimensions[
                        "cross"
                    ] += 1

            # ------------------------------------------------
            # Aspect thresholds
            # ------------------------------------------------

            for name, limit in ASPECT_LIMITS:

                if aspect <= limit + 1e-12:

                    aspect_counts[
                        name
                    ] += 1

                    if cross:
                        aspect_cross[
                            name
                        ] += 1

            # ------------------------------------------------
            # Hash threshold matrix
            # Both hashes must satisfy threshold here.
            # ------------------------------------------------

            for name, pmax, dmax in THRESHOLDS:

                if (
                    pd <= pmax
                    and dd <= dmax
                ):

                    threshold_counts[
                        name
                    ] += 1

                    if cross:

                        threshold_cross[
                            name
                        ] += 1

                    # Strongest practical gate:
                    # same dimensions too.
                    if exact_dims:

                        key = (
                            name
                            + "_exactdims"
                        )

                        strict_combo[
                            key
                        ] += 1

                        strict_combo_relation[
                            key
                        ][relation] += 1

            if total % 500000 == 0:

                print(
                    f"\rProcessed "
                    f"{total:,} pairs",
                    end="",
                    flush=True,
                )

    print(
        f"\rProcessed "
        f"{total:,} pairs"
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 92)
    print("GLOBAL")
    print("=" * 92)

    print(
        "Candidate pairs      :",
        f"{total:,}",
    )

    print(
        "Exact dimensions     :",
        f"{same_dimensions['all']:,}",
    )

    print(
        "Exact dimensions cross:",
        f"{same_dimensions['cross']:,}",
    )

    print()
    print("RELATIONS")
    print("-" * 92)

    for rel in RELATIONS:

        print(
            f"{rel:16}: "
            f"{relation_counts[rel]:,}"
        )

    # ========================================================
    # HASH HISTOGRAM
    # ========================================================

    print()
    print("=" * 92)
    print("HASH DISTANCE HISTOGRAM")
    print("=" * 92)

    print()
    print("distance | pHash       | dHash")
    print("-" * 44)

    max_distance = max(
        max(phash_hist, default=0),
        max(dhash_hist, default=0),
    )

    for d in range(
        min(max_distance, 20) + 1
    ):

        print(
            f"{d:8} | "
            f"{phash_hist[d]:11,} | "
            f"{dhash_hist[d]:11,}"
        )

    # ========================================================
    # THRESHOLD COUNTS
    # ========================================================

    print()
    print("=" * 92)
    print("HASH THRESHOLD PROFILE")
    print("=" * 92)

    print(
        f"{'threshold':16}"
        f"{'all':>14}"
        f"{'cross':>14}"
        f"{'exact-dim':>14}"
    )

    print("-" * 58)

    for name, _, _ in THRESHOLDS:

        exact_key = (
            name
            + "_exactdims"
        )

        print(
            f"{name:16}"
            f"{threshold_counts[name]:14,}"
            f"{threshold_cross[name]:14,}"
            f"{strict_combo[exact_key]:14,}"
        )

    # ========================================================
    # EXACT-DIM RELATION PROFILE
    # ========================================================

    print()
    print("=" * 92)
    print("STRICT CROSS-SPLIT PROFILES")
    print("=" * 92)

    # Only print useful stricter settings.
    for name in (
        "p2_d2",
        "p3_d3",
        "p4_d4",
        "p4_d6",
        "p5_d5",
        "p6_d6",
    ):

        key = (
            name
            + "_exactdims"
        )

        print()
        print(key)

        for rel in (
            "train_val",
            "train_test",
            "val_test",
        ):

            print(
                f"  {rel:14}: "
                f"{strict_combo_relation[key][rel]:,}"
            )

    # ========================================================
    # ASPECT PROFILE
    # ========================================================

    print()
    print("=" * 92)
    print("ASPECT PROFILE")
    print("=" * 92)

    print(
        f"{'gate':18}"
        f"{'all':>14}"
        f"{'cross':>14}"
    )

    print("-" * 46)

    for name, _ in ASPECT_LIMITS:

        print(
            f"{name:18}"
            f"{aspect_counts[name]:14,}"
            f"{aspect_cross[name]:14,}"
        )

    # ========================================================
    # JSON
    # ========================================================

    report = {
        "candidate_pairs":
            total,

        "relations":
            dict(relation_counts),

        "exact_dimensions": {
            "all":
                same_dimensions["all"],

            "cross":
                same_dimensions["cross"],

            "relations":
                dict(
                    relation_same_dimensions
                ),
        },

        "phash_histogram": {
            str(k): v
            for k, v
            in sorted(
                phash_hist.items()
            )
        },

        "dhash_histogram": {
            str(k): v
            for k, v
            in sorted(
                dhash_hist.items()
            )
        },

        "joint_histogram": {
            f"{p},{d}": n
            for (p, d), n
            in sorted(
                joint_hist.items()
            )
        },

        "thresholds": {
            name: {
                "all":
                    threshold_counts[name],

                "cross":
                    threshold_cross[name],

                "exact_dimensions":
                    strict_combo[
                        name
                        + "_exactdims"
                    ],

                "exact_dimensions_by_relation":
                    dict(
                        strict_combo_relation[
                            name
                            + "_exactdims"
                        ]
                    ),
            }
            for name, _, _
            in THRESHOLDS
        },

        "aspect": {
            name: {
                "all":
                    aspect_counts[name],

                "cross":
                    aspect_cross[name],
            }
            for name, _
            in ASPECT_LIMITS
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

    print()
    print(
        "JSON:",
        OUT_JSON
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
