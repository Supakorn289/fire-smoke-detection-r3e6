#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

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
    / "near_duplicates"
)

OUT_CSV = (
    OUT_DIR
    / "candidate_pairs.csv"
)

OUT_JSON = (
    OUT_DIR
    / "candidate_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

# Query radius for either perceptual hash.
PRIMARY_RADIUS = 10

# If one hash is close, the other must not be wildly different.
SECONDARY_MAX = 20

# Relative aspect ratio difference.
MAX_ASPECT_DIFF = 0.08


# ============================================================
# HAMMING
# ============================================================

def hamming(a, b):
    return (a ^ b).bit_count()


# ============================================================
# BK TREE
# ============================================================

class BKNode:

    __slots__ = (
        "value",
        "indices",
        "children",
    )

    def __init__(self, value, index):

        self.value = value

        # multiple images can have the same perceptual hash
        self.indices = [index]

        self.children = {}


class BKTree:

    def __init__(self):
        self.root = None

    def add(self, value, index):

        if self.root is None:
            self.root = BKNode(
                value,
                index,
            )
            return

        node = self.root

        while True:

            d = hamming(
                value,
                node.value,
            )

            if d == 0:

                node.indices.append(
                    index
                )
                return

            child = node.children.get(d)

            if child is None:

                node.children[d] = BKNode(
                    value,
                    index,
                )
                return

            node = child

    def search(self, value, radius):

        if self.root is None:
            return []

        results = []

        stack = [
            self.root
        ]

        while stack:

            node = stack.pop()

            d = hamming(
                value,
                node.value,
            )

            if d <= radius:

                for index in node.indices:

                    results.append(
                        (
                            index,
                            d,
                        )
                    )

            low = d - radius
            high = d + radius

            for edge_distance, child in (
                node.children.items()
            ):

                if (
                    low
                    <= edge_distance
                    <= high
                ):

                    stack.append(child)

        return results


# ============================================================
# RELATION
# ============================================================

ORDER = {
    "train": 0,
    "val": 1,
    "test": 2,
}


def relation(a, b):

    if a == b:
        return f"same_{a}"

    x, y = sorted(
        [a, b],
        key=lambda s: ORDER.get(
            s,
            99,
        ),
    )

    return f"{x}_{y}"


# ============================================================
# ASPECT DIFFERENCE
# ============================================================

def aspect_difference(a, b):

    denom = max(
        abs(a),
        abs(b),
        1e-12,
    )

    return abs(a - b) / denom


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("FASDD NEAR-DUPLICATE CANDIDATE SEARCH")
    print("=" * 88)

    rows = []

    with HASH_CSV.open(
        encoding="utf-8",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append({
                "stem":
                    row["stem"],

                "split":
                    row["split"],

                "width":
                    int(row["width"]),

                "height":
                    int(row["height"]),

                "aspect":
                    float(row["aspect"]),

                "phash":
                    int(
                        row["phash"],
                        16,
                    ),

                "dhash":
                    int(
                        row["dhash"],
                        16,
                    ),
            })

    total = len(rows)

    print(
        "Images:",
        total
    )

    print(
        "Primary Hamming radius:",
        PRIMARY_RADIUS
    )

    print(
        "Secondary hash maximum:",
        SECONDARY_MAX
    )

    print(
        "Max aspect difference:",
        f"{MAX_ASPECT_DIFF:.1%}"
    )

    phash_tree = BKTree()
    dhash_tree = BKTree()

    candidate_pairs = {}

    raw_phash_hits = 0
    raw_dhash_hits = 0

    # ========================================================
    # SEARCH
    # ========================================================

    for i, current in enumerate(
        rows
    ):

        candidate_indices = set()

        # ----------------------------------------------------
        # pHash neighborhood
        # ----------------------------------------------------

        p_hits = phash_tree.search(
            current["phash"],
            PRIMARY_RADIUS,
        )

        raw_phash_hits += len(
            p_hits
        )

        for idx, _ in p_hits:
            candidate_indices.add(idx)

        # ----------------------------------------------------
        # dHash neighborhood
        # ----------------------------------------------------

        d_hits = dhash_tree.search(
            current["dhash"],
            PRIMARY_RADIUS,
        )

        raw_dhash_hits += len(
            d_hits
        )

        for idx, _ in d_hits:
            candidate_indices.add(idx)

        # ----------------------------------------------------
        # Verify candidate hash conditions
        # ----------------------------------------------------

        for j in candidate_indices:

            other = rows[j]

            pd = hamming(
                current["phash"],
                other["phash"],
            )

            dd = hamming(
                current["dhash"],
                other["dhash"],
            )

            # One hash must be strongly close.
            if (
                pd > PRIMARY_RADIUS
                and
                dd > PRIMARY_RADIUS
            ):
                continue

            # Companion hash must still be reasonably close.
            if (
                pd > SECONDARY_MAX
                or
                dd > SECONDARY_MAX
            ):
                continue

            aspect_diff = aspect_difference(
                current["aspect"],
                other["aspect"],
            )

            if (
                aspect_diff
                > MAX_ASPECT_DIFF
            ):
                continue

            # canonical pair ordering
            a = min(i, j)
            b = max(i, j)

            key = (
                a,
                b,
            )

            candidate_pairs[key] = {
                "a":
                    a,

                "b":
                    b,

                "phash_distance":
                    pd,

                "dhash_distance":
                    dd,

                "aspect_difference":
                    aspect_diff,
            }

        # ----------------------------------------------------
        # Insert current image AFTER query
        # avoids matching itself.
        # ----------------------------------------------------

        phash_tree.add(
            current["phash"],
            i,
        )

        dhash_tree.add(
            current["dhash"],
            i,
        )

        if (
            (i + 1) % 5000 == 0
            or i + 1 == total
        ):

            print(
                f"\rSearching "
                f"{i + 1}/{total} "
                f"({(i + 1) / total * 100:5.1f}%) "
                f"candidates={len(candidate_pairs)}",
                end="",
                flush=True,
            )

    print()

    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    output_rows = []

    relation_counts = Counter()

    cross_split = 0
    same_split = 0

    distance_counts = Counter()

    for pair in candidate_pairs.values():

        a = rows[
            pair["a"]
        ]

        b = rows[
            pair["b"]
        ]

        rel = relation(
            a["split"],
            b["split"],
        )

        relation_counts[
            rel
        ] += 1

        if a["split"] == b["split"]:
            same_split += 1
        else:
            cross_split += 1

        pd = pair[
            "phash_distance"
        ]

        dd = pair[
            "dhash_distance"
        ]

        distance_counts[
            f"phash_{pd}"
        ] += 1

        distance_counts[
            f"dhash_{dd}"
        ] += 1

        output_rows.append({
            "stem_a":
                a["stem"],

            "split_a":
                a["split"],

            "width_a":
                a["width"],

            "height_a":
                a["height"],

            "stem_b":
                b["stem"],

            "split_b":
                b["split"],

            "width_b":
                b["width"],

            "height_b":
                b["height"],

            "relation":
                rel,

            "phash_distance":
                pd,

            "dhash_distance":
                dd,

            "aspect_difference":
                f"{pair['aspect_difference']:.8f}",
        })

    # Sort critical cross-split pairs first.
    output_rows.sort(
        key=lambda x: (
            0
            if x["split_a"]
            != x["split_b"]
            else 1,

            x["phash_distance"]
            + x["dhash_distance"],

            x["stem_a"],
            x["stem_b"],
        )
    )

    # ========================================================
    # CSV
    # ========================================================

    with OUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "stem_a",
            "split_a",
            "width_a",
            "height_a",
            "stem_b",
            "split_b",
            "width_b",
            "height_b",
            "relation",
            "phash_distance",
            "dhash_distance",
            "aspect_difference",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    # ========================================================
    # JSON
    # ========================================================

    summary = {
        "images":
            total,

        "config": {
            "primary_radius":
                PRIMARY_RADIUS,

            "secondary_max":
                SECONDARY_MAX,

            "max_aspect_difference":
                MAX_ASPECT_DIFF,
        },

        "raw_hits": {
            "phash":
                raw_phash_hits,

            "dhash":
                raw_dhash_hits,
        },

        "candidate_pairs":
            len(output_rows),

        "same_split":
            same_split,

        "cross_split":
            cross_split,

        "relations":
            dict(
                relation_counts
            ),

        "distance_distribution":
            dict(
                sorted(
                    distance_counts.items()
                )
            ),
    }

    OUT_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 88)
    print("NEAR-DUPLICATE CANDIDATE SUMMARY")
    print("=" * 88)

    print(
        "Candidate pairs :",
        len(output_rows)
    )

    print(
        "Same split      :",
        same_split
    )

    print(
        "Cross split     :",
        cross_split
    )

    print()
    print("RELATIONS")
    print("-" * 88)

    for rel in (
        "same_train",
        "same_val",
        "same_test",
        "train_val",
        "train_test",
        "val_test",
    ):

        print(
            f"{rel:16}: "
            f"{relation_counts[rel]}"
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
        "No image, label, or manifest was modified."
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
