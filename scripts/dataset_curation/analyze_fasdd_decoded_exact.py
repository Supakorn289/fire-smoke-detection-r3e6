#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

METRICS_CSV = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "p0d0_pixel_metrics.csv"
)

ORIGINAL_LABELS = (
    ROOT
    / "sources"
    / "fasdd"
    / "FASDD_CV"
    / "annotations"
    / "YOLO_CV"
    / "labels"
)

OUT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "decoded_exact"
)

OUT_JSON = OUT_DIR / "analysis.json"
OUT_CSV = OUT_DIR / "analysis.csv"

TOL = 1e-6


# ============================================================
# UNION FIND
# ============================================================

class DSU:

    def __init__(self):
        self.parent = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x):
        p = self.parent[x]

        if p != x:
            self.parent[x] = self.find(p)

        return self.parent[x]

    def union(self, a, b):
        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[rb] = ra


# ============================================================
# LABEL READER
# ============================================================

def read_label(stem):

    path = ORIGINAL_LABELS / f"{stem}.txt"

    if not path.exists():
        return {
            "exists": False,
            "text": "",
            "boxes": [],
            "classes": [],
        }

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    boxes = []

    if text:

        for line in text.splitlines():

            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                cls = int(parts[0])

                coords = tuple(
                    float(x)
                    for x in parts[1:]
                )

            except ValueError:
                continue

            boxes.append(
                (cls, *coords)
            )

    boxes.sort()

    return {
        "exists": True,
        "text": text,
        "boxes": boxes,
        "classes": sorted(
            b[0]
            for b in boxes
        ),
    }


def boxes_equal(a, b):

    if len(a) != len(b):
        return False

    for x, y in zip(a, b):

        if x[0] != y[0]:
            return False

        for xa, ya in zip(
            x[1:],
            y[1:],
        ):

            if abs(xa - ya) > TOL:
                return False

    return True


def compare_labels(a, b):

    if (
        not a["exists"]
        or not b["exists"]
    ):
        return "missing_label"

    if a["text"] == b["text"]:
        return "exact_annotation"

    if boxes_equal(
        a["boxes"],
        b["boxes"],
    ):
        return "equivalent_annotation"

    if len(a["boxes"]) != len(b["boxes"]):
        return "box_count_conflict"

    if a["classes"] != b["classes"]:
        return "class_conflict"

    return "geometry_conflict"


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 86)
    print("FASDD DECODED-PIXEL EXACT ANALYSIS")
    print("=" * 86)

    edges = []

    dsu = DSU()

    stem_split = {}

    with METRICS_CSV.open(
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if int(
                row["decoded_pixel_exact"]
            ) != 1:
                continue

            a = row["stem_a"]
            b = row["stem_b"]

            stem_split[a] = row["split_a"]
            stem_split[b] = row["split_b"]

            dsu.union(a, b)

            label_a = read_label(a)
            label_b = read_label(b)

            result = compare_labels(
                label_a,
                label_b,
            )

            edges.append({
                "stem_a": a,
                "split_a": row["split_a"],
                "stem_b": b,
                "split_b": row["split_b"],
                "relation": row["relation"],
                "annotation_result": result,
                "boxes_a": len(
                    label_a["boxes"]
                ),
                "boxes_b": len(
                    label_b["boxes"]
                ),
                "classes_a": ",".join(
                    map(
                        str,
                        label_a["classes"],
                    )
                ),
                "classes_b": ",".join(
                    map(
                        str,
                        label_b["classes"],
                    )
                ),
            })

    # ========================================================
    # COMPONENTS
    # ========================================================

    components = defaultdict(list)

    for stem in stem_split:

        components[
            dsu.find(stem)
        ].append(stem)

    groups = []

    for group_id, stems in enumerate(
        sorted(
            components.values(),
            key=lambda x: (
                -len(x),
                sorted(x)[0],
            ),
        ),
        1,
    ):

        stems = sorted(stems)

        splits = sorted({
            stem_split[s]
            for s in stems
        })

        groups.append({
            "group_id": group_id,
            "size": len(stems),
            "splits": splits,
            "members": [
                {
                    "stem": stem,
                    "split": stem_split[stem],
                }
                for stem in stems
            ],
        })

    # ========================================================
    # COUNTS
    # ========================================================

    stats = Counter()

    relation_counts = Counter()
    annotation_counts = Counter()

    for row in edges:

        stats["pairs"] += 1

        relation_counts[
            row["relation"]
        ] += 1

        annotation_counts[
            row["annotation_result"]
        ] += 1

    stats["groups"] = len(groups)

    stats["unique_images"] = len(
        stem_split
    )

    stats["cross_split_pairs"] = sum(
        relation_counts[x]
        for x in (
            "train_val",
            "train_test",
            "val_test",
        )
    )

    conflict_types = {
        "geometry_conflict",
        "box_count_conflict",
        "class_conflict",
        "missing_label",
    }

    stats["annotation_conflicts"] = sum(
        annotation_counts[x]
        for x in conflict_types
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
            "stem_b",
            "split_b",
            "relation",
            "annotation_result",
            "boxes_a",
            "boxes_b",
            "classes_a",
            "classes_b",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(edges)

    # ========================================================
    # JSON
    # ========================================================

    report = {
        "stats":
            dict(stats),

        "relations":
            dict(relation_counts),

        "annotations":
            dict(annotation_counts),

        "groups":
            groups,

        "pairs":
            edges,
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
    # OUTPUT
    # ========================================================

    print()
    print("DECODED EXACT STRUCTURE")
    print("-" * 86)

    print(
        "Pair edges       :",
        stats["pairs"],
    )

    print(
        "Unique images    :",
        stats["unique_images"],
    )

    print(
        "Connected groups :",
        stats["groups"],
    )

    print(
        "Cross-split pairs:",
        stats["cross_split_pairs"],
    )

    print()
    print("RELATIONS")
    print("-" * 86)

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
    print("ANNOTATION COMPARISON")
    print("-" * 86)

    for result in (
        "exact_annotation",
        "equivalent_annotation",
        "geometry_conflict",
        "box_count_conflict",
        "class_conflict",
        "missing_label",
    ):

        print(
            f"{result:24}: "
            f"{annotation_counts[result]}"
        )

    print()
    print(
        "Annotation conflicts:",
        stats["annotation_conflicts"],
    )

    print()
    print("GROUP SIZE DISTRIBUTION")
    print("-" * 86)

    size_counts = Counter(
        g["size"]
        for g in groups
    )

    for size in sorted(size_counts):

        print(
            f"size {size}: "
            f"{size_counts[size]} groups"
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

    print("=" * 86)


if __name__ == "__main__":
    main()
