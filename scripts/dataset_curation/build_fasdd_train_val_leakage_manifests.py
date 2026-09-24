#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

BASE_DIR = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_direct_test"
)

EDGES_CSV = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "cross_split_graph"
    / "edges.csv"
)

OUT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_train_val"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DECISIONS_CSV = (
    OUT_DIR
    / "train_val_edges.csv"
)

SUMMARY_JSON = (
    OUT_DIR
    / "summary.json"
)


VALID_PRIORITIES = {
    "VERY_HIGH",
    "HIGH",
}


def load_set(path):

    return {
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    }


def write_set(path, values):

    path.write_text(
        "".join(
            f"{x}\n"
            for x in sorted(values)
        ),
        encoding="utf-8",
    )


def endpoint(row, split):

    if row["split_a"] == split:
        return row["stem_a"]

    if row["split_b"] == split:
        return row["stem_b"]

    raise RuntimeError(
        f"Missing {split} endpoint: "
        f"{row['stem_a']} / "
        f"{row['stem_b']}"
    )


def main():

    print("=" * 94)
    print("FASDD TRAIN-VAL LEAKAGE MANIFEST BUILDER")
    print("=" * 94)

    # ========================================================
    # CURRENT CLEAN BASE
    # ========================================================

    base = {
        split: load_set(
            BASE_DIR
            / f"{split}_clean_test_protected.txt"
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    print(
        "Current direct-test-clean base:"
    )

    for split in (
        "train",
        "val",
        "test",
    ):

        print(
            f"  {split:5}: "
            f"{len(base[split]):,}"
        )

    print(
        "  total:",
        f"{sum(len(x) for x in base.values()):,}"
    )

    # ========================================================
    # READ HIGH-PRIORITY TRAIN-VAL EDGES
    # ========================================================

    with EDGES_CSV.open(
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    source_train_val_edges = 0
    already_filtered = 0

    usable_edges = []

    excluded_train = set()
    val_anchors = set()

    priority_counts = Counter()
    visual_counts = Counter()
    annotation_counts = Counter()

    # How many edges hit the same train / val nodes?
    train_degree = Counter()
    val_degree = Counter()

    for row in rows:

        if row["relation"] != "train_val":
            continue

        if (
            row["leakage_priority"]
            not in VALID_PRIORITIES
        ):
            continue

        source_train_val_edges += 1

        train_stem = endpoint(
            row,
            "train",
        )

        val_stem = endpoint(
            row,
            "val",
        )

        # ----------------------------------------------------
        # Use current clean layer only.
        # Anything already removed due to Test leakage
        # is ignored here.
        # ----------------------------------------------------

        if (
            train_stem not in base["train"]
            or
            val_stem not in base["val"]
        ):

            already_filtered += 1
            continue

        excluded_train.add(
            train_stem
        )

        val_anchors.add(
            val_stem
        )

        train_degree[
            train_stem
        ] += 1

        val_degree[
            val_stem
        ] += 1

        priority_counts[
            row["leakage_priority"]
        ] += 1

        visual_counts[
            row["visual_tier"]
        ] += 1

        annotation_counts[
            row["annotation_state"]
        ] += 1

        usable_edges.append({
            **row,

            "excluded_train_stem":
                train_stem,

            "protected_val_stem":
                val_stem,
        })

    # ========================================================
    # BUILD NEW CLEAN LAYER
    # ========================================================

    clean = {
        "train":
            base["train"]
            - excluded_train,

        # VAL anchor is protected.
        "val":
            set(
                base["val"]
            ),

        # TEST remains protected.
        "test":
            set(
                base["test"]
            ),
    }

    # ========================================================
    # VALIDATION
    # ========================================================

    accounting_pass = (
        len(clean["train"])
        + len(excluded_train)
        == len(base["train"])
        and
        clean["val"]
        == base["val"]
        and
        clean["test"]
        == base["test"]
    )

    val_protected_pass = (
        clean["val"]
        == base["val"]
    )

    test_protected_pass = (
        clean["test"]
        == base["test"]
    )

    filename_isolation_pass = (
        not (
            clean["train"]
            & clean["val"]
        )
        and
        not (
            clean["train"]
            & clean["test"]
        )
        and
        not (
            clean["val"]
            & clean["test"]
        )
    )

    # Every usable direct train-val edge
    # must now have its train endpoint removed.
    surviving_edges = []

    for row in usable_edges:

        train_stem = row[
            "excluded_train_stem"
        ]

        val_stem = row[
            "protected_val_stem"
        ]

        if (
            train_stem
            in clean["train"]
            and
            val_stem
            in clean["val"]
        ):

            surviving_edges.append(
                row
            )

    leakage_pass = (
        len(surviving_edges) == 0
    )

    # ========================================================
    # SAVE MANIFESTS
    # ========================================================

    for split in (
        "train",
        "val",
        "test",
    ):

        write_set(
            OUT_DIR
            / f"{split}_base_direct_test_clean.txt",
            base[split],
        )

        write_set(
            OUT_DIR
            / f"{split}_clean_cross_split.txt",
            clean[split],
        )

    write_set(
        OUT_DIR
        / "train_exclude_direct_val.txt",
        excluded_train,
    )

    write_set(
        OUT_DIR
        / "val_anchors_direct_train.txt",
        val_anchors,
    )

    # ========================================================
    # EDGE CSV
    # ========================================================

    if usable_edges:

        fields = list(
            usable_edges[0].keys()
        )

        with DECISIONS_CSV.open(
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
                usable_edges
            )

    # ========================================================
    # STATS
    # ========================================================

    total_base = sum(
        len(x)
        for x in base.values()
    )

    total_clean = sum(
        len(x)
        for x in clean.values()
    )

    removed = len(
        excluded_train
    )

    max_train_edges = max(
        train_degree.values(),
        default=0,
    )

    max_val_edges = max(
        val_degree.values(),
        default=0,
    )

    # ========================================================
    # JSON
    # ========================================================

    summary = {
        "source_layer":
            "direct-test-clean",

        "policy":
            (
                "Protect VAL and TEST. "
                "Remove only TRAIN nodes "
                "with direct HIGH/VERY_HIGH "
                "near-identical edge to VAL. "
                "No transitive removal."
            ),

        "edges": {
            "source_high_priority_train_val":
                source_train_val_edges,

            "already_filtered_by_previous_layer":
                already_filtered,

            "usable":
                len(usable_edges),
        },

        "unique_nodes": {
            "train_removed":
                len(excluded_train),

            "val_anchors":
                len(val_anchors),
        },

        "edge_priority":
            dict(priority_counts),

        "visual_tier":
            dict(visual_counts),

        "annotation_state":
            dict(annotation_counts),

        "degree": {
            "max_edges_per_train_node":
                max_train_edges,

            "max_edges_per_val_node":
                max_val_edges,
        },

        "splits": {
            split: {
                "base":
                    len(base[split]),

                "clean":
                    len(clean[split]),

                "removed":
                    (
                        len(base[split])
                        - len(clean[split])
                    ),
            }
            for split in (
                "train",
                "val",
                "test",
            )
        },

        "totals": {
            "base":
                total_base,

            "clean":
                total_clean,

            "removed":
                removed,
        },

        "validation": {
            "accounting":
                accounting_pass,

            "val_protected":
                val_protected_pass,

            "test_protected":
                test_protected_pass,

            "filename_isolation":
                filename_isolation_pass,

            "train_val_leakage":
                leakage_pass,

            "surviving_direct_edges":
                len(
                    surviving_edges
                ),
        },
    }

    SUMMARY_JSON.write_text(
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
    print("=" * 94)
    print("TRAIN-VAL LEAKAGE SUMMARY")
    print("=" * 94)

    print(
        "Source HIGH/VERY_HIGH edges:",
        f"{source_train_val_edges:,}"
    )

    print(
        "Filtered by Test-clean layer:",
        f"{already_filtered:,}"
    )

    print(
        "Usable direct edges         :",
        f"{len(usable_edges):,}"
    )

    print()
    print("UNIQUE NODES")
    print("-" * 94)

    print(
        "Train removed:",
        f"{len(excluded_train):,}"
    )

    print(
        "Val anchors  :",
        f"{len(val_anchors):,}"
    )

    print()
    print("EDGE PRIORITY")
    print("-" * 94)

    print(
        "VERY_HIGH:",
        f"{priority_counts['VERY_HIGH']:,}"
    )

    print(
        "HIGH     :",
        f"{priority_counts['HIGH']:,}"
    )

    print()
    print("CLEAN SPLITS")
    print("-" * 94)

    for split in (
        "train",
        "val",
        "test",
    ):

        print(
            f"{split.upper():5} "
            f"base={len(base[split]):6} "
            f"removed="
            f"{len(base[split]) - len(clean[split]):4} "
            f"clean={len(clean[split]):6}"
        )

    print()

    print(
        "Base total :",
        f"{total_base:,}"
    )

    print(
        "Clean total:",
        f"{total_clean:,}"
    )

    print(
        "Removed    :",
        f"{removed:,}"
    )

    print()
    print("NODE DEGREE")
    print("-" * 94)

    print(
        "Max edges / Train node:",
        max_train_edges
    )

    print(
        "Max edges / Val node  :",
        max_val_edges
    )

    print()
    print("VALIDATION")
    print("-" * 94)

    print(
        "Accounting           :",
        "PASS"
        if accounting_pass
        else "FAIL"
    )

    print(
        "VAL protected        :",
        "PASS"
        if val_protected_pass
        else "FAIL"
    )

    print(
        "TEST protected       :",
        "PASS"
        if test_protected_pass
        else "FAIL"
    )

    print(
        "Filename isolation   :",
        "PASS"
        if filename_isolation_pass
        else "FAIL"
    )

    print(
        "Direct Train-Val leak:",
        "PASS"
        if leakage_pass
        else "FAIL"
    )

    print(
        "Surviving edges      :",
        len(
            surviving_edges
        )
    )

    final_pass = (
        accounting_pass
        and
        val_protected_pass
        and
        test_protected_pass
        and
        filename_isolation_pass
        and
        leakage_pass
    )

    print()
    print(
        "RESULT:",
        (
            "CROSS-SPLIT CLEAN PASS"
            if final_pass
            else
            "CROSS-SPLIT CLEAN FAIL"
        )
    )

    print()
    print(
        "Output:",
        OUT_DIR
    )

    print()
    print(
        "No image or annotation file was modified."
    )

    print("=" * 94)


if __name__ == "__main__":
    main()
