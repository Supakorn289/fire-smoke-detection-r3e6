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
    / "dedup_visual_exact"
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
    / "dedup_direct_test"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DECISIONS_CSV = OUT_DIR / "direct_test_edges.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"


VALID_PRIORITIES = {
    "VERY_HIGH",
    "HIGH",
}

TEST_RELATIONS = {
    "train_test",
    "val_test",
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


def endpoint_for_split(row, wanted_split):

    if row["split_a"] == wanted_split:
        return row["stem_a"]

    if row["split_b"] == wanted_split:
        return row["stem_b"]

    raise RuntimeError(
        f"Pair does not contain split "
        f"{wanted_split}: "
        f"{row['stem_a']} / {row['stem_b']}"
    )


def main():

    print("=" * 94)
    print("FASDD DIRECT TEST-LEAKAGE MANIFEST BUILDER")
    print("=" * 94)

    # ========================================================
    # LOAD VISUAL-EXACT CLEAN BASE
    # ========================================================

    base = {
        split: load_set(
            BASE_DIR
            / f"{split}_clean_visual_exact.txt"
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    print(
        "Base visual-exact clean:"
    )

    print(
        "  train:",
        f"{len(base['train']):,}"
    )

    print(
        "  val  :",
        f"{len(base['val']):,}"
    )

    print(
        "  test :",
        f"{len(base['test']):,}"
    )

    print(
        "  total:",
        f"{sum(len(x) for x in base.values()):,}"
    )

    # ========================================================
    # READ HIGH-PRIORITY GRAPH EDGES
    # ========================================================

    with EDGES_CSV.open(
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    direct_edges = []

    excluded_train = set()
    excluded_val = set()

    test_anchors_train = set()
    test_anchors_val = set()

    relation_counts = Counter()
    priority_counts = Counter()
    visual_counts = Counter()
    annotation_counts = Counter()

    for row in rows:

        relation = row[
            "relation"
        ]

        priority = row[
            "leakage_priority"
        ]

        if relation not in TEST_RELATIONS:
            continue

        if priority not in VALID_PRIORITIES:
            continue

        test_stem = endpoint_for_split(
            row,
            "test",
        )

        if relation == "train_test":

            non_test_stem = endpoint_for_split(
                row,
                "train",
            )

            non_test_split = "train"

            excluded_train.add(
                non_test_stem
            )

            test_anchors_train.add(
                test_stem
            )

        elif relation == "val_test":

            non_test_stem = endpoint_for_split(
                row,
                "val",
            )

            non_test_split = "val"

            excluded_val.add(
                non_test_stem
            )

            test_anchors_val.add(
                test_stem
            )

        else:
            raise RuntimeError(
                relation
            )

        # ----------------------------------------------------
        # Verify membership in current clean layer
        # ----------------------------------------------------

        if (
            non_test_stem
            not in base[
                non_test_split
            ]
        ):

            raise RuntimeError(
                f"{non_test_stem} "
                f"not in base "
                f"{non_test_split}"
            )

        if test_stem not in base["test"]:

            raise RuntimeError(
                f"Test anchor "
                f"{test_stem} "
                f"not in current clean test"
            )

        relation_counts[
            relation
        ] += 1

        priority_counts[
            priority
        ] += 1

        visual_counts[
            row["visual_tier"]
        ] += 1

        annotation_counts[
            row["annotation_state"]
        ] += 1

        direct_edges.append({
            **row,

            "protected_test_stem":
                test_stem,

            "excluded_non_test_stem":
                non_test_stem,

            "excluded_split":
                non_test_split,
        })

    # ========================================================
    # BUILD NEW CLEAN LAYER
    # ========================================================

    clean = {
        "train":
            base["train"]
            - excluded_train,

        "val":
            base["val"]
            - excluded_val,

        # TEST IS PROTECTED
        "test":
            set(
                base["test"]
            ),
    }

    test_anchors = (
        test_anchors_train
        | test_anchors_val
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    accounting_pass = (
        len(clean["train"])
        + len(excluded_train)
        == len(base["train"])
        and
        len(clean["val"])
        + len(excluded_val)
        == len(base["val"])
        and
        clean["test"]
        == base["test"]
    )

    test_immutable_pass = (
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

    # Every direct HIGH/VERY_HIGH Test edge
    # must have its lower-priority endpoint removed.
    surviving_direct_edges = []

    for row in direct_edges:

        split = row[
            "excluded_split"
        ]

        stem = row[
            "excluded_non_test_stem"
        ]

        if stem in clean[split]:

            surviving_direct_edges.append(
                row
            )

    direct_leakage_pass = (
        len(
            surviving_direct_edges
        ) == 0
    )

    # ========================================================
    # WRITE MANIFESTS
    # ========================================================

    for split in (
        "train",
        "val",
        "test",
    ):

        write_set(
            OUT_DIR
            / f"{split}_base_visual_exact.txt",
            base[split],
        )

        write_set(
            OUT_DIR
            / f"{split}_clean_test_protected.txt",
            clean[split],
        )

    write_set(
        OUT_DIR
        / "train_exclude_direct_test.txt",
        excluded_train,
    )

    write_set(
        OUT_DIR
        / "val_exclude_direct_test.txt",
        excluded_val,
    )

    write_set(
        OUT_DIR
        / "test_anchors_direct_leakage.txt",
        test_anchors,
    )

    write_set(
        OUT_DIR
        / "test_anchors_from_train.txt",
        test_anchors_train,
    )

    write_set(
        OUT_DIR
        / "test_anchors_from_val.txt",
        test_anchors_val,
    )

    # ========================================================
    # EDGE CSV
    # ========================================================

    if direct_edges:

        fields = list(
            direct_edges[0].keys()
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
                direct_edges
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    total_base = sum(
        len(x)
        for x in base.values()
    )

    total_clean = sum(
        len(x)
        for x in clean.values()
    )

    removed = (
        len(excluded_train)
        + len(excluded_val)
    )

    summary = {
        "source_layer":
            "visual-exact clean",

        "policy":
            (
                "Protect TEST. Remove only "
                "non-test nodes with direct "
                "HIGH/VERY_HIGH edge to TEST. "
                "No transitive removal."
            ),

        "edges": {
            "direct_test_edges":
                len(direct_edges),

            "train_test":
                relation_counts[
                    "train_test"
                ],

            "val_test":
                relation_counts[
                    "val_test"
                ],
        },

        "unique_nodes": {
            "train_removed":
                len(excluded_train),

            "val_removed":
                len(excluded_val),

            "test_anchors":
                len(test_anchors),

            "test_anchors_from_train":
                len(
                    test_anchors_train
                ),

            "test_anchors_from_val":
                len(
                    test_anchors_val
                ),
        },

        "edge_priority":
            dict(
                priority_counts
            ),

        "visual_tier":
            dict(
                visual_counts
            ),

        "annotation_state":
            dict(
                annotation_counts
            ),

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

            "test_immutable":
                test_immutable_pass,

            "filename_isolation":
                filename_isolation_pass,

            "direct_test_leakage":
                direct_leakage_pass,

            "surviving_direct_edges":
                len(
                    surviving_direct_edges
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
    print("DIRECT TEST-LEAKAGE SUMMARY")
    print("=" * 94)

    print(
        "Direct HIGH/VERY_HIGH edges:",
        f"{len(direct_edges):,}"
    )

    print(
        "  train ↔ test:",
        f"{relation_counts['train_test']:,}"
    )

    print(
        "  val   ↔ test:",
        f"{relation_counts['val_test']:,}"
    )

    print()
    print("UNIQUE NODES")
    print("-" * 94)

    print(
        "Train removed :",
        f"{len(excluded_train):,}"
    )

    print(
        "Val removed   :",
        f"{len(excluded_val):,}"
    )

    print(
        "Test anchors  :",
        f"{len(test_anchors):,}"
    )

    print(
        "  from train  :",
        f"{len(test_anchors_train):,}"
    )

    print(
        "  from val    :",
        f"{len(test_anchors_val):,}"
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
    print("VALIDATION")
    print("-" * 94)

    print(
        "Accounting          :",
        "PASS"
        if accounting_pass
        else "FAIL"
    )

    print(
        "TEST protected      :",
        "PASS"
        if test_immutable_pass
        else "FAIL"
    )

    print(
        "Filename isolation  :",
        "PASS"
        if filename_isolation_pass
        else "FAIL"
    )

    print(
        "Direct Test leakage :",
        "PASS"
        if direct_leakage_pass
        else "FAIL"
    )

    print(
        "Surviving edges     :",
        len(
            surviving_direct_edges
        )
    )

    final_pass = (
        accounting_pass
        and
        test_immutable_pass
        and
        filename_isolation_pass
        and
        direct_leakage_pass
    )

    print()
    print(
        "RESULT:",
        (
            "DIRECT TEST-LEAKAGE CLEAN PASS"
            if final_pass
            else
            "DIRECT TEST-LEAKAGE CLEAN FAIL"
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
