#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

DATA = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)

YOLO = DATA / "annotations" / "YOLO_CV"

ANALYSIS_CSV = (
    ROOT
    / "reports"
    / "fasdd"
    / "duplicate_label_analysis.csv"
)

OUT = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_exact"
)

SUMMARY_JSON = OUT / "summary.json"
DECISIONS_CSV = OUT / "decisions.csv"
CONFLICTS_CSV = OUT / "conflicts.csv"


PRIORITY = {
    "train": 0,
    "val": 1,
    "test": 2,
}

SAFE_RESULTS = {
    "exact_annotation",
    "equivalent_annotation",
}

CONFLICT_RESULTS = {
    "geometry_conflict",
    "box_count_conflict",
    "class_conflict",
    "missing_label",
}


def read_split(split):

    path = YOLO / f"{split}.txt"

    lines = [
        x.strip()
        for x in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if x.strip()
    ]

    return {
        Path(x).stem
        for x in lines
    }


def write_set(path, values):

    path.write_text(
        "".join(
            f"{x}\n"
            for x in sorted(values)
        ),
        encoding="utf-8",
    )


def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 84)
    print("FASDD SAFE EXACT-DEDUP MANIFEST BUILDER")
    print("=" * 84)

    originals = {
        split: read_split(split)
        for split in (
            "train",
            "val",
            "test",
        )
    }

    exclude = defaultdict(set)
    quarantine = defaultdict(set)

    conflict_all = set()

    stats = Counter()
    matrix = Counter()

    decisions = []
    conflict_rows = []

    with ANALYSIS_CSV.open(
        encoding="utf-8",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    for row in rows:

        group_id = int(
            row["group_id"]
        )

        relation = row["relation"]
        result = row["result"]

        stem_a = row["stem_a"]
        stem_b = row["stem_b"]

        split_a = row["split_a"]
        split_b = row["split_b"]

        matrix[
            (relation, result)
        ] += 1

        stats["pairs"] += 1

        # ====================================================
        # SAFE DUPLICATE
        # ====================================================

        if result in SAFE_RESULTS:

            stats[
                "safe_pairs"
            ] += 1

            # -----------------------------------------------
            # Same split:
            # deterministic canonical by lexical stem
            # -----------------------------------------------

            if split_a == split_b:

                keep = min(
                    stem_a,
                    stem_b,
                )

                remove = max(
                    stem_a,
                    stem_b,
                )

                exclude[
                    split_a
                ].add(remove)

                action = (
                    "safe_same_split_"
                    "keep_lexical_canonical"
                )

                decisions.append({
                    "group_id":
                        group_id,

                    "relation":
                        relation,

                    "result":
                        result,

                    "action":
                        action,

                    "keep_stem":
                        keep,

                    "keep_split":
                        split_a,

                    "exclude_stem":
                        remove,

                    "exclude_split":
                        split_a,

                    "quarantine_stem":
                        "",
                    "quarantine_split":
                        "",
                })

                continue

            # -----------------------------------------------
            # Cross split:
            # TEST > VAL > TRAIN
            # -----------------------------------------------

            if (
                PRIORITY[split_a]
                >
                PRIORITY[split_b]
            ):

                high_stem = stem_a
                high_split = split_a

                low_stem = stem_b
                low_split = split_b

            else:

                high_stem = stem_b
                high_split = split_b

                low_stem = stem_a
                low_split = split_a

            exclude[
                low_split
            ].add(low_stem)

            decisions.append({
                "group_id":
                    group_id,

                "relation":
                    relation,

                "result":
                    result,

                "action":
                    "safe_cross_split_keep_higher_priority",

                "keep_stem":
                    high_stem,

                "keep_split":
                    high_split,

                "exclude_stem":
                    low_stem,

                "exclude_split":
                    low_split,

                "quarantine_stem":
                    "",
                "quarantine_split":
                    "",
            })

            continue

        # ====================================================
        # CONFLICT DUPLICATE
        # ====================================================

        stats[
            "conflict_pairs"
        ] += 1

        conflict_all.update(
            [
                stem_a,
                stem_b,
            ]
        )

        # -----------------------------------------------
        # Same split conflict:
        # quarantine BOTH
        # -----------------------------------------------

        if split_a == split_b:

            quarantine[
                split_a
            ].update(
                [
                    stem_a,
                    stem_b,
                ]
            )

            action = (
                "conflict_same_split_"
                "quarantine_both"
            )

            decisions.append({
                "group_id":
                    group_id,

                "relation":
                    relation,

                "result":
                    result,

                "action":
                    action,

                "keep_stem":
                    "",

                "keep_split":
                    "",

                "exclude_stem":
                    "",

                "exclude_split":
                    "",

                "quarantine_stem":
                    f"{stem_a}|{stem_b}",

                "quarantine_split":
                    split_a,
            })

        # -----------------------------------------------
        # Cross split conflict:
        #
        # lower-priority copy is excluded for leakage
        # higher-priority copy is quarantined until
        # annotation adjudication.
        # -----------------------------------------------

        else:

            if (
                PRIORITY[split_a]
                >
                PRIORITY[split_b]
            ):

                high_stem = stem_a
                high_split = split_a

                low_stem = stem_b
                low_split = split_b

            else:

                high_stem = stem_b
                high_split = split_b

                low_stem = stem_a
                low_split = split_a

            exclude[
                low_split
            ].add(low_stem)

            quarantine[
                high_split
            ].add(high_stem)

            decisions.append({
                "group_id":
                    group_id,

                "relation":
                    relation,

                "result":
                    result,

                "action":
                    "conflict_cross_split_exclude_low_quarantine_high",

                "keep_stem":
                    "",

                "keep_split":
                    "",

                "exclude_stem":
                    low_stem,

                "exclude_split":
                    low_split,

                "quarantine_stem":
                    high_stem,

                "quarantine_split":
                    high_split,
            })

        conflict_rows.append({
            "group_id":
                group_id,

            "relation":
                relation,

            "result":
                result,

            "stem_a":
                stem_a,

            "split_a":
                split_a,

            "boxes_a":
                row["boxes_a"],

            "classes_a":
                row["classes_a"],

            "stem_b":
                stem_b,

            "split_b":
                split_b,

            "boxes_b":
                row["boxes_b"],

            "classes_b":
                row["classes_b"],
        })

    # ========================================================
    # CLEAN SETS
    # ========================================================

    clean = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        clean[split] = (
            originals[split]
            - exclude[split]
            - quarantine[split]
        )

        # Safety
        assert (
            clean[split]
            & exclude[split]
        ) == set()

        assert (
            clean[split]
            & quarantine[split]
        ) == set()

    # ========================================================
    # WRITE MANIFESTS
    # ========================================================

    for split in (
        "train",
        "val",
        "test",
    ):

        write_set(
            OUT
            / f"{split}_original_stems.txt",
            originals[split],
        )

        write_set(
            OUT
            / f"{split}_clean_exact_stems.txt",
            clean[split],
        )

        write_set(
            OUT
            / f"{split}_exclude_exact_stems.txt",
            exclude[split],
        )

        write_set(
            OUT
            / f"{split}_quarantine_exact_stems.txt",
            quarantine[split],
        )

    # ========================================================
    # DECISIONS CSV
    # ========================================================

    with DECISIONS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "group_id",
            "relation",
            "result",
            "action",
            "keep_stem",
            "keep_split",
            "exclude_stem",
            "exclude_split",
            "quarantine_stem",
            "quarantine_split",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            decisions
        )

    # ========================================================
    # CONFLICT CSV
    # ========================================================

    with CONFLICTS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "group_id",
            "relation",
            "result",
            "stem_a",
            "split_a",
            "boxes_a",
            "classes_a",
            "stem_b",
            "split_b",
            "boxes_b",
            "classes_b",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            conflict_rows
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "policy":
            "conservative_exact_dedup",

        "priority":
            "test > val > train",

        "stats":
            dict(stats),

        "matrix": {
            f"{relation}|{result}":
                count

            for (
                relation,
                result
            ), count
            in sorted(
                matrix.items()
            )
        },

        "splits": {},
    }

    for split in (
        "train",
        "val",
        "test",
    ):

        summary[
            "splits"
        ][split] = {
            "original":
                len(
                    originals[split]
                ),

            "excluded":
                len(
                    exclude[split]
                ),

            "quarantine":
                len(
                    quarantine[split]
                ),

            "clean_exact":
                len(
                    clean[split]
                ),
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
    print("RELATION × ANNOTATION RESULT")
    print("-" * 84)

    relations = (
        "same_train",
        "same_val",
        "same_test",
        "train_val",
        "train_test",
        "val_test",
    )

    results = (
        "exact_annotation",
        "geometry_conflict",
        "box_count_conflict",
        "class_conflict",
    )

    for relation in relations:

        print()
        print(relation)

        for result in results:

            print(
                f"  {result:22}: "
                f"{matrix[(relation, result)]}"
            )

    print()
    print("=" * 84)
    print("SAFE EXACT-DEDUP SUMMARY")
    print("=" * 84)

    print(
        "Duplicate pairs :",
        stats["pairs"],
    )

    print(
        "Safe pairs      :",
        stats["safe_pairs"],
    )

    print(
        "Conflict pairs  :",
        stats["conflict_pairs"],
    )

    print()

    for split in (
        "train",
        "val",
        "test",
    ):

        print(
            f"{split.upper():5} "
            f"original={len(originals[split]):6} "
            f"exclude={len(exclude[split]):4} "
            f"quarantine={len(quarantine[split]):4} "
            f"clean={len(clean[split]):6}"
        )

    print()
    print(
        "Important: official TEST files were NOT modified."
    )

    print(
        "test_clean_exact is only a clean benchmark manifest."
    )

    print()
    print(
        "Output:",
        OUT,
    )

    print("=" * 84)


if __name__ == "__main__":
    main()
