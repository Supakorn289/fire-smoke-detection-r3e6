#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_exact"
)

ANALYSIS = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "decoded_exact"
    / "analysis.csv"
)

OUT = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_visual_exact"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_JSON = OUT / "summary.json"
DECISIONS_CSV = OUT / "decisions.csv"


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


def main():

    print("=" * 88)
    print("FASDD VISUAL-EXACT MANIFEST BUILDER")
    print("=" * 88)

    # ========================================================
    # Base = already byte-exact-clean manifests
    # ========================================================

    base_clean = {
        split: load_set(
            BASE
            / f"{split}_clean_exact_stems.txt"
        )
        for split in (
            "train",
            "val",
            "test",
        )
    }

    exclude = defaultdict(set)
    quarantine = defaultdict(set)

    stats = Counter()
    decisions = []

    with ANALYSIS.open(
        encoding="utf-8",
        newline="",
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    print(
        "Decoded-exact pairs:",
        len(rows)
    )

    # ========================================================
    # Decisions
    # ========================================================

    for row in rows:

        a = row["stem_a"]
        b = row["stem_b"]

        split_a = row["split_a"]
        split_b = row["split_b"]

        relation = row["relation"]
        result = row["annotation_result"]

        stats["pairs"] += 1
        stats[f"annotation_{result}"] += 1

        # ----------------------------------------------------
        # Safety: both must belong to current clean layer
        # ----------------------------------------------------

        if a not in base_clean[split_a]:
            raise RuntimeError(
                f"{a} is not in base clean {split_a}"
            )

        if b not in base_clean[split_b]:
            raise RuntimeError(
                f"{b} is not in base clean {split_b}"
            )

        # ====================================================
        # SAFE ANNOTATION
        # ====================================================

        if result in SAFE_RESULTS:

            stats["safe_pairs"] += 1

            if split_a == split_b:

                keep = min(a, b)
                remove = max(a, b)

                exclude[
                    split_a
                ].add(remove)

                decisions.append({
                    "stem_a": a,
                    "split_a": split_a,
                    "stem_b": b,
                    "split_b": split_b,
                    "relation": relation,
                    "annotation_result": result,
                    "action":
                        "safe_same_split_keep_canonical",
                    "keep": keep,
                    "exclude": remove,
                    "quarantine": "",
                })

            else:

                if (
                    PRIORITY[split_a]
                    >
                    PRIORITY[split_b]
                ):

                    high_stem = a
                    high_split = split_a

                    low_stem = b
                    low_split = split_b

                else:

                    high_stem = b
                    high_split = split_b

                    low_stem = a
                    low_split = split_a

                exclude[
                    low_split
                ].add(low_stem)

                decisions.append({
                    "stem_a": a,
                    "split_a": split_a,
                    "stem_b": b,
                    "split_b": split_b,
                    "relation": relation,
                    "annotation_result": result,
                    "action":
                        "safe_cross_split_keep_higher_priority",
                    "keep":
                        f"{high_split}:{high_stem}",
                    "exclude":
                        f"{low_split}:{low_stem}",
                    "quarantine": "",
                })

            continue

        # ====================================================
        # CONFLICT ANNOTATION
        # ====================================================

        if result not in CONFLICT_RESULTS:

            raise RuntimeError(
                f"Unknown annotation result: {result}"
            )

        stats["conflict_pairs"] += 1

        # ----------------------------------------------------
        # Same split conflict:
        # quarantine both
        # ----------------------------------------------------

        if split_a == split_b:

            quarantine[
                split_a
            ].update(
                [a, b]
            )

            decisions.append({
                "stem_a": a,
                "split_a": split_a,
                "stem_b": b,
                "split_b": split_b,
                "relation": relation,
                "annotation_result": result,
                "action":
                    "conflict_same_split_quarantine_both",
                "keep": "",
                "exclude": "",
                "quarantine":
                    f"{split_a}:{a}|{split_b}:{b}",
            })

            continue

        # ----------------------------------------------------
        # Cross-split conflict:
        # remove lower-priority side from clean;
        # quarantine higher-priority side.
        # ----------------------------------------------------

        if (
            PRIORITY[split_a]
            >
            PRIORITY[split_b]
        ):

            high_stem = a
            high_split = split_a

            low_stem = b
            low_split = split_b

        else:

            high_stem = b
            high_split = split_b

            low_stem = a
            low_split = split_a

        exclude[
            low_split
        ].add(low_stem)

        quarantine[
            high_split
        ].add(high_stem)

        decisions.append({
            "stem_a": a,
            "split_a": split_a,
            "stem_b": b,
            "split_b": split_b,
            "relation": relation,
            "annotation_result": result,
            "action":
                "conflict_cross_exclude_low_quarantine_high",
            "keep": "",
            "exclude":
                f"{low_split}:{low_stem}",
            "quarantine":
                f"{high_split}:{high_stem}",
        })

    # ========================================================
    # New clean manifests
    # ========================================================

    new_clean = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        new_clean[split] = (
            base_clean[split]
            - exclude[split]
            - quarantine[split]
        )

        assert not (
            exclude[split]
            & quarantine[split]
        )

        assert not (
            new_clean[split]
            & exclude[split]
        )

        assert not (
            new_clean[split]
            & quarantine[split]
        )

    # ========================================================
    # Save manifests
    # ========================================================

    for split in (
        "train",
        "val",
        "test",
    ):

        write_set(
            OUT
            / f"{split}_base_exact_clean.txt",
            base_clean[split],
        )

        write_set(
            OUT
            / f"{split}_exclude_visual_exact.txt",
            exclude[split],
        )

        write_set(
            OUT
            / f"{split}_quarantine_visual_exact.txt",
            quarantine[split],
        )

        write_set(
            OUT
            / f"{split}_clean_visual_exact.txt",
            new_clean[split],
        )

    # ========================================================
    # Decision CSV
    # ========================================================

    with DECISIONS_CSV.open(
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
            "action",
            "keep",
            "exclude",
            "quarantine",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(decisions)

    # ========================================================
    # Summary
    # ========================================================

    removed_additional = sum(
        len(base_clean[s])
        - len(new_clean[s])
        for s in (
            "train",
            "val",
            "test",
        )
    )

    summary = {
        "source_layer":
            "dedup_exact clean manifests",

        "policy":
            "conservative decoded-pixel exact dedup",

        "stats":
            dict(stats),

        "splits": {
            split: {
                "base_clean":
                    len(base_clean[split]),

                "exclude":
                    len(exclude[split]),

                "quarantine":
                    len(quarantine[split]),

                "clean_visual_exact":
                    len(new_clean[split]),
            }
            for split in (
                "train",
                "val",
                "test",
            )
        },

        "additional_removed_from_clean":
            removed_additional,

        "total_clean_visual_exact":
            sum(
                len(new_clean[s])
                for s in (
                    "train",
                    "val",
                    "test",
                )
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
    # Print
    # ========================================================

    print()
    print("=" * 88)
    print("VISUAL-EXACT CLEAN SUMMARY")
    print("=" * 88)

    print(
        "Pairs          :",
        stats["pairs"]
    )

    print(
        "Safe pairs     :",
        stats["safe_pairs"]
    )

    print(
        "Conflict pairs :",
        stats["conflict_pairs"]
    )

    print()

    for split in (
        "train",
        "val",
        "test",
    ):

        print(
            f"{split.upper():5} "
            f"base={len(base_clean[split]):6} "
            f"exclude={len(exclude[split]):3} "
            f"quarantine={len(quarantine[split]):3} "
            f"clean={len(new_clean[split]):6}"
        )

    print()
    print(
        "Additional images removed from clean:",
        removed_additional
    )

    print(
        "Total visual-exact clean:",
        summary[
            "total_clean_visual_exact"
        ]
    )

    print()
    print(
        "Official dataset files were NOT modified."
    )

    print(
        "Official TEST files were NOT modified."
    )

    print()
    print(
        "Output:",
        OUT
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
