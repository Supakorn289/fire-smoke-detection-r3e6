#!/usr/bin/env python3

from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[2]

BASE = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_exact"
)

VISUAL = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_visual_exact"
)

ANALYSIS = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "decoded_exact"
    / "analysis.csv"
)


EXPECTED = {
    "train": 47503,
    "val": 31693,
    "test": 15863,
}


def load(path):

    return {
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    }


def main():

    print("=" * 86)
    print("FASDD VISUAL-EXACT MANIFEST VALIDATION")
    print("=" * 86)

    base = {}
    clean = {}
    exclude = {}
    quarantine = {}

    accounting_pass = True

    # ========================================================
    # ACCOUNTING
    # ========================================================

    print()
    print("SPLIT ACCOUNTING")
    print("-" * 86)

    for split in (
        "train",
        "val",
        "test",
    ):

        base[split] = load(
            VISUAL
            / f"{split}_base_exact_clean.txt"
        )

        clean[split] = load(
            VISUAL
            / f"{split}_clean_visual_exact.txt"
        )

        exclude[split] = load(
            VISUAL
            / f"{split}_exclude_visual_exact.txt"
        )

        quarantine[split] = load(
            VISUAL
            / f"{split}_quarantine_visual_exact.txt"
        )

        expected_base = load(
            BASE
            / f"{split}_clean_exact_stems.txt"
        )

        union = (
            clean[split]
            | exclude[split]
            | quarantine[split]
        )

        overlap_ce = (
            clean[split]
            & exclude[split]
        )

        overlap_cq = (
            clean[split]
            & quarantine[split]
        )

        overlap_eq = (
            exclude[split]
            & quarantine[split]
        )

        ok = (
            base[split]
            == expected_base
            and
            union == base[split]
            and
            not overlap_ce
            and
            not overlap_cq
            and
            not overlap_eq
            and
            len(clean[split])
            == EXPECTED[split]
        )

        if not ok:
            accounting_pass = False

        print(
            f"{split.upper():5} "
            f"base={len(base[split]):6} "
            f"clean={len(clean[split]):6} "
            f"exclude={len(exclude[split]):3} "
            f"quarantine={len(quarantine[split]):3} "
            f"status={'PASS' if ok else 'FAIL'}"
        )

    # ========================================================
    # CLEAN SPLIT ISOLATION
    # ========================================================

    train_val = (
        clean["train"]
        & clean["val"]
    )

    train_test = (
        clean["train"]
        & clean["test"]
    )

    val_test = (
        clean["val"]
        & clean["test"]
    )

    print()
    print("CLEAN FILENAME OVERLAP")
    print("-" * 86)

    print(
        "train ∩ val :",
        len(train_val)
    )

    print(
        "train ∩ test:",
        len(train_test)
    )

    print(
        "val ∩ test  :",
        len(val_test)
    )

    split_pass = (
        not train_val
        and not train_test
        and not val_test
    )

    # ========================================================
    # DECODED-PIXEL EXACT SURVIVAL CHECK
    # ========================================================

    clean_lookup = {
        stem: split
        for split, stems
        in clean.items()
        for stem in stems
    }

    surviving_pairs = []

    total_pairs = 0

    with ANALYSIS.open(
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            total_pairs += 1

            a = row["stem_a"]
            b = row["stem_b"]

            if (
                a in clean_lookup
                and b in clean_lookup
            ):

                surviving_pairs.append({
                    "stem_a": a,
                    "split_a":
                        clean_lookup[a],

                    "stem_b": b,
                    "split_b":
                        clean_lookup[b],

                    "annotation_result":
                        row[
                            "annotation_result"
                        ],
                })

    print()
    print("DECODED-PIXEL EXACT CHECK")
    print("-" * 86)

    print(
        "Decoded-exact pairs checked :",
        total_pairs
    )

    print(
        "Pairs with both copies clean:",
        len(surviving_pairs)
    )

    visual_dedup_pass = (
        len(surviving_pairs) == 0
    )

    # ========================================================
    # TOTALS
    # ========================================================

    clean_total = sum(
        len(x)
        for x in clean.values()
    )

    base_total = sum(
        len(x)
        for x in base.values()
    )

    removed = (
        base_total
        - clean_total
    )

    print()
    print("TOTALS")
    print("-" * 86)

    print(
        "Exact-clean base     :",
        base_total
    )

    print(
        "Visual-exact clean   :",
        clean_total
    )

    print(
        "Additional removed   :",
        removed
    )

    totals_pass = (
        base_total == 95097
        and
        clean_total == 95059
        and
        removed == 38
    )

    # ========================================================
    # FINAL
    # ========================================================

    final_pass = (
        accounting_pass
        and split_pass
        and visual_dedup_pass
        and totals_pass
    )

    print()
    print("=" * 86)

    print(
        "Manifest accounting   :",
        "PASS"
        if accounting_pass
        else "FAIL"
    )

    print(
        "Filename isolation    :",
        "PASS"
        if split_pass
        else "FAIL"
    )

    print(
        "Decoded visual dedup  :",
        "PASS"
        if visual_dedup_pass
        else "FAIL"
    )

    print(
        "Expected totals       :",
        "PASS"
        if totals_pass
        else "FAIL"
    )

    print()

    if final_pass:

        print(
            "RESULT: VISUAL-EXACT MANIFEST PASS"
        )

    else:

        print(
            "RESULT: VISUAL-EXACT MANIFEST FAIL"
        )

        if surviving_pairs:

            print()
            print(
                "Surviving pairs:"
            )

            for x in surviving_pairs[:20]:
                print(x)

    print("=" * 86)


if __name__ == "__main__":
    main()
