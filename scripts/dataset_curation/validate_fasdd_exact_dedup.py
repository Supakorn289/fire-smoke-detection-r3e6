#!/usr/bin/env python3

from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[2]

DEDUP = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_exact"
)

DUP_JSON = (
    ROOT
    / "reports"
    / "fasdd"
    / "exact_duplicates.json"
)


EXPECTED = {
    "train": 47520,
    "val": 31712,
    "test": 15865,
}


def load(name):

    path = DEDUP / name

    return {
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    }


def main():

    print("=" * 82)
    print("FASDD EXACT-DEDUP MANIFEST VALIDATION")
    print("=" * 82)

    clean = {}
    exclude = {}
    quarantine = {}
    original = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        original[split] = load(
            f"{split}_original_stems.txt"
        )

        clean[split] = load(
            f"{split}_clean_exact_stems.txt"
        )

        exclude[split] = load(
            f"{split}_exclude_exact_stems.txt"
        )

        quarantine[split] = load(
            f"{split}_quarantine_exact_stems.txt"
        )

    # ========================================================
    # MANIFEST ACCOUNTING
    # ========================================================

    accounting_pass = True

    print()
    print("SPLIT ACCOUNTING")
    print("-" * 82)

    for split in (
        "train",
        "val",
        "test",
    ):

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

        missing = (
            original[split]
            - union
        )

        extra = (
            union
            - original[split]
        )

        ok = (
            not overlap_ce
            and not overlap_cq
            and not overlap_eq
            and not missing
            and not extra
            and len(clean[split])
                == EXPECTED[split]
        )

        if not ok:
            accounting_pass = False

        print(
            f"{split.upper():5} "
            f"original={len(original[split]):6} "
            f"clean={len(clean[split]):6} "
            f"exclude={len(exclude[split]):4} "
            f"quarantine={len(quarantine[split]):4} "
            f"status={'PASS' if ok else 'FAIL'}"
        )

    # ========================================================
    # CLEAN SPLIT OVERLAP
    # ========================================================

    tv = clean["train"] & clean["val"]
    tt = clean["train"] & clean["test"]
    vt = clean["val"] & clean["test"]

    print()
    print("CLEAN FILENAME OVERLAP")
    print("-" * 82)

    print("train ∩ val :", len(tv))
    print("train ∩ test:", len(tt))
    print("val ∩ test  :", len(vt))

    filename_pass = (
        len(tv) == 0
        and len(tt) == 0
        and len(vt) == 0
    )

    # ========================================================
    # EXACT DUPLICATE GROUP CHECK
    # ========================================================

    data = json.loads(
        DUP_JSON.read_text(
            encoding="utf-8"
        )
    )

    clean_all = {
        stem: split
        for split, stems in clean.items()
        for stem in stems
    }

    surviving_groups = []

    for group in data[
        "duplicate_groups"
    ]:

        survivors = []

        for member in group[
            "members"
        ]:

            stem = member["stem"]

            if stem in clean_all:

                survivors.append({
                    "stem": stem,
                    "split": clean_all[stem],
                })

        if len(survivors) > 1:

            surviving_groups.append({
                "group_id":
                    group["group_id"],

                "survivors":
                    survivors,
            })

    print()
    print("EXACT DUPLICATE CHECK")
    print("-" * 82)

    print(
        "Original duplicate groups :",
        len(
            data["duplicate_groups"]
        ),
    )

    print(
        "Groups with >1 clean copy :",
        len(surviving_groups),
    )

    duplicate_pass = (
        len(surviving_groups) == 0
    )

    # ========================================================
    # CLEAN TOTAL
    # ========================================================

    total_clean = sum(
        len(x)
        for x in clean.values()
    )

    removed = (
        95314
        - total_clean
    )

    print()
    print("TOTALS")
    print("-" * 82)

    print(
        "Original:",
        95314
    )

    print(
        "Clean   :",
        total_clean
    )

    print(
        "Removed from clean manifests:",
        removed
    )

    expected_total_pass = (
        total_clean == 95097
        and removed == 217
    )

    # ========================================================
    # FINAL
    # ========================================================

    final_pass = (
        accounting_pass
        and filename_pass
        and duplicate_pass
        and expected_total_pass
    )

    print()
    print("=" * 82)

    print(
        "Manifest accounting :",
        "PASS"
        if accounting_pass
        else "FAIL"
    )

    print(
        "Filename isolation  :",
        "PASS"
        if filename_pass
        else "FAIL"
    )

    print(
        "Exact deduplication :",
        "PASS"
        if duplicate_pass
        else "FAIL"
    )

    print(
        "Expected clean size :",
        "PASS"
        if expected_total_pass
        else "FAIL"
    )

    print()

    if final_pass:

        print(
            "RESULT: EXACT-DEDUP MANIFEST PASS"
        )

    else:

        print(
            "RESULT: EXACT-DEDUP MANIFEST FAIL"
        )

        if surviving_groups:

            print()
            print(
                "Surviving duplicate groups:"
            )

            for x in surviving_groups[:20]:
                print(x)

    print("=" * 82)


if __name__ == "__main__":
    main()
