#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import hashlib
import json


ROOT = Path(__file__).resolve().parents[2]

INVENTORY = (
    ROOT
    / "reports"
    / "fasdd"
    / "r1_validation"
    / "val_inventory.csv"
)

CLEAN_VAL = (
    ROOT
    / "reports"
    / "fasdd"
    / "clean_v1"
    / "val.txt"
)

CLEAN_TEST = (
    ROOT
    / "reports"
    / "fasdd"
    / "clean_v1"
    / "test.txt"
)

R1_TRAIN = (
    ROOT
    / "reports"
    / "fasdd"
    / "r1_selection_v1"
    / "fasdd_r1_train_12000.txt"
)

OUT = (
    ROOT
    / "reports"
    / "fasdd"
    / "r1_validation_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MAIN_TXT = (
    OUT
    / "fasdd_r1_val_main_3000.txt"
)

MAIN_CSV = (
    OUT
    / "fasdd_r1_val_main_3000.csv"
)

SMALL_SMOKE_TXT = (
    OUT
    / "fasdd_r1_val_small_smoke_222.txt"
)

SUMMARY_JSON = (
    OUT
    / "summary.json"
)


QUOTAS = {
    "fire_only": 396,
    "smoke_only": 742,
    "fire_smoke": 612,
    "negative": 1250,
}

TARGET = sum(QUOTAS.values())


def load_set(path):
    return {
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    }


def stable_rank(stem):
    return hashlib.sha256(
        (
            "FASDD_R1_VAL_MAIN_V1|"
            + stem
        ).encode("utf-8")
    ).hexdigest()


def main():

    print("=" * 94)
    print("FASDD R1 VALIDATION V1 SELECTOR")
    print("=" * 94)

    clean_val = load_set(CLEAN_VAL)
    clean_test = load_set(CLEAN_TEST)
    r1_train = load_set(R1_TRAIN)

    if len(clean_val) != 31254:
        raise RuntimeError(
            f"Expected 31,254 clean VAL, got {len(clean_val)}"
        )

    if len(r1_train) != 12000:
        raise RuntimeError(
            f"Expected 12,000 R1 train, got {len(r1_train)}"
        )

    if clean_val & r1_train:
        raise RuntimeError(
            "Clean VAL overlaps R1 TRAIN"
        )

    if clean_val & clean_test:
        raise RuntimeError(
            "Clean VAL overlaps TEST"
        )

    rows = []

    with INVENTORY.open(
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            stem = row["stem"]

            if stem not in clean_val:
                raise RuntimeError(
                    f"Inventory stem outside clean VAL: {stem}"
                )

            row["_rank"] = stable_rank(stem)

            rows.append(row)

    if len(rows) != 31254:
        raise RuntimeError(
            f"Expected 31,254 inventory rows, got {len(rows)}"
        )

    # ========================================================
    # CATEGORY-STRATIFIED REPRESENTATIVE MAIN VAL
    # ========================================================

    pools = {
        category: []
        for category in QUOTAS
    }

    for row in rows:
        pools[row["category"]].append(row)

    selected = []

    for category, quota in QUOTAS.items():

        pool = sorted(
            pools[category],
            key=lambda r: (
                r["_rank"],
                r["stem"],
            ),
        )

        if len(pool) < quota:
            raise RuntimeError(
                f"{category}: quota exceeds available"
            )

        selected.extend(
            pool[:quota]
        )

    selected_stems = {
        row["stem"]
        for row in selected
    }

    if len(selected_stems) != TARGET:
        raise RuntimeError(
            f"Expected {TARGET} unique VAL images, "
            f"got {len(selected_stems)}"
        )

    # ========================================================
    # SMALL-SMOKE DIAGNOSTIC SLICE
    # ========================================================

    small_smoke_rows = [
        row
        for row in rows
        if int(row["smoke_small"]) > 0
    ]

    small_smoke_stems = {
        row["stem"]
        for row in small_smoke_rows
    }

    if len(small_smoke_stems) != 222:
        raise RuntimeError(
            f"Expected 222 small-smoke VAL images, "
            f"got {len(small_smoke_stems)}"
        )

    # ========================================================
    # HARD SPLIT VALIDATION
    # ========================================================

    train_overlap = (
        selected_stems
        & r1_train
    )

    test_overlap = (
        selected_stems
        & clean_test
    )

    if train_overlap:
        raise RuntimeError(
            "Main VAL overlaps R1 TRAIN"
        )

    if test_overlap:
        raise RuntimeError(
            "Main VAL overlaps TEST"
        )

    # ========================================================
    # PROFILE MAIN VAL
    # ========================================================

    categories = Counter()
    image_signals = Counter()
    object_counts = Counter()

    for row in selected:

        categories[
            row["category"]
        ] += 1

        for signal in (
            "fire_small",
            "fire_medium",
            "fire_large",
            "smoke_small",
            "smoke_medium",
            "smoke_large",
        ):

            count = int(
                row[signal]
            )

            object_counts[
                signal
            ] += count

            if count > 0:
                image_signals[
                    signal
                ] += 1

        if (
            row["category"]
            == "fire_smoke"
        ):
            image_signals[
                "fire_smoke"
            ] += 1

    # ========================================================
    # WRITE MAIN MANIFEST
    # ========================================================

    MAIN_TXT.write_text(
        "".join(
            f"{stem}\n"
            for stem
            in sorted(selected_stems)
        ),
        encoding="utf-8",
    )

    # ========================================================
    # WRITE SMALL-SMOKE SLICE
    # ========================================================

    SMALL_SMOKE_TXT.write_text(
        "".join(
            f"{stem}\n"
            for stem
            in sorted(
                small_smoke_stems
            )
        ),
        encoding="utf-8",
    )

    # ========================================================
    # WRITE MAIN CSV
    # ========================================================

    selected.sort(
        key=lambda r: (
            r["category"],
            r["_rank"],
        )
    )

    fields = [
        key
        for key in rows[0].keys()
        if not key.startswith("_")
    ]

    with MAIN_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in selected:

            writer.writerow({
                key: row[key]
                for key in fields
            })

    # ========================================================
    # HASHES
    # ========================================================

    main_sha256 = hashlib.sha256(
        MAIN_TXT.read_bytes()
    ).hexdigest()

    small_smoke_sha256 = (
        hashlib.sha256(
            SMALL_SMOKE_TXT.read_bytes()
        ).hexdigest()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    main_small_smoke = len(
        selected_stems
        & small_smoke_stems
    )

    summary = {
        "version":
            "FASDD_R1_VAL_V1",

        "main_validation": {
            "images":
                len(selected_stems),

            "quotas":
                QUOTAS,

            "categories":
                dict(categories),

            "image_signals":
                dict(image_signals),

            "object_counts":
                dict(object_counts),

            "small_smoke_images":
                main_small_smoke,

            "manifest":
                str(MAIN_TXT),

            "sha256":
                main_sha256,

            "purpose":
                (
                    "Primary representative validation "
                    "for model selection and early stopping."
                ),
        },

        "small_smoke_diagnostic": {
            "images":
                len(small_smoke_stems),

            "overlap_with_main":
                main_small_smoke,

            "manifest":
                str(SMALL_SMOKE_TXT),

            "sha256":
                small_smoke_sha256,

            "purpose":
                (
                    "Diagnostic rare-case slice only. "
                    "Not used for early stopping "
                    "or best-model selection."
                ),
        },

        "validation": {
            "main_count":
                len(selected_stems),

            "unique_main":
                len(selected_stems),

            "train_overlap":
                len(train_overlap),

            "test_overlap":
                len(test_overlap),

            "inside_clean_val":
                selected_stems
                <= clean_val,

            "small_smoke_count":
                len(small_smoke_stems),
        },

        "selection_policy": {
            "method":
                (
                    "category-stratified deterministic "
                    "sampling preserving full VAL "
                    "category proportions"
                ),

            "rare_signal_prioritization":
                False,

            "test_used":
                False,

            "train_used_for_ranking":
                False,

            "tie_break":
                "SHA256 deterministic",
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
    print("FASDD R1 VAL V1 SUMMARY")
    print("=" * 94)

    print(
        "Main VAL:",
        f"{len(selected_stems):,}"
    )

    print()
    print("CATEGORY")
    print("-" * 94)

    for category in (
        "fire_only",
        "smoke_only",
        "fire_smoke",
        "negative",
    ):

        print(
            f"{category:16}: "
            f"{categories[category]:,}"
        )

    print()
    print("IMAGE SIGNAL COVERAGE")
    print("-" * 94)

    for signal in (
        "fire_small",
        "fire_medium",
        "fire_large",
        "smoke_small",
        "smoke_medium",
        "smoke_large",
        "fire_smoke",
    ):

        print(
            f"{signal:16}: "
            f"{image_signals[signal]:,}"
        )

    print()
    print("OBJECT COUNTS")
    print("-" * 94)

    for signal in (
        "fire_small",
        "fire_medium",
        "fire_large",
        "smoke_small",
        "smoke_medium",
        "smoke_large",
    ):

        print(
            f"{signal:16}: "
            f"{object_counts[signal]:,}"
        )

    print()
    print("SMALL-SMOKE DIAGNOSTIC")
    print("-" * 94)

    print(
        "All clean VAL small-smoke:",
        len(
            small_smoke_stems
        )
    )

    print(
        "Present in Main VAL       :",
        main_small_smoke
    )

    print()
    print("VALIDATION")
    print("-" * 94)

    print(
        "TRAIN overlap:",
        len(train_overlap)
    )

    print(
        "TEST overlap :",
        len(test_overlap)
    )

    print(
        "Inside clean VAL:",
        (
            "PASS"
            if selected_stems
            <= clean_val
            else "FAIL"
        )
    )

    final_pass = (
        len(selected_stems)
        == 3000
        and
        len(small_smoke_stems)
        == 222
        and
        not train_overlap
        and
        not test_overlap
        and
        selected_stems
        <= clean_val
    )

    print()
    print(
        "RESULT:",
        (
            "FASDD R1 VAL V1 PASS"
            if final_pass
            else
            "FASDD R1 VAL V1 FAIL"
        )
    )

    print()
    print(
        "Main manifest :",
        MAIN_TXT
    )

    print(
        "Small-smoke   :",
        SMALL_SMOKE_TXT
    )

    print(
        "Summary       :",
        SUMMARY_JSON
    )

    print("=" * 94)


if __name__ == "__main__":
    main()
