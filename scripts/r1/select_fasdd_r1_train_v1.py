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
    / "r1_candidates"
    / "train_inventory.csv"
)

CLEAN_TRAIN = (
    ROOT
    / "reports"
    / "fasdd"
    / "clean_v1"
    / "train.txt"
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

OUT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
    / "r1_selection_v1"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SELECTED_TXT = (
    OUT_DIR
    / "fasdd_r1_train_12000.txt"
)

SELECTED_CSV = (
    OUT_DIR
    / "selected.csv"
)

REJECTED_TXT = (
    OUT_DIR
    / "not_selected.txt"
)

SUMMARY_JSON = (
    OUT_DIR
    / "summary.json"
)


QUOTAS = {
    "fire_only": 2500,
    "smoke_only": 2500,
    "fire_smoke": 5000,
    "negative": 2000,
}

TARGET = sum(
    QUOTAS.values()
)


def load_set(path):

    return {
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    }


def stable_rank(stem):

    value = hashlib.sha256(
        (
            "FASDD_R1_V1|"
            + stem
        ).encode(
            "utf-8"
        )
    ).hexdigest()

    return value


def score_row(row):

    category = row[
        "category"
    ]

    fire_total = int(
        row["fire_total"]
    )

    fire_small = int(
        row["fire_small"]
    )

    fire_medium = int(
        row["fire_medium"]
    )

    fire_large = int(
        row["fire_large"]
    )

    smoke_total = int(
        row["smoke_total"]
    )

    smoke_small = int(
        row["smoke_small"]
    )

    smoke_medium = int(
        row["smoke_medium"]
    )

    smoke_large = int(
        row["smoke_large"]
    )

    box_count = int(
        row["box_count"]
    )

    # ========================================================
    # FIRE ONLY
    #
    # Strong preference:
    # small fire > medium fire > large fire
    # ========================================================

    if category == "fire_only":

        return (
            150 * int(
                fire_small > 0
            )
            +
            70 * int(
                fire_medium > 0
            )
            +
            10 * int(
                fire_large > 0
            )
            +
            min(
                fire_total,
                10,
            )
        )

    # ========================================================
    # SMOKE ONLY
    #
    # Preserve rare small smoke aggressively.
    # Then medium smoke.
    # Large-smoke-only is lower priority.
    # ========================================================

    if category == "smoke_only":

        return (
            5000 * int(
                smoke_small > 0
            )
            +
            150 * int(
                smoke_medium > 0
            )
            +
            10 * int(
                smoke_large > 0
            )
            +
            min(
                smoke_total,
                10,
            )
        )

    # ========================================================
    # FIRE + SMOKE
    #
    # Best source for:
    # - small fire
    # - complex scenes
    # - medium smoke
    # ========================================================

    if category == "fire_smoke":

        return (
            5000 * int(
                smoke_small > 0
            )
            +
            200 * int(
                fire_small > 0
            )
            +
            140 * int(
                smoke_medium > 0
            )
            +
            70 * int(
                fire_medium > 0
            )
            +
            15 * int(
                fire_large > 0
            )
            +
            5 * int(
                smoke_large > 0
            )
            +
            min(
                box_count,
                15,
            )
        )

    # ========================================================
    # NEGATIVE
    #
    # No semantic score.
    # Stable hash is used for deterministic sampling.
    # ========================================================

    if category == "negative":
        return 0

    raise RuntimeError(
        f"Unknown category: {category}"
    )


def hash_key(row):

    p = row[
        "phash"
    ]

    d = row[
        "dhash"
    ]

    if not p or not d:
        return None

    return (
        p,
        d,
    )


def select_category(
    rows,
    quota,
):

    ranked = sorted(
        rows,
        key=lambda r: (
            -r["_score"],
            r["_stable_rank"],
            r["stem"],
        ),
    )

    selected = []
    deferred = []

    used_hashes = set()

    # ========================================================
    # Pass 1
    #
    # Prefer perceptual-hash diversity.
    # This is a SOFT diversity rule only.
    # ========================================================

    for row in ranked:

        if len(selected) >= quota:
            break

        key = hash_key(row)

        # Rare small-smoke images are never
        # rejected because of hash collision.
        rare_small_smoke = (
            int(
                row["smoke_small"]
            ) > 0
        )

        if (
            key is not None
            and key in used_hashes
            and not rare_small_smoke
        ):

            deferred.append(
                row
            )

            continue

        selected.append(
            row
        )

        if key is not None:

            used_hashes.add(
                key
            )

    # ========================================================
    # Pass 2
    #
    # Fill quota if soft diversity skipped too many.
    # ========================================================

    if len(selected) < quota:

        selected_stems = {
            r["stem"]
            for r in selected
        }

        remaining = [
            r
            for r in ranked
            if r["stem"]
            not in selected_stems
        ]

        for row in remaining:

            if len(selected) >= quota:
                break

            selected.append(
                row
            )

    if len(selected) != quota:

        raise RuntimeError(
            f"Could not satisfy quota "
            f"{quota}; selected "
            f"{len(selected)}"
        )

    return selected


def main():

    print("=" * 94)
    print("FASDD R1 TRAIN V1 SELECTOR")
    print("=" * 94)

    clean_train = load_set(
        CLEAN_TRAIN
    )

    clean_val = load_set(
        CLEAN_VAL
    )

    clean_test = load_set(
        CLEAN_TEST
    )

    if len(clean_train) != 45658:

        raise RuntimeError(
            f"Unexpected clean train size: "
            f"{len(clean_train)}"
        )

    rows = []

    with INVENTORY.open(
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            stem = row[
                "stem"
            ]

            if stem not in clean_train:

                raise RuntimeError(
                    f"Inventory stem not in "
                    f"clean train: {stem}"
                )

            if (
                stem in clean_val
                or stem in clean_test
            ):

                raise RuntimeError(
                    f"Split leakage before "
                    f"selection: {stem}"
                )

            if (
                not row["phash"]
                or not row["dhash"]
            ):

                raise RuntimeError(
                    f"Missing perceptual hash: "
                    f"{stem}"
                )

            row[
                "_score"
            ] = score_row(
                row
            )

            row[
                "_stable_rank"
            ] = stable_rank(
                stem
            )

            rows.append(
                row
            )

    if len(rows) != 45658:

        raise RuntimeError(
            f"Expected 45,658 inventory "
            f"rows; got {len(rows)}"
        )

    # ========================================================
    # CATEGORY POOLS
    # ========================================================

    pools = {
        category: []
        for category in QUOTAS
    }

    for row in rows:

        pools[
            row["category"]
        ].append(
            row
        )

    print()
    print("AVAILABLE")
    print("-" * 94)

    for category in QUOTAS:

        print(
            f"{category:16}: "
            f"{len(pools[category]):,}"
        )

    # ========================================================
    # SELECT
    # ========================================================

    selected = []

    for category, quota in (
        QUOTAS.items()
    ):

        if len(
            pools[category]
        ) < quota:

            raise RuntimeError(
                f"{category}: "
                f"quota {quota} exceeds "
                f"available "
                f"{len(pools[category])}"
            )

        chosen = select_category(
            pools[category],
            quota,
        )

        selected.extend(
            chosen
        )

    if len(selected) != TARGET:

        raise RuntimeError(
            f"Expected {TARGET}, "
            f"got {len(selected)}"
        )

    selected_stems = {
        r["stem"]
        for r in selected
    }

    if len(
        selected_stems
    ) != TARGET:

        raise RuntimeError(
            "Duplicate selected stems"
        )

    # ========================================================
    # HARD VALIDATION
    # ========================================================

    if not selected_stems <= clean_train:

        raise RuntimeError(
            "Selected data outside train"
        )

    if (
        selected_stems
        & clean_val
    ):

        raise RuntimeError(
            "Selected data overlaps VAL"
        )

    if (
        selected_stems
        & clean_test
    ):

        raise RuntimeError(
            "Selected data overlaps TEST"
        )

    # ========================================================
    # PROFILE
    # ========================================================

    category_counts = Counter()

    image_signals = Counter()

    object_counts = Counter()

    rare_small_smoke_available = 0
    rare_small_smoke_selected = 0

    for row in rows:

        if int(
            row["smoke_small"]
        ) > 0:

            rare_small_smoke_available += 1

    for row in selected:

        category_counts[
            row["category"]
        ] += 1

        for key in (
            "fire_small",
            "fire_medium",
            "fire_large",
            "smoke_small",
            "smoke_medium",
            "smoke_large",
        ):

            count = int(
                row[key]
            )

            object_counts[
                key
            ] += count

            if count > 0:

                image_signals[
                    key
                ] += 1

        if (
            row["category"]
            == "fire_smoke"
        ):

            image_signals[
                "fire_smoke"
            ] += 1

        if int(
            row["smoke_small"]
        ) > 0:

            rare_small_smoke_selected += 1

    # Every small-smoke image must survive.
    rare_retention_pass = (
        rare_small_smoke_selected
        ==
        rare_small_smoke_available
        ==
        333
    )

    if not rare_retention_pass:

        raise RuntimeError(
            "Small-smoke retention failed: "
            f"available="
            f"{rare_small_smoke_available}, "
            f"selected="
            f"{rare_small_smoke_selected}"
        )

    # ========================================================
    # WRITE MANIFEST
    # ========================================================

    ordered_selected = sorted(
        selected_stems
    )

    SELECTED_TXT.write_text(
        "".join(
            f"{stem}\n"
            for stem
            in ordered_selected
        ),
        encoding="utf-8",
    )

    rejected = sorted(
        clean_train
        - selected_stems
    )

    REJECTED_TXT.write_text(
        "".join(
            f"{stem}\n"
            for stem in rejected
        ),
        encoding="utf-8",
    )

    # ========================================================
    # WRITE CSV
    # ========================================================

    selected.sort(
        key=lambda r: (
            r["category"],
            -r["_score"],
            r["_stable_rank"],
        )
    )

    csv_fields = [
        "stem",
        "category",
        "selection_score",
        "box_count",
        "fire_total",
        "fire_small",
        "fire_medium",
        "fire_large",
        "smoke_total",
        "smoke_small",
        "smoke_medium",
        "smoke_large",
        "has_small_fire",
        "has_medium_fire",
        "has_large_fire",
        "has_small_smoke",
        "has_medium_smoke",
        "has_large_smoke",
        "has_fire_smoke",
        "phash",
        "dhash",
        "width",
        "height",
    ]

    with SELECTED_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in selected:

            out = {
                key:
                    row.get(
                        key,
                        ""
                    )
                for key
                in csv_fields
            }

            out[
                "selection_score"
            ] = row[
                "_score"
            ]

            writer.writerow(
                out
            )

    # ========================================================
    # HASH MANIFEST
    # ========================================================

    manifest_sha256 = (
        hashlib.sha256(
            SELECTED_TXT.read_bytes()
        ).hexdigest()
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "version":
            "FASDD_R1_TRAIN_V1",

        "target_images":
            TARGET,

        "quotas":
            QUOTAS,

        "selected_categories":
            dict(
                category_counts
            ),

        "selected_image_signals":
            dict(
                image_signals
            ),

        "selected_object_counts":
            dict(
                object_counts
            ),

        "small_smoke": {
            "available_images":
                rare_small_smoke_available,

            "selected_images":
                rare_small_smoke_selected,

            "retention_pass":
                rare_retention_pass,
        },

        "validation": {
            "selected_count":
                len(
                    selected_stems
                ),

            "unique_selected":
                len(
                    selected_stems
                ),

            "inside_clean_train":
                selected_stems
                <= clean_train,

            "val_overlap":
                len(
                    selected_stems
                    & clean_val
                ),

            "test_overlap":
                len(
                    selected_stems
                    & clean_test
                ),
        },

        "manifest": {
            "path":
                str(
                    SELECTED_TXT
                ),

            "sha256":
                manifest_sha256,
        },

        "selection_policy": {
            "val_used":
                False,

            "test_used":
                False,

            "rare_small_smoke":
                "retain all",

            "perceptual_hash":
                (
                    "soft diversity only; "
                    "never blocks quota fill"
                ),

            "tie_break":
                "deterministic SHA256",
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
    print("FASDD R1 TRAIN V1 SUMMARY")
    print("=" * 94)

    print(
        "Selected:",
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
            f"{category_counts[category]:,}"
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
    print("SMALL-SMOKE RETENTION")
    print("-" * 94)

    print(
        "Available:",
        rare_small_smoke_available
    )

    print(
        "Selected :",
        rare_small_smoke_selected
    )

    print(
        "Retention:",
        (
            "PASS"
            if rare_retention_pass
            else "FAIL"
        )
    )

    print()
    print("VALIDATION")
    print("-" * 94)

    print(
        "Unique selected :",
        len(
            selected_stems
        )
    )

    print(
        "VAL overlap     :",
        len(
            selected_stems
            & clean_val
        )
    )

    print(
        "TEST overlap    :",
        len(
            selected_stems
            & clean_test
        )
    )

    final_pass = (
        len(selected_stems)
        == TARGET
        and
        selected_stems
        <= clean_train
        and
        not (
            selected_stems
            & clean_val
        )
        and
        not (
            selected_stems
            & clean_test
        )
        and
        rare_retention_pass
    )

    print()
    print(
        "RESULT:",
        (
            "FASDD R1 TRAIN V1 PASS"
            if final_pass
            else
            "FASDD R1 TRAIN V1 FAIL"
        )
    )

    print()
    print(
        "Manifest:",
        SELECTED_TXT
    )

    print(
        "CSV     :",
        SELECTED_CSV
    )

    print(
        "JSON    :",
        SUMMARY_JSON
    )

    print("=" * 94)


if __name__ == "__main__":
    main()
