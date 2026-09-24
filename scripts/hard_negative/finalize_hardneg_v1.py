#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math


ROOT = Path(__file__).resolve().parents[2]

MINED = (
    ROOT
    / "reports/negative_v1/mining/"
    "fasdd_unseen_negative_mined.csv"
)

MINING_DIR = (
    ROOT
    / "reports/negative_v1/mining"
)

OUT = (
    ROOT
    / "reports/negative_v1/final_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

EXPECTED_TOTAL = 472
VAL_RATIO = 0.20
SEED = 42

THRESHOLDS = [
    0.10,
    0.25,
    0.50,
]


def stable_score(stem):
    text = (
        f"{SEED}:{stem}"
    ).encode("utf-8")

    return hashlib.sha256(
        text
    ).hexdigest()


if not MINED.exists():
    raise FileNotFoundError(MINED)


with MINED.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:
    all_rows = list(
        csv.DictReader(f)
    )


selected = [
    row
    for row in all_rows
    if row["severity"]
    in {
        "EXTREME",
        "HARD",
    }
]


if len(selected) != EXPECTED_TOTAL:
    raise RuntimeError(
        f"Expected {EXPECTED_TOTAL} "
        f"EXTREME+HARD images, "
        f"got {len(selected)}"
    )


# ============================================================
# HUMAN ADJUDICATION
#
# User manually reviewed all 472 images and confirmed:
# NO actual fire and NO actual smoke.
# ============================================================

for row in selected:

    image = Path(
        row["image"]
    )

    if not image.exists():
        raise FileNotFoundError(
            image
        )

    row[
        "human_decision"
    ] = "APPROVE_NEGATIVE"

    row[
        "human_review"
    ] = "MANUAL_VISUAL_REVIEW"

    row[
        "review_date"
    ] = "2026-08-15"

    row[
        "review_note"
    ] = (
        "No real fire or smoke visible."
    )


# ============================================================
# STRATIFIED GROUPS
# severity + consensus class
# ============================================================

groups = defaultdict(list)

for row in selected:

    key = (
        row["severity"],
        row["consensus_class"],
    )

    groups[key].append(
        row
    )


for key in groups:

    groups[key].sort(
        key=lambda r:
            stable_score(
                r["stem"]
            )
    )


target_val = round(
    EXPECTED_TOTAL
    * VAL_RATIO
)


# Initial floor allocation
quotas = {}
fractions = []

for key, rows in groups.items():

    exact = (
        len(rows)
        * VAL_RATIO
    )

    base = math.floor(
        exact
    )

    quotas[
        key
    ] = base

    fractions.append(
        (
            exact - base,
            key,
        )
    )


# Distribute remaining validation slots
remaining = (
    target_val
    - sum(
        quotas.values()
    )
)

fractions.sort(
    key=lambda x:
        (
            -x[0],
            str(x[1]),
        )
)

for _, key in (
    fractions[:remaining]
):
    quotas[
        key
    ] += 1


train_rows = []
val_rows = []


for key, rows in groups.items():

    n_val = quotas[
        key
    ]

    val_rows.extend(
        rows[:n_val]
    )

    train_rows.extend(
        rows[n_val:]
    )


if (
    len(train_rows)
    + len(val_rows)
    != EXPECTED_TOTAL
):
    raise RuntimeError(
        "Split count mismatch"
    )


if len(val_rows) != target_val:
    raise RuntimeError(
        f"Expected {target_val} "
        f"validation images, "
        f"got {len(val_rows)}"
    )


train_stems = {
    row["stem"]
    for row in train_rows
}

val_stems = {
    row["stem"]
    for row in val_rows
}


if (
    train_stems
    & val_stems
):
    raise RuntimeError(
        "Hard-negative train/val leakage"
    )


# ============================================================
# SORT FOR REPRODUCIBILITY
# ============================================================

train_rows.sort(
    key=lambda r:
        (
            r["severity"],
            r["consensus_class"],
            r["stem"],
        )
)

val_rows.sort(
    key=lambda r:
        (
            r["severity"],
            r["consensus_class"],
            r["stem"],
        )
)


# ============================================================
# SAVE CSV
# ============================================================

all_fields = list(
    selected[0].keys()
)


def write_csv(path, rows):

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=all_fields,
        )

        writer.writeheader()
        writer.writerows(rows)


write_csv(
    OUT / "hardneg_verified_all_472.csv",
    selected,
)

write_csv(
    OUT / "hardneg_train.csv",
    train_rows,
)

write_csv(
    OUT / "hardneg_val.csv",
    val_rows,
)


# ============================================================
# SAVE MANIFESTS
# ============================================================

def write_manifest(
    path,
    rows,
):

    path.write_text(
        "\n".join(
            row["image"]
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


write_manifest(
    OUT / "hardneg_train.txt",
    train_rows,
)

write_manifest(
    OUT / "hardneg_val.txt",
    val_rows,
)


# ============================================================
# BASELINE R1 HARD-NEGATIVE VALIDATION
#
# We already have N/S/M predictions from mining.
# No inference needed again.
# ============================================================

val_stem_set = set(
    val_stems
)


baseline_rows = []


for teacher in (
    "n",
    "s",
    "m",
):

    prediction_file = (
        MINING_DIR
        / (
            f"teacher_{teacher}_"
            "negative_predictions.csv"
        )
    )

    if not prediction_file.exists():
        raise FileNotFoundError(
            prediction_file
        )

    predictions = {}

    with prediction_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            if (
                row["stem"]
                not in val_stem_set
            ):
                continue

            predictions[
                row["stem"]
            ] = row


    if (
        len(predictions)
        != len(val_rows)
    ):
        raise RuntimeError(
            f"Teacher {teacher}: "
            "missing validation predictions"
        )


    for threshold in THRESHOLDS:

        any_alert = 0
        fire_alert = 0
        smoke_alert = 0

        for row in predictions.values():

            fire_conf = float(
                row["fire_conf"]
            )

            smoke_conf = float(
                row["smoke_conf"]
            )

            if (
                max(
                    fire_conf,
                    smoke_conf,
                )
                >= threshold
            ):
                any_alert += 1

            if (
                fire_conf
                >= threshold
            ):
                fire_alert += 1

            if (
                smoke_conf
                >= threshold
            ):
                smoke_alert += 1


        total = len(
            predictions
        )

        baseline_rows.append({
            "model":
                teacher,

            "threshold":
                threshold,

            "images":
                total,

            "false_alert_images":
                any_alert,

            "image_fpr":
                any_alert / total,

            "fire_false_alert_images":
                fire_alert,

            "fire_image_fpr":
                fire_alert / total,

            "smoke_false_alert_images":
                smoke_alert,

            "smoke_image_fpr":
                smoke_alert / total,
        })


baseline_csv = (
    OUT
    / "r1_hardneg_val_baseline.csv"
)


with baseline_csv.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            baseline_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        baseline_rows
    )


summary = {
    "verified_total":
        len(selected),

    "train":
        len(train_rows),

    "validation":
        len(val_rows),

    "validation_ratio":
        VAL_RATIO,

    "seed":
        SEED,

    "human_decision":
        "APPROVE_NEGATIVE",

    "real_fire_found":
        0,

    "real_smoke_found":
        0,
}


(
    OUT / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT
# ============================================================

print("=" * 92)
print(
    "HARD NEGATIVE V1 — FINALIZED"
)
print("=" * 92)

print(
    "Human verified:",
    len(selected)
)

print(
    "Train         :",
    len(train_rows)
)

print(
    "Validation    :",
    len(val_rows)
)

print(
    "Train ∩ Val   :",
    len(
        train_stems
        & val_stems
    )
)

print()
print(
    "R1 BASELINE ON HELD-OUT "
    "HARD NEGATIVES"
)
print("-" * 92)

for row in baseline_rows:

    print(
        f"{row['model'].upper()} "
        f"conf={row['threshold']:.2f} "
        f"FPR="
        f"{row['image_fpr']:.4f} "
        f"("
        f"{row['false_alert_images']}/"
        f"{row['images']}"
        f")"
    )

print()
print(
    "Baseline:",
    baseline_csv
)

print()
print(
    "RESULT: HARD NEGATIVE V1 "
    "FINAL PASS"
)

print("=" * 92)
