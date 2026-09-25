#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
from itertools import product
import csv
import json


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SPATIAL_CSV = (
    ROOT
    / "reports"
    / "auto_curator_r0"
    / "spatial_scores.csv"
)

HUMAN_CSV = (
    ROOT
    / "reports"
    / "human_reference"
    / "dfire_human_qa_200.csv"
)

OUT_DIR = (
    ROOT
    / "reports"
    / "auto_curator_r0"
)

REPORT_JSON = (
    OUT_DIR
    / "grouped_calibration.json"
)

DECISIONS_CSV = (
    OUT_DIR
    / "spatial_decisions_grouped.csv"
)


# เป้าหมายหลัก
TARGET_RECALL = 0.85

# ไม่ยอมให้การไล่ Recall
# ทำให้ Precision พังจนเกินไป
MIN_PRECISION = 0.35


# Search 0.00 -> 10.00
THRESHOLDS = [
    i * 0.25
    for i in range(41)
]

# ตัวเลือก "แทบไม่ flag group นี้เลย"
THRESHOLDS.append(999.0)


# ============================================================
# HELPERS
# ============================================================

def normalize(path):
    return str(Path(path).resolve())


def category_group(category):

    if category == "fire_only":
        return "fire_only"

    if category == "smoke_only":
        return "smoke_only"

    # fire_smoke + negative
    return "stable"


def fbeta(
    precision,
    recall,
    beta=2.0,
):

    if precision == 0 and recall == 0:
        return 0.0

    b2 = beta * beta

    return (
        (1 + b2)
        * precision
        * recall
        /
        (
            b2 * precision
            + recall
        )
    )


def metrics_from_counts(
    tp,
    fp,
    tn,
    fn,
):

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / (tp + fp + tn + fn)
        if tp + fp + tn + fn
        else 0.0
    )

    f2 = fbeta(
        precision,
        recall,
        beta=2.0,
    )

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f2": f2,
    }


# ============================================================
# LOAD
# ============================================================

def load_spatial():

    rows = list(
        csv.DictReader(
            SPATIAL_CSV.open(
                encoding="utf-8"
            )
        )
    )

    data = {}

    for row in rows:

        row["image"] = normalize(
            row["image"]
        )

        row["risk_score"] = float(
            row["risk_score"]
        )

        # numeric diagnostics
        for key in [
            "missing_fire",
            "missing_smoke",
            "unsupported_gt",
            "class_conflicts",
            "strong_singletons",
            "gt_fire",
            "gt_smoke",
            "consensus_fire",
            "consensus_smoke",
        ]:

            row[key] = int(
                float(row[key])
            )

        data[row["image"]] = row

    return data


def load_human():

    rows = list(
        csv.DictReader(
            HUMAN_CSV.open(
                encoding="utf-8"
            )
        )
    )

    data = {}

    for row in rows:

        status = (
            row["status"]
            .strip()
            .lower()
        )

        if status not in {
            "approve",
            "relabel",
            "quarantine",
        }:
            continue

        image = normalize(
            row["image"]
        )

        data[image] = {
            "status": status,
            "attention":
                status in {
                    "relabel",
                    "quarantine",
                },
            "category":
                row["category"],
            "note":
                row.get(
                    "note",
                    ""
                ).strip(),
        }

    return data


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    samples,
    thresholds,
):

    tp = fp = tn = fn = 0

    by_category = defaultdict(
        lambda: Counter()
    )

    for sample in samples:

        category = sample["category"]

        group = category_group(
            category
        )

        threshold = thresholds[
            group
        ]

        predicted_attention = (
            sample["risk_score"]
            >= threshold
        )

        actual_attention = (
            sample["attention"]
        )

        if (
            predicted_attention
            and actual_attention
        ):
            tp += 1
            by_category[category]["tp"] += 1

        elif (
            predicted_attention
            and not actual_attention
        ):
            fp += 1
            by_category[category]["fp"] += 1

        elif (
            not predicted_attention
            and actual_attention
        ):
            fn += 1
            by_category[category]["fn"] += 1

        else:
            tn += 1
            by_category[category]["tn"] += 1

    result = metrics_from_counts(
        tp,
        fp,
        tn,
        fn,
    )

    result.update({
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    })

    return result, by_category


# ============================================================
# GRID SEARCH
# ============================================================

def search_best(samples):

    best_target = None
    best_any = None

    combinations = (
        len(THRESHOLDS) ** 3
    )

    print(
        "Threshold combinations:",
        combinations
    )

    checked = 0

    for (
        fire_t,
        smoke_t,
        stable_t,
    ) in product(
        THRESHOLDS,
        THRESHOLDS,
        THRESHOLDS,
    ):

        thresholds = {
            "fire_only": fire_t,
            "smoke_only": smoke_t,
            "stable": stable_t,
        }

        metrics, _ = evaluate(
            samples,
            thresholds,
        )

        checked += 1

        candidate = {
            "thresholds":
                thresholds.copy(),
            "metrics":
                metrics.copy(),
        }

        # --------------------------------------------
        # Best unrestricted
        # --------------------------------------------

        if best_any is None:

            best_any = candidate

        else:

            new_key = (
                metrics["f2"],
                metrics["recall"],
                metrics["precision"],
                metrics["accuracy"],
            )

            old = (
                best_any["metrics"]
            )

            old_key = (
                old["f2"],
                old["recall"],
                old["precision"],
                old["accuracy"],
            )

            if new_key > old_key:
                best_any = candidate

        # --------------------------------------------
        # Target mode
        # --------------------------------------------

        if (
            metrics["recall"]
            < TARGET_RECALL
        ):
            continue

        if (
            metrics["precision"]
            < MIN_PRECISION
        ):
            continue

        if best_target is None:

            best_target = candidate

        else:

            # เมื่อ Recall ผ่านเป้าแล้ว
            # ใช้ F2 เป็นหลัก
            # แล้วเลือก Precision สูงกว่า
            new_key = (
                metrics["f2"],
                metrics["precision"],
                metrics["accuracy"],
                metrics["recall"],
            )

            old = (
                best_target["metrics"]
            )

            old_key = (
                old["f2"],
                old["precision"],
                old["accuracy"],
                old["recall"],
            )

            if new_key > old_key:
                best_target = candidate

    print(
        "Grid search checked:",
        checked
    )

    if best_target is not None:
        return (
            best_target,
            True,
            best_any,
        )

    return (
        best_any,
        False,
        best_any,
    )


# ============================================================
# REASONS
# ============================================================

def build_reason(row):

    reasons = []

    if row["missing_fire"] > 0:
        reasons.append(
            "missing_fire"
        )

    if row["missing_smoke"] > 0:
        reasons.append(
            "missing_smoke"
        )

    if row["class_conflicts"] > 0:
        reasons.append(
            "class_conflict"
        )

    if row["unsupported_gt"] > 0:
        reasons.append(
            "unsupported_gt"
        )

    if row["strong_singletons"] > 0:
        reasons.append(
            "strong_singleton"
        )

    if not reasons:
        reasons.append(
            "risk_prior"
        )

    return "|".join(reasons)


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("GROUPED SPATIAL CALIBRATION R0")
    print("=" * 78)

    spatial = load_spatial()
    human = load_human()

    samples = []

    for image, human_row in (
        human.items()
    ):

        spatial_row = spatial.get(
            image
        )

        if spatial_row is None:
            continue

        samples.append({
            "image": image,
            "category":
                human_row["category"],
            "attention":
                human_row["attention"],
            "status":
                human_row["status"],
            "risk_score":
                spatial_row["risk_score"],
        })

    print(
        "Human calibration samples:",
        len(samples)
    )

    print()

    print("Human category distribution")
    print("-" * 78)

    category_stats = defaultdict(
        Counter
    )

    for row in samples:

        cat = row["category"]

        category_stats[
            cat
        ]["total"] += 1

        if row["attention"]:
            category_stats[
                cat
            ]["attention"] += 1
        else:
            category_stats[
                cat
            ]["approve"] += 1

    for category in [
        "fire_only",
        "smoke_only",
        "fire_smoke",
        "negative",
    ]:

        s = category_stats[
            category
        ]

        print(
            f"{category:14} "
            f"total={s['total']:3} "
            f"attention={s['attention']:3} "
            f"approve={s['approve']:3}"
        )

    print()
    print("=" * 78)
    print("GRID SEARCH")
    print("=" * 78)

    best, target_met, best_any = (
        search_best(samples)
    )

    thresholds = best[
        "thresholds"
    ]

    metrics, by_category = (
        evaluate(
            samples,
            thresholds,
        )
    )

    print()
    print("=" * 78)
    print("BEST GROUPED CALIBRATION")
    print("=" * 78)

    print(
        "Target recall met :",
        target_met
    )

    print()

    print(
        "fire_only threshold :",
        thresholds["fire_only"]
    )

    print(
        "smoke_only threshold:",
        thresholds["smoke_only"]
    )

    print(
        "stable threshold    :",
        thresholds["stable"]
    )

    print()

    print(
        "TP / FP :",
        metrics["tp"],
        "/",
        metrics["fp"]
    )

    print(
        "TN / FN :",
        metrics["tn"],
        "/",
        metrics["fn"]
    )

    print()

    print(
        "Precision:",
        f"{metrics['precision']:.4f}"
    )

    print(
        "Recall   :",
        f"{metrics['recall']:.4f}"
    )

    print(
        "F2       :",
        f"{metrics['f2']:.4f}"
    )

    print(
        "Accuracy :",
        f"{metrics['accuracy']:.4f}"
    )

    print()
    print("Per-category confusion")
    print("-" * 78)

    for category in [
        "fire_only",
        "smoke_only",
        "fire_smoke",
        "negative",
    ]:

        s = by_category[
            category
        ]

        print(
            f"{category:14} "
            f"TP={s['tp']:2} "
            f"FP={s['fp']:2} "
            f"TN={s['tn']:2} "
            f"FN={s['fn']:2}"
        )

    # ========================================================
    # Apply to all 2000
    # ========================================================

    decision_rows = []

    counts = Counter()
    counts_by_category = defaultdict(
        Counter
    )

    for image, row in spatial.items():

        category = row[
            "category"
        ]

        group = category_group(
            category
        )

        threshold = thresholds[
            group
        ]

        flagged = (
            row["risk_score"]
            >= threshold
        )

        if flagged:

            action = "R0_HOLD"

        else:

            action = (
                "R0_CLEAN_CANDIDATE"
            )

        counts[action] += 1

        counts_by_category[
            category
        ][action] += 1

        decision_rows.append({
            "image":
                image,

            "category":
                category,

            "source_split":
                row[
                    "source_split"
                ],

            "risk_score":
                row[
                    "risk_score"
                ],

            "threshold":
                threshold,

            "action":
                action,

            "reason":
                build_reason(row),

            "missing_fire":
                row["missing_fire"],

            "missing_smoke":
                row["missing_smoke"],

            "unsupported_gt":
                row["unsupported_gt"],

            "class_conflicts":
                row["class_conflicts"],

            "strong_singletons":
                row["strong_singletons"],
        })

    fields = list(
        decision_rows[0].keys()
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
            decision_rows
        )

    print()
    print("=" * 78)
    print("D-FIRE 2000 GROUPED DECISIONS")
    print("=" * 78)

    print(
        "R0 CLEAN CANDIDATE:",
        counts[
            "R0_CLEAN_CANDIDATE"
        ]
    )

    print(
        "R0 HOLD           :",
        counts[
            "R0_HOLD"
        ]
    )

    print()

    for category in [
        "fire_only",
        "smoke_only",
        "fire_smoke",
        "negative",
    ]:

        s = counts_by_category[
            category
        ]

        print(
            f"{category:14} "
            f"clean="
            f"{s['R0_CLEAN_CANDIDATE']:4} "
            f"hold="
            f"{s['R0_HOLD']:4}"
        )

    # ========================================================
    # False negatives diagnostics
    # ========================================================

    fn_rows = []

    for sample in samples:

        group = category_group(
            sample["category"]
        )

        predicted = (
            sample["risk_score"]
            >= thresholds[group]
        )

        if (
            sample["attention"]
            and not predicted
        ):

            fn_rows.append(
                sample
            )

    print()
    print("=" * 78)
    print("MISSED HUMAN ATTENTION CASES")
    print("=" * 78)

    missed_by_category = Counter(
        row["category"]
        for row in fn_rows
    )

    print(
        "Total FN:",
        len(fn_rows)
    )

    for cat, n in (
        missed_by_category.items()
    ):
        print(
            f"{cat:14}: {n}"
        )

    # ========================================================
    # Save report
    # ========================================================

    report = {
        "target_recall":
            TARGET_RECALL,

        "min_precision":
            MIN_PRECISION,

        "target_met":
            target_met,

        "thresholds":
            thresholds,

        "metrics":
            metrics,

        "human_category_stats": {
            cat: dict(value)
            for cat, value
            in category_stats.items()
        },

        "all_2000": {
            "clean_candidate":
                counts[
                    "R0_CLEAN_CANDIDATE"
                ],

            "hold":
                counts[
                    "R0_HOLD"
                ],
        },

        "false_negative_categories":
            dict(
                missed_by_category
            ),
    }

    REPORT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Decisions:",
        DECISIONS_CSV
    )

    print(
        "Report   :",
        REPORT_JSON
    )

    print("=" * 78)

    print(
        "NOTE: Human 200 is calibration data, "
        "not an independent final benchmark."
    )


if __name__ == "__main__":
    main()
