#!/usr/bin/env python3

from pathlib import Path
from itertools import product
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

HUMAN = (
    ROOT
    / "reports/human_reference"
    / "dfire_human_qa_200.csv"
)

CACHE_DIR = (
    ROOT
    / "reports/auto_curator_r0"
    / "predictions"
)

OUT = (
    ROOT
    / "reports/auto_curator_r0"
    / "calibration.json"
)


# Test-set performance ใกล้กัน
# จึงใช้ soft weighting เท่านั้น
WEIGHTS = {
    "teacher_n_r0": 0.31,
    "teacher_s_r0": 0.33,
    "teacher_m_r0": 0.36,
}


def load_cache(model):

    path = CACHE_DIR / f"{model}.jsonl"

    if not path.exists():
        raise FileNotFoundError(path)

    data = {}

    with path.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(line)

            key = str(
                Path(
                    row["image"]
                ).resolve()
            )

            data[key] = row["predictions"]

    return data


def load_gt_classes(label):

    if not label:
        return set()

    path = Path(label)

    if not path.exists():
        return set()

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    if not text:
        return set()

    classes = set()

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        try:
            classes.add(
                int(parts[0])
            )
        except ValueError:
            pass

    return classes


def max_confidence(predictions, cls):

    values = [
        float(p["confidence"])
        for p in predictions
        if int(p["class_id"]) == cls
    ]

    return max(
        values,
        default=0.0,
    )


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


def main():

    caches = {
        model: load_cache(model)
        for model in WEIGHTS
    }

    rows = list(
        csv.DictReader(
            HUMAN.open(
                encoding="utf-8"
            )
        )
    )

    samples = []

    for row in rows:

        status = row["status"].strip().lower()

        if status == "skip":
            continue

        if status not in {
            "approve",
            "relabel",
            "quarantine",
        }:
            continue

        image = str(
            Path(row["image"]).resolve()
        )

        gt = load_gt_classes(
            row.get("label", "")
        )

        confidences = {}

        for model in WEIGHTS:

            preds = caches[model].get(
                image,
                []
            )

            confidences[model] = {
                0: max_confidence(
                    preds,
                    0,
                ),
                1: max_confidence(
                    preds,
                    1,
                ),
            }

        samples.append({
            "image": image,
            "gt": gt,
            "human_attention":
                status
                in {"relabel", "quarantine"},
            "confidences": confidences,
        })

    print("=" * 72)
    print("AUTO-CURATOR CALIBRATION")
    print("=" * 72)

    print(
        "Human samples :",
        len(samples)
    )

    print(
        "Attention     :",
        sum(
            s["human_attention"]
            for s in samples
        )
    )

    print(
        "Approve       :",
        sum(
            not s["human_attention"]
            for s in samples
        )
    )

    # --------------------------------------------------------
    # Grid search
    # --------------------------------------------------------

    vote_conf_values = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
    ]

    missing_score_values = [
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
    ]

    unsupported_values = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
    ]

    best = None

    for (
        vote_conf,
        missing_score,
        unsupported_score,
    ) in product(
        vote_conf_values,
        missing_score_values,
        unsupported_values,
    ):

        tp = fp = tn = fn = 0

        for sample in samples:

            gt = sample["gt"]

            auto_attention = False

            for cls in (0, 1):

                weighted = 0.0
                votes = 0

                for model, weight in (
                    WEIGHTS.items()
                ):

                    conf = (
                        sample["confidences"]
                        [model]
                        [cls]
                    )

                    weighted += (
                        weight * conf
                    )

                    if conf >= vote_conf:
                        votes += 1

                # --------------------------------------------
                # Teachers เห็น object ที่ GT ไม่มี
                # --------------------------------------------

                if cls not in gt:

                    if (
                        votes >= 2
                        and
                        weighted
                        >= missing_score
                    ):
                        auto_attention = True

                # --------------------------------------------
                # GT บอกว่ามี แต่ Teachers แทบไม่เห็น
                # --------------------------------------------

                else:

                    if (
                        weighted
                        < unsupported_score
                    ):
                        auto_attention = True

            human = sample[
                "human_attention"
            ]

            if auto_attention and human:
                tp += 1

            elif auto_attention and not human:
                fp += 1

            elif (
                not auto_attention
                and human
            ):
                fn += 1

            else:
                tn += 1

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

        f2 = fbeta(
            precision,
            recall,
            beta=2.0,
        )

        accuracy = (
            (tp + tn)
            / len(samples)
            if samples
            else 0.0
        )

        candidate = {
            "vote_conf": vote_conf,
            "missing_score":
                missing_score,
            "unsupported_score":
                unsupported_score,

            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,

            "precision": precision,
            "recall": recall,
            "f2": f2,
            "accuracy": accuracy,
        }

        if best is None:

            best = candidate
            continue

        # F2 ให้ Recall สำคัญกว่า Precision
        key_new = (
            candidate["f2"],
            candidate["recall"],
            candidate["precision"],
        )

        key_old = (
            best["f2"],
            best["recall"],
            best["precision"],
        )

        if key_new > key_old:
            best = candidate

    result = {
        "weights": WEIGHTS,
        "human_samples":
            len(samples),
        "best_thresholds": {
            "vote_conf":
                best["vote_conf"],

            "missing_score":
                best["missing_score"],

            "unsupported_score":
                best[
                    "unsupported_score"
                ],
        },
        "metrics": {
            k: best[k]
            for k in [
                "tp",
                "fp",
                "tn",
                "fn",
                "precision",
                "recall",
                "f2",
                "accuracy",
            ]
        },
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BEST CALIBRATION")
    print("=" * 72)

    print(
        "vote_conf        :",
        best["vote_conf"]
    )

    print(
        "missing_score    :",
        best["missing_score"]
    )

    print(
        "unsupported_score:",
        best["unsupported_score"]
    )

    print()

    print(
        "TP / FP :",
        best["tp"],
        "/",
        best["fp"],
    )

    print(
        "TN / FN :",
        best["tn"],
        "/",
        best["fn"],
    )

    print()

    print(
        "Precision :",
        f"{best['precision']:.4f}"
    )

    print(
        "Recall    :",
        f"{best['recall']:.4f}"
    )

    print(
        "F2        :",
        f"{best['f2']:.4f}"
    )

    print(
        "Accuracy  :",
        f"{best['accuracy']:.4f}"
    )

    print()
    print("Report:", OUT)
    print("=" * 72)

    # Human calibration set ไม่ใช่ independent test
    print(
        "NOTE: This is calibration performance, "
        "not final model accuracy."
    )


if __name__ == "__main__":
    main()
