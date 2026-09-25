#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import json


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT
    / "reports/auto_curator_r0"
    / "dfire_2000_manifest.csv"
)

CACHE_DIR = (
    ROOT
    / "reports/auto_curator_r0"
    / "predictions"
)

HUMAN = (
    ROOT
    / "reports/human_reference"
    / "dfire_human_qa_200.csv"
)

OUT_DIR = (
    ROOT
    / "reports/auto_curator_r0"
)

SCORES_CSV = (
    OUT_DIR
    / "spatial_scores.csv"
)

SUMMARY_JSON = (
    OUT_DIR
    / "spatial_calibration.json"
)


WEIGHTS = {
    "teacher_n_r0": 0.31,
    "teacher_s_r0": 0.33,
    "teacher_m_r0": 0.36,
}


# Prediction ต่ำกว่านี้ไม่เอามาสร้าง spatial cluster
PRED_MIN_CONF = 0.10

# bbox จากคนละ Teacher ถือว่าเป็น object เดียวกัน
CLUSTER_IOU = 0.25

# Ground truth ถือว่ามี Teacher สนับสนุน
GT_SUPPORT_IOU = 0.20

# Consensus ถือว่าไม่มี GT รองรับ
MISSING_GT_IOU = 0.15

# ครอบ GT คนละ class มากพอให้สงสัย class conflict
CONFLICT_IOU = 0.30

# Consensus confidence ขั้นต่ำ
MISSING_MIN_SCORE = 0.35

# Detection จาก Teacher ตัวเดียวที่น่าสนใจ
SINGLETON_STRONG_CONF = 0.55


# ============================================================
# GEOMETRY
# ============================================================

def iou(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    inter = w * h

    area_a = max(
        0.0,
        a[2] - a[0]
    ) * max(
        0.0,
        a[3] - a[1]
    )

    area_b = max(
        0.0,
        b[2] - b[0]
    ) * max(
        0.0,
        b[3] - b[1]
    )

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def weighted_box(members):

    total = 0.0
    result = [0.0, 0.0, 0.0, 0.0]

    for m in members:

        factor = (
            WEIGHTS[m["model"]]
            * m["confidence"]
        )

        total += factor

        for i in range(4):
            result[i] += (
                m["box"][i]
                * factor
            )

    if total <= 0:
        return members[0]["box"]

    return [
        value / total
        for value in result
    ]


# ============================================================
# LOAD DATA
# ============================================================

def load_cache(model):

    path = (
        CACHE_DIR
        / f"{model}.jsonl"
    )

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

            data[key] = (
                row["predictions"]
            )

    return data


def load_gt(label_path):

    if not label_path:
        return []

    path = Path(label_path)

    if not path.exists():
        return []

    text = path.read_text(
        encoding="utf-8",
        errors="replace"
    ).strip()

    if not text:
        return []

    boxes = []

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        try:

            cls = int(parts[0])

            xc, yc, w, h = map(
                float,
                parts[1:]
            )

        except ValueError:
            continue

        boxes.append({
            "class_id": cls,
            "box": [
                xc - w / 2,
                yc - h / 2,
                xc + w / 2,
                yc + h / 2,
            ],
        })

    return boxes


# ============================================================
# ENSEMBLE CLUSTERING
# ============================================================

def build_clusters(
    image,
    caches,
):

    detections = []

    for model in WEIGHTS:

        preds = caches[
            model
        ].get(
            image,
            []
        )

        for pred in preds:

            conf = float(
                pred["confidence"]
            )

            if conf < PRED_MIN_CONF:
                continue

            detections.append({
                "model": model,
                "class_id":
                    int(pred["class_id"]),
                "confidence": conf,
                "box": [
                    float(x)
                    for x
                    in pred["xyxy"]
                ],
            })

    detections.sort(
        key=lambda x: x["confidence"],
        reverse=True,
    )

    clusters = []

    for det in detections:

        best_cluster = None
        best_iou = 0.0

        for cluster in clusters:

            if (
                cluster["class_id"]
                != det["class_id"]
            ):
                continue

            existing_models = {
                x["model"]
                for x
                in cluster["members"]
            }

            # 1 Teacher ให้ได้สูงสุด 1 vote ต่อ cluster
            if det["model"] in existing_models:
                continue

            current_box = weighted_box(
                cluster["members"]
            )

            score = iou(
                det["box"],
                current_box,
            )

            if (
                score >= CLUSTER_IOU
                and score > best_iou
            ):
                best_cluster = cluster
                best_iou = score

        if best_cluster is None:

            clusters.append({
                "class_id":
                    det["class_id"],

                "members": [
                    det
                ],
            })

        else:

            best_cluster[
                "members"
            ].append(det)

    final = []

    for cluster in clusters:

        members = (
            cluster["members"]
        )

        votes = len({
            m["model"]
            for m in members
        })

        score = sum(
            WEIGHTS[m["model"]]
            * m["confidence"]
            for m in members
        )

        final.append({
            "class_id":
                cluster["class_id"],

            "votes": votes,

            "score": score,

            "box":
                weighted_box(members),

            "max_conf":
                max(
                    m["confidence"]
                    for m in members
                ),
        })

    return final


# ============================================================
# SPATIAL FEATURES
# ============================================================

def extract_features(
    row,
    caches,
):

    image = str(
        Path(
            row["image"]
        ).resolve()
    )

    gt = load_gt(
        row["label"]
    )

    clusters = build_clusters(
        image,
        caches,
    )

    consensus = [
        c
        for c in clusters
        if c["votes"] >= 2
    ]

    singletons = [
        c
        for c in clusters
        if (
            c["votes"] == 1
            and c["max_conf"]
            >= SINGLETON_STRONG_CONF
        )
    ]

    # --------------------------------------------------------
    # GT support
    # --------------------------------------------------------

    unsupported_gt = 0
    support_ious = []

    conflicts = 0

    for gt_box in gt:

        cls = gt_box["class_id"]
        box = gt_box["box"]

        same = [
            c
            for c in consensus
            if c["class_id"] == cls
        ]

        other = [
            c
            for c in consensus
            if c["class_id"] != cls
        ]

        same_iou = max(
            (
                iou(
                    box,
                    c["box"]
                )
                for c in same
            ),
            default=0.0,
        )

        support_ious.append(
            same_iou
        )

        if (
            same_iou
            < GT_SUPPORT_IOU
        ):
            unsupported_gt += 1

        other_iou = max(
            (
                iou(
                    box,
                    c["box"]
                )
                for c in other
                if c["score"]
                >= MISSING_MIN_SCORE
            ),
            default=0.0,
        )

        if (
            other_iou
            >= CONFLICT_IOU
            and
            same_iou
            < GT_SUPPORT_IOU
        ):
            conflicts += 1

    # --------------------------------------------------------
    # Consensus objects missing from GT
    # --------------------------------------------------------

    missing_fire = 0
    missing_smoke = 0

    for cluster in consensus:

        if (
            cluster["score"]
            < MISSING_MIN_SCORE
        ):
            continue

        cls = cluster["class_id"]

        same_gt = [
            g
            for g in gt
            if g["class_id"] == cls
        ]

        best_gt_iou = max(
            (
                iou(
                    cluster["box"],
                    g["box"]
                )
                for g in same_gt
            ),
            default=0.0,
        )

        if (
            best_gt_iou
            < MISSING_GT_IOU
        ):

            if cls == 0:
                missing_fire += 1

            elif cls == 1:
                missing_smoke += 1

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    gt_fire = sum(
        g["class_id"] == 0
        for g in gt
    )

    gt_smoke = sum(
        g["class_id"] == 1
        for g in gt
    )

    consensus_fire = sum(
        c["class_id"] == 0
        for c in consensus
    )

    consensus_smoke = sum(
        c["class_id"] == 1
        for c in consensus
    )

    min_support = (
        min(support_ious)
        if support_ious
        else 1.0
    )

    mean_support = (
        sum(support_ious)
        / len(support_ious)
        if support_ious
        else 1.0
    )

    # --------------------------------------------------------
    # Risk Score
    # --------------------------------------------------------

    risk = 0.0

    # Missing annotation is high-risk
    risk += (
        missing_fire
        + missing_smoke
    ) * 2.5

    # GT that 3 teachers cannot spatially support
    risk += (
        unsupported_gt
        * 1.5
    )

    # Same region but opposite class
    risk += (
        conflicts
        * 2.5
    )

    # One strong teacher disagrees
    risk += min(
        len(singletons) * 0.50,
        2.0,
    )

    # Single-class source is known to be less reliable
    if (
        row["category"]
        in {
            "fire_only",
            "smoke_only",
        }
        and (
            missing_fire
            + missing_smoke
        ) > 0
    ):
        risk += 0.75

    return {
        "image": image,
        "category": row["category"],
        "source_split":
            row["source_split"],

        "gt_boxes":
            len(gt),

        "gt_fire":
            gt_fire,

        "gt_smoke":
            gt_smoke,

        "consensus_fire":
            consensus_fire,

        "consensus_smoke":
            consensus_smoke,

        "missing_fire":
            missing_fire,

        "missing_smoke":
            missing_smoke,

        "unsupported_gt":
            unsupported_gt,

        "class_conflicts":
            conflicts,

        "strong_singletons":
            len(singletons),

        "min_gt_support_iou":
            round(
                min_support,
                6
            ),

        "mean_gt_support_iou":
            round(
                mean_support,
                6
            ),

        "risk_score":
            round(
                risk,
                4
            ),
    }


# ============================================================
# CALIBRATION
# ============================================================

def fbeta(
    precision,
    recall,
    beta=2.0,
):

    if (
        precision == 0
        and recall == 0
    ):
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


def calibrate(
    features,
):

    human_rows = list(
        csv.DictReader(
            HUMAN.open(
                encoding="utf-8"
            )
        )
    )

    human = {}

    for row in human_rows:

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

        image = str(
            Path(
                row["image"]
            ).resolve()
        )

        human[image] = (
            status
            in {
                "relabel",
                "quarantine",
            }
        )

    samples = [
        f
        for f in features
        if f["image"] in human
    ]

    best = None

    # 0.25 -> 10.00
    thresholds = [
        i / 4
        for i in range(
            1,
            41
        )
    ]

    for threshold in thresholds:

        tp = fp = tn = fn = 0

        for row in samples:

            predicted = (
                row["risk_score"]
                >= threshold
            )

            actual = human[
                row["image"]
            ]

            if predicted and actual:
                tp += 1

            elif predicted and not actual:
                fp += 1

            elif (
                not predicted
                and actual
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

        result = {
            "threshold":
                threshold,

            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,

            "precision":
                precision,

            "recall":
                recall,

            "f2":
                f2,

            "accuracy":
                accuracy,
        }

        if best is None:

            best = result
            continue

        new_key = (
            result["f2"],
            result["recall"],
            result["precision"],
        )

        old_key = (
            best["f2"],
            best["recall"],
            best["precision"],
        )

        if new_key > old_key:
            best = result

    return best, samples


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("SPATIAL AUTO-CURATOR R0")
    print("=" * 76)

    caches = {}

    for model in WEIGHTS:

        print(
            f"Loading {model}..."
        )

        caches[model] = (
            load_cache(model)
        )

    manifest = list(
        csv.DictReader(
            MANIFEST.open(
                encoding="utf-8"
            )
        )
    )

    print(
        "Images:",
        len(manifest)
    )

    features = []

    for i, row in enumerate(
        manifest,
        start=1,
    ):

        features.append(
            extract_features(
                row,
                caches,
            )
        )

        if (
            i % 200 == 0
            or i == len(manifest)
        ):
            print(
                f"\rSpatial analysis: "
                f"{i}/{len(manifest)}",
                end="",
                flush=True,
            )

    print()

    # --------------------------------------------------------
    # Save feature table
    # --------------------------------------------------------

    fields = list(
        features[0].keys()
    )

    with SCORES_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(features)

    # --------------------------------------------------------
    # Calibrate against Human 200
    # --------------------------------------------------------

    best, samples = calibrate(
        features
    )

    threshold = (
        best["threshold"]
    )

    flagged = [
        row
        for row in features
        if (
            row["risk_score"]
            >= threshold
        )
    ]

    clean = (
        len(features)
        - len(flagged)
    )

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    stats = Counter()

    for row in features:

        stats[
            "missing_fire"
        ] += row[
            "missing_fire"
        ]

        stats[
            "missing_smoke"
        ] += row[
            "missing_smoke"
        ]

        stats[
            "unsupported_gt"
        ] += row[
            "unsupported_gt"
        ]

        stats[
            "class_conflicts"
        ] += row[
            "class_conflicts"
        ]

        stats[
            "strong_singletons"
        ] += row[
            "strong_singletons"
        ]

    summary = {
        "config": {
            "weights":
                WEIGHTS,

            "pred_min_conf":
                PRED_MIN_CONF,

            "cluster_iou":
                CLUSTER_IOU,

            "gt_support_iou":
                GT_SUPPORT_IOU,

            "missing_gt_iou":
                MISSING_GT_IOU,

            "conflict_iou":
                CONFLICT_IOU,

            "missing_min_score":
                MISSING_MIN_SCORE,
        },

        "human_samples":
            len(samples),

        "best_calibration":
            best,

        "all_2000": {
            "flagged":
                len(flagged),

            "clean":
                clean,

            "flagged_percent":
                (
                    len(flagged)
                    / len(features)
                    * 100
                ),
        },

        "feature_totals":
            dict(stats),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("SPATIAL FEATURES")
    print("=" * 76)

    print(
        "Likely missing fire :",
        stats["missing_fire"]
    )

    print(
        "Likely missing smoke:",
        stats["missing_smoke"]
    )

    print(
        "Unsupported GT      :",
        stats["unsupported_gt"]
    )

    print(
        "Class conflicts     :",
        stats["class_conflicts"]
    )

    print(
        "Strong singletons   :",
        stats["strong_singletons"]
    )

    print()
    print("=" * 76)
    print("HUMAN CALIBRATION")
    print("=" * 76)

    print(
        "Best threshold :",
        best["threshold"]
    )

    print(
        "TP / FP        :",
        best["tp"],
        "/",
        best["fp"]
    )

    print(
        "TN / FN        :",
        best["tn"],
        "/",
        best["fn"]
    )

    print()

    print(
        "Precision      :",
        f"{best['precision']:.4f}"
    )

    print(
        "Recall         :",
        f"{best['recall']:.4f}"
    )

    print(
        "F2             :",
        f"{best['f2']:.4f}"
    )

    print(
        "Accuracy       :",
        f"{best['accuracy']:.4f}"
    )

    print()
    print("=" * 76)
    print("D-FIRE 2000")
    print("=" * 76)

    print(
        "AUTO FLAGGED   :",
        len(flagged)
    )

    print(
        "AUTO CLEAN     :",
        clean
    )

    print(
        "Flagged %      :",
        f"{len(flagged)/len(features)*100:.1f}%"
    )

    print()

    print(
        "Scores :",
        SCORES_CSV
    )

    print(
        "Report :",
        SUMMARY_JSON
    )

    print("=" * 76)


if __name__ == "__main__":
    main()
