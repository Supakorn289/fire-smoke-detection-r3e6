#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import json

import cv2
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)

IMAGES_DIR = DATA / "images"

LABELS_DIR = (
    DATA
    / "annotations"
    / "YOLO_CV"
    / "labels"
)

CONFLICTS_CSV = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_exact"
    / "conflicts.csv"
)

OUT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_exact"
    / "r0_adjudication"
)

OUT_JSON = OUT_DIR / "adjudication.json"
OUT_CSV = OUT_DIR / "adjudication.csv"


# ============================================================
# TEACHERS
# ============================================================

TEACHERS = {
    "N": (
        ROOT
        / "runs"
        / "teachers_r0"
        / "teacher_n_r0"
        / "weights"
        / "best.pt"
    ),

    "S": (
        ROOT
        / "runs"
        / "teachers_r0"
        / "teacher_s_r0"
        / "weights"
        / "best.pt"
    ),

    "M": (
        ROOT
        / "runs"
        / "teachers_r0"
        / "teacher_m_r0"
        / "weights"
        / "best.pt"
    ),
}

# R0 test-based soft weights
TEACHER_WEIGHTS = {
    "N": 0.31,
    "S": 0.33,
    "M": 0.36,
}


# ============================================================
# CONFIG
# ============================================================

IMGSZ = 768
BATCH = 16

PRED_CONF = 0.05
PRED_IOU = 0.60

CLUSTER_IOU = 0.30

CONSENSUS_MIN_TEACHERS = 2
CONSENSUS_MIN_MEAN_CONF = 0.10

ANNOTATION_MATCH_IOU = 0.20

STRONG_MIN_SCORE = 0.60
STRONG_MIN_MARGIN = 0.15

BOTH_REASONABLE_MIN = 0.60
BOTH_REASONABLE_MAX_MARGIN = 0.10


# ============================================================
# GEOMETRY
# ============================================================

def iou_xyxy(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    inter = iw * ih

    area_a = max(
        0.0,
        (a[2] - a[0])
        * (a[3] - a[1])
    )

    area_b = max(
        0.0,
        (b[2] - b[0])
        * (b[3] - b[1])
    )

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def build_image_map():

    result = {}

    for p in IMAGES_DIR.iterdir():

        if (
            p.is_file()
            and p.suffix.lower()
            in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp",
            }
        ):

            result[p.stem] = p

    return result


# ============================================================
# YOLO LABEL → PIXEL XYXY
# ============================================================

def read_annotation(
    stem,
    width,
    height,
):

    path = LABELS_DIR / f"{stem}.txt"

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    boxes = []

    if not text:
        return boxes

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            continue

        cls = int(parts[0])

        xc, yc, bw, bh = map(
            float,
            parts[1:],
        )

        x1 = (
            xc - bw / 2
        ) * width

        y1 = (
            yc - bh / 2
        ) * height

        x2 = (
            xc + bw / 2
        ) * width

        y2 = (
            yc + bh / 2
        ) * height

        boxes.append({
            "cls": cls,
            "box": [
                x1,
                y1,
                x2,
                y2,
            ],
        })

    return boxes


# ============================================================
# CONSENSUS CLUSTERING
# ============================================================

def cluster_box(cluster):

    members = cluster["members"]

    total_weight = sum(
        m["merge_weight"]
        for m in members
    )

    if total_weight <= 0:

        total_weight = len(members)

        weights = [
            1.0
            for _ in members
        ]

    else:

        weights = [
            m["merge_weight"]
            for m in members
        ]

    coords = []

    for idx in range(4):

        value = sum(
            m["box"][idx] * w
            for m, w
            in zip(
                members,
                weights,
            )
        ) / sum(weights)

        coords.append(value)

    return coords


def build_consensus(
    teacher_predictions,
):

    detections = []

    for teacher, preds in (
        teacher_predictions.items()
    ):

        for p in preds:

            detections.append({
                **p,

                "teacher":
                    teacher,

                "merge_weight":
                    (
                        p["conf"]
                        * TEACHER_WEIGHTS[
                            teacher
                        ]
                    ),
            })

    detections.sort(
        key=lambda x:
            x["merge_weight"],
        reverse=True,
    )

    clusters = []

    for det in detections:

        best_cluster = None
        best_iou = 0.0

        for cluster in clusters:

            if (
                cluster["cls"]
                != det["cls"]
            ):
                continue

            # one detection per teacher
            # in a consensus cluster
            teachers = {
                x["teacher"]
                for x
                in cluster["members"]
            }

            if det["teacher"] in teachers:
                continue

            rep = cluster_box(
                cluster
            )

            overlap = iou_xyxy(
                det["box"],
                rep,
            )

            if (
                overlap >= CLUSTER_IOU
                and overlap > best_iou
            ):

                best_iou = overlap
                best_cluster = cluster

        if best_cluster is None:

            clusters.append({
                "cls":
                    det["cls"],

                "members": [
                    det
                ],
            })

        else:

            best_cluster[
                "members"
            ].append(det)

    consensus = []

    for cluster in clusters:

        members = cluster[
            "members"
        ]

        teachers = sorted({
            m["teacher"]
            for m in members
        })

        mean_conf = (
            sum(
                m["conf"]
                for m in members
            )
            / len(members)
        )

        if (
            len(teachers)
            < CONSENSUS_MIN_TEACHERS
        ):
            continue

        if (
            mean_conf
            < CONSENSUS_MIN_MEAN_CONF
        ):
            continue

        consensus.append({
            "cls":
                cluster["cls"],

            "box":
                cluster_box(
                    cluster
                ),

            "teachers":
                teachers,

            "teacher_count":
                len(teachers),

            "mean_conf":
                mean_conf,

            "support_weight":
                sum(
                    TEACHER_WEIGHTS[t]
                    for t in teachers
                ),
        })

    return consensus


# ============================================================
# ANNOTATION SCORING
# ============================================================

def score_annotation(
    gt_boxes,
    consensus,
):

    gt_count = len(gt_boxes)
    consensus_count = len(consensus)

    # --------------------------------------------------------
    # Negative annotation
    # --------------------------------------------------------

    if gt_count == 0:

        if consensus_count == 0:

            return {
                "score": 1.0,
                "gt_coverage": 1.0,
                "consensus_coverage": 1.0,
                "mean_iou": 1.0,
                "supported_gt": 0,
                "missing_consensus": 0,
                "unsupported_gt": 0,
            }

        return {
            "score": 0.0,
            "gt_coverage": 0.0,
            "consensus_coverage": 0.0,
            "mean_iou": 0.0,
            "supported_gt": 0,
            "missing_consensus":
                consensus_count,
            "unsupported_gt": 0,
        }

    # --------------------------------------------------------
    # Teachers produced no consensus
    # --------------------------------------------------------

    if consensus_count == 0:

        return {
            "score": 0.0,
            "gt_coverage": 0.0,
            "consensus_coverage": 0.0,
            "mean_iou": 0.0,
            "supported_gt": 0,
            "missing_consensus": 0,
            "unsupported_gt":
                gt_count,
        }

    # --------------------------------------------------------
    # GT -> consensus
    # --------------------------------------------------------

    gt_best_ious = []

    for gt in gt_boxes:

        candidates = [
            iou_xyxy(
                gt["box"],
                c["box"],
            )
            for c in consensus
            if c["cls"] == gt["cls"]
        ]

        best = (
            max(candidates)
            if candidates
            else 0.0
        )

        gt_best_ious.append(
            best
        )

    supported_gt = sum(
        x >= ANNOTATION_MATCH_IOU
        for x in gt_best_ious
    )

    unsupported_gt = (
        gt_count
        - supported_gt
    )

    gt_coverage = (
        supported_gt
        / gt_count
    )

    mean_iou = (
        sum(gt_best_ious)
        / gt_count
    )

    # --------------------------------------------------------
    # Consensus -> GT
    # --------------------------------------------------------

    consensus_supported = 0

    for c in consensus:

        candidates = [
            iou_xyxy(
                c["box"],
                gt["box"],
            )
            for gt in gt_boxes
            if gt["cls"] == c["cls"]
        ]

        best = (
            max(candidates)
            if candidates
            else 0.0
        )

        if (
            best
            >= ANNOTATION_MATCH_IOU
        ):

            consensus_supported += 1

    consensus_coverage = (
        consensus_supported
        / consensus_count
    )

    missing_consensus = (
        consensus_count
        - consensus_supported
    )

    # --------------------------------------------------------
    # Composite score
    # --------------------------------------------------------

    score = (
        0.45 * gt_coverage
        +
        0.35 * consensus_coverage
        +
        0.20 * mean_iou
    )

    return {
        "score":
            score,

        "gt_coverage":
            gt_coverage,

        "consensus_coverage":
            consensus_coverage,

        "mean_iou":
            mean_iou,

        "supported_gt":
            supported_gt,

        "unsupported_gt":
            unsupported_gt,

        "missing_consensus":
            missing_consensus,
    }


# ============================================================
# DECISION
# ============================================================

def decide(
    relation,
    conflict_type,
    score_a,
    score_b,
    consensus_count,
):

    # --------------------------------------------------------
    # Benchmark-sensitive conflicts are never
    # auto-recovered from teacher evidence.
    # --------------------------------------------------------

    if "test" in relation:

        return (
            "BENCHMARK_QUARANTINE",
            False,
        )

    # --------------------------------------------------------
    # Class conflicts always remain review-required.
    # --------------------------------------------------------

    if conflict_type == "class_conflict":

        return (
            "CLASS_CONFLICT_REVIEW",
            False,
        )

    if consensus_count == 0:

        return (
            "TEACHER_UNCERTAIN",
            False,
        )

    a = score_a["score"]
    b = score_b["score"]

    margin = abs(a - b)

    if (
        a >= STRONG_MIN_SCORE
        and
        a - b >= STRONG_MIN_MARGIN
    ):

        decision = "A_STRONG"

    elif (
        b >= STRONG_MIN_SCORE
        and
        b - a >= STRONG_MIN_MARGIN
    ):

        decision = "B_STRONG"

    elif (
        a >= BOTH_REASONABLE_MIN
        and
        b >= BOTH_REASONABLE_MIN
        and
        margin
        <= BOTH_REASONABLE_MAX_MARGIN
    ):

        decision = "BOTH_REASONABLE"

    else:

        decision = "TEACHER_UNCERTAIN"

    # --------------------------------------------------------
    # Only SAME_TRAIN can later become
    # automatic recovery candidate.
    # --------------------------------------------------------

    recovery_eligible = (
        relation == "same_train"
        and decision
        in {
            "A_STRONG",
            "B_STRONG",
        }
    )

    return (
        decision,
        recovery_eligible,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 88)
    print("FASDD R0 TEACHER EXACT-CONFLICT ADJUDICATOR")
    print("=" * 88)

    # --------------------------------------------------------
    # Verify models
    # --------------------------------------------------------

    for name, path in TEACHERS.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Teacher {name}: {path}"
            )

        print(
            f"Teacher {name}: {path}"
        )

    # --------------------------------------------------------
    # Read conflicts
    # --------------------------------------------------------

    with CONFLICTS_CSV.open(
        encoding="utf-8",
    ) as f:

        conflicts = list(
            csv.DictReader(f)
        )

    print()
    print(
        "Conflict groups:",
        len(conflicts)
    )

    if len(conflicts) != 67:

        print(
            "WARNING: expected 67 conflict groups."
        )

    image_map = build_image_map()

    # one physical image per exact duplicate group
    image_paths = []

    for row in conflicts:

        stem = row["stem_a"]

        image_path = image_map.get(
            stem
        )

        if image_path is None:

            raise FileNotFoundError(
                f"Image not found: {stem}"
            )

        image_paths.append(
            image_path
        )

    # ========================================================
    # TEACHER INFERENCE
    # ========================================================

    predictions = {
        i: {}
        for i in range(
            len(conflicts)
        )
    }

    for teacher_name, model_path in (
        TEACHERS.items()
    ):

        print()
        print("=" * 88)
        print(
            f"Running Teacher {teacher_name}"
        )
        print("=" * 88)

        model = YOLO(
            str(model_path)
        )

        results = model.predict(
            source=[
                str(p)
                for p in image_paths
            ],
            imgsz=IMGSZ,
            conf=PRED_CONF,
            iou=PRED_IOU,
            device=0,
            batch=BATCH,
            verbose=False,
            stream=True,
        )

        count = 0

        for idx, result in enumerate(
            results
        ):

            preds = []

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                xyxy = (
                    result.boxes.xyxy
                    .cpu()
                    .tolist()
                )

                confs = (
                    result.boxes.conf
                    .cpu()
                    .tolist()
                )

                classes = (
                    result.boxes.cls
                    .cpu()
                    .tolist()
                )

                for box, conf, cls in zip(
                    xyxy,
                    confs,
                    classes,
                ):

                    cls = int(cls)

                    if cls not in {
                        0,
                        1,
                    }:
                        continue

                    preds.append({
                        "cls":
                            cls,

                        "conf":
                            float(conf),

                        "box": [
                            float(x)
                            for x in box
                        ],
                    })

            predictions[
                idx
            ][teacher_name] = preds

            count += 1

        print(
            f"Teacher {teacher_name}: "
            f"{count} images complete"
        )

        del model

    # ========================================================
    # ADJUDICATE
    # ========================================================

    records = []
    csv_rows = []

    decision_counts = Counter()
    relation_counts = Counter()
    type_counts = Counter()

    recovery_candidates = []

    for idx, row in enumerate(
        conflicts
    ):

        group_id = int(
            row["group_id"]
        )

        relation = row[
            "relation"
        ]

        conflict_type = row[
            "result"
        ]

        stem_a = row[
            "stem_a"
        ]

        stem_b = row[
            "stem_b"
        ]

        image_path = image_paths[
            idx
        ]

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"Cannot decode: {image_path}"
            )

        height, width = (
            image.shape[:2]
        )

        gt_a = read_annotation(
            stem_a,
            width,
            height,
        )

        gt_b = read_annotation(
            stem_b,
            width,
            height,
        )

        teacher_preds = predictions[
            idx
        ]

        consensus = build_consensus(
            teacher_preds
        )

        score_a = score_annotation(
            gt_a,
            consensus,
        )

        score_b = score_annotation(
            gt_b,
            consensus,
        )

        decision, recovery_eligible = (
            decide(
                relation,
                conflict_type,
                score_a,
                score_b,
                len(consensus),
            )
        )

        decision_counts[
            decision
        ] += 1

        relation_counts[
            relation
        ] += 1

        type_counts[
            conflict_type
        ] += 1

        selected_stem = ""

        if decision == "A_STRONG":
            selected_stem = stem_a

        elif decision == "B_STRONG":
            selected_stem = stem_b

        if recovery_eligible:

            recovery_candidates.append({
                "group_id":
                    group_id,

                "selected_stem":
                    selected_stem,

                "decision":
                    decision,

                "score_a":
                    score_a["score"],

                "score_b":
                    score_b["score"],
            })

        record = {
            "group_id":
                group_id,

            "relation":
                relation,

            "conflict_type":
                conflict_type,

            "stem_a":
                stem_a,

            "split_a":
                row["split_a"],

            "stem_b":
                stem_b,

            "split_b":
                row["split_b"],

            "teacher_predictions":
                teacher_preds,

            "consensus":
                consensus,

            "annotation_a":
                {
                    "boxes":
                        gt_a,

                    "metrics":
                        score_a,
                },

            "annotation_b":
                {
                    "boxes":
                        gt_b,

                    "metrics":
                        score_b,
                },

            "decision":
                decision,

            "selected_stem":
                selected_stem,

            "recovery_eligible":
                recovery_eligible,
        }

        records.append(
            record
        )

        csv_rows.append({
            "group_id":
                group_id,

            "relation":
                relation,

            "conflict_type":
                conflict_type,

            "stem_a":
                stem_a,

            "stem_b":
                stem_b,

            "consensus_boxes":
                len(consensus),

            "score_a":
                f"{score_a['score']:.6f}",

            "score_b":
                f"{score_b['score']:.6f}",

            "margin":
                f"{abs(score_a['score'] - score_b['score']):.6f}",

            "a_gt_coverage":
                f"{score_a['gt_coverage']:.6f}",

            "b_gt_coverage":
                f"{score_b['gt_coverage']:.6f}",

            "a_consensus_coverage":
                f"{score_a['consensus_coverage']:.6f}",

            "b_consensus_coverage":
                f"{score_b['consensus_coverage']:.6f}",

            "a_mean_iou":
                f"{score_a['mean_iou']:.6f}",

            "b_mean_iou":
                f"{score_b['mean_iou']:.6f}",

            "decision":
                decision,

            "selected_stem":
                selected_stem,

            "recovery_eligible":
                int(
                    recovery_eligible
                ),
        })

    # ========================================================
    # SAVE CSV
    # ========================================================

    with OUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fields = [
            "group_id",
            "relation",
            "conflict_type",
            "stem_a",
            "stem_b",
            "consensus_boxes",
            "score_a",
            "score_b",
            "margin",
            "a_gt_coverage",
            "b_gt_coverage",
            "a_consensus_coverage",
            "b_consensus_coverage",
            "a_mean_iou",
            "b_mean_iou",
            "decision",
            "selected_stem",
            "recovery_eligible",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            csv_rows
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    report = {
        "config": {
            "imgsz":
                IMGSZ,

            "prediction_conf":
                PRED_CONF,

            "prediction_iou":
                PRED_IOU,

            "cluster_iou":
                CLUSTER_IOU,

            "consensus_min_teachers":
                CONSENSUS_MIN_TEACHERS,

            "consensus_min_mean_conf":
                CONSENSUS_MIN_MEAN_CONF,

            "annotation_match_iou":
                ANNOTATION_MATCH_IOU,

            "teacher_weights":
                TEACHER_WEIGHTS,

            "automatic_recovery_policy":
                (
                    "same_train only; "
                    "never class_conflict; "
                    "never test-related"
                ),
        },

        "summary": {
            "groups":
                len(records),

            "decisions":
                dict(
                    decision_counts
                ),

            "relations":
                dict(
                    relation_counts
                ),

            "conflict_types":
                dict(
                    type_counts
                ),

            "recovery_candidates":
                len(
                    recovery_candidates
                ),
        },

        "recovery_candidates":
            recovery_candidates,

        "groups":
            records,
    }

    OUT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 88)
    print("R0 ADJUDICATION SUMMARY")
    print("=" * 88)

    print(
        "Conflict groups      :",
        len(records)
    )

    print()
    print("DECISIONS")
    print("-" * 88)

    for decision in (
        "A_STRONG",
        "B_STRONG",
        "BOTH_REASONABLE",
        "TEACHER_UNCERTAIN",
        "CLASS_CONFLICT_REVIEW",
        "BENCHMARK_QUARANTINE",
    ):

        print(
            f"{decision:26}: "
            f"{decision_counts[decision]}"
        )

    print()
    print(
        "Same-train recovery candidates:",
        len(recovery_candidates)
    )

    print()
    print(
        "Important:"
    )

    print(
        "  No manifest has been modified."
    )

    print(
        "  No annotation has been modified."
    )

    print(
        "  VAL/TEST are not auto-recovered."
    )

    print(
        "  Class conflicts are not auto-relabelled."
    )

    print()
    print(
        "CSV :",
        OUT_CSV
    )

    print(
        "JSON:",
        OUT_JSON
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
