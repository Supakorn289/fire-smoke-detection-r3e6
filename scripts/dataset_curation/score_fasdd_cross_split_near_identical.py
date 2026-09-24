#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import json
import re


ROOT = Path(__file__).resolve().parents[2]

LABELS = (
    Path("/path/to/fire_datasets/fasdd/curated_v1")
    / "annotations"
    / "YOLO_CV"
    / "labels"
)

VISUAL_CLEAN = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_visual_exact"
)

PIXEL_METRICS = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "p0d0_pixel_metrics.csv"
)

OUT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
    / "cross_split_scoring"
)

OUT_CSV = OUT_DIR / "scored_pairs.csv"
HIGH_CSV = OUT_DIR / "high_priority_pairs.csv"
OUT_JSON = OUT_DIR / "summary.json"


CROSS_RELATIONS = {
    "train_val",
    "train_test",
    "val_test",
}


# ============================================================
# MANIFEST
# ============================================================

def load_manifest(split):

    path = (
        VISUAL_CLEAN
        / f"{split}_clean_visual_exact.txt"
    )

    return {
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    }


# ============================================================
# YOLO LABELS
# ============================================================

def read_boxes(stem):

    path = LABELS / f"{stem}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing label: {path}"
        )

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

        xc, yc, w, h = map(
            float,
            parts[1:],
        )

        boxes.append({
            "cls": cls,
            "xyxy": (
                xc - w / 2,
                yc - h / 2,
                xc + w / 2,
                yc + h / 2,
            ),
            "raw": (
                cls,
                xc,
                yc,
                w,
                h,
            ),
        })

    return boxes


# ============================================================
# IOU
# ============================================================

def iou(a, b):

    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])

    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    iw = max(
        0.0,
        x2 - x1,
    )

    ih = max(
        0.0,
        y2 - y1,
    )

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

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


# ============================================================
# CLASS-AWARE GREEDY MATCH
# ============================================================

def match_boxes(
    boxes_a,
    boxes_b,
    threshold,
):

    # Both negative images
    if (
        len(boxes_a) == 0
        and len(boxes_b) == 0
    ):

        return {
            "matched": 0,
            "coverage_a": 1.0,
            "coverage_b": 1.0,
            "mean_iou": 1.0,
        }

    # One negative, one positive
    if (
        len(boxes_a) == 0
        or len(boxes_b) == 0
    ):

        return {
            "matched": 0,
            "coverage_a": (
                1.0
                if len(boxes_a) == 0
                else 0.0
            ),
            "coverage_b": (
                1.0
                if len(boxes_b) == 0
                else 0.0
            ),
            "mean_iou": 0.0,
        }

    candidates = []

    for ia, a in enumerate(boxes_a):

        for ib, b in enumerate(boxes_b):

            if a["cls"] != b["cls"]:
                continue

            overlap = iou(
                a["xyxy"],
                b["xyxy"],
            )

            if overlap >= threshold:

                candidates.append(
                    (
                        overlap,
                        ia,
                        ib,
                    )
                )

    candidates.sort(
        reverse=True
    )

    used_a = set()
    used_b = set()

    matches = []

    for overlap, ia, ib in candidates:

        if ia in used_a:
            continue

        if ib in used_b:
            continue

        used_a.add(ia)
        used_b.add(ib)

        matches.append(
            overlap
        )

    matched = len(matches)

    return {
        "matched":
            matched,

        "coverage_a":
            matched / len(boxes_a),

        "coverage_b":
            matched / len(boxes_b),

        "mean_iou":
            (
                sum(matches)
                / len(matches)
                if matches
                else 0.0
            ),
    }


# ============================================================
# ANNOTATION EQUIVALENCE
# ============================================================

def annotation_equivalent(
    boxes_a,
    boxes_b,
    tol=1e-6,
):

    if len(boxes_a) != len(boxes_b):
        return False

    raw_a = sorted(
        x["raw"]
        for x in boxes_a
    )

    raw_b = sorted(
        x["raw"]
        for x in boxes_b
    )

    for a, b in zip(
        raw_a,
        raw_b,
    ):

        if a[0] != b[0]:
            return False

        for xa, xb in zip(
            a[1:],
            b[1:],
        ):

            if abs(xa - xb) > tol:
                return False

    return True


# ============================================================
# CLASS PROFILE
# ============================================================

def class_counter(boxes):

    return Counter(
        x["cls"]
        for x in boxes
    )


# ============================================================
# FILENAME SEQUENCE HINT
# ============================================================

TRAILING_NUMBER = re.compile(
    r"^(.*?)(\d+)$"
)


def filename_sequence(
    stem_a,
    stem_b,
):

    ma = TRAILING_NUMBER.match(
        stem_a
    )

    mb = TRAILING_NUMBER.match(
        stem_b
    )

    if not ma or not mb:

        return {
            "same_prefix": False,
            "index_gap": None,
            "close_index": False,
        }

    prefix_a = ma.group(1)
    prefix_b = mb.group(1)

    if prefix_a != prefix_b:

        return {
            "same_prefix": False,
            "index_gap": None,
            "close_index": False,
        }

    num_a = int(
        ma.group(2)
    )

    num_b = int(
        mb.group(2)
    )

    gap = abs(
        num_a - num_b
    )

    return {
        "same_prefix": True,
        "index_gap": gap,
        "close_index": (
            gap <= 5
        ),
    }


# ============================================================
# IMAGE TIER
# ============================================================

def image_tier(
    corr,
    mae,
    psnr,
    color,
):

    # Extremely similar decoded appearance.
    if (
        corr >= 0.9995
        and
        mae <= 0.005
        and
        psnr >= 40.0
        and
        color >= 0.999
    ):

        return "EXTREME"

    # All rows entering this script should
    # already satisfy ultra_strict.
    if (
        corr >= 0.999
        and
        mae <= 0.010
        and
        color >= 0.995
    ):

        return "ULTRA"

    return "OTHER"


# ============================================================
# ANNOTATION STATE
# ============================================================

def annotation_state(
    boxes_a,
    boxes_b,
    match30,
    match50,
):

    if annotation_equivalent(
        boxes_a,
        boxes_b,
    ):

        return "EXACT"

    classes_equal = (
        class_counter(boxes_a)
        ==
        class_counter(boxes_b)
    )

    box_count_equal = (
        len(boxes_a)
        ==
        len(boxes_b)
    )

    strong = (
        classes_equal
        and
        box_count_equal
        and
        match50["coverage_a"] >= 0.80
        and
        match50["coverage_b"] >= 0.80
        and
        match50["mean_iou"] >= 0.70
    )

    if strong:
        return "STRONG"

    moderate = (
        classes_equal
        and
        match30["coverage_a"] >= 0.70
        and
        match30["coverage_b"] >= 0.70
        and
        match30["mean_iou"] >= 0.50
    )

    if moderate:
        return "MODERATE"

    return "CONFLICT"


# ============================================================
# PRIORITY
# ============================================================

def leakage_priority(
    visual_tier,
    annotation,
):

    if (
        visual_tier == "EXTREME"
        and
        annotation
        in {
            "EXACT",
            "STRONG",
        }
    ):

        return "VERY_HIGH"

    if visual_tier == "EXTREME":

        return "HIGH"

    if (
        visual_tier == "ULTRA"
        and
        annotation
        in {
            "EXACT",
            "STRONG",
        }
    ):

        return "HIGH"

    if (
        visual_tier == "ULTRA"
        and
        annotation == "MODERATE"
    ):

        return "MEDIUM"

    return "REVIEW"


# ============================================================
# MAIN
# ============================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 92)
    print("FASDD CROSS-SPLIT NEAR-IDENTICAL LEAKAGE SCORING")
    print("=" * 92)

    clean = {
        split:
            load_manifest(split)

        for split in (
            "train",
            "val",
            "test",
        )
    }

    clean_lookup = {
        stem: split
        for split, stems
        in clean.items()
        for stem in stems
    }

    print(
        "Visual-exact clean images:",
        f"{len(clean_lookup):,}"
    )

    # ========================================================
    # INPUT FILTER
    # ========================================================

    candidates = []

    original_ultra_cross = 0
    removed_by_clean_layer = 0

    with PIXEL_METRICS.open(
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            relation = row[
                "relation"
            ]

            if relation not in CROSS_RELATIONS:
                continue

            gates = {
                x
                for x in row[
                    "passed_gates"
                ].split("|")
                if x
            }

            if "ultra_strict" not in gates:
                continue

            original_ultra_cross += 1

            a = row[
                "stem_a"
            ]

            b = row[
                "stem_b"
            ]

            if (
                a not in clean_lookup
                or
                b not in clean_lookup
            ):

                removed_by_clean_layer += 1
                continue

            candidates.append(
                row
            )

    print(
        "Original ultra-strict cross:",
        f"{original_ultra_cross:,}"
    )

    print(
        "Filtered by visual-clean    :",
        f"{removed_by_clean_layer:,}"
    )

    print(
        "Pairs to score              :",
        f"{len(candidates):,}"
    )

    # ========================================================
    # CACHE LABELS
    # ========================================================

    label_cache = {}

    def get_boxes(stem):

        if stem not in label_cache:

            label_cache[
                stem
            ] = read_boxes(
                stem
            )

        return label_cache[
            stem
        ]

    # ========================================================
    # SCORE
    # ========================================================

    output = []

    priority_counts = Counter()
    visual_counts = Counter()
    annotation_counts = Counter()
    relation_counts = Counter()

    relation_priority = {
        rel: Counter()
        for rel in sorted(
            CROSS_RELATIONS
        )
    }

    sequence_counts = Counter()

    for idx, row in enumerate(
        candidates,
        1,
    ):

        stem_a = row[
            "stem_a"
        ]

        stem_b = row[
            "stem_b"
        ]

        relation = row[
            "relation"
        ]

        boxes_a = get_boxes(
            stem_a
        )

        boxes_b = get_boxes(
            stem_b
        )

        m30 = match_boxes(
            boxes_a,
            boxes_b,
            0.30,
        )

        m50 = match_boxes(
            boxes_a,
            boxes_b,
            0.50,
        )

        corr = float(
            row["gray_corr"]
        )

        mae = float(
            row["gray_mae"]
        )

        pair_psnr = float(
            row["psnr"]
        )

        color = float(
            row["color_corr"]
        )

        visual = image_tier(
            corr,
            mae,
            pair_psnr,
            color,
        )

        annotation = (
            annotation_state(
                boxes_a,
                boxes_b,
                m30,
                m50,
            )
        )

        priority = leakage_priority(
            visual,
            annotation,
        )

        seq = filename_sequence(
            stem_a,
            stem_b,
        )

        classes_a = class_counter(
            boxes_a
        )

        classes_b = class_counter(
            boxes_b
        )

        class_profile_a = ";".join(
            f"{k}:{v}"
            for k, v
            in sorted(
                classes_a.items()
            )
        )

        class_profile_b = ";".join(
            f"{k}:{v}"
            for k, v
            in sorted(
                classes_b.items()
            )
        )

        output.append({
            "stem_a":
                stem_a,

            "split_a":
                row["split_a"],

            "stem_b":
                stem_b,

            "split_b":
                row["split_b"],

            "relation":
                relation,

            "visual_tier":
                visual,

            "annotation_state":
                annotation,

            "leakage_priority":
                priority,

            "gray_corr":
                f"{corr:.8f}",

            "gray_mae":
                f"{mae:.8f}",

            "psnr":
                f"{pair_psnr:.6f}",

            "color_corr":
                f"{color:.8f}",

            "boxes_a":
                len(boxes_a),

            "boxes_b":
                len(boxes_b),

            "class_profile_a":
                class_profile_a,

            "class_profile_b":
                class_profile_b,

            "match30":
                m30["matched"],

            "coverage30_a":
                f"{m30['coverage_a']:.6f}",

            "coverage30_b":
                f"{m30['coverage_b']:.6f}",

            "mean_iou30":
                f"{m30['mean_iou']:.6f}",

            "match50":
                m50["matched"],

            "coverage50_a":
                f"{m50['coverage_a']:.6f}",

            "coverage50_b":
                f"{m50['coverage_b']:.6f}",

            "mean_iou50":
                f"{m50['mean_iou']:.6f}",

            "same_filename_prefix":
                int(
                    seq["same_prefix"]
                ),

            "filename_index_gap":
                (
                    ""
                    if seq["index_gap"] is None
                    else seq["index_gap"]
                ),

            "close_filename_index":
                int(
                    seq["close_index"]
                ),
        })

        priority_counts[
            priority
        ] += 1

        visual_counts[
            visual
        ] += 1

        annotation_counts[
            annotation
        ] += 1

        relation_counts[
            relation
        ] += 1

        relation_priority[
            relation
        ][priority] += 1

        if seq[
            "same_prefix"
        ]:

            sequence_counts[
                "same_prefix"
            ] += 1

        if seq[
            "close_index"
        ]:

            sequence_counts[
                "close_index"
            ] += 1

        if (
            idx % 500 == 0
            or idx == len(candidates)
        ):

            print(
                f"\rScoring "
                f"{idx}/{len(candidates)} "
                f"({idx/len(candidates)*100:5.1f}%)",
                end="",
                flush=True,
            )

    print()

    # ========================================================
    # SORT
    # ========================================================

    priority_order = {
        "VERY_HIGH": 0,
        "HIGH": 1,
        "MEDIUM": 2,
        "REVIEW": 3,
    }

    relation_order = {
        "train_test": 0,
        "val_test": 1,
        "train_val": 2,
    }

    output.sort(
        key=lambda x: (
            priority_order[
                x[
                    "leakage_priority"
                ]
            ],

            relation_order[
                x["relation"]
            ],

            -float(
                x["gray_corr"]
            ),

            float(
                x["gray_mae"]
            ),

            x["stem_a"],
            x["stem_b"],
        )
    )

    # ========================================================
    # SAVE ALL
    # ========================================================

    fields = [
        "stem_a",
        "split_a",
        "stem_b",
        "split_b",
        "relation",
        "visual_tier",
        "annotation_state",
        "leakage_priority",
        "gray_corr",
        "gray_mae",
        "psnr",
        "color_corr",
        "boxes_a",
        "boxes_b",
        "class_profile_a",
        "class_profile_b",
        "match30",
        "coverage30_a",
        "coverage30_b",
        "mean_iou30",
        "match50",
        "coverage50_a",
        "coverage50_b",
        "mean_iou50",
        "same_filename_prefix",
        "filename_index_gap",
        "close_filename_index",
    ]

    with OUT_CSV.open(
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
            output
        )

    # ========================================================
    # SAVE HIGH PRIORITY
    # ========================================================

    high_rows = [
        x
        for x in output
        if x[
            "leakage_priority"
        ] in {
            "VERY_HIGH",
            "HIGH",
        }
    ]

    with HIGH_CSV.open(
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
            high_rows
        )

    # ========================================================
    # JSON
    # ========================================================

    report = {
        "source":
            "p0/d0 ultra_strict cross-split pairs",

        "input": {
            "original_ultra_strict_cross":
                original_ultra_cross,

            "removed_by_visual_clean":
                removed_by_clean_layer,

            "scored":
                len(output),
        },

        "priority":
            dict(
                priority_counts
            ),

        "visual_tier":
            dict(
                visual_counts
            ),

        "annotation_state":
            dict(
                annotation_counts
            ),

        "relations":
            dict(
                relation_counts
            ),

        "relation_priority": {
            relation:
                dict(counts)

            for relation, counts
            in relation_priority.items()
        },

        "filename_sequence_hint":
            dict(
                sequence_counts
            ),

        "important":
            (
                "This report is diagnostic only. "
                "No manifest or dataset file was modified."
            ),
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
    # PRINT
    # ========================================================

    print()
    print("=" * 92)
    print("CROSS-SPLIT LEAKAGE SCORE SUMMARY")
    print("=" * 92)

    print(
        "Original ultra-strict cross :",
        f"{original_ultra_cross:,}"
    )

    print(
        "Removed by visual-clean     :",
        f"{removed_by_clean_layer:,}"
    )

    print(
        "Scored pairs                :",
        f"{len(output):,}"
    )

    print()
    print("VISUAL TIER")
    print("-" * 92)

    for name in (
        "EXTREME",
        "ULTRA",
        "OTHER",
    ):

        print(
            f"{name:16}: "
            f"{visual_counts[name]:,}"
        )

    print()
    print("ANNOTATION STATE")
    print("-" * 92)

    for name in (
        "EXACT",
        "STRONG",
        "MODERATE",
        "CONFLICT",
    ):

        print(
            f"{name:16}: "
            f"{annotation_counts[name]:,}"
        )

    print()
    print("LEAKAGE PRIORITY")
    print("-" * 92)

    for name in (
        "VERY_HIGH",
        "HIGH",
        "MEDIUM",
        "REVIEW",
    ):

        print(
            f"{name:16}: "
            f"{priority_counts[name]:,}"
        )

    print()
    print("RELATION × PRIORITY")
    print("-" * 92)

    for relation in (
        "train_val",
        "train_test",
        "val_test",
    ):

        counts = relation_priority[
            relation
        ]

        print()
        print(relation)

        for name in (
            "VERY_HIGH",
            "HIGH",
            "MEDIUM",
            "REVIEW",
        ):

            print(
                f"  {name:12}: "
                f"{counts[name]:,}"
            )

    print()
    print("FILENAME SEQUENCE HINT")
    print("-" * 92)

    print(
        "Same prefix       :",
        f"{sequence_counts['same_prefix']:,}"
    )

    print(
        "Index gap <= 5    :",
        f"{sequence_counts['close_index']:,}"
    )

    print()
    print(
        "High-priority pairs:",
        f"{len(high_rows):,}"
    )

    print()
    print(
        "CSV all :",
        OUT_CSV
    )

    print(
        "CSV high:",
        HIGH_CSV
    )

    print(
        "JSON    :",
        OUT_JSON
    )

    print()
    print(
        "No dataset, annotation, or manifest was modified."
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
