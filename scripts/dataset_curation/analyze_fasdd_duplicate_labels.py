#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import json


ROOT = Path(__file__).resolve().parents[2]

DUP_JSON = (
    ROOT
    / "reports"
    / "fasdd"
    / "exact_duplicates.json"
)

ORIGINAL_LABELS = (
    ROOT
    / "sources"
    / "fasdd"
    / "FASDD_CV"
    / "annotations"
    / "YOLO_CV"
    / "labels"
)

REPORT_JSON = (
    ROOT
    / "reports"
    / "fasdd"
    / "duplicate_label_analysis.json"
)

REPORT_CSV = (
    ROOT
    / "reports"
    / "fasdd"
    / "duplicate_label_analysis.csv"
)

TOL = 1e-9


def read_label(stem):

    path = ORIGINAL_LABELS / f"{stem}.txt"

    if not path.exists():
        return {
            "exists": False,
            "text": "",
            "boxes": [],
            "classes": [],
        }

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()

    boxes = []

    if text:

        for line in text.splitlines():

            parts = line.split()

            if len(parts) != 5:
                continue

            try:
                cls = int(parts[0])

                coords = tuple(
                    float(x)
                    for x in parts[1:]
                )

            except ValueError:
                continue

            boxes.append(
                (cls, *coords)
            )

    boxes.sort()

    return {
        "exists": True,
        "text": text,
        "boxes": boxes,
        "classes": sorted(
            box[0]
            for box in boxes
        ),
    }


def boxes_equal(a, b, tol=TOL):

    if len(a) != len(b):
        return False

    for box_a, box_b in zip(a, b):

        if box_a[0] != box_b[0]:
            return False

        for x, y in zip(
            box_a[1:],
            box_b[1:],
        ):

            if abs(x - y) > tol:
                return False

    return True


def classify(a, b):

    if (
        not a["exists"]
        or not b["exists"]
    ):
        return "missing_label"

    if a["text"] == b["text"]:
        return "exact_annotation"

    if boxes_equal(
        a["boxes"],
        b["boxes"],
    ):
        return "equivalent_annotation"

    if len(a["boxes"]) != len(b["boxes"]):
        return "box_count_conflict"

    if a["classes"] != b["classes"]:
        return "class_conflict"

    return "geometry_conflict"


def pair_type(split_a, split_b):

    if split_a == split_b:
        return f"same_{split_a}"

    return "_".join(
        sorted(
            [split_a, split_b],
            key=lambda x: {
                "train": 0,
                "val": 1,
                "test": 2,
            }.get(x, 99),
        )
    )


def main():

    print("=" * 84)
    print("FASDD DUPLICATE ANNOTATION ANALYSIS")
    print("=" * 84)

    data = json.loads(
        DUP_JSON.read_text(
            encoding="utf-8"
        )
    )

    groups = data[
        "duplicate_groups"
    ]

    stats = Counter()

    rows = []

    for group in groups:

        members = group[
            "members"
        ]

        stats[
            "duplicate_groups"
        ] += 1

        if len(members) != 2:

            stats[
                "non_pair_groups"
            ] += 1

        # Handle all possible combinations even though
        # current FASDD result consists of pairs.
        for i in range(len(members)):

            for j in range(
                i + 1,
                len(members)
            ):

                a = members[i]
                b = members[j]

                stem_a = a["stem"]
                stem_b = b["stem"]

                split_a = a["split"]
                split_b = b["split"]

                label_a = read_label(
                    stem_a
                )

                label_b = read_label(
                    stem_b
                )

                result = classify(
                    label_a,
                    label_b,
                )

                relation = pair_type(
                    split_a,
                    split_b,
                )

                stats[
                    "pairs"
                ] += 1

                stats[
                    f"relation_{relation}"
                ] += 1

                stats[
                    f"result_{result}"
                ] += 1

                conflict = result in {
                    "missing_label",
                    "box_count_conflict",
                    "class_conflict",
                    "geometry_conflict",
                }

                if conflict:
                    stats[
                        "annotation_conflicts"
                    ] += 1

                rows.append({
                    "group_id":
                        group["group_id"],

                    "relation":
                        relation,

                    "stem_a":
                        stem_a,

                    "split_a":
                        split_a,

                    "boxes_a":
                        len(
                            label_a["boxes"]
                        ),

                    "classes_a":
                        ",".join(
                            map(
                                str,
                                label_a[
                                    "classes"
                                ],
                            )
                        ),

                    "stem_b":
                        stem_b,

                    "split_b":
                        split_b,

                    "boxes_b":
                        len(
                            label_b["boxes"]
                        ),

                    "classes_b":
                        ",".join(
                            map(
                                str,
                                label_b[
                                    "classes"
                                ],
                            )
                        ),

                    "result":
                        result,

                    "conflict":
                        int(conflict),
                })

    REPORT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "relation",
                "stem_a",
                "split_a",
                "boxes_a",
                "classes_a",
                "stem_b",
                "split_b",
                "boxes_b",
                "classes_b",
                "result",
                "conflict",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    report = {
        "stats": dict(stats),
        "pairs": rows,
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
    print("PAIR RELATIONS")
    print("-" * 84)

    for relation in (
        "same_train",
        "same_val",
        "same_test",
        "train_val",
        "train_test",
        "val_test",
    ):

        print(
            f"{relation:18}: "
            f"{stats[f'relation_{relation}']}"
        )

    print()
    print("ANNOTATION COMPARISON")
    print("-" * 84)

    for result in (
        "exact_annotation",
        "equivalent_annotation",
        "geometry_conflict",
        "box_count_conflict",
        "class_conflict",
        "missing_label",
    ):

        print(
            f"{result:24}: "
            f"{stats[f'result_{result}']}"
        )

    print()
    print(
        "Total duplicate pairs :",
        stats["pairs"],
    )

    print(
        "Annotation conflicts   :",
        stats[
            "annotation_conflicts"
        ],
    )

    print()
    print(
        "CSV :",
        REPORT_CSV,
    )

    print(
        "JSON:",
        REPORT_JSON,
    )

    print("=" * 84)


if __name__ == "__main__":
    main()
