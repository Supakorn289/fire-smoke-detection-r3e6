#!/usr/bin/env python3

from pathlib import Path
from collections import Counter, defaultdict
import json
import zipfile


ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = (
    ROOT
    / "sources"
    / "fasdd"
    / "FASDD_CV.zip"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
)

REPORT_FILE = (
    REPORT_DIR
    / "yolo_mapping_verification.json"
)

YOLO_PREFIX = (
    "FASDD_CV/annotations/YOLO_CV/labels/"
)

SPLIT_PREFIX = (
    "FASDD_CV/annotations/YOLO_CV/"
)


def category_from_stem(stem):

    s = stem.lower()

    if s.startswith("bothfireandsmoke"):
        return "fire_smoke"

    if s.startswith("fire"):
        return "fire_only"

    if s.startswith("smoke"):
        return "smoke_only"

    if (
        "neither" in s
        or "nofire" in s
        or "negative" in s
    ):
        return "negative"

    return "other"


def parse_label(text):

    classes = []
    invalid = 0
    boxes = 0

    text = text.strip()

    if not text:
        return {
            "classes": [],
            "boxes": 0,
            "invalid": 0,
        }

    for line in text.splitlines():

        parts = line.split()

        if len(parts) != 5:
            invalid += 1
            continue

        try:
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
        except ValueError:
            invalid += 1
            continue

        if cls not in {0, 1}:
            invalid += 1

        if not all(
            -0.001 <= x <= 1.001
            for x in coords
        ):
            invalid += 1

        classes.append(cls)
        boxes += 1

    return {
        "classes": classes,
        "boxes": boxes,
        "invalid": invalid,
    }


def read_text(zf, name):

    return zf.read(name).decode(
        "utf-8",
        errors="replace"
    )


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 78)
    print("FASDD YOLO CLASS-MAPPING VERIFICATION")
    print("=" * 78)

    with zipfile.ZipFile(
        ZIP_PATH,
        "r"
    ) as zf:

        names = zf.namelist()

        labels = sorted(
            name
            for name in names
            if (
                name.startswith(YOLO_PREFIX)
                and name.endswith(".txt")
            )
        )

        print("YOLO labels :", len(labels))

        category_stats = defaultdict(Counter)
        class_instances = Counter()

        invalid_total = 0
        empty_total = 0

        for i, name in enumerate(
            labels,
            start=1
        ):

            stem = Path(name).stem

            category = category_from_stem(
                stem
            )

            result = parse_label(
                read_text(zf, name)
            )

            classes = result["classes"]
            unique = set(classes)

            category_stats[
                category
            ]["files"] += 1

            category_stats[
                category
            ]["boxes"] += result["boxes"]

            invalid_total += result[
                "invalid"
            ]

            if not classes:

                category_stats[
                    category
                ]["empty"] += 1

                empty_total += 1

            elif unique == {0}:

                category_stats[
                    category
                ]["only_class_0"] += 1

            elif unique == {1}:

                category_stats[
                    category
                ]["only_class_1"] += 1

            elif unique == {0, 1}:

                category_stats[
                    category
                ]["both_classes"] += 1

            else:

                category_stats[
                    category
                ]["other_class_set"] += 1

            class_instances.update(
                classes
            )

            if (
                i % 10000 == 0
                or i == len(labels)
            ):

                print(
                    f"\rLabels checked: "
                    f"{i}/{len(labels)}",
                    end="",
                    flush=True
                )

        print()

        # ====================================================
        # SPLITS
        # ====================================================

        split_stats = {}
        split_sets = {}

        for split in [
            "train",
            "val",
            "test"
        ]:

            path = (
                SPLIT_PREFIX
                + f"{split}.txt"
            )

            text = read_text(
                zf,
                path
            )

            lines = [
                x.strip()
                for x in text.splitlines()
                if x.strip()
            ]

            # basename/stem ใช้ตรวจ overlap
            stems = {
                Path(x).stem
                for x in lines
            }

            split_sets[
                split
            ] = stems

            split_stats[
                split
            ] = {
                "lines": len(lines),
                "unique_stems": len(stems),
            }

        train = split_sets["train"]
        val = split_sets["val"]
        test = split_sets["test"]

        overlap = {
            "train_val":
                len(train & val),

            "train_test":
                len(train & test),

            "val_test":
                len(val & test),
        }

        split_union = (
            train
            | val
            | test
        )

        # ====================================================
        # MAPPING DECISION
        # ====================================================

        fire_stats = (
            category_stats[
                "fire_only"
            ]
        )

        smoke_stats = (
            category_stats[
                "smoke_only"
            ]
        )

        fire_class0 = (
            fire_stats[
                "only_class_0"
            ]
        )

        fire_class1 = (
            fire_stats[
                "only_class_1"
            ]
        )

        smoke_class0 = (
            smoke_stats[
                "only_class_0"
            ]
        )

        smoke_class1 = (
            smoke_stats[
                "only_class_1"
            ]
        )

        mapping = "UNRESOLVED"

        # Filename-semantic consistency
        if (
            fire_class0 > fire_class1
            and smoke_class1 > smoke_class0
        ):

            mapping = (
                "0=fire, 1=smoke"
            )

        elif (
            fire_class1 > fire_class0
            and smoke_class0 > smoke_class1
        ):

            mapping = (
                "0=smoke, 1=fire"
            )

        # ====================================================
        # PRINT
        # ====================================================

        print()
        print("=" * 78)
        print("CATEGORY ↔ YOLO CLASS CONSISTENCY")
        print("=" * 78)

        for category in [
            "fire_only",
            "smoke_only",
            "fire_smoke",
            "negative",
            "other"
        ]:

            s = category_stats[
                category
            ]

            if not s["files"]:
                continue

            print()
            print(f"[{category}]")

            for key in [
                "files",
                "empty",
                "only_class_0",
                "only_class_1",
                "both_classes",
                "other_class_set",
                "boxes"
            ]:

                print(
                    f"{key:18}: "
                    f"{s[key]}"
                )

        print()
        print("=" * 78)
        print("INSTANCE COUNTS")
        print("=" * 78)

        print(
            "Class 0:",
            class_instances[0]
        )

        print(
            "Class 1:",
            class_instances[1]
        )

        print(
            "Invalid:",
            invalid_total
        )

        print(
            "Empty labels:",
            empty_total
        )

        print()
        print("=" * 78)
        print("SOURCE SPLITS")
        print("=" * 78)

        for split in [
            "train",
            "val",
            "test"
        ]:

            s = split_stats[
                split
            ]

            print(
                f"{split:6}: "
                f"{s['lines']:6} lines, "
                f"{s['unique_stems']:6} unique"
            )

        print()
        print(
            "Split union:",
            len(split_union)
        )

        print(
            "train ∩ val :",
            overlap["train_val"]
        )

        print(
            "train ∩ test:",
            overlap["train_test"]
        )

        print(
            "val ∩ test  :",
            overlap["val_test"]
        )

        print()
        print("=" * 78)
        print("MAPPING DECISION")
        print("=" * 78)

        print(
            "YOLO mapping:",
            mapping
        )

        print(
            "COCO reference:",
            "0=fire, 1=smoke"
        )

        mapping_pass = (
            mapping
            == "0=fire, 1=smoke"
        )

        split_pass = (
            len(split_union)
            == len(labels)
            and
            all(
                value == 0
                for value
                in overlap.values()
            )
        )

        print()
        print(
            "Mapping check:",
            "PASS"
            if mapping_pass
            else "FAIL"
        )

        print(
            "Split check  :",
            "PASS"
            if split_pass
            else "FAIL"
        )

        # ====================================================
        # REPORT
        # ====================================================

        report = {
            "labels":
                len(labels),

            "category_stats": {
                category:
                    dict(stats)

                for category, stats
                in category_stats.items()
            },

            "class_instances":
                dict(class_instances),

            "invalid":
                invalid_total,

            "empty_labels":
                empty_total,

            "splits":
                split_stats,

            "split_union":
                len(split_union),

            "split_overlap":
                overlap,

            "mapping":
                mapping,

            "mapping_pass":
                mapping_pass,

            "split_pass":
                split_pass,
        }

        REPORT_FILE.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print()
        print(
            "Report:",
            REPORT_FILE
        )

        print("=" * 78)


if __name__ == "__main__":
    main()
