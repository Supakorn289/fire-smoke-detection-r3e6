#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import hashlib
import json
import shutil


ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_train_val"
)

DATA = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)

LABELS = (
    DATA
    / "annotations"
    / "YOLO_CV"
    / "labels"
)

OUT = (
    ROOT
    / "reports"
    / "fasdd"
    / "clean_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# normalized bbox area thresholds
SMALL_MAX = 0.005
MEDIUM_MAX = 0.05


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_manifest(path):

    return [
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def size_class(area):

    if area < SMALL_MAX:
        return "small"

    if area < MEDIUM_MAX:
        return "medium"

    return "large"


def analyze_split(stems):

    stats = Counter()
    object_sizes = {
        0: Counter(),
        1: Counter(),
    }

    for stem in stems:

        label_path = (
            LABELS
            / f"{stem}.txt"
        )

        if not label_path.exists():
            stats["missing_label"] += 1
            continue

        text = label_path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        if not text:

            stats["negative"] += 1
            continue

        classes = set()
        valid_boxes = 0

        for line in text.splitlines():

            parts = line.split()

            if len(parts) != 5:
                stats["malformed"] += 1
                continue

            try:
                cls = int(parts[0])
                _, _, w, h = map(
                    float,
                    parts[1:],
                )
            except ValueError:
                stats["parse_error"] += 1
                continue

            if cls not in {0, 1}:
                stats["invalid_class"] += 1
                continue

            classes.add(cls)
            valid_boxes += 1

            stats[
                f"class_{cls}_instances"
            ] += 1

            area = w * h

            object_sizes[
                cls
            ][
                size_class(area)
            ] += 1

        if valid_boxes == 0:
            stats[
                "positive_but_no_valid_boxes"
            ] += 1
            continue

        if classes == {0}:
            stats["fire_only"] += 1

        elif classes == {1}:
            stats["smoke_only"] += 1

        elif classes == {0, 1}:
            stats["fire_smoke"] += 1

        else:
            stats["unknown_category"] += 1

    stats["images"] = len(stems)

    stats["positive"] = (
        stats["fire_only"]
        + stats["smoke_only"]
        + stats["fire_smoke"]
    )

    return {
        "stats":
            dict(stats),

        "fire_sizes":
            dict(
                object_sizes[0]
            ),

        "smoke_sizes":
            dict(
                object_sizes[1]
            ),
    }


def main():

    print("=" * 92)
    print("FASDD CLEAN V1 FREEZE + PROFILE")
    print("=" * 92)

    manifests = {}

    expected = {
        "train": 45658,
        "val": 31254,
        "test": 15863,
    }

    # ========================================================
    # FREEZE MANIFESTS
    # ========================================================

    for split in (
        "train",
        "val",
        "test",
    ):

        source = (
            SOURCE
            / f"{split}_clean_cross_split.txt"
        )

        destination = (
            OUT
            / f"{split}.txt"
        )

        shutil.copy2(
            source,
            destination,
        )

        stems = read_manifest(
            destination
        )

        manifests[
            split
        ] = stems

        if len(stems) != expected[split]:

            raise RuntimeError(
                f"{split}: expected "
                f"{expected[split]}, "
                f"got {len(stems)}"
            )

    # ========================================================
    # SPLIT ISOLATION
    # ========================================================

    sets = {
        k: set(v)
        for k, v in manifests.items()
    }

    overlaps = {
        "train_val":
            len(
                sets["train"]
                & sets["val"]
            ),

        "train_test":
            len(
                sets["train"]
                & sets["test"]
            ),

        "val_test":
            len(
                sets["val"]
                & sets["test"]
            ),
    }

    if any(
        overlaps.values()
    ):
        raise RuntimeError(
            f"Split overlap: {overlaps}"
        )

    # ========================================================
    # PROFILE ALL SPLITS
    # ========================================================

    profiles = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        print()
        print(
            f"Profiling {split}: "
            f"{len(manifests[split]):,}"
        )

        profiles[
            split
        ] = analyze_split(
            manifests[split]
        )

    # ========================================================
    # HASH FROZEN MANIFESTS
    # ========================================================

    hashes = {
        split:
            sha256(
                OUT
                / f"{split}.txt"
            )

        for split in (
            "train",
            "val",
            "test",
        )
    }

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "version":
            "FASDD_CLEAN_V1",

        "source":
            "dedup_train_val clean-cross-split manifests",

        "class_contract": {
            "0": "fire",
            "1": "smoke",
        },

        "object_size_definition": {
            "small":
                "normalized bbox area < 0.005",

            "medium":
                "0.005 <= area < 0.05",

            "large":
                "area >= 0.05",
        },

        "splits": {
            split: {
                "manifest":
                    str(
                        OUT
                        / f"{split}.txt"
                    ),

                "sha256":
                    hashes[split],

                **profiles[split],
            }
            for split in (
                "train",
                "val",
                "test",
            )
        },

        "split_overlap":
            overlaps,

        "total_images":
            sum(
                len(x)
                for x
                in manifests.values()
            ),

        "policy": {
            "test_files_modified":
                False,

            "dataset_files_deleted":
                False,

            "automatic_r0_recovery":
                False,
        },
    }

    report_path = (
        OUT
        / "profile.json"
    )

    report_path.write_text(
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
    print("FASDD CLEAN V1")
    print("=" * 92)

    for split in (
        "train",
        "val",
        "test",
    ):

        p = profiles[
            split
        ]

        s = p[
            "stats"
        ]

        print()
        print(
            split.upper()
        )

        print(
            f"  images      : "
            f"{s.get('images', 0):,}"
        )

        print(
            f"  fire_only   : "
            f"{s.get('fire_only', 0):,}"
        )

        print(
            f"  smoke_only  : "
            f"{s.get('smoke_only', 0):,}"
        )

        print(
            f"  fire_smoke  : "
            f"{s.get('fire_smoke', 0):,}"
        )

        print(
            f"  negative    : "
            f"{s.get('negative', 0):,}"
        )

        print(
            f"  fire boxes  : "
            f"{s.get('class_0_instances', 0):,}"
        )

        print(
            f"  smoke boxes : "
            f"{s.get('class_1_instances', 0):,}"
        )

        fs = p[
            "fire_sizes"
        ]

        ss = p[
            "smoke_sizes"
        ]

        print(
            "  FIRE size   : "
            f"small={fs.get('small', 0):,} "
            f"medium={fs.get('medium', 0):,} "
            f"large={fs.get('large', 0):,}"
        )

        print(
            "  SMOKE size  : "
            f"small={ss.get('small', 0):,} "
            f"medium={ss.get('medium', 0):,} "
            f"large={ss.get('large', 0):,}"
        )

    print()
    print("SPLIT ISOLATION")
    print("-" * 92)

    print(
        "train ∩ val :",
        overlaps["train_val"]
    )

    print(
        "train ∩ test:",
        overlaps["train_test"]
    )

    print(
        "val ∩ test  :",
        overlaps["val_test"]
    )

    print()
    print(
        "TOTAL:",
        f"{report['total_images']:,}"
    )

    print()
    print(
        "RESULT: FASDD CLEAN V1 FROZEN"
    )

    print()
    print(
        "Output:",
        OUT
    )

    print("=" * 92)


if __name__ == "__main__":
    main()
