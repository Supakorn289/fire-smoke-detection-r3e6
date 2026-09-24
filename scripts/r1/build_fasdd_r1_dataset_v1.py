#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os
import shutil


ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)

SOURCE_IMAGES = SOURCE_ROOT / "images"

SOURCE_LABELS = (
    SOURCE_ROOT
    / "annotations"
    / "YOLO_CV"
    / "labels"
)

TRAIN_MANIFEST = (
    ROOT
    / "reports"
    / "fasdd"
    / "r1_selection_v1"
    / "fasdd_r1_train_12000.txt"
)

VAL_MANIFEST = (
    ROOT
    / "reports"
    / "fasdd"
    / "r1_validation_v1"
    / "fasdd_r1_val_main_3000.txt"
)

SMALL_SMOKE_MANIFEST = (
    ROOT
    / "reports"
    / "fasdd"
    / "r1_validation_v1"
    / "fasdd_r1_val_small_smoke_222.txt"
)

CLEAN_TEST_MANIFEST = (
    ROOT
    / "reports"
    / "fasdd"
    / "clean_v1"
    / "test.txt"
)

TARGET = (
    ROOT
    / "datasets"
    / "fasdd_r1_v1"
)

REPORT = (
    ROOT
    / "reports"
    / "fasdd"
    / "r1_dataset_v1"
)

REPORT.mkdir(
    parents=True,
    exist_ok=True,
)


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def load_manifest(path):

    values = [
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    if len(values) != len(set(values)):
        raise RuntimeError(
            f"Duplicate stems in manifest: {path}"
        )

    return values


def sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def build_image_map():

    result = {}

    for path in SOURCE_IMAGES.iterdir():

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTS
        ):

            if path.stem in result:

                raise RuntimeError(
                    f"Duplicate image stem: "
                    f"{path.stem}"
                )

            result[path.stem] = path

    return result


def validate_label(path):

    text = path.read_text(
        encoding="utf-8",
        errors="strict",
    ).strip()

    if not text:

        return {
            "boxes": 0,
            "negative": True,
        }

    boxes = 0

    for line_no, line in enumerate(
        text.splitlines(),
        1,
    ):

        parts = line.split()

        if len(parts) != 5:

            raise RuntimeError(
                f"Malformed label "
                f"{path}:{line_no}"
            )

        cls = int(parts[0])

        if cls not in {0, 1}:

            raise RuntimeError(
                f"Invalid class "
                f"{cls} in {path}"
            )

        xc, yc, w, h = map(
            float,
            parts[1:],
        )

        if not (
            0.0 <= xc <= 1.0
            and
            0.0 <= yc <= 1.0
            and
            0.0 < w <= 1.0
            and
            0.0 < h <= 1.0
        ):

            raise RuntimeError(
                f"Invalid normalized bbox "
                f"in {path}:{line_no}"
            )

        boxes += 1

    return {
        "boxes": boxes,
        "negative": False,
    }


def make_link(src, dst):

    if not src.exists():

        raise FileNotFoundError(
            src
        )

    os.symlink(
        str(src),
        str(dst),
    )

    if not dst.exists():

        raise RuntimeError(
            f"Broken symlink: {dst}"
        )


def build_split(
    split_name,
    stems,
    image_map,
):

    images_dir = (
        TARGET
        / "images"
        / split_name
    )

    labels_dir = (
        TARGET
        / "labels"
        / split_name
    )

    images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats = {
        "images": 0,
        "labels": 0,
        "boxes": 0,
        "negative": 0,
    }

    for index, stem in enumerate(
        stems,
        1,
    ):

        image = image_map.get(
            stem
        )

        if image is None:

            raise FileNotFoundError(
                f"Missing image: {stem}"
            )

        label = (
            SOURCE_LABELS
            / f"{stem}.txt"
        )

        if not label.exists():

            raise FileNotFoundError(
                f"Missing label: {stem}"
            )

        label_info = validate_label(
            label
        )

        image_dst = (
            images_dir
            / image.name
        )

        label_dst = (
            labels_dir
            / label.name
        )

        make_link(
            image,
            image_dst,
        )

        make_link(
            label,
            label_dst,
        )

        stats["images"] += 1
        stats["labels"] += 1
        stats["boxes"] += (
            label_info["boxes"]
        )

        if label_info["negative"]:

            stats["negative"] += 1

        if (
            index % 2000 == 0
            or index == len(stems)
        ):

            print(
                f"\r{split_name:16} "
                f"{index}/{len(stems)} "
                f"({index/len(stems)*100:5.1f}%)",
                end="",
                flush=True,
            )

    print()

    return stats


def main():

    print("=" * 94)
    print("FASDD R1 DATASET V1 BUILDER")
    print("=" * 94)

    train = load_manifest(
        TRAIN_MANIFEST
    )

    val = load_manifest(
        VAL_MANIFEST
    )

    small_smoke = load_manifest(
        SMALL_SMOKE_MANIFEST
    )

    clean_test = set(
        load_manifest(
            CLEAN_TEST_MANIFEST
        )
    )

    if len(train) != 12000:

        raise RuntimeError(
            f"Expected 12,000 TRAIN, "
            f"got {len(train)}"
        )

    if len(val) != 3000:

        raise RuntimeError(
            f"Expected 3,000 VAL, "
            f"got {len(val)}"
        )

    if len(small_smoke) != 222:

        raise RuntimeError(
            f"Expected 222 diagnostic, "
            f"got {len(small_smoke)}"
        )

    train_set = set(train)
    val_set = set(val)
    small_smoke_set = set(
        small_smoke
    )

    if train_set & val_set:

        raise RuntimeError(
            "TRAIN overlaps MAIN VAL"
        )

    if train_set & clean_test:

        raise RuntimeError(
            "TRAIN overlaps TEST"
        )

    if val_set & clean_test:

        raise RuntimeError(
            "MAIN VAL overlaps TEST"
        )

    if not (
        small_smoke_set
        <= (
            set(
                load_manifest(
                    ROOT
                    / "reports"
                    / "fasdd"
                    / "clean_v1"
                    / "val.txt"
                )
            )
        )
    ):

        raise RuntimeError(
            "Small-smoke slice is not "
            "inside clean VAL"
        )

    # ========================================================
    # REBUILD GENERATED DATASET
    # ========================================================

    if TARGET.exists():

        expected_parent = (
            ROOT
            / "datasets"
        ).resolve()

        if TARGET.parent.resolve() != (
            expected_parent
        ):

            raise RuntimeError(
                "Safety check failed "
                "before removing target."
            )

        shutil.rmtree(
            TARGET
        )

    TARGET.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_map = build_image_map()

    print()
    print(
        "Source image stems:",
        f"{len(image_map):,}"
    )

    # ========================================================
    # BUILD
    # ========================================================

    print()
    train_stats = build_split(
        "train",
        train,
        image_map,
    )

    val_stats = build_split(
        "val",
        val,
        image_map,
    )

    diagnostic_stats = build_split(
        "val_small_smoke",
        small_smoke,
        image_map,
    )

    # ========================================================
    # DATA YAML — PRIMARY
    # ========================================================

    data_yaml = (
        f"path: {TARGET}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"  0: fire\n"
        f"  1: smoke\n"
    )

    (
        TARGET
        / "data.yaml"
    ).write_text(
        data_yaml,
        encoding="utf-8",
    )

    # ========================================================
    # DATA YAML — SMALL-SMOKE DIAGNOSTIC
    #
    # train is included for dataset-schema compatibility.
    # Validation uses val_small_smoke.
    # ========================================================

    diagnostic_yaml = (
        f"path: {TARGET}\n"
        f"train: images/train\n"
        f"val: images/val_small_smoke\n"
        f"\n"
        f"names:\n"
        f"  0: fire\n"
        f"  1: smoke\n"
    )

    (
        TARGET
        / "data_small_smoke.yaml"
    ).write_text(
        diagnostic_yaml,
        encoding="utf-8",
    )

    # ========================================================
    # FINAL FILE/LINK COUNTS
    # ========================================================

    expected_counts = {
        "train": 12000,
        "val": 3000,
        "val_small_smoke": 222,
    }

    link_validation = {}

    for split, expected in (
        expected_counts.items()
    ):

        image_dir = (
            TARGET
            / "images"
            / split
        )

        label_dir = (
            TARGET
            / "labels"
            / split
        )

        image_links = list(
            image_dir.iterdir()
        )

        label_links = list(
            label_dir.iterdir()
        )

        broken_images = [
            str(p)
            for p in image_links
            if (
                not p.is_symlink()
                or not p.exists()
            )
        ]

        broken_labels = [
            str(p)
            for p in label_links
            if (
                not p.is_symlink()
                or not p.exists()
            )
        ]

        image_stems = {
            p.stem
            for p in image_links
        }

        label_stems = {
            p.stem
            for p in label_links
        }

        passed = (
            len(image_links) == expected
            and
            len(label_links) == expected
            and
            not broken_images
            and
            not broken_labels
            and
            image_stems == label_stems
        )

        link_validation[
            split
        ] = {
            "expected":
                expected,

            "images":
                len(image_links),

            "labels":
                len(label_links),

            "broken_images":
                len(broken_images),

            "broken_labels":
                len(broken_labels),

            "stem_match":
                image_stems
                == label_stems,

            "pass":
                passed,
        }

    final_pass = all(
        x["pass"]
        for x in link_validation.values()
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "version":
            "FASDD_R1_DATASET_V1",

        "dataset_root":
            str(TARGET),

        "class_contract": {
            "0": "fire",
            "1": "smoke",
        },

        "primary": {
            "train":
                train_stats,

            "val":
                val_stats,
        },

        "diagnostic": {
            "small_smoke":
                diagnostic_stats,

            "overlap_with_main_val":
                len(
                    val_set
                    & small_smoke_set
                ),

            "used_for_model_selection":
                False,
        },

        "split_validation": {
            "train_val_overlap":
                len(
                    train_set
                    & val_set
                ),

            "train_test_overlap":
                len(
                    train_set
                    & clean_test
                ),

            "val_test_overlap":
                len(
                    val_set
                    & clean_test
                ),
        },

        "link_validation":
            link_validation,

        "manifest_sha256": {
            "train":
                sha256(
                    TRAIN_MANIFEST
                ),

            "val":
                sha256(
                    VAL_MANIFEST
                ),

            "small_smoke":
                sha256(
                    SMALL_SMOKE_MANIFEST
                ),
        },

        "test_policy": {
            "test_directory_created":
                False,

            "test_used":
                False,
        },

        "pass":
            final_pass,
    }

    report_path = (
        REPORT
        / "build_report.json"
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
    print("=" * 94)
    print("FASDD R1 DATASET V1 SUMMARY")
    print("=" * 94)

    for split in (
        "train",
        "val",
        "val_small_smoke",
    ):

        x = link_validation[
            split
        ]

        print(
            f"{split:18} "
            f"images={x['images']:5} "
            f"labels={x['labels']:5} "
            f"broken_img={x['broken_images']:2} "
            f"broken_lbl={x['broken_labels']:2} "
            f"stem_match="
            f"{'PASS' if x['stem_match'] else 'FAIL'}"
        )

    print()
    print("PRIMARY LABEL PROFILE")
    print("-" * 94)

    print(
        "TRAIN boxes     :",
        f"{train_stats['boxes']:,}"
    )

    print(
        "TRAIN negatives :",
        f"{train_stats['negative']:,}"
    )

    print(
        "VAL boxes       :",
        f"{val_stats['boxes']:,}"
    )

    print(
        "VAL negatives   :",
        f"{val_stats['negative']:,}"
    )

    print()
    print("SPLIT GUARD")
    print("-" * 94)

    print(
        "TRAIN ∩ VAL :",
        len(
            train_set
            & val_set
        )
    )

    print(
        "TRAIN ∩ TEST:",
        len(
            train_set
            & clean_test
        )
    )

    print(
        "VAL ∩ TEST  :",
        len(
            val_set
            & clean_test
        )
    )

    print(
        "Diagnostic ∩ Main VAL:",
        len(
            val_set
            & small_smoke_set
        )
    )

    print()
    print(
        "TEST directory created: NO"
    )

    print()
    print(
        "data.yaml:",
        TARGET
        / "data.yaml"
    )

    print(
        "diagnostic:",
        TARGET
        / "data_small_smoke.yaml"
    )

    print()
    print(
        "RESULT:",
        (
            "FASDD R1 DATASET V1 PASS"
            if final_pass
            else
            "FASDD R1 DATASET V1 FAIL"
        )
    )

    print("=" * 94)


if __name__ == "__main__":
    main()
