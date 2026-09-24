#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import json


ROOT = Path(__file__).resolve().parents[2]

DATA = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)

IMAGES = DATA / "images"

YOLO = DATA / "annotations" / "YOLO_CV"

REPORT_DIR = ROOT / "reports" / "fasdd"

REPORT_JSON = REPORT_DIR / "exact_duplicates.json"
REPORT_CSV = REPORT_DIR / "exact_duplicates.csv"

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def read_split(split):

    path = YOLO / f"{split}.txt"

    return {
        Path(x.strip()).stem
        for x in path.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        if x.strip()
    }


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                4 * 1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 84)
    print("FASDD EXACT DUPLICATE DETECTOR")
    print("=" * 84)

    splits = {
        split: read_split(split)
        for split in (
            "train",
            "val",
            "test",
        )
    }

    split_lookup = {}

    for split, stems in splits.items():

        for stem in stems:

            if stem in split_lookup:
                raise RuntimeError(
                    f"Split overlap: {stem}"
                )

            split_lookup[stem] = split

    images = sorted(
        p
        for p in IMAGES.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTS
        )
    )

    print("Images:", len(images))

    # --------------------------------------------------------
    # Stage 1:
    # group by file size first.
    #
    # Exact duplicate files MUST have identical byte size,
    # so files with unique sizes do not need hashing.
    # --------------------------------------------------------

    size_groups = defaultdict(list)

    for p in images:

        size_groups[
            p.stat().st_size
        ].append(p)

    candidate_groups = {
        size: paths
        for size, paths
        in size_groups.items()
        if len(paths) > 1
    }

    candidate_images = sum(
        len(paths)
        for paths
        in candidate_groups.values()
    )

    unique_size_images = (
        len(images)
        - candidate_images
    )

    print(
        "Unique-size images :",
        unique_size_images,
    )

    print(
        "Hash candidates    :",
        candidate_images,
    )

    # --------------------------------------------------------
    # Stage 2:
    # SHA-256 only within repeated-size groups
    # --------------------------------------------------------

    hashes = defaultdict(list)

    done = 0

    for size, paths in sorted(
        candidate_groups.items()
    ):

        for p in paths:

            digest = sha256(p)

            hashes[
                (size, digest)
            ].append(p)

            done += 1

            if (
                done % 5000 == 0
                or done == candidate_images
            ):

                pct = (
                    done
                    / candidate_images
                    * 100
                    if candidate_images
                    else 100
                )

                print(
                    f"\rHashing "
                    f"{done}/{candidate_images} "
                    f"({pct:5.1f}%)",
                    end="",
                    flush=True,
                )

    print()

    # --------------------------------------------------------
    # Duplicate groups
    # --------------------------------------------------------

    duplicate_groups = [
        paths
        for paths
        in hashes.values()
        if len(paths) > 1
    ]

    duplicate_groups.sort(
        key=lambda x: (
            -len(x),
            x[0].name,
        )
    )

    stats = Counter()

    rows = []

    group_records = []

    for group_id, paths in enumerate(
        duplicate_groups,
        start=1,
    ):

        members = []

        group_splits = set()

        for p in paths:

            stem = p.stem

            split = split_lookup.get(
                stem,
                "unknown",
            )

            group_splits.add(split)

            members.append({
                "stem": stem,
                "split": split,
                "path": str(p),
            })

        split_signature = "+".join(
            sorted(group_splits)
        )

        stats["duplicate_groups"] += 1
        stats["duplicate_images"] += len(paths)
        stats["redundant_images"] += (
            len(paths) - 1
        )

        if len(group_splits) == 1:

            stats[
                "same_split_groups"
            ] += 1

        else:

            stats[
                "cross_split_groups"
            ] += 1

        if (
            "train" in group_splits
            and "val" in group_splits
        ):

            stats[
                "train_val_leakage_groups"
            ] += 1

        if (
            "train" in group_splits
            and "test" in group_splits
        ):

            stats[
                "train_test_leakage_groups"
            ] += 1

        if (
            "val" in group_splits
            and "test" in group_splits
        ):

            stats[
                "val_test_leakage_groups"
            ] += 1

        group_records.append({
            "group_id":
                group_id,

            "count":
                len(paths),

            "splits":
                sorted(group_splits),

            "members":
                members,
        })

        for member in members:

            rows.append({
                "group_id":
                    group_id,

                "group_size":
                    len(paths),

                "group_splits":
                    split_signature,

                "stem":
                    member["stem"],

                "split":
                    member["split"],

                "path":
                    member["path"],
            })

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with REPORT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group_id",
                "group_size",
                "group_splits",
                "stem",
                "split",
                "path",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    report = {
        "images":
            len(images),

        "unique_size_images":
            unique_size_images,

        "hash_candidates":
            candidate_images,

        "stats":
            dict(stats),

        "duplicate_groups":
            group_records,
    }

    REPORT_JSON.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 84)
    print("EXACT DUPLICATE SUMMARY")
    print("=" * 84)

    print(
        "Images                   :",
        len(images),
    )

    print(
        "Duplicate groups         :",
        stats["duplicate_groups"],
    )

    print(
        "Images inside dup groups :",
        stats["duplicate_images"],
    )

    print(
        "Redundant image copies   :",
        stats["redundant_images"],
    )

    print()
    print(
        "Same-split groups        :",
        stats["same_split_groups"],
    )

    print(
        "Cross-split groups       :",
        stats["cross_split_groups"],
    )

    print()
    print(
        "train ↔ val leakage      :",
        stats[
            "train_val_leakage_groups"
        ],
    )

    print(
        "train ↔ test leakage     :",
        stats[
            "train_test_leakage_groups"
        ],
    )

    print(
        "val ↔ test leakage       :",
        stats[
            "val_test_leakage_groups"
        ],
    )

    print()
    print(
        "JSON:",
        REPORT_JSON,
    )

    print(
        "CSV :",
        REPORT_CSV,
    )

    print("=" * 84)


if __name__ == "__main__":
    main()
