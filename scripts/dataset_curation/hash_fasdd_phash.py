#!/usr/bin/env python3

from pathlib import Path
import csv
import json

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

DATA = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)

IMAGES = DATA / "images"

YOLO = DATA / "annotations" / "YOLO_CV"

DEDUP = (
    ROOT
    / "reports"
    / "fasdd"
    / "dedup_exact"
)

OUT = (
    ROOT
    / "reports"
    / "fasdd"
    / "near_duplicates"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

HASH_CSV = OUT / "phash.csv"
SUMMARY_JSON = OUT / "phash_summary.json"


IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def load_manifest(name):

    path = DEDUP / name

    return {
        x.strip()
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    }


def phash64(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    small = cv2.resize(
        gray,
        (32, 32),
        interpolation=cv2.INTER_AREA,
    )

    small = np.float32(
        small
    )

    dct = cv2.dct(
        small
    )

    low = dct[:8, :8].copy()

    values = low.flatten()

    # Ignore DC coefficient for median
    median = np.median(
        values[1:]
    )

    bits = (
        values > median
    )

    value = 0

    for bit in bits:

        value = (
            (value << 1)
            | int(bit)
        )

    return value


def dhash64(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    small = cv2.resize(
        gray,
        (9, 8),
        interpolation=cv2.INTER_AREA,
    )

    diff = (
        small[:, 1:]
        >
        small[:, :-1]
    ).flatten()

    value = 0

    for bit in diff:

        value = (
            (value << 1)
            | int(bit)
        )

    return value


def main():

    print("=" * 84)
    print("FASDD PERCEPTUAL HASH GENERATOR")
    print("=" * 84)

    clean = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        clean[split] = load_manifest(
            f"{split}_clean_exact_stems.txt"
        )

    split_lookup = {
        stem: split
        for split, stems
        in clean.items()
        for stem in stems
    }

    image_map = {
        p.stem: p
        for p in IMAGES.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTS
        )
    }

    stems = sorted(
        split_lookup
    )

    print(
        "Exact-clean images:",
        len(stems)
    )

    rows = []

    failures = []

    cv2.setNumThreads(0)

    for i, stem in enumerate(
        stems,
        1,
    ):

        path = image_map.get(stem)

        if path is None:

            failures.append(stem)
            continue

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            failures.append(stem)
            continue

        h, w = image.shape[:2]

        p = phash64(image)
        d = dhash64(image)

        rows.append({
            "stem":
                stem,

            "split":
                split_lookup[stem],

            "width":
                w,

            "height":
                h,

            "aspect":
                f"{w / h:.8f}",

            "phash":
                f"{p:016x}",

            "dhash":
                f"{d:016x}",
        })

        if (
            i % 5000 == 0
            or i == len(stems)
        ):

            print(
                f"\rHashing "
                f"{i}/{len(stems)} "
                f"({i/len(stems)*100:5.1f}%)",
                end="",
                flush=True,
            )

    print()

    with HASH_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem",
                "split",
                "width",
                "height",
                "aspect",
                "phash",
                "dhash",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "exact_clean_images":
            len(stems),

        "hashed":
            len(rows),

        "failures":
            failures,

        "hashes": {
            "phash": "64-bit DCT perceptual hash",
            "dhash": "64-bit difference hash",
        },
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
    print("=" * 84)

    print(
        "Hashed  :",
        len(rows)
    )

    print(
        "Failures:",
        len(failures)
    )

    print(
        "CSV     :",
        HASH_CSV
    )

    print("=" * 84)


if __name__ == "__main__":
    main()
