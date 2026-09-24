#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os
import shutil

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "datasets/internet_proxy_negative_v1/"
      "extracted"
)

OUT = (
    ROOT
    / "datasets/internet_proxy_negative_v1/"
      "raw"
)

REPORT = (
    ROOT
    / "reports/internet_proxy_negative_v1"
)

CAMERAS = [
    "858",
    "3888",
    "4795",
    "5021",
    "1093",
    "3297",
    "4232",
    "19106",
    "4801",
    "21444",
]

BASE_PER_CAMERA = 200
TARGET_TOTAL = 2000

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def valid_image(path):

    try:
        with Image.open(path) as im:
            im.verify()

        return True

    except Exception:
        return False


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def evenly_sample(items, count):

    if count <= 0:
        return []

    if len(items) < count:

        raise RuntimeError(
            f"Requested {count}, "
            f"but only {len(items)} "
            "items are available"
        )

    if count == 1:

        return [
            items[
                len(items) // 2
            ]
        ]

    indices = [
        round(
            i
            * (len(items) - 1)
            / (count - 1)
        )
        for i in range(count)
    ]

    # Protect against rounding duplicates.
    unique_indices = []

    seen = set()

    for idx in indices:

        if idx not in seen:

            unique_indices.append(
                idx
            )

            seen.add(
                idx
            )

    # If rounding produced fewer indices,
    # fill deterministically.
    if len(unique_indices) < count:

        for idx in range(
            len(items)
        ):

            if idx in seen:
                continue

            unique_indices.append(
                idx
            )

            seen.add(
                idx
            )

            if (
                len(unique_indices)
                == count
            ):
                break

    unique_indices.sort()

    return [
        items[idx]
        for idx in unique_indices
    ]


# ============================================================
# RESET OUTPUT
# ============================================================

if OUT.exists():
    shutil.rmtree(
        OUT
    )

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 96)
print(
    "BUILD INTERNET PROXY NEGATIVE V1"
)
print(
    "ADAPTIVE CAMERA SAMPLING"
)
print("=" * 96)


# ============================================================
# SCAN ALL CAMERAS
# ============================================================

camera_images = {}

invalid_count = 0


for camera in CAMERAS:

    camera_root = (
        SOURCE / camera
    )

    if not camera_root.exists():

        raise FileNotFoundError(
            camera_root
        )


    candidates = sorted(
        [
            p
            for p in camera_root.rglob("*")
            if (
                p.is_file()
                and
                p.suffix.lower()
                in IMAGE_EXTS
            )
        ],
        key=lambda p:
            str(p),
    )


    valid = []

    for image in candidates:

        if valid_image(image):

            valid.append(
                image.resolve()
            )

        else:

            invalid_count += 1


    camera_images[
        camera
    ] = valid


    print(
        f"Camera {camera:>5}: "
        f"{len(candidates):5d} files | "
        f"{len(valid):5d} valid"
    )


# ============================================================
# INITIAL QUOTA
#
# Each camera gets up to 200.
# Cameras with fewer images contribute all available.
# ============================================================

quota = {
    camera:
        min(
            BASE_PER_CAMERA,
            len(
                camera_images[
                    camera
                ]
            ),
        )
    for camera in CAMERAS
}


initial_total = sum(
    quota.values()
)


if initial_total > TARGET_TOTAL:

    raise RuntimeError(
        "Initial quota unexpectedly "
        "exceeds target"
    )


deficit = (
    TARGET_TOTAL
    - initial_total
)


print()
print(
    "Initial quota total:",
    initial_total
)

print(
    "Need redistribution:",
    deficit
)


# ============================================================
# REDISTRIBUTE SHORTFALL
#
# Round-robin across cameras that have
# remaining unused images.
# ============================================================

while deficit > 0:

    progress = False

    for camera in CAMERAS:

        capacity = len(
            camera_images[
                camera
            ]
        )

        if quota[
            camera
        ] >= capacity:

            continue


        quota[
            camera
        ] += 1

        deficit -= 1

        progress = True


        if deficit == 0:
            break


    if not progress:

        raise RuntimeError(
            "Not enough valid images "
            "to reach 2,000 total"
        )


if sum(
    quota.values()
) != TARGET_TOTAL:

    raise RuntimeError(
        "Quota allocation failed"
    )


print()
print("=" * 96)
print(
    "FINAL CAMERA QUOTAS"
)
print("=" * 96)


for camera in CAMERAS:

    print(
        f"Camera {camera:>5}: "
        f"{quota[camera]:4d}"
    )


print("-" * 96)

print(
    "Total:",
    sum(
        quota.values()
    )
)


# ============================================================
# SELECT IMAGES
# ============================================================

selected_by_camera = {}

for camera in CAMERAS:

    selected_by_camera[
        camera
    ] = evenly_sample(
        camera_images[
            camera
        ],
        quota[
            camera
        ],
    )


# ============================================================
# GLOBAL EXACT-DUPLICATE FILTER
# ============================================================

manifest = []

hashes = set()

used_paths = set()

duplicate_count = 0


for camera in CAMERAS:

    accepted_index = 0

    for source in selected_by_camera[
        camera
    ]:

        digest = sha256(
            source
        )

        if digest in hashes:

            duplicate_count += 1
            continue


        hashes.add(
            digest
        )

        used_paths.add(
            source
        )


        suffix = (
            source.suffix.lower()
        )


        new_name = (
            f"skyfinder_{camera}_"
            f"{accepted_index:04d}"
            f"{suffix}"
        )


        destination = (
            OUT
            / new_name
        )


        os.symlink(
            source,
            destination,
        )


        manifest.append({
            "image":
                str(
                    destination.resolve()
                ),

            "proxy_path":
                str(destination),

            "source":
                "SkyFinder",

            "camera_id":
                camera,

            "sha256":
                digest,

            "initial_label":
                "UNVERIFIED_PROXY_NEGATIVE",
        })


        accepted_index += 1


# ============================================================
# TOP-UP IF EXACT DUPLICATES CAUSED SHORTFALL
# ============================================================

shortfall = (
    TARGET_TOTAL
    - len(manifest)
)


if shortfall > 0:

    print()
    print(
        "Exact duplicates caused "
        f"a shortfall of {shortfall}."
    )

    print(
        "Selecting replacement "
        "images automatically..."
    )


    while shortfall > 0:

        progress = False

        for camera in CAMERAS:

            for source in camera_images[
                camera
            ]:

                if source in used_paths:
                    continue


                digest = sha256(
                    source
                )

                used_paths.add(
                    source
                )


                if digest in hashes:

                    duplicate_count += 1
                    continue


                hashes.add(
                    digest
                )


                current_count = sum(
                    1
                    for row in manifest
                    if (
                        row[
                            "camera_id"
                        ]
                        == camera
                    )
                )


                suffix = (
                    source.suffix.lower()
                )


                new_name = (
                    f"skyfinder_{camera}_"
                    f"{current_count:04d}"
                    f"{suffix}"
                )


                destination = (
                    OUT
                    / new_name
                )


                os.symlink(
                    source,
                    destination,
                )


                manifest.append({
                    "image":
                        str(
                            destination.resolve()
                        ),

                    "proxy_path":
                        str(
                            destination
                        ),

                    "source":
                        "SkyFinder",

                    "camera_id":
                        camera,

                    "sha256":
                        digest,

                    "initial_label":
                        "UNVERIFIED_PROXY_NEGATIVE",
                })


                shortfall -= 1
                progress = True

                break


            if shortfall == 0:
                break


        if not progress:

            raise RuntimeError(
                "Unable to replace "
                "duplicate images"
            )


# ============================================================
# FINAL VALIDATION
# ============================================================

if len(manifest) != TARGET_TOTAL:

    raise RuntimeError(
        f"Expected {TARGET_TOTAL} "
        f"unique images, "
        f"got {len(manifest)}"
    )


if (
    len(
        {
            row["sha256"]
            for row in manifest
        }
    )
    != TARGET_TOTAL
):

    raise RuntimeError(
        "Exact duplicate remained "
        "after filtering"
    )


# ============================================================
# MANIFEST TXT
# ============================================================

manifest_txt = (
    REPORT
    / "internet_proxy_v1.txt"
)


manifest_txt.write_text(
    "\n".join(
        row["image"]
        for row in manifest
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# METADATA JSON
# ============================================================

manifest_json = (
    REPORT
    / "internet_proxy_v1.json"
)


manifest_json.write_text(
    json.dumps(
        manifest,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# SUMMARY
# ============================================================

final_counts = {
    camera:
        sum(
            1
            for row in manifest
            if (
                row[
                    "camera_id"
                ]
                == camera
            )
        )
    for camera in CAMERAS
}


summary = {
    "source":
        "SkyFinder",

    "role":
        "internet_proxy_negative_pool",

    "ground_truth_status":
        "UNVERIFIED_PROXY_NEGATIVE",

    "internet_only":
        True,

    "camera_count":
        len(CAMERAS),

    "target_total":
        TARGET_TOTAL,

    "actual_total":
        len(manifest),

    "camera_counts":
        final_counts,

    "exact_duplicates_skipped":
        duplicate_count,

    "invalid_images":
        invalid_count,
}


summary_path = (
    REPORT
    / "summary.json"
)


summary_path.write_text(
    json.dumps(
        summary,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# PRINT FINAL REPORT
# ============================================================

print()
print("=" * 96)
print(
    "INTERNET PROXY V1 SUMMARY"
)
print("=" * 96)


for camera in CAMERAS:

    print(
        f"Camera {camera:>5}: "
        f"{final_counts[camera]:4d}"
    )


print("-" * 96)

print(
    "Total unique images    :",
    len(manifest)
)

print(
    "Exact duplicates skipped:",
    duplicate_count
)

print(
    "Invalid images         :",
    invalid_count
)

print()

print(
    "Manifest:",
    manifest_txt
)

print(
    "Metadata:",
    manifest_json
)

print(
    "Summary :",
    summary_path
)

print()

print(
    "RESULT: INTERNET PROXY "
    "NEGATIVE V1 BUILD PASS"
)

print("=" * 96)
