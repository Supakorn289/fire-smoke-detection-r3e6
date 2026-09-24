#!/usr/bin/env python3

from pathlib import Path
from collections import Counter
import csv
import json
import os
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "sources"
    / "fasdd"
    / "FASDD_CV"
)

SOURCE_IMAGES = SOURCE / "images"

SOURCE_YOLO = (
    SOURCE
    / "annotations"
    / "YOLO_CV"
)

SOURCE_LABELS = SOURCE_YOLO / "labels"


DEST = Path(
    "/path/to/fire_datasets/fasdd/curated_v1"
)

DEST_IMAGES = DEST / "images"

DEST_YOLO = (
    DEST
    / "annotations"
    / "YOLO_CV"
)

DEST_LABELS = DEST_YOLO / "labels"


REPORT_DIR = (
    ROOT
    / "reports"
    / "fasdd"
)

REPORT_JSON = (
    REPORT_DIR
    / "sanitize_v1_report.json"
)

REPORT_CSV = (
    REPORT_DIR
    / "sanitize_v1_changes.csv"
)


EPS = 1e-12


def read_split(path: Path):

    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    return {
        Path(x.strip()).stem
        for x in lines
        if x.strip()
    }


def format_box(cls, xc, yc, w, h):

    return (
        f"{cls} "
        f"{xc:.10f} "
        f"{yc:.10f} "
        f"{w:.10f} "
        f"{h:.10f}"
    )


def main():

    print("=" * 82)
    print("FASDD CURATED V1 SANITIZER")
    print("=" * 82)

    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    if not SOURCE_LABELS.exists():
        raise FileNotFoundError(SOURCE_LABELS)

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Destination protection
    # --------------------------------------------------------

    if DEST.exists():

        print()
        print(
            "ERROR: destination already exists:"
        )

        print(
            DEST
        )

        print()
        print(
            "Refusing to overwrite curated dataset."
        )

        sys.exit(1)

    DEST_LABELS.mkdir(
        parents=True,
        exist_ok=False,
    )

    # --------------------------------------------------------
    # Images: symlink instead of 95k-file duplication
    # --------------------------------------------------------

    os.symlink(
        SOURCE_IMAGES.resolve(),
        DEST_IMAGES,
        target_is_directory=True,
    )

    # --------------------------------------------------------
    # Split metadata
    # --------------------------------------------------------

    split_sets = {}

    for split in (
        "train",
        "val",
        "test",
    ):

        src = (
            SOURCE_YOLO
            / f"{split}.txt"
        )

        dst = (
            DEST_YOLO
            / f"{split}.txt"
        )

        shutil.copy2(
            src,
            dst,
        )

        split_sets[
            split
        ] = read_split(src)

    stem_to_split = {}

    for split, stems in split_sets.items():

        for stem in stems:

            if stem in stem_to_split:

                raise RuntimeError(
                    f"Split overlap: {stem}"
                )

            stem_to_split[
                stem
            ] = split

    stats = Counter()

    changes = []

    source_labels = sorted(
        SOURCE_LABELS.glob("*.txt")
    )

    total = len(source_labels)

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for index, src_label in enumerate(
        source_labels,
        start=1,
    ):

        stem = src_label.stem

        split = stem_to_split.get(stem)

        if split is None:

            raise RuntimeError(
                f"No split for: {stem}"
            )

        dst_label = (
            DEST_LABELS
            / src_label.name
        )

        # ====================================================
        # TEST MUST STAY BYTE-IDENTICAL
        # ====================================================

        if split == "test":

            shutil.copy2(
                src_label,
                dst_label,
            )

            stats[
                "test_labels_exact_copy"
            ] += 1

            continue

        # ====================================================
        # TRAIN / VAL SANITIZATION
        # ====================================================

        original_text = src_label.read_text(
            encoding="utf-8",
            errors="replace",
        )

        original_lines = (
            original_text.splitlines()
        )

        output_lines = []

        file_modified = False

        for line_no, line in enumerate(
            original_lines,
            start=1,
        ):

            stripped = line.strip()

            if not stripped:

                continue

            parts = stripped.split()

            if len(parts) != 5:

                raise RuntimeError(
                    f"Malformed annotation: "
                    f"{src_label}:{line_no}"
                )

            cls = int(parts[0])

            xc, yc, w, h = map(
                float,
                parts[1:],
            )

            stats[
                f"{split}_boxes_input"
            ] += 1

            # ------------------------------------------------
            # Drop zero-area boxes
            # ------------------------------------------------

            if w <= 0 or h <= 0:

                stats[
                    f"{split}_zero_area_dropped"
                ] += 1

                stats[
                    "zero_area_dropped"
                ] += 1

                file_modified = True

                changes.append({
                    "stem":
                        stem,

                    "split":
                        split,

                    "line":
                        line_no,

                    "action":
                        "drop_zero_area",

                    "before":
                        stripped,

                    "after":
                        "",
                })

                continue

            # ------------------------------------------------
            # XYWH -> corners
            # ------------------------------------------------

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            outside = (
                x1 < -EPS
                or y1 < -EPS
                or x2 > 1 + EPS
                or y2 > 1 + EPS
            )

            if not outside:

                # Preserve original annotation exactly
                output_lines.append(line)

                stats[
                    f"{split}_boxes_unchanged"
                ] += 1

                continue

            # ------------------------------------------------
            # Clip to legal image boundary
            # ------------------------------------------------

            cx1 = min(
                1.0,
                max(0.0, x1),
            )

            cy1 = min(
                1.0,
                max(0.0, y1),
            )

            cx2 = min(
                1.0,
                max(0.0, x2),
            )

            cy2 = min(
                1.0,
                max(0.0, y2),
            )

            new_w = cx2 - cx1
            new_h = cy2 - cy1

            # Defensive check:
            # clipping must not create zero-area boxes
            if new_w <= 0 or new_h <= 0:

                stats[
                    "post_clip_zero_area_dropped"
                ] += 1

                stats[
                    f"{split}_post_clip_zero_area_dropped"
                ] += 1

                file_modified = True

                changes.append({
                    "stem":
                        stem,

                    "split":
                        split,

                    "line":
                        line_no,

                    "action":
                        "drop_after_clip",

                    "before":
                        stripped,

                    "after":
                        "",
                })

                continue

            new_xc = (
                cx1 + cx2
            ) / 2

            new_yc = (
                cy1 + cy2
            ) / 2

            new_line = format_box(
                cls,
                new_xc,
                new_yc,
                new_w,
                new_h,
            )

            output_lines.append(
                new_line
            )

            stats[
                "boxes_clipped"
            ] += 1

            stats[
                f"{split}_boxes_clipped"
            ] += 1

            file_modified = True

            changes.append({
                "stem":
                    stem,

                "split":
                    split,

                "line":
                    line_no,

                "action":
                    "clip_boundary",

                "before":
                    stripped,

                "after":
                    new_line,
            })

        # ----------------------------------------------------
        # Write sanitized label
        # ----------------------------------------------------

        if output_lines:

            dst_label.write_text(
                "\n".join(output_lines)
                + "\n",
                encoding="utf-8",
            )

        else:

            # Negative label must remain a zero-byte file
            dst_label.write_bytes(b"")

        if file_modified:

            stats[
                "modified_label_files"
            ] += 1

        else:

            stats[
                "unchanged_label_files"
            ] += 1

        stats[
            f"{split}_labels"
        ] += 1

        if (
            index % 10000 == 0
            or index == total
        ):

            print(
                f"\rProcessing "
                f"{index}/{total}",
                end="",
                flush=True,
            )

    print()

    # --------------------------------------------------------
    # Count output
    # --------------------------------------------------------

    output_labels = list(
        DEST_LABELS.glob("*.txt")
    )

    stats[
        "source_labels"
    ] = len(source_labels)

    stats[
        "output_labels"
    ] = len(output_labels)

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
                "stem",
                "split",
                "line",
                "action",
                "before",
                "after",
            ],
        )

        writer.writeheader()
        writer.writerows(
            changes
        )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    report = {
        "source":
            str(SOURCE.resolve()),

        "destination":
            str(DEST),

        "policy": {
            "train":
                "drop zero-area + clip boundary",

            "val":
                "drop zero-area + clip boundary",

            "test":
                "byte-identical official labels",
        },

        "stats":
            dict(stats),

        "changes":
            len(changes),
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
    print("=" * 82)
    print("SANITIZATION SUMMARY")
    print("=" * 82)

    print(
        "Source labels              :",
        stats["source_labels"],
    )

    print(
        "Output labels              :",
        stats["output_labels"],
    )

    print(
        "Modified label files       :",
        stats["modified_label_files"],
    )

    print(
        "Zero-area boxes dropped    :",
        stats["zero_area_dropped"],
    )

    print(
        "Boundary boxes clipped     :",
        stats["boxes_clipped"],
    )

    print(
        "Post-clip boxes dropped    :",
        stats[
            "post_clip_zero_area_dropped"
        ],
    )

    print()
    print("TRAIN")
    print(
        "  labels       :",
        stats["train_labels"],
    )
    print(
        "  boxes clipped:",
        stats["train_boxes_clipped"],
    )
    print(
        "  zero dropped :",
        stats["train_zero_area_dropped"],
    )

    print()
    print("VAL")
    print(
        "  labels       :",
        stats["val_labels"],
    )
    print(
        "  boxes clipped:",
        stats["val_boxes_clipped"],
    )
    print(
        "  zero dropped :",
        stats["val_zero_area_dropped"],
    )

    print()
    print("TEST")
    print(
        "  exact copies :",
        stats["test_labels_exact_copy"],
    )

    print()
    print(
        "Dataset:",
        DEST,
    )

    print(
        "Report :",
        REPORT_JSON,
    )

    print(
        "Changes:",
        REPORT_CSV,
    )

    print("=" * 82)


if __name__ == "__main__":
    main()
