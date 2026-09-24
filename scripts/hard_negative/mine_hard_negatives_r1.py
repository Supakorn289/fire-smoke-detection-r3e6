#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv
import gc
import json
import os

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

POOL = (
    ROOT
    / "reports/negative_v1/pool/"
    "fasdd_unseen_negative_train.txt"
)

OUT = (
    ROOT
    / "reports/negative_v1/mining"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = {
    "n": (
        ROOT
        / "runs/teachers_r1/"
        "teacher_n_r1/weights/best.pt"
    ),
    "s": (
        ROOT
        / "runs/teachers_r1/"
        "teacher_s_r1/weights/best.pt"
    ),
    "m": (
        ROOT
        / "runs/teachers_r1/"
        "teacher_m_r1/weights/best.pt"
    ),
}

EXPECTED_POOL_SIZE = 17510

PRED_CONF_FLOOR = 0.05

IMAGE_FIELDS = [
    "stem",
    "image",
    "fire_conf",
    "smoke_conf",
    "top_conf",
    "top_class",
    "fire_count",
    "smoke_count",
]


def load_pool():
    if not POOL.exists():
        raise FileNotFoundError(
            f"Negative pool missing: {POOL}"
        )

    paths = []

    seen = set()

    for raw in POOL.read_text(
        encoding="utf-8"
    ).splitlines():

        raw = raw.strip()

        if not raw:
            continue

        p = Path(raw)

        if not p.exists():
            raise FileNotFoundError(
                f"Pool image missing: {p}"
            )

        if p.stem in seen:
            raise RuntimeError(
                f"Duplicate pool stem: {p.stem}"
            )

        seen.add(p.stem)
        paths.append(p)

    if len(paths) != EXPECTED_POOL_SIZE:
        raise RuntimeError(
            f"Expected {EXPECTED_POOL_SIZE} "
            f"negative images, got {len(paths)}"
        )

    return paths


def prediction_summary(result):
    fire_conf = 0.0
    smoke_conf = 0.0

    fire_count = 0
    smoke_count = 0

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):
        confs = (
            result.boxes.conf
            .detach()
            .cpu()
            .tolist()
        )

        classes = (
            result.boxes.cls
            .detach()
            .cpu()
            .tolist()
        )

        for conf, cls in zip(
            confs,
            classes,
        ):
            conf = float(conf)
            cls = int(cls)

            if cls == 0:
                fire_count += 1
                fire_conf = max(
                    fire_conf,
                    conf,
                )

            elif cls == 1:
                smoke_count += 1
                smoke_conf = max(
                    smoke_conf,
                    conf,
                )

    if (
        fire_conf == 0.0
        and smoke_conf == 0.0
    ):
        top_class = "none"
        top_conf = 0.0

    elif fire_conf >= smoke_conf:
        top_class = "fire"
        top_conf = fire_conf

    else:
        top_class = "smoke"
        top_conf = smoke_conf

    return {
        "fire_conf": fire_conf,
        "smoke_conf": smoke_conf,
        "top_conf": top_conf,
        "top_class": top_class,
        "fire_count": fire_count,
        "smoke_count": smoke_count,
    }


def checkpoint_path(model_name):
    return (
        OUT
        / f"teacher_{model_name}_negative_predictions.csv"
    )


def checkpoint_valid(
    model_name,
    paths,
):
    path = checkpoint_path(
        model_name
    )

    if not path.exists():
        return False

    try:
        with path.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            reader = csv.DictReader(f)

            rows = list(reader)

        if len(rows) != len(paths):
            return False

        expected = {
            p.stem
            for p in paths
        }

        actual = {
            row["stem"]
            for row in rows
        }

        return expected == actual

    except Exception:
        return False


def run_teacher(
    model_name,
    paths,
    batch,
    chunk,
):
    final_path = checkpoint_path(
        model_name
    )

    if checkpoint_valid(
        model_name,
        paths,
    ):
        print()
        print(
            f"Teacher {model_name.upper()} "
            "checkpoint already complete."
        )
        print(
            "Skipping inference:",
            final_path,
        )
        return

    temp_path = final_path.with_suffix(
        ".tmp"
    )

    if temp_path.exists():
        temp_path.unlink()

    model_path = MODELS[
        model_name
    ]

    if not model_path.exists():
        raise FileNotFoundError(
            model_path
        )

    print()
    print("=" * 100)
    print(
        f"MINING TEACHER "
        f"{model_name.upper()}"
    )
    print("=" * 100)

    print(
        "Model:",
        model_path
    )

    print(
        "Images:",
        len(paths)
    )

    print(
        "Batch:",
        batch
    )

    print(
        "Chunk:",
        chunk
    )

    print(
        "Prediction floor:",
        PRED_CONF_FLOOR
    )

    model = YOLO(
        str(model_path)
    )

    completed = 0

    with temp_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=IMAGE_FIELDS,
        )

        writer.writeheader()

        for start in range(
            0,
            len(paths),
            chunk,
        ):
            current_paths = paths[
                start:
                start + chunk
            ]

            with torch.inference_mode():
                results = model.predict(
                    source=[
                        str(p)
                        for p in current_paths
                    ],
                    imgsz=768,
                    conf=PRED_CONF_FLOOR,
                    iou=0.70,
                    max_det=100,
                    device=0,
                    batch=min(
                        batch,
                        len(current_paths),
                    ),
                    stream=False,
                    verbose=False,
                )

            if (
                len(results)
                != len(current_paths)
            ):
                raise RuntimeError(
                    "Prediction result count "
                    "does not match input count"
                )

            for image_path, result in zip(
                current_paths,
                results,
            ):
                summary = (
                    prediction_summary(
                        result
                    )
                )

                writer.writerow({
                    "stem":
                        image_path.stem,

                    "image":
                        str(image_path),

                    **summary,
                })

                completed += 1

            f.flush()

            del results
            del current_paths

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (
                completed % 250 == 0
                or
                completed == len(paths)
            ):
                print(
                    f"\rTeacher "
                    f"{model_name.upper()} "
                    f"{completed}/"
                    f"{len(paths)} "
                    f"("
                    f"{completed/len(paths)*100:5.1f}%"
                    f")",
                    end="",
                    flush=True,
                )

    print()

    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.replace(
        temp_path,
        final_path,
    )

    if not checkpoint_valid(
        model_name,
        paths,
    ):
        raise RuntimeError(
            f"Checkpoint validation failed: "
            f"{final_path}"
        )

    print(
        f"Teacher {model_name.upper()} "
        "checkpoint PASS"
    )

    print(
        final_path
    )


def load_teacher_predictions(
    model_name,
):
    path = checkpoint_path(
        model_name
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    records = {}

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            stem = row["stem"]

            records[stem] = {
                "image":
                    row["image"],

                "fire_conf":
                    float(
                        row["fire_conf"]
                    ),

                "smoke_conf":
                    float(
                        row["smoke_conf"]
                    ),

                "top_conf":
                    float(
                        row["top_conf"]
                    ),

                "top_class":
                    row["top_class"],

                "fire_count":
                    int(
                        row["fire_count"]
                    ),

                "smoke_count":
                    int(
                        row["smoke_count"]
                    ),
            }

    return records


def classify_record(record):
    confs = [
        record["n_top_conf"],
        record["s_top_conf"],
        record["m_top_conf"],
    ]

    votes_010 = sum(
        conf >= 0.10
        for conf in confs
    )

    votes_025 = sum(
        conf >= 0.25
        for conf in confs
    )

    votes_050 = sum(
        conf >= 0.50
        for conf in confs
    )

    max_conf = max(
        confs
    )

    mean_conf = (
        sum(confs)
        / 3.0
    )

    fire_votes = 0
    smoke_votes = 0

    for name in (
        "n",
        "s",
        "m",
    ):
        if (
            record[
                f"{name}_top_conf"
            ] < 0.25
        ):
            continue

        cls = record[
            f"{name}_top_class"
        ]

        if cls == "fire":
            fire_votes += 1

        elif cls == "smoke":
            smoke_votes += 1

    if (
        fire_votes == 0
        and smoke_votes == 0
    ):
        consensus_class = "none"

    elif fire_votes > smoke_votes:
        consensus_class = "fire"

    elif smoke_votes > fire_votes:
        consensus_class = "smoke"

    else:
        consensus_class = "mixed"

    # --------------------------------------------------------
    # Severity rules
    # --------------------------------------------------------

    if (
        votes_025 == 3
        or votes_050 >= 2
    ):
        severity = "EXTREME"

    elif (
        votes_025 >= 2
        or max_conf >= 0.75
    ):
        severity = "HARD"

    elif (
        votes_025 == 1
        or votes_010 >= 1
    ):
        severity = "MODERATE"

    else:
        severity = "EASY"

    # Ranking only.
    hardness_score = (
        0.30 * mean_conf
        + 0.25 * (
            votes_025 / 3.0
        )
        + 0.25 * (
            votes_050 / 3.0
        )
        + 0.20 * max_conf
    )

    record[
        "votes_conf_010"
    ] = votes_010

    record[
        "votes_conf_025"
    ] = votes_025

    record[
        "votes_conf_050"
    ] = votes_050

    record[
        "max_conf"
    ] = max_conf

    record[
        "mean_conf"
    ] = mean_conf

    record[
        "fire_votes_025"
    ] = fire_votes

    record[
        "smoke_votes_025"
    ] = smoke_votes

    record[
        "consensus_class"
    ] = consensus_class

    record[
        "severity"
    ] = severity

    record[
        "hardness_score"
    ] = hardness_score


def merge_results(paths):
    teacher_data = {
        name:
            load_teacher_predictions(
                name
            )
        for name in (
            "n",
            "s",
            "m",
        )
    }

    rows = []

    for image_path in paths:
        stem = image_path.stem

        row = {
            "stem":
                stem,

            "image":
                str(image_path),

            "source":
                "FASDD_CLEAN_V1_TRAIN",

            "category":
                "N00_fasdd_unclassified",
        }

        for name in (
            "n",
            "s",
            "m",
        ):
            data = teacher_data[
                name
            ][stem]

            row[
                f"{name}_fire_conf"
            ] = data[
                "fire_conf"
            ]

            row[
                f"{name}_smoke_conf"
            ] = data[
                "smoke_conf"
            ]

            row[
                f"{name}_top_conf"
            ] = data[
                "top_conf"
            ]

            row[
                f"{name}_top_class"
            ] = data[
                "top_class"
            ]

            row[
                f"{name}_fire_count"
            ] = data[
                "fire_count"
            ]

            row[
                f"{name}_smoke_count"
            ] = data[
                "smoke_count"
            ]

        classify_record(
            row
        )

        rows.append(
            row
        )

    rows.sort(
        key=lambda r:
            r[
                "hardness_score"
            ],
        reverse=True,
    )

    return rows


def save_final(rows):
    csv_path = (
        OUT
        / "fasdd_unseen_negative_mined.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    severities = [
        "EXTREME",
        "HARD",
        "MODERATE",
        "EASY",
    ]

    counts = {}

    for severity in severities:
        selected = [
            row
            for row in rows
            if (
                row["severity"]
                == severity
            )
        ]

        counts[
            severity
        ] = len(selected)

        manifest = (
            OUT
            / f"{severity.lower()}.txt"
        )

        text = ""

        if selected:
            text = (
                "\n".join(
                    row["image"]
                    for row in selected
                )
                + "\n"
            )

        manifest.write_text(
            text,
            encoding="utf-8",
        )

    summary = {
        "total":
            len(rows),

        "extreme":
            counts["EXTREME"],

        "hard":
            counts["HARD"],

        "moderate":
            counts["MODERATE"],

        "easy":
            counts["EASY"],

        "prediction_conf_floor":
            PRED_CONF_FLOOR,
    }

    summary_path = (
        OUT
        / "mining_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print(
        "HARD-NEGATIVE MINING SUMMARY"
    )
    print("=" * 100)

    print(
        "TOTAL    :",
        len(rows)
    )

    for severity in severities:
        print(
            f"{severity:9}:",
            counts[
                severity
            ]
        )

    print()
    print(
        "TOP 20 FALSE POSITIVE CANDIDATES"
    )
    print("-" * 100)

    for index, row in enumerate(
        rows[:20],
        start=1,
    ):
        print(
            f"{index:02d} "
            f"{row['severity']:8} "
            f"score="
            f"{row['hardness_score']:.4f} "
            f"N="
            f"{row['n_top_conf']:.3f} "
            f"S="
            f"{row['s_top_conf']:.3f} "
            f"M="
            f"{row['m_top_conf']:.3f} "
            f"class="
            f"{row['consensus_class']:5} "
            f"{Path(row['image']).name}"
        )

    print()
    print(
        "CSV:",
        csv_path
    )

    print(
        "Summary:",
        summary_path
    )

    print()
    print(
        "RESULT: HARD NEGATIVE "
        "MINING V1 PASS"
    )

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--chunk",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete existing teacher "
            "checkpoints and rerun mining"
        ),
    )

    args = parser.parse_args()

    if args.batch < 1:
        raise ValueError(
            "--batch must be >= 1"
        )

    if args.chunk < args.batch:
        raise ValueError(
            "--chunk must be >= --batch"
        )

    paths = load_pool()

    print("=" * 100)
    print(
        "R1 HARD-NEGATIVE MINER"
    )
    print("=" * 100)

    print(
        "Negative pool:",
        len(paths)
    )

    print(
        "Batch:",
        args.batch
    )

    print(
        "Chunk:",
        args.chunk
    )

    print(
        "Models:",
        "N, S, M"
    )

    if args.force:
        for name in (
            "n",
            "s",
            "m",
        ):
            path = checkpoint_path(
                name
            )

            if path.exists():
                path.unlink()

    for name in (
        "n",
        "s",
        "m",
    ):
        run_teacher(
            name,
            paths,
            args.batch,
            args.chunk,
        )

    print()
    print(
        "Merging N/S/M checkpoints..."
    )

    rows = merge_results(
        paths
    )

    save_final(
        rows
    )


if __name__ == "__main__":
    main()
