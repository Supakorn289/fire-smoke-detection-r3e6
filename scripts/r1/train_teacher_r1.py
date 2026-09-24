#!/usr/bin/env python3

from pathlib import Path
import argparse
import json
import sys

import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

DATA = (
    ROOT
    / "datasets"
    / "fasdd_r1_v1"
    / "data.yaml"
)

PROJECT = (
    ROOT
    / "runs"
    / "teachers_r1"
)


MODELS = {
    "n": "yolov8n.pt",
    "s": "yolov8s.pt",
    "m": "yolov8m.pt",
}


# ============================================================
# FROZEN R0 TRAINING CONFIG
# ============================================================

TRAIN_ARGS = {
    "epochs": 150,
    "patience": 35,

    "batch": -1,
    "imgsz": 768,

    "cache": False,
    "device": 0,
    "workers": 8,

    "pretrained": True,
    "optimizer": "AdamW",

    "seed": 42,
    "deterministic": True,

    "rect": False,
    "cos_lr": True,
    "close_mosaic": 15,

    "amp": True,
    "multi_scale": 0.0,

    "lr0": 0.001,
    "lrf": 0.01,

    "momentum": 0.937,
    "weight_decay": 0.0005,

    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,

    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,

    "hsv_h": 0.015,
    "hsv_s": 0.65,
    "hsv_v": 0.4,

    "degrees": 3.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 1.0,
    "perspective": 0.0,

    "flipud": 0.0,
    "fliplr": 0.5,

    "mosaic": 1.0,
    "mixup": 0.05,
    "copy_paste": 0.0,
}


EXPECTED = {
    "train": 12000,
    "val": 3000,
}


def count_files(path):

    return sum(
        1
        for p in path.iterdir()
        if p.is_file()
        or p.is_symlink()
    )


def check_dataset():

    if not DATA.exists():

        raise RuntimeError(
            f"Missing dataset YAML: {DATA}"
        )

    config = yaml.safe_load(
        DATA.read_text(
            encoding="utf-8"
        )
    )

    names = config.get(
        "names"
    )

    if names not in (
        {
            0: "fire",
            1: "smoke",
        },
        {
            "0": "fire",
            "1": "smoke",
        },
    ):

        raise RuntimeError(
            f"Unexpected class contract: {names}"
        )

    # R1 training dataset must have no TEST entry.
    if config.get("test"):

        raise RuntimeError(
            "data.yaml unexpectedly contains TEST"
        )

    dataset_root = Path(
        config["path"]
    )

    for split, expected in (
        EXPECTED.items()
    ):

        image_dir = (
            dataset_root
            / "images"
            / split
        )

        label_dir = (
            dataset_root
            / "labels"
            / split
        )

        if not image_dir.exists():
            raise RuntimeError(
                f"Missing {image_dir}"
            )

        if not label_dir.exists():
            raise RuntimeError(
                f"Missing {label_dir}"
            )

        images = count_files(
            image_dir
        )

        labels = count_files(
            label_dir
        )

        if images != expected:

            raise RuntimeError(
                f"{split}: expected "
                f"{expected} images, "
                f"got {images}"
            )

        if labels != expected:

            raise RuntimeError(
                f"{split}: expected "
                f"{expected} labels, "
                f"got {labels}"
            )

    print(
        "Dataset preflight : PASS"
    )


def check_gpu():

    print()
    print("GPU")
    print("-" * 88)

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available"
        )

    print(
        "CUDA:",
        torch.version.cuda
    )

    print(
        "Device:",
        torch.cuda.get_device_name(0)
    )

    total = (
        torch.cuda.get_device_properties(
            0
        ).total_memory
        / 1024**3
    )

    allocated = (
        torch.cuda.memory_allocated(
            0
        )
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved(
            0
        )
        / 1024**3
    )

    print(
        f"VRAM total    : "
        f"{total:.2f} GB"
    )

    print(
        f"VRAM allocated: "
        f"{allocated:.2f} GB"
    )

    print(
        f"VRAM reserved : "
        f"{reserved:.2f} GB"
    )


def print_config(
    size,
    model_name,
    run_name,
):

    print()
    print("=" * 88)
    print("TEACHER R1 TRAINING CONFIG")
    print("=" * 88)

    print(
        "Teacher :",
        size.upper()
    )

    print(
        "Model   :",
        model_name
    )

    print(
        "Data    :",
        DATA
    )

    print(
        "Project :",
        PROJECT
    )

    print(
        "Run     :",
        run_name
    )

    print()

    for key, value in (
        TRAIN_ARGS.items()
    ):

        print(
            f"{key:20}: "
            f"{value}"
        )

    print("=" * 88)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--size",
        required=True,
        choices=[
            "n",
            "s",
            "m",
        ],
    )

    args = parser.parse_args()

    size = args.size

    model_name = MODELS[
        size
    ]

    run_name = (
        f"teacher_{size}_r1"
    )

    run_dir = (
        PROJECT
        / run_name
    )

    # --------------------------------------------------------
    # Safety against accidentally overwriting an experiment
    # --------------------------------------------------------

    if run_dir.exists():

        print(
            f"ERROR: run directory already exists:\n"
            f"{run_dir}",
            file=sys.stderr,
        )

        print(
            "\nDo not overwrite an existing R1 run.",
            file=sys.stderr,
        )

        print(
            "If this was an interrupted run, "
            "resume from weights/last.pt instead.",
            file=sys.stderr,
        )

        raise SystemExit(2)

    check_dataset()
    check_gpu()

    print_config(
        size,
        model_name,
        run_name,
    )

    PROJECT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save frozen requested configuration before training
    # --------------------------------------------------------

    requested_config = {
        "round": "R1",
        "teacher": size.upper(),
        "model": model_name,
        "data": str(DATA),
        "project": str(PROJECT),
        "name": run_name,
        **TRAIN_ARGS,
    }

    config_dir = (
        ROOT
        / "reports"
        / "fasdd"
        / "r1_training_configs"
    )

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        config_dir
        / f"teacher_{size}_r1_requested.json"
    ).write_text(
        json.dumps(
            requested_config,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # MODEL
    # ========================================================

    print()
    print("=" * 88)
    print(
        f"STARTING TEACHER {size.upper()} R1"
    )
    print("=" * 88)

    model = YOLO(
        model_name
    )

    model.train(
        data=str(DATA),

        project=str(PROJECT),
        name=run_name,

        exist_ok=False,

        **TRAIN_ARGS,
    )

    # ========================================================
    # POST-TRAIN CHECK
    # ========================================================

    best = (
        run_dir
        / "weights"
        / "best.pt"
    )

    last = (
        run_dir
        / "weights"
        / "last.pt"
    )

    print()
    print("=" * 88)
    print("TRAINING COMPLETE")
    print("=" * 88)

    print(
        "BEST:",
        best
    )

    print(
        "LAST:",
        last
    )

    print(
        "best.pt exists:",
        best.exists()
    )

    print(
        "last.pt exists:",
        last.exists()
    )

    if not best.exists():

        raise RuntimeError(
            "Training ended but best.pt "
            "was not found"
        )

    print()
    print(
        f"RESULT: TEACHER "
        f"{size.upper()} R1 COMPLETE"
    )

    print("=" * 88)


if __name__ == "__main__":
    main()
