#!/usr/bin/env python3

from pathlib import Path
import csv
import gc
import json
import re

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

BASELINE_MODEL = (
    ROOT
    / "runs/teachers_r21/"
      "teacher_s_r21_preserve_v1/"
      "weights/best.pt"
)

R3_WEIGHTS = (
    ROOT
    / "runs/teachers_r3/"
      "teacher_s_r3_proxy_v1/"
      "weights"
)

DATA = (
    ROOT
    / "datasets/fasdd_r21_preserve_v1/"
      "data.yaml"
)

HOLDOUT = (
    ROOT
    / "reports/fasdd/r3_proxy_v1/"
      "proxy_holdout_all.csv"
)

OUT = (
    ROOT
    / "reports/fasdd/r3_proxy_v1/"
      "checkpoint_benchmark"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
]

PRED_FLOOR = 0.05

EXPECTED_HOLDOUT = 406


# ============================================================
# CHECK INPUTS
# ============================================================

for path in [
    BASELINE_MODEL,
    R3_WEIGHTS,
    DATA,
    HOLDOUT,
]:
    if not path.exists():
        raise FileNotFoundError(path)


with HOLDOUT.open(
    "r",
    encoding="utf-8",
    newline="",
) as f:

    holdout_rows = list(
        csv.DictReader(f)
    )


if len(holdout_rows) != EXPECTED_HOLDOUT:

    raise RuntimeError(
        f"Expected {EXPECTED_HOLDOUT} holdout images, "
        f"got {len(holdout_rows)}"
    )


holdout_images = []

for row in holdout_rows:

    # proxy_path points to the selected image
    image = Path(
        row["proxy_path"]
    )

    if not image.exists():
        raise FileNotFoundError(image)

    holdout_images.append(image)


# ============================================================
# DISCOVER R3 EPOCH CHECKPOINTS
# ============================================================

def epoch_number(path):

    m = re.fullmatch(
        r"epoch(\d+)\.pt",
        path.name,
    )

    if not m:
        return 10**9

    return int(m.group(1))


epoch_models = sorted(
    R3_WEIGHTS.glob("epoch*.pt"),
    key=epoch_number,
)


if not epoch_models:

    raise RuntimeError(
        "No epoch*.pt checkpoints found. "
        "Check save_period=1 output."
    )


models = [
    (
        "R2.1_BASELINE",
        BASELINE_MODEL,
    )
]


for model_path in epoch_models:

    ep = epoch_number(
        model_path
    )

    models.append(
        (
            f"R3_EPOCH_{ep}",
            model_path,
        )
    )


print("=" * 112)
print(
    "R3 CHECKPOINT PARETO BENCHMARK V1"
)
print("=" * 112)

print(
    "Human-verified negative holdout:",
    EXPECTED_HOLDOUT
)

print(
    "R3 epoch checkpoints:",
    len(epoch_models)
)

print()

for name, path in models:

    print(
        f"{name:<18} -> {path}"
    )

print("=" * 112)


all_rows = []


# ============================================================
# EVALUATE EACH MODEL
# ============================================================

for model_index, (
    model_name,
    model_path,
) in enumerate(
    models,
    start=1,
):

    print()
    print("=" * 112)

    print(
        f"[{model_index}/{len(models)}] "
        f"{model_name}"
    )

    print(
        "Model:",
        model_path
    )

    print("=" * 112)


    model = YOLO(
        str(model_path)
    )


    # --------------------------------------------------------
    # A) MAIN FASDD VAL
    # --------------------------------------------------------

    print()
    print(
        "A) FASDD MAIN VAL 3000"
    )


    metrics = model.val(
        data=str(DATA),
        imgsz=768,
        batch=2,
        device=0,
        workers=2,
        plots=False,
        verbose=False,

        project=str(OUT),
        name=f"{model_name.lower()}_val",
        exist_ok=True,
    )


    all_p = float(
        metrics.box.mp
    )

    all_r = float(
        metrics.box.mr
    )

    all_map50 = float(
        metrics.box.map50
    )

    all_map5095 = float(
        metrics.box.map
    )


    p_class = [
        float(x)
        for x in metrics.box.p
    ]

    r_class = [
        float(x)
        for x in metrics.box.r
    ]

    ap50_class = [
        float(x)
        for x in metrics.box.ap50
    ]

    map5095_class = [
        float(x)
        for x in metrics.box.maps
    ]


    if not (
        len(p_class) >= 2
        and len(r_class) >= 2
        and len(ap50_class) >= 2
        and len(map5095_class) >= 2
    ):

        raise RuntimeError(
            "Expected metrics for 2 classes"
        )


    print(
        f"ALL   "
        f"P={all_p:.4f} "
        f"R={all_r:.4f} "
        f"mAP50={all_map50:.4f} "
        f"mAP50-95={all_map5095:.4f}"
    )

    print(
        f"FIRE  "
        f"P={p_class[0]:.4f} "
        f"R={r_class[0]:.4f} "
        f"mAP50={ap50_class[0]:.4f} "
        f"mAP50-95={map5095_class[0]:.4f}"
    )

    print(
        f"SMOKE "
        f"P={p_class[1]:.4f} "
        f"R={r_class[1]:.4f} "
        f"mAP50={ap50_class[1]:.4f} "
        f"mAP50-95={map5095_class[1]:.4f}"
    )


    # --------------------------------------------------------
    # B) HUMAN-VERIFIED PROXY HOLDOUT
    # --------------------------------------------------------

    print()
    print(
        "B) INTERNET PROXY HOLDOUT 406"
    )


    stats = {
        threshold: {
            "alerts": 0,
            "fire_alerts": 0,
            "smoke_alerts": 0,
            "boxes": 0,
        }
        for threshold in THRESHOLDS
    }


    for image_index, image in enumerate(
        holdout_images,
        start=1,
    ):

        with torch.inference_mode():

            result = model.predict(
                source=str(image),
                imgsz=768,
                conf=PRED_FLOOR,
                iou=0.70,
                max_det=100,
                device=0,
                batch=1,
                verbose=False,
            )[0]


        detections = []


        if (
            result.boxes is not None
            and len(result.boxes) > 0
        ):

            classes = (
                result.boxes.cls
                .detach()
                .cpu()
                .tolist()
            )

            confs = (
                result.boxes.conf
                .detach()
                .cpu()
                .tolist()
            )


            detections = [
                (
                    int(cls),
                    float(conf),
                )
                for cls, conf
                in zip(
                    classes,
                    confs,
                )
            ]


        for threshold in THRESHOLDS:

            selected = [
                (
                    cls,
                    conf,
                )
                for cls, conf
                in detections
                if conf >= threshold
            ]


            if selected:

                stats[
                    threshold
                ][
                    "alerts"
                ] += 1


            if any(
                cls == 0
                for cls, conf
                in selected
            ):

                stats[
                    threshold
                ][
                    "fire_alerts"
                ] += 1


            if any(
                cls == 1
                for cls, conf
                in selected
            ):

                stats[
                    threshold
                ][
                    "smoke_alerts"
                ] += 1


            stats[
                threshold
            ][
                "boxes"
            ] += len(
                selected
            )


        del result


        if (
            image_index % 50 == 0
            and torch.cuda.is_available()
        ):

            gc.collect()
            torch.cuda.empty_cache()


        if (
            image_index % 50 == 0
            or image_index
            == EXPECTED_HOLDOUT
        ):

            print(
                f"\rProxy "
                f"{image_index}/"
                f"{EXPECTED_HOLDOUT}",
                end="",
                flush=True,
            )


    print()


    result_row = {
        "model":
            model_name,

        "path":
            str(model_path),

        "main_p":
            all_p,

        "main_r":
            all_r,

        "main_map50":
            all_map50,

        "main_map5095":
            all_map5095,

        "fire_p":
            p_class[0],

        "fire_r":
            r_class[0],

        "fire_map50":
            ap50_class[0],

        "fire_map5095":
            map5095_class[0],

        "smoke_p":
            p_class[1],

        "smoke_r":
            r_class[1],

        "smoke_map50":
            ap50_class[1],

        "smoke_map5095":
            map5095_class[1],
    }


    for threshold in THRESHOLDS:

        s = stats[
            threshold
        ]

        tag = str(
            threshold
        ).replace(
            ".",
            ""
        )

        fpr = (
            s["alerts"]
            / EXPECTED_HOLDOUT
        )

        fire_fpr = (
            s["fire_alerts"]
            / EXPECTED_HOLDOUT
        )

        smoke_fpr = (
            s["smoke_alerts"]
            / EXPECTED_HOLDOUT
        )


        result_row[
            f"fpr_{tag}"
        ] = fpr

        result_row[
            f"fire_fpr_{tag}"
        ] = fire_fpr

        result_row[
            f"smoke_fpr_{tag}"
        ] = smoke_fpr

        result_row[
            f"alerts_{tag}"
        ] = s["alerts"]

        result_row[
            f"boxes_{tag}"
        ] = s["boxes"]


    all_rows.append(
        result_row
    )


    print(
        "Proxy FPR:"
    )

    for threshold in THRESHOLDS:

        s = stats[
            threshold
        ]

        print(
            f"  conf={threshold:.2f} "
            f"{s['alerts']:3d}/"
            f"{EXPECTED_HOLDOUT} "
            f"FPR="
            f"{s['alerts']/EXPECTED_HOLDOUT:.4f} "
            f"fire="
            f"{s['fire_alerts']:3d} "
            f"smoke="
            f"{s['smoke_alerts']:3d}"
        )


    del model
    del metrics

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# SAVE FULL CSV
# ============================================================

csv_path = (
    OUT
    / "r3_checkpoint_benchmark.csv"
)


with csv_path.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            all_rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        all_rows
    )


# ============================================================
# COMPACT SUMMARY
# ============================================================

print()
print("=" * 140)
print(
    "R3 CHECKPOINT COMPACT COMPARISON"
)
print("=" * 140)

print(
    f"{'MODEL':<18}"
    f"{'P':>8}"
    f"{'R':>8}"
    f"{'mAP50':>9}"
    f"{'mAP95':>9}"
    f"{'FireR':>9}"
    f"{'SmokeR':>9}"
    f"{'FPR.20':>9}"
    f"{'FPR.25':>9}"
    f"{'FPR.30':>9}"
    f"{'FPR.40':>9}"
    f"{'FPR.50':>9}"
)

print("-" * 140)


def get_fpr(row, threshold):

    tag = str(
        threshold
    ).replace(
        ".",
        ""
    )

    return float(
        row[
            f"fpr_{tag}"
        ]
    )


for row in all_rows:

    print(
        f"{row['model']:<18}"
        f"{row['main_p']:>8.3f}"
        f"{row['main_r']:>8.3f}"
        f"{row['main_map50']:>9.3f}"
        f"{row['main_map5095']:>9.3f}"
        f"{row['fire_r']:>9.3f}"
        f"{row['smoke_r']:>9.3f}"
        f"{get_fpr(row,0.20):>9.3f}"
        f"{get_fpr(row,0.25):>9.3f}"
        f"{get_fpr(row,0.30):>9.3f}"
        f"{get_fpr(row,0.40):>9.3f}"
        f"{get_fpr(row,0.50):>9.3f}"
    )


# ============================================================
# DELTAS VS R2.1 BASELINE @ .25
# ============================================================

baseline = all_rows[0]

baseline_fpr25 = get_fpr(
    baseline,
    0.25,
)


print()
print("=" * 112)
print(
    "DELTA VS R2.1 BASELINE @ CONF=.25"
)
print("=" * 112)


for row in all_rows[1:]:

    fpr25 = get_fpr(
        row,
        0.25,
    )

    relative_reduction = (
        (
            baseline_fpr25
            - fpr25
        )
        / baseline_fpr25
        if baseline_fpr25 > 0
        else 0.0
    )


    print(
        f"{row['model']:<18} "
        f"ΔRecall="
        f"{row['main_r'] - baseline['main_r']:+.4f} "
        f"ΔmAP50="
        f"{row['main_map50'] - baseline['main_map50']:+.4f} "
        f"FPR25="
        f"{fpr25:.4f} "
        f"relative_FPR_reduction="
        f"{relative_reduction:+.2%}"
    )


# ============================================================
# JSON
# ============================================================

json_path = (
    OUT
    / "r3_checkpoint_benchmark.json"
)


json_path.write_text(
    json.dumps(
        {
            "holdout_status":
                "HUMAN_VERIFIED_NEGATIVE",

            "holdout_images":
                EXPECTED_HOLDOUT,

            "models":
                all_rows,
        },
        indent=2,
    ),
    encoding="utf-8",
)


print()
print(
    "CSV :",
    csv_path
)

print(
    "JSON:",
    json_path
)

print()

print(
    "RESULT: R3 CHECKPOINT "
    "BENCHMARK PASS"
)

print("=" * 112)
