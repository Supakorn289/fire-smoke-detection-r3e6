#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import statistics
import time

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]

MODEL = (
    ROOT
    / "models/final/onnx_v1/"
      "fire_smoke_r3_e6_final_768_fp32.onnx"
)

EXPECTED_SHA256 = (
    "e887a759bcc65c89803c8ea0aa3d8891bcb0e60a"
    "5fd334433b2d6d912ba3bc79"
)

VAL_IMAGES = (
    ROOT
    / "datasets/fasdd_r21_preserve_v1/"
      "images/val"
)

OUT = (
    ROOT
    / "reports/deployment_v1/"
      "cpu_benchmark"
)

IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

IMGSZ = 768

CONF = 0.25
NMS_IOU = 0.70

PAD_VALUE = 114

THREAD_CONFIGS = [
    1,
    2,
    4,
]


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


def percentile(values, q):

    a = np.asarray(
        values,
        dtype=np.float64,
    )

    return float(
        np.percentile(
            a,
            q,
        )
    )


def process_rss_mb():

    status = Path(
        "/proc/self/status"
    )

    if not status.exists():
        return None

    for line in status.read_text().splitlines():

        if line.startswith("VmRSS:"):

            kb = int(
                line.split()[1]
            )

            return (
                kb / 1024.0
            )

    return None


def process_peak_rss_mb():

    status = Path(
        "/proc/self/status"
    )

    if not status.exists():
        return None

    for line in status.read_text().splitlines():

        if line.startswith("VmHWM:"):

            kb = int(
                line.split()[1]
            )

            return (
                kb / 1024.0
            )

    return None


def letterbox_768(image):

    h, w = image.shape[:2]

    scale = min(
        IMGSZ / w,
        IMGSZ / h,
    )

    new_w = int(
        round(w * scale)
    )

    new_h = int(
        round(h * scale)
    )


    if (
        new_w != w
        or new_h != h
    ):

        resized = cv2.resize(
            image,
            (
                new_w,
                new_h,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    else:

        resized = image


    dw = (
        IMGSZ - new_w
    )

    dh = (
        IMGSZ - new_h
    )

    left = int(
        round(
            dw / 2 - 0.1
        )
    )

    right = int(
        round(
            dw / 2 + 0.1
        )
    )

    top = int(
        round(
            dh / 2 - 0.1
        )
    )

    bottom = int(
        round(
            dh / 2 + 0.1
        )
    )


    padded = cv2.copyMakeBorder(
        resized,

        top,
        bottom,
        left,
        right,

        cv2.BORDER_CONSTANT,

        value=(
            PAD_VALUE,
            PAD_VALUE,
            PAD_VALUE,
        ),
    )


    # BGR -> RGB
    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB,
    )


    tensor = (
        rgb
        .transpose(
            2,
            0,
            1,
        )
        .astype(
            np.float32,
            copy=False,
        )
        / 255.0
    )


    tensor = np.expand_dims(
        tensor,
        axis=0,
    )


    return np.ascontiguousarray(
        tensor
    )


def xywh_to_xyxy(boxes):

    result = np.empty_like(
        boxes
    )

    result[:, 0] = (
        boxes[:, 0]
        - boxes[:, 2] / 2
    )

    result[:, 1] = (
        boxes[:, 1]
        - boxes[:, 3] / 2
    )

    result[:, 2] = (
        boxes[:, 0]
        + boxes[:, 2] / 2
    )

    result[:, 3] = (
        boxes[:, 1]
        + boxes[:, 3] / 2
    )

    return result


def iou_one_to_many(
    box,
    boxes,
):

    xx1 = np.maximum(
        box[0],
        boxes[:, 0],
    )

    yy1 = np.maximum(
        box[1],
        boxes[:, 1],
    )

    xx2 = np.minimum(
        box[2],
        boxes[:, 2],
    )

    yy2 = np.minimum(
        box[3],
        boxes[:, 3],
    )


    w = np.maximum(
        0.0,
        xx2 - xx1,
    )

    h = np.maximum(
        0.0,
        yy2 - yy1,
    )

    inter = (
        w * h
    )


    area_a = (
        (box[2] - box[0])
        * (box[3] - box[1])
    )

    area_b = (
        (boxes[:, 2] - boxes[:, 0])
        * (boxes[:, 3] - boxes[:, 1])
    )


    union = (
        area_a
        + area_b
        - inter
    )


    return np.divide(
        inter,
        union,
        out=np.zeros_like(
            inter,
        ),
        where=union > 0,
    )


def nms(
    boxes,
    scores,
    iou_threshold,
):

    if len(boxes) == 0:
        return []


    order = np.argsort(
        scores
    )[::-1]


    keep = []


    while len(order) > 0:

        i = int(
            order[0]
        )

        keep.append(
            i
        )


        if len(order) == 1:
            break


        remaining = order[
            1:
        ]


        ious = iou_one_to_many(
            boxes[i],
            boxes[
                remaining
            ],
        )


        order = remaining[
            ious
            <= iou_threshold
        ]


    return keep


def postprocess(raw):

    # YOLOv8 detect output:
    # [1, 4 + nc, N]
    pred = raw[
        0
    ].T


    boxes_xywh = pred[
        :,
        :4
    ]

    class_scores = pred[
        :,
        4:
    ]


    classes = np.argmax(
        class_scores,
        axis=1,
    )


    scores = np.max(
        class_scores,
        axis=1,
    )


    mask = (
        scores >= CONF
    )


    if not np.any(mask):
        return 0


    boxes = xywh_to_xyxy(
        boxes_xywh[
            mask
        ]
    )

    scores = scores[
        mask
    ]

    classes = classes[
        mask
    ]


    total_keep = 0


    # Class-aware NMS
    for cls in np.unique(
        classes
    ):

        cls_mask = (
            classes == cls
        )

        cls_boxes = boxes[
            cls_mask
        ]

        cls_scores = scores[
            cls_mask
        ]


        keep = nms(
            cls_boxes,
            cls_scores,
            NMS_IOU,
        )


        total_keep += len(
            keep
        )


    return total_keep


def build_session(
    threads,
):

    options = (
        ort.SessionOptions()
    )

    options.intra_op_num_threads = (
        threads
    )

    options.inter_op_num_threads = 1

    options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel
        .ORT_ENABLE_ALL
    )


    return ort.InferenceSession(
        str(MODEL),

        sess_options=options,

        providers=[
            "CPUExecutionProvider"
        ],
    )


def summarize(values):

    return {
        "mean_ms":
            statistics.mean(
                values
            ),

        "median_ms":
            statistics.median(
                values
            ),

        "p95_ms":
            percentile(
                values,
                95,
            ),

        "p99_ms":
            percentile(
                values,
                99,
            ),

        "min_ms":
            min(values),

        "max_ms":
            max(values),
    }


parser = argparse.ArgumentParser()

parser.add_argument(
    "--images",
    type=int,
    default=300,
)

parser.add_argument(
    "--warmup",
    type=int,
    default=20,
)

args = parser.parse_args()


# ============================================================
# PRE-FLIGHT
# ============================================================

if not MODEL.exists():
    raise FileNotFoundError(
        MODEL
    )


model_hash = sha256(
    MODEL
)


if model_hash != EXPECTED_SHA256:

    raise RuntimeError(
        "ONNX SHA256 mismatch"
    )


all_images = sorted([
    p
    for p in VAL_IMAGES.iterdir()
    if (
        p.is_file()
        and p.suffix.lower()
        in IMAGE_EXTS
    )
])


if len(all_images) < args.images:

    raise RuntimeError(
        "Not enough benchmark images"
    )


# Deterministic evenly distributed sample.
indices = np.linspace(
    0,
    len(all_images) - 1,
    num=args.images,
    dtype=int,
)


images = [
    all_images[
        int(i)
    ]
    for i in indices
]


OUT.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 110)
print(
    "FINAL ONNX CPU BENCHMARK V1"
)
print("=" * 110)

print(
    "Machine:",
    platform.platform()
)

print(
    "Processor:",
    platform.processor()
)

print(
    "Logical CPUs:",
    os.cpu_count()
)

print(
    "ORT:",
    ort.__version__
)

print(
    "ONNX:",
    MODEL
)

print(
    "SHA256:",
    model_hash
)

print(
    "Images:",
    len(images)
)

print(
    "Warmup:",
    args.warmup
)

print(
    "Input:",
    "1x3x768x768 FP32"
)

print(
    "Conf:",
    CONF
)

print(
    "NMS IoU:",
    NMS_IOU
)

print()

print(
    "FASDD TEST IS NOT USED."
)

print("=" * 110)


results = []


for threads in THREAD_CONFIGS:

    print()
    print("=" * 110)

    print(
        f"ORT CPU — {threads} intra-op thread(s)"
    )

    print("=" * 110)


    session = build_session(
        threads
    )

    input_name = (
        session
        .get_inputs()[0]
        .name
    )


    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    warm_image = cv2.imread(
        str(images[0])
    )

    warm_tensor = letterbox_768(
        warm_image
    )


    for _ in range(
        args.warmup
    ):

        raw = session.run(
            None,
            {
                input_name:
                    warm_tensor
            },
        )[0]

        postprocess(
            raw
        )


    # --------------------------------------------------------
    # TIMED LOOP
    # --------------------------------------------------------

    decode_ms = []
    preprocess_ms = []
    inference_ms = []
    postprocess_ms = []
    total_ms = []

    detections = 0


    rss_start = process_rss_mb()


    for index, image_path in enumerate(
        images,
        start=1,
    ):

        total_start = (
            time.perf_counter_ns()
        )


        # Decode
        t0 = (
            time.perf_counter_ns()
        )

        image = cv2.imread(
            str(image_path)
        )

        t1 = (
            time.perf_counter_ns()
        )


        if image is None:

            raise RuntimeError(
                f"Cannot decode: "
                f"{image_path}"
            )


        # Preprocess
        tensor = letterbox_768(
            image
        )

        t2 = (
            time.perf_counter_ns()
        )


        # ONNX Runtime
        raw = session.run(
            None,
            {
                input_name:
                    tensor
            },
        )[0]

        t3 = (
            time.perf_counter_ns()
        )


        # Postprocess + NMS
        count = postprocess(
            raw
        )

        t4 = (
            time.perf_counter_ns()
        )


        detections += count


        decode_ms.append(
            (t1 - t0)
            / 1_000_000
        )

        preprocess_ms.append(
            (t2 - t1)
            / 1_000_000
        )

        inference_ms.append(
            (t3 - t2)
            / 1_000_000
        )

        postprocess_ms.append(
            (t4 - t3)
            / 1_000_000
        )

        total_ms.append(
            (t4 - total_start)
            / 1_000_000
        )


        if (
            index % 50 == 0
            or index == len(images)
        ):

            print(
                f"\r{index}/{len(images)}",
                end="",
                flush=True,
            )


    print()


    rss_end = process_rss_mb()

    peak_rss = (
        process_peak_rss_mb()
    )


    decode_summary = summarize(
        decode_ms
    )

    pre_summary = summarize(
        preprocess_ms
    )

    infer_summary = summarize(
        inference_ms
    )

    post_summary = summarize(
        postprocess_ms
    )

    total_summary = summarize(
        total_ms
    )


    fps_median = (
        1000.0
        / total_summary[
            "median_ms"
        ]
    )

    fps_mean = (
        1000.0
        / total_summary[
            "mean_ms"
        ]
    )


    row = {
        "threads":
            threads,

        "images":
            len(images),

        "detections":
            detections,

        "decode":
            decode_summary,

        "preprocess":
            pre_summary,

        "inference":
            infer_summary,

        "postprocess":
            post_summary,

        "end_to_end":
            total_summary,

        "fps_from_median":
            fps_median,

        "fps_from_mean":
            fps_mean,

        "rss_start_mb":
            rss_start,

        "rss_end_mb":
            rss_end,

        "peak_rss_mb":
            peak_rss,
    }


    results.append(
        row
    )


    print(
        f"Decode median     : "
        f"{decode_summary['median_ms']:.2f} ms"
    )

    print(
        f"Preprocess median : "
        f"{pre_summary['median_ms']:.2f} ms"
    )

    print(
        f"Inference median  : "
        f"{infer_summary['median_ms']:.2f} ms"
    )

    print(
        f"Inference p95     : "
        f"{infer_summary['p95_ms']:.2f} ms"
    )

    print(
        f"Postprocess median: "
        f"{post_summary['median_ms']:.2f} ms"
    )

    print(
        f"E2E median        : "
        f"{total_summary['median_ms']:.2f} ms"
    )

    print(
        f"E2E p95           : "
        f"{total_summary['p95_ms']:.2f} ms"
    )

    print(
        f"FPS (median)      : "
        f"{fps_median:.2f}"
    )

    print(
        f"Peak RSS          : "
        f"{peak_rss:.1f} MB"
        if peak_rss is not None
        else
        "Peak RSS          : unavailable"
    )


# ============================================================
# COMPACT TABLE
# ============================================================

print()
print("=" * 120)
print(
    "CPU BENCHMARK COMPARISON"
)
print("=" * 120)

print(
    f"{'THREADS':>8}"
    f"{'INFER MED':>14}"
    f"{'INFER P95':>14}"
    f"{'E2E MED':>14}"
    f"{'E2E P95':>14}"
    f"{'FPS':>10}"
    f"{'PEAK RSS':>14}"
)

print("-" * 120)


for row in results:

    print(
        f"{row['threads']:>8}"
        f"{row['inference']['median_ms']:>14.2f}"
        f"{row['inference']['p95_ms']:>14.2f}"
        f"{row['end_to_end']['median_ms']:>14.2f}"
        f"{row['end_to_end']['p95_ms']:>14.2f}"
        f"{row['fps_from_median']:>10.2f}"
        f"{row['peak_rss_mb']:>14.1f}"
    )


best = min(
    results,
    key=lambda x:
        x["end_to_end"][
            "median_ms"
        ],
)


print()
print(
    "BEST CONFIG:",
    best["threads"],
    "thread(s)"
)

print(
    "BEST E2E MEDIAN:",
    f"{best['end_to_end']['median_ms']:.2f} ms"
)

print(
    "BEST FPS:",
    f"{best['fps_from_median']:.2f}"
)


# ============================================================
# SAVE
# ============================================================

record = {
    "status":
        "CPU_BENCHMARK_COMPLETE",

    "machine": {
        "platform":
            platform.platform(),

        "processor":
            platform.processor(),

        "logical_cpus":
            os.cpu_count(),
    },

    "onnx": {
        "path":
            str(MODEL),

        "sha256":
            model_hash,
    },

    "settings": {
        "imgsz": IMGSZ,
        "confidence": CONF,
        "nms_iou": NMS_IOU,
        "images": len(images),
        "warmup": args.warmup,
    },

    "results":
        results,

    "best_threads":
        best["threads"],

    "test_used":
        False,
}


record_path = (
    OUT
    / "CPU_BENCHMARK.json"
)


record_path.write_text(
    json.dumps(
        record,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print(
    "Record:",
    record_path
)

print(
    "RESULT: CPU BENCHMARK PASS"
)

print("=" * 120)
