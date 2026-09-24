#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import shutil

import onnx
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

SOURCE = (
    ROOT
    / "models/final/"
      "fire_smoke_r3_e6_final.pt"
)

EXPECTED_PT_SHA256 = (
    "49dc0464d99a6c250cf3c3e305d4149c3d4ce3ee"
    "354d9d7a5ae1cb8c53a22183"
)

OUT_DIR = (
    ROOT
    / "models/final/onnx_v1"
)

FINAL_ONNX = (
    OUT_DIR
    / "fire_smoke_r3_e6_final_768_fp32.onnx"
)

REPORT_DIR = (
    ROOT
    / "reports/deployment_v1/onnx_v1"
)


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


# ------------------------------------------------------------
# VERIFY FROZEN PT
# ------------------------------------------------------------

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)


pt_hash = sha256(
    SOURCE
)


if pt_hash != EXPECTED_PT_SHA256:

    raise RuntimeError(
        "FINAL PT SHA256 MISMATCH\n"
        f"Expected: {EXPECTED_PT_SHA256}\n"
        f"Actual  : {pt_hash}"
    )


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print("=" * 100)
print("FINAL ONNX EXPORT V1")
print("=" * 100)

print(
    "Source:",
    SOURCE
)

print(
    "PT SHA256:",
    pt_hash
)

print()

print(
    "Export configuration:"
)

print(
    "  imgsz    = 768"
)

print(
    "  batch    = 1"
)

print(
    "  dynamic  = False"
)

print(
    "  simplify = False"
)

print(
    "  opset    = 17"
)

print(
    "  half     = False"
)

print(
    "  nms      = False"
)

print("=" * 100)


# ------------------------------------------------------------
# EXPORT
# ------------------------------------------------------------

model = YOLO(
    str(SOURCE)
)


exported = model.export(
    format="onnx",

    imgsz=768,
    batch=1,

    dynamic=False,
    simplify=False,

    opset=17,

    half=False,

    nms=False,

    device="cpu",
)


exported = Path(
    exported
)


if not exported.exists():

    raise RuntimeError(
        f"Export output missing: {exported}"
    )


# ------------------------------------------------------------
# COPY TO LOCKED DEPLOYMENT LOCATION
# ------------------------------------------------------------

shutil.copy2(
    exported,
    FINAL_ONNX,
)


# ------------------------------------------------------------
# ONNX STRUCTURAL VALIDATION
# ------------------------------------------------------------

onnx_model = onnx.load(
    FINAL_ONNX
)


onnx.checker.check_model(
    onnx_model
)


onnx_hash = sha256(
    FINAL_ONNX
)


def tensor_shape(value_info):

    shape = []

    tensor_type = (
        value_info
        .type
        .tensor_type
    )

    for dim in tensor_type.shape.dim:

        if dim.HasField(
            "dim_value"
        ):

            shape.append(
                int(dim.dim_value)
            )

        elif dim.HasField(
            "dim_param"
        ):

            shape.append(
                dim.dim_param
            )

        else:

            shape.append(
                None
            )

    return shape


inputs = [
    {
        "name":
            x.name,

        "shape":
            tensor_shape(x),
    }

    for x in onnx_model.graph.input
]


outputs = [
    {
        "name":
            x.name,

        "shape":
            tensor_shape(x),
    }

    for x in onnx_model.graph.output
]


record = {
    "status":
        "ONNX_EXPORT_COMPLETE",

    "source_pt":
        str(SOURCE),

    "source_pt_sha256":
        pt_hash,

    "onnx":
        str(FINAL_ONNX),

    "onnx_sha256":
        onnx_hash,

    "configuration": {
        "imgsz": 768,
        "batch": 1,
        "dynamic": False,
        "simplify": False,
        "opset": 17,
        "fp16": False,
        "embedded_nms": False,
    },

    "inputs":
        inputs,

    "outputs":
        outputs,

    "onnx_checker":
        "PASS",
}


record_path = (
    REPORT_DIR
    / "ONNX_EXPORT_RECORD.json"
)


record_path.write_text(
    json.dumps(
        record,
        indent=2,
    ),
    encoding="utf-8",
)


print()
print("=" * 100)
print("ONNX EXPORT RESULT")
print("=" * 100)

print(
    "ONNX:",
    FINAL_ONNX
)

print(
    "SHA256:",
    onnx_hash
)

print(
    "Size:",
    f"{FINAL_ONNX.stat().st_size / 1024 / 1024:.2f} MiB"
)

print(
    "Inputs :",
    inputs
)

print(
    "Outputs:",
    outputs
)

print(
    "ONNX checker: PASS"
)

print(
    "Record:",
    record_path
)

print()

print(
    "RESULT: FINAL ONNX EXPORT PASS"
)

print("=" * 100)
