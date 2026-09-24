#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ARCHIVES="$ROOT/datasets/internet_proxy_negative_v1/archives"
EXTRACTED="$ROOT/datasets/internet_proxy_negative_v1/extracted"

mkdir -p "$ARCHIVES" "$EXTRACTED"

CAMERAS=(
    858
    3888
    4795
    5021
    1093
    3297
    4232
    19106
    4801
    21444
)

BASE_URL="https://zenodo.org/records/5884485/files"

echo "============================================================"
echo "SKYFINDER INTERNET PROXY V1"
echo "============================================================"

for CAM in "${CAMERAS[@]}"; do

    ZIP="$ARCHIVES/${CAM}.zip"

    echo
    echo "Downloading camera $CAM ..."

    wget \
        --continue \
        --tries=5 \
        --timeout=30 \
        --show-progress \
        -O "$ZIP" \
        "${BASE_URL}/${CAM}.zip?download=1"

done

echo
echo "============================================================"
echo "VERIFYING ZIP FILES"
echo "============================================================"

for CAM in "${CAMERAS[@]}"; do

    ZIP="$ARCHIVES/${CAM}.zip"

    echo "Testing $ZIP"

    unzip -tq "$ZIP" >/dev/null

done

echo
echo "============================================================"
echo "EXTRACTING"
echo "============================================================"

for CAM in "${CAMERAS[@]}"; do

    ZIP="$ARCHIVES/${CAM}.zip"
    DEST="$EXTRACTED/$CAM"

    mkdir -p "$DEST"

    echo "Extracting camera $CAM ..."

    unzip \
        -q \
        -o \
        "$ZIP" \
        -d "$DEST"

done

echo
echo "============================================================"
echo "SKYFINDER DOWNLOAD PASS"
echo "============================================================"

du -sh "$ARCHIVES"
du -sh "$EXTRACTED"
