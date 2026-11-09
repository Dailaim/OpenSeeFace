#!/bin/bash
# Usage: ./make_exe.sh [cpu|openvino|rocm|torch-rocm|xpu]   (default: cpu)
set -e
FLAVOR=${1:-cpu}
command -v uv >/dev/null || { echo "Installing uv"; pip install uv; }

case "$FLAVOR" in
    xpu)        uv sync --locked --extra openvino --extra xpu --extra mediapipe --group build; EXTRA="--collect-all openvino --collect-all torch --collect-all onnx2torch" ;;
    torch-rocm) uv sync --locked --extra rocm --extra torch-rocm --extra mediapipe --group build; EXTRA="--collect-all torch --collect-all onnx2torch" ;;
    openvino)   uv sync --locked --extra openvino --extra mediapipe --group build; EXTRA="--collect-all openvino" ;;
    *)          uv sync --locked --extra "$FLAVOR" --extra mediapipe --group build; EXTRA="" ;;
esac

uv run --no-sync python tools/fetch_mediapipe_models.py
uv run --no-sync python tools/convert_mediapipe_models.py
echo "Generating portable models"
uv run --no-sync --with onnx python tools/portable_models.py --no-verify

echo "Creating binary"
uv run --no-sync pyinstaller --onedir --clean --noconfirm --specpath build facetracker.py --collect-all onnxruntime --collect-all mediapipe $EXTRA
cp -r models dist/facetracker/models

echo "Files should be available in the dist/ folder"
