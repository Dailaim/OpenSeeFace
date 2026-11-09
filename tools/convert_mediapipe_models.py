"""Convert the MediaPipe .task bundles into ONNX models for the GPU backends.

Writes into models/portable/:
    mp_hand_detector.onnx, mp_hand_landmarks.onnx,
    mp_face_detector.onnx, mp_face_landmarks.onnx, mp_face_blendshapes.onnx,
    mp_face_geometry.npz   (canonical face mesh + Procrustes weights for the pose matrix)

The conversion needs TensorFlow, which is only used at build time, so it runs
tf2onnx in a throwaway uv environment:
    uv run python tools/convert_mediapipe_models.py
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
DST = os.path.join(MODELS, "portable")

BUNDLES = {
    "hand_landmarker.task": {"hand_detector.tflite": "mp_hand_detector.onnx",
                             "hand_landmarks_detector.tflite": "mp_hand_landmarks.onnx"},
    "face_landmarker_v2_with_blendshapes.task": {"face_detector.tflite": "mp_face_detector.onnx",
                                                 "face_landmarks_detector.tflite": "mp_face_landmarks.onnx",
                                                 "face_blendshapes.tflite": "mp_face_blendshapes.onnx"},
}


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _varint(b, i)
        elif wire == 2:
            n, i = _varint(b, i)
            v = b[i:i + n]
            i += n
        elif wire == 5:
            v = struct.unpack_from("<f", b, i)[0]
            i += 4
        elif wire == 1:
            v = struct.unpack_from("<d", b, i)[0]
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield field, wire, v


def parse_geometry(data):
    """GeometryPipelineMetadata: 1 = canonical_mesh (Mesh3d: 3 = vertex_buffer [x,y,z,u,v]),
    2 = procrustes_landmark_basis (1 = landmark_id, 2 = weight)."""
    vertices, ids, weights = [], [], []
    for field, wire, value in _fields(data):
        if field == 1 and wire == 2:
            for f2, w2, v2 in _fields(value):
                if f2 == 3:
                    if w2 == 5:
                        vertices.append(v2)
                    elif w2 == 2:
                        vertices.extend(struct.unpack(f"<{len(v2) // 4}f", v2))
        elif field == 2 and wire == 2:
            lid, w = 0, 0.0
            for f2, _, v2 in _fields(value):
                if f2 == 1:
                    lid = v2
                elif f2 == 2:
                    w = v2
            ids.append(lid)
            weights.append(w)
    mesh = np.asarray(vertices, np.float32).reshape(-1, 5)
    full_weights = np.zeros(len(mesh), np.float32)
    full_weights[ids] = weights
    return mesh[:, :3].copy(), full_weights


def convert(tflite, onnx_path):
    cmd = ["uv", "run", "--no-project", "--python", "3.12", "--with", "tf2onnx", "--with", "tensorflow-cpu",
           "--with", "numpy<2.2", "python", "-m", "tf2onnx.convert", "--tflite", tflite, "--output", onnx_path, "--opset", "17"]
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH")}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"tf2onnx failed for {tflite}:\n{r.stderr[-2000:]}")


def main():
    os.makedirs(DST, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for bundle, members in BUNDLES.items():
            path = os.path.join(MODELS, bundle)
            if not os.path.exists(path):
                raise SystemExit(f"{path} missing, run tools/fetch_mediapipe_models.py first")
            with zipfile.ZipFile(path) as z:
                z.extractall(tmp)
            for tfl, out in members.items():
                dst = os.path.join(DST, out)
                convert(os.path.join(tmp, tfl), dst)
                print(f"{out}: ok")
        canonical, weights = parse_geometry(open(os.path.join(tmp, "geometry_pipeline_metadata_landmarks.binarypb"), "rb").read())
        np.savez(os.path.join(DST, "mp_face_geometry.npz"), canonical=canonical, weights=weights)
        print(f"mp_face_geometry.npz: {len(canonical)} vertices, {int((weights > 0).sum())} weighted")


if __name__ == "__main__":
    main()
