"""Compare every available backend against ONNX Runtime CPU and time it.

Usage:
    uv run python tools/check_backends.py [--backends openvino torch] [--precision fp16] [--iters 200]
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import backends  # noqa: E402

MODELS = ["lm_model3_opt.onnx", "lm_modelT_opt.onnx", "mnv3_detection_opt.onnx", "mnv3_gaze32_split_opt.onnx", "retinaface_640x640_opt.onnx"]


def load(backend, name, precision, device):
    backends.configure(backend, device=device, precision=precision)
    s = backends.create_session(backends.model_path(os.path.join(ROOT, "models"), name), threads=1)
    return s


def run(s, x):
    return s.run([], {s.get_inputs()[0].name: x})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="*", default=None)
    ap.add_argument("--precision", default="fp32", choices=backends.PRECISIONS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()
    avail = backends.available_backends()
    targets = args.backends or avail
    print("available:", ", ".join(avail))
    rng = np.random.default_rng(0)
    for name in MODELS:
        ref = load("cpu", name, "fp32", None)
        shape = [d if isinstance(d, int) else 1 for d in ref.get_inputs()[0].shape]
        x = (rng.random(shape, dtype=np.float32) * 4 - 2)
        want = run(ref, x)
        for b in targets:
            try:
                s = load(b, name, args.precision if b != "cpu" else "fp32", args.device if b != "cpu" else None)
                for _ in range(5):
                    got = run(s, x)
                t = time.perf_counter()
                for _ in range(args.iters):
                    got = run(s, x)
                ms = (time.perf_counter() - t) / args.iters * 1000
                err = max(float(np.max(np.abs(g - w))) / (float(np.max(np.abs(w))) + 1e-6) for g, w in zip(got, want))
                print(f"{name:32s} {backends.describe():28s} {ms:7.2f} ms  max rel err {err:.1e}")
            except Exception as e:
                print(f"{name:32s} {b:28s} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:150]}")


if __name__ == "__main__":
    main()
