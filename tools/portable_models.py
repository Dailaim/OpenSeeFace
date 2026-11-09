"""Generate portable (standard ONNX) copies of the OpenSeeFace models.

The shipped ``*_opt.onnx`` models were optimized offline by ONNX Runtime's CPU
provider and contain the ``com.microsoft::FusedConv`` contrib op. Only ORT's
CPU/CUDA providers implement it, so OpenVINO (oneAPI), MIGraphX (ROCm),
DirectML and PyTorch either reject the model or bounce every conv back to the
CPU. This script rewrites every FusedConv into ``Conv`` + activation, drops the
unused custom domains and upgrades the opset, then checks the result against
the original model on ORT CPU.

Usage:
    uv run python tools/portable_models.py [--src models] [--dst models/portable]
"""
import argparse
import glob
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, version_converter

TARGET_OPSET = 13


def _attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def defuse(model):
    graph = model.graph
    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    new_nodes = []
    extra_inits = []
    for i, node in enumerate(graph.node):
        if node.domain != "com.microsoft" or node.op_type != "FusedConv":
            if node.domain not in ("", "ai.onnx"):
                raise RuntimeError(f"Unsupported custom op {node.domain}::{node.op_type}")
            new_nodes.append(node)
            continue
        a = _attrs(node)
        act = a.pop("activation", b"").decode()
        params = list(a.pop("activation_params", []) or [])
        if len(node.input) > 3 and node.input[3]:
            raise RuntimeError("FusedConv with sum input (Z) is not handled")
        conv_out = f"{node.output[0]}__conv"
        name = node.name or f"fusedconv_{i}"
        conv_attrs = {k: v for k, v in a.items() if not (k == "auto_pad" and v in (b"NOTSET", "NOTSET"))}
        new_nodes.append(helper.make_node("Conv", list(node.input[:3]), [conv_out], name=f"{name}_conv", **conv_attrs))
        if act == "Relu":
            new_nodes.append(helper.make_node("Relu", [conv_out], list(node.output), name=f"{name}_relu"))
        elif act == "LeakyRelu":
            new_nodes.append(helper.make_node("LeakyRelu", [conv_out], list(node.output), name=f"{name}_lrelu", alpha=float(params[0]) if params else 0.01))
        elif act == "Clip":
            lo, hi = (params + [0.0, 6.0])[:2] if params else (0.0, 6.0)
            if opset >= 11:
                lo_n, hi_n = f"{name}_clip_min", f"{name}_clip_max"
                extra_inits += [numpy_helper.from_array(np.array(lo, np.float32), lo_n),
                                numpy_helper.from_array(np.array(hi, np.float32), hi_n)]
                new_nodes.append(helper.make_node("Clip", [conv_out, lo_n, hi_n], list(node.output), name=f"{name}_clip"))
            else:
                new_nodes.append(helper.make_node("Clip", [conv_out], list(node.output), name=f"{name}_clip", min=float(lo), max=float(hi)))
        elif act in ("", "Identity"):
            new_nodes[-1].output[0] = node.output[0]
        else:
            raise RuntimeError(f"Unhandled FusedConv activation {act!r}")
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(extra_inits)
    del model.opset_import[:]
    model.opset_import.extend([helper.make_opsetid("", opset)])
    return model


def verify(src, dst, rtol=1e-3, atol=1e-4):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    a = ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(dst, so, providers=["CPUExecutionProvider"])
    inp = a.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.random.default_rng(0).random(shape, dtype=np.float32) * 4 - 2
    ra = a.run(None, {inp.name: x})
    rb = b.run(None, {b.get_inputs()[0].name: x})
    return max(float(np.max(np.abs(p - q))) for p, q in zip(ra, rb)), all(np.allclose(p, q, rtol=rtol, atol=atol) for p, q in zip(ra, rb))


def main():
    ap = argparse.ArgumentParser()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--src", default=os.path.join(root, "models"))
    ap.add_argument("--dst", default=os.path.join(root, "models", "portable"))
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    ok = True
    for src in sorted(glob.glob(os.path.join(args.src, "*.onnx"))):
        dst = os.path.join(args.dst, os.path.basename(src))
        model = defuse(onnx.load(src))
        model = version_converter.convert_version(model, TARGET_OPSET)
        model.ir_version = min(model.ir_version, 8)
        onnx.checker.check_model(model)
        onnx.save(model, dst)
        msg = f"{os.path.basename(src)}: ok"
        if not args.no_verify:
            diff, good = verify(src, dst)
            ok &= good
            msg = f"{os.path.basename(src)}: max abs diff {diff:.2e} {'ok' if good else 'MISMATCH'}"
        print(msg)
    for extra in glob.glob(os.path.join(args.src, "*.json")):
        with open(extra, "rb") as fi, open(os.path.join(args.dst, os.path.basename(extra)), "wb") as fo:
            fo.write(fi.read())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
