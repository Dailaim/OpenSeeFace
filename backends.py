"""Inference backend selection for OpenSeeFace.

Backends:
    cpu       ONNX Runtime CPU (the original behaviour, uses the *_opt models)
    openvino  ONNX Runtime + OpenVINO EP (Intel oneAPI stack: Arc/Iris Xe GPUs, NPUs, CPUs)
    rocm      ONNX Runtime + MIGraphX EP (AMD ROCm), falls back to the legacy ROCm EP
    dml       ONNX Runtime + DirectML (any DX12 GPU on Windows)
    torch     PyTorch running the models natively on XPU (oneAPI/Level Zero) or
              on ROCm/HIP, converted from ONNX with onnx2torch
    auto      first available GPU backend in the order above, else cpu

GPU backends load the standard-ONNX copies from ``models/portable`` (generated
by ``tools/portable_models.py``) because the shipped models use the
ORT-CPU-only ``com.microsoft::FusedConv`` op.
"""
import os
import sys
import threading

import numpy as np

BACKENDS = ["auto", "cpu", "openvino", "rocm", "dml", "torch"]
PRECISIONS = ["fp32", "fp16"]

_config = {"backend": "cpu", "device": None, "precision": "fp32", "cache_dir": None}
_resolved = None
_lock = threading.Lock()


def configure(backend="auto", device=None, precision="fp32", cache_dir=None):
    global _resolved
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}, choose from {BACKENDS}")
    _config.update(backend=backend, device=device, precision=precision, cache_dir=cache_dir)
    _resolved = None


_ov_dll_dirs_added = False


def _add_openvino_dll_dirs():
    """onnxruntime-openvino on Windows needs the OpenVINO DLLs from the openvino wheel.
    ORT's own helper (onnxruntime.tools.add_openvino_win_libs) calls sys.exit() when it
    cannot find them, which breaks frozen (PyInstaller) builds, so locate them here."""
    global _ov_dll_dirs_added
    if _ov_dll_dirs_added or sys.platform != "win32":
        return
    _ov_dll_dirs_added = True
    candidates = []
    if getattr(sys, "frozen", False):
        candidates.append(os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)), "openvino", "libs"))
    try:
        import importlib.util
        spec = importlib.util.find_spec("openvino")
        if spec and spec.submodule_search_locations:
            candidates += [os.path.join(p, "libs") for p in spec.submodule_search_locations]
    except Exception:
        pass
    candidates += [p for p in os.environ.get("OPENVINO_LIB_PATHS", "").split(";") if p]
    for path in candidates:
        if os.path.isdir(path):
            os.environ["PATH"] = os.path.abspath(path) + os.pathsep + os.environ.get("PATH", "")
            os.add_dll_directory(os.path.abspath(path))


def _ort():
    _add_openvino_dll_dirs()
    import onnxruntime
    return onnxruntime


def _ort_providers():
    try:
        return set(_ort().get_available_providers())
    except Exception:
        return set()


def _torch_device():
    try:
        import torch
    except Exception:
        return None
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available() and getattr(torch.version, "hip", None):
        return "cuda"  # ROCm builds expose HIP devices through the cuda API
    return None


def available_backends():
    out = ["cpu"]
    p = _ort_providers()
    if "OpenVINOExecutionProvider" in p:
        out.append("openvino")
    if p & {"MIGraphXExecutionProvider", "ROCMExecutionProvider"}:
        out.append("rocm")
    if "DmlExecutionProvider" in p:
        out.append("dml")
    if _torch_device():
        out.append("torch")
    return out


def resolve_backend():
    global _resolved
    if _resolved is None:
        want = _config["backend"]
        avail = available_backends()
        if want == "auto":
            want = next((b for b in ("openvino", "rocm", "torch", "dml") if b in avail), "cpu")
        elif want not in avail:
            raise RuntimeError(f"Backend {want!r} is not available in this environment (available: {', '.join(avail)}). "
                               f"Install the matching extra, e.g. `uv sync --extra {want}`.")
        _resolved = want
    return _resolved


def describe():
    b = resolve_backend()
    dev = _config["device"]
    if b == "torch":
        dev = dev or _torch_device()
    elif b == "openvino":
        dev = dev or _openvino_default_device()
    return f"{b}" + (f" ({dev}, {_config['precision']})" if b != "cpu" else "")


def model_path(model_base_path, filename):
    if resolve_backend() == "cpu":
        return os.path.join(model_base_path, filename)
    portable = os.path.join(model_base_path, "portable", filename)
    if not os.path.exists(portable):
        raise FileNotFoundError(f"{portable} not found. Run `uv run python tools/portable_models.py` to generate the portable models.")
    return portable


def session_options(threads, sequential=True):
    ort = _ort()
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = max(1, threads)
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return options


_ov_default = None


def _openvino_default_device():
    """Prefer a discrete GPU (Arc) over the iGPU that OpenVINO calls plain "GPU"."""
    global _ov_default
    if _ov_default is None:
        _ov_default = "GPU"
        try:
            import openvino as ov
            core = ov.Core()
            gpus = [d for d in core.available_devices if d.startswith("GPU")]
            discrete = [d for d in gpus if "DISCRETE" in str(core.get_property(d, "DEVICE_TYPE")).upper()]
            if discrete:
                _ov_default = discrete[0]
            elif gpus:
                _ov_default = gpus[0]
            else:
                _ov_default = "CPU"
        except Exception:
            pass
    return _ov_default


def _input_spec(path, batch):
    """name[batch,c,h,w] for the model's first input, used to make OpenVINO compile a static graph."""
    try:
        import onnx
        inp = onnx.load(path, load_external_data=False).graph.input[0]
        dims = [d.dim_value if d.dim_value else batch for d in inp.type.tensor_type.shape.dim]
        return f"{inp.name}[{','.join(str(d) for d in dims)}]"
    except Exception:
        return None


def _ort_provider_list(backend, path=None, batch=1, device=None):
    dev = device or _config["device"]
    fp16 = _config["precision"] == "fp16"
    cache = _config["cache_dir"]
    avail = _ort_providers()
    if backend == "openvino":
        device = dev or _openvino_default_device()
        opts = {"device_type": device, "precision": "FP16" if fp16 else "FP32"}
        if device.startswith("GPU"):
            # Less host-side polling while waiting on the GPU: lower CPU use and slightly lower latency
            import json
            opts["load_config"] = json.dumps({"GPU": {"GPU_QUEUE_THROTTLE": "LOW", "GPU_HOST_TASK_PRIORITY": "LOW"}})  # keyed by device type, not GPU.1
        # Static shapes roughly halve GPU latency for these small models
        spec = _input_spec(path, batch) if path else None
        if spec:
            opts["reshape_input"] = spec
        if cache:
            opts["cache_dir"] = cache
        return [("OpenVINOExecutionProvider", opts), "CPUExecutionProvider"]
    if backend == "rocm":
        dev_id = int(dev) if dev not in (None, "") else 0
        if "MIGraphXExecutionProvider" in avail:
            opts = {"device_id": dev_id, "migraphx_fp16_enable": int(fp16)}
            if cache:
                opts["migraphx_model_cache_dir"] = cache
            return [("MIGraphXExecutionProvider", opts), "CPUExecutionProvider"]
        return [("ROCMExecutionProvider", {"device_id": dev_id}), "CPUExecutionProvider"]
    if backend == "dml":
        return [("DmlExecutionProvider", {"device_id": int(dev) if dev not in (None, "") else 0}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class _Input:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class TorchSession:
    """Minimal onnxruntime.InferenceSession look-alike backed by PyTorch."""
    _cache = {}
    _traced_cache = {}

    def __init__(self, path):
        import onnx
        import torch
        from onnx2torch import convert

        self.torch = torch
        self.device = torch.device(_config["device"] or _torch_device())
        self.dtype = torch.float16 if _config["precision"] == "fp16" else torch.float32
        key = (path, str(self.device), self.dtype)
        model = onnx.load(path)
        self._inputs = [_Input(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]
        with _lock:
            if key not in TorchSession._cache:
                net = convert(model).eval().to(self.device, self.dtype)
                TorchSession._cache[key] = net
            self.net = TorchSession._cache[key]
        self._traced = TorchSession._traced_cache.setdefault(key, {})

    def get_inputs(self):
        return self._inputs

    def _module_for(self, args):
        """Trace + freeze once per input shape: onnx2torch's eager graph re-evaluates
        Shape/Slice/Resize on the host every call (~20 ms on Arc), the frozen trace
        bakes the shapes in (~8 ms). OSF_TORCH_COMPILE=1 uses torch.compile instead
        (needs a C compiler for Triton, e.g. MSVC on Windows)."""
        key = tuple(tuple(a.shape) for a in args)
        mod = self._traced.get(key)
        if mod is None:
            torch = self.torch
            with _lock:
                mod = self._traced.get(key)
                if mod is None:
                    if os.environ.get("OSF_TORCH_COMPILE") == "1":
                        mod = torch.compile(self.net, dynamic=False)
                    else:
                        try:
                            mod = torch.jit.freeze(torch.jit.trace(self.net, tuple(args), check_trace=False))
                        except Exception:
                            mod = self.net
                    self._traced[key] = mod
        return mod

    def run(self, output_names, feed):
        torch = self.torch
        with torch.inference_mode():
            args = [torch.from_numpy(np.ascontiguousarray(feed[i.name])).to(self.device, self.dtype) for i in self._inputs]
            out = self._module_for(args)(*args)
            if not isinstance(out, (tuple, list)):
                out = (out,)
            return [o.float().cpu().numpy() for o in out]


_EXPECTED_PROVIDERS = {
    "openvino": ("OpenVINOExecutionProvider",),
    "rocm": ("MIGraphXExecutionProvider", "ROCMExecutionProvider"),
    "dml": ("DmlExecutionProvider",),
}


def create_session(path, threads=1, batch=1, device=None):
    """device overrides the configured device for this session only (e.g. OpenVINO "GPU.0")."""
    backend = resolve_backend()
    if backend == "torch":
        return TorchSession(path)
    ort = _ort()
    session = ort.InferenceSession(path, sess_options=session_options(threads), providers=_ort_provider_list(backend, path, batch, device))
    # ORT silently falls back to CPU when a provider fails to load; make that loud.
    active = session.get_providers()[0]
    expected = _EXPECTED_PROVIDERS.get(backend)
    if expected and active not in expected:
        raise RuntimeError(f"{backend} backend requested but ONNX Runtime fell back to {active}. "
                           "Check the GPU driver and that the openvino/ROCm runtime versions match onnxruntime.")
    return session
