#!/usr/bin/env python3
# realtime_dashboard_unbounded_full.py
# Updated: robust loading that doesn't assume add_safe_globals returns a context manager
from threading import Lock, RLock
import dash
from dash.dependencies import Output, Input
from dash import dcc, html
from datetime import datetime, timedelta
import json
import plotly.graph_objs as go
from collections import deque
from flask import Flask, request, jsonify
import socket
import numpy as np
import torch
from pathlib import Path
import traceback
import sys
import threading
import time as time_module

# -----------------------
# Ensure project src folders are importable (adjust if your layout differs)
# -----------------------
ROOT = Path.cwd()
for p in (ROOT, ROOT / "finetune" / "src", ROOT / "pretrain" / "src", ROOT):
    sp = str(p.resolve())
    if sp not in sys.path:
        sys.path.insert(0, sp)

# -----------------------
# Basic config
# -----------------------
hostname = socket.gethostname()
try:
    local_ip = socket.gethostbyname(hostname)
except Exception:
    local_ip = "127.0.0.1"
print(f"Server running at: http://{local_ip}:8000")
print(f"Configure Sensor Logger to POST to: http://{local_ip}:8000/data")

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# -----------------------
# Configuration (tweak as desired)
# -----------------------
MAX_DATA_POINTS = None  # None => unbounded; or set integer to cap storage
PLOT_MAX_POINTS = 5000
PREDICTION_PLOT_MAX = 500
UPDATE_FREQ_MS = 200
WINDOW_SIZE = 150
STEP_SIZE_RT = 75
TARGET_HZ = 50

ACTIVITY_LABELS = ["Walk", "Run", "Sit", "Stand", "Lie"]
ACTIVITY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# -----------------------
# Shared buffers & locks
# -----------------------
buffer_lock = RLock()


def make_deque(maxlen):
    return deque(maxlen=maxlen) if (maxlen is not None) else deque()


# -----------------------
# Time series / buffers
# -----------------------
phone_time_accel = make_deque(MAX_DATA_POINTS)
phone_accel_x = make_deque(MAX_DATA_POINTS)
phone_accel_y = make_deque(MAX_DATA_POINTS)
phone_accel_z = make_deque(MAX_DATA_POINTS)

phone_time_gyro = make_deque(MAX_DATA_POINTS)
phone_gyro_x = make_deque(MAX_DATA_POINTS)
phone_gyro_y = make_deque(MAX_DATA_POINTS)
phone_gyro_z = make_deque(MAX_DATA_POINTS)

watch_time_accel = make_deque(MAX_DATA_POINTS)
watch_accel_x = make_deque(MAX_DATA_POINTS)
watch_accel_y = make_deque(MAX_DATA_POINTS)
watch_accel_z = make_deque(MAX_DATA_POINTS)

watch_time_rot = make_deque(MAX_DATA_POINTS)
watch_rot_x = make_deque(MAX_DATA_POINTS)
watch_rot_y = make_deque(MAX_DATA_POINTS)
watch_rot_z = make_deque(MAX_DATA_POINTS)

phone_buffer = make_deque(MAX_DATA_POINTS)
watch_buffer = make_deque(MAX_DATA_POINTS)

accel_cache = {"phone": make_deque(2000), "watch": make_deque(2000)}
gyro_cache = {"phone": make_deque(2000), "watch": make_deque(2000)}

sample_counts = {"phone": 0, "watch": 0}

phone_predictions = make_deque(MAX_DATA_POINTS)
watch_predictions = make_deque(MAX_DATA_POINTS)
fusion_predictions = make_deque(MAX_DATA_POINTS)
prediction_times = make_deque(MAX_DATA_POINTS)

phone_probs = make_deque(MAX_DATA_POINTS)
watch_probs = make_deque(MAX_DATA_POINTS)
fusion_probs = make_deque(MAX_DATA_POINTS)

recent_events = deque(maxlen=200)
recent_device_strings = deque(maxlen=200)

frame_count = 0
both_ready_flag = False
last_both_ready_time = None

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------
# Try import your model constructors (they may import pretrain.* internally)
# -----------------------
SingleStreamClassifier = None
FusionClassifier = None
load_pretrained_backbone = None
try:
    try:
        from finetune.src.finetune.models import SingleStreamClassifier, FusionClassifier, load_pretrained_backbone  # type: ignore

        print("[info] Imported finetune.src.finetune.models")
    except Exception:
        from finetune.models import SingleStreamClassifier, FusionClassifier, load_pretrained_backbone  # type: ignore

        print("[info] Imported finetune.models")
except Exception as e:
    print("[warning] Could not import finetune model classes automatically:", e)

MODEL_DIR = Path("finetune/models/dashboard_models")
phone_model = None
watch_model = None
fusion_model = None

model_load_info = {
    "phone": {"loaded": False, "path": None, "why_failed": None},
    "watch": {"loaded": False, "path": None, "why_failed": None},
    "fusion": {"loaded": False, "path": None, "why_failed": None},
}


def is_model_loaded(m):
    return isinstance(m, torch.nn.Module)


# -----------------------
# Helper: robust torch.load (handles PyTorch 2.6+ weights_only / safe globals)
# -----------------------
def robust_torch_load(path: Path):
    """
    Attempt torch.load. If loading fails with the 'Unsupported global: numpy.core.multiarray.scalar'
    or weights_only message (PyTorch 2.6+), retry with weights_only=False and attempt to register
    the safe global (but do not assume a context manager is returned).
    """
    path = Path(path)
    try:
        return torch.load(path, map_location=device)
    except Exception as e:
        msg = str(e)
        # If the error indicates the new safe globals / weights_only behavior, retry
        if (
            "multiarray.scalar" in msg
            or "Weights only load failed" in msg
            or "Unsupported global" in msg
            or "weights_only" in msg
        ):
            print(
                f"[info] retrying torch.load for {path} with weights_only=False due to: {msg.splitlines()[0]}"
            )
            try:
                # Try to register safe globals if API present (call it but don't assume it returns a context manager)
                try:
                    if hasattr(torch, "serialization") and hasattr(
                        torch.serialization, "add_safe_globals"
                    ):
                        # some torch versions return a context manager, some return None after registering
                        try:
                            maybe_ctx = torch.serialization.add_safe_globals(
                                [np.core.multiarray.scalar]
                            )
                            # If it returned a context, attempt to use it (guarded)
                            if maybe_ctx is not None and hasattr(
                                maybe_ctx, "__enter__"
                            ):
                                try:
                                    maybe_ctx.__enter__()
                                    try:
                                        return torch.load(
                                            path,
                                            map_location=device,
                                            weights_only=False,
                                        )
                                    finally:
                                        try:
                                            maybe_ctx.__exit__(None, None, None)
                                        except Exception:
                                            pass
                                except Exception:
                                    # fallback to calling torch.load after registration
                                    pass
                            else:
                                # registration done (no context returned) — proceed to load
                                return torch.load(
                                    path, map_location=device, weights_only=False
                                )
                        except Exception as ex_reg:
                            print(
                                "[warning] add_safe_globals call failed or behaved unexpectedly:",
                                ex_reg,
                            )
                            # still try loading without registration
                            return torch.load(
                                path, map_location=device, weights_only=False
                            )
                    elif hasattr(torch, "serialization") and hasattr(
                        torch.serialization, "safe_globals"
                    ):
                        try:
                            maybe_ctx = torch.serialization.safe_globals(
                                [np.core.multiarray.scalar]
                            )
                            if maybe_ctx is not None and hasattr(
                                maybe_ctx, "__enter__"
                            ):
                                try:
                                    maybe_ctx.__enter__()
                                    try:
                                        return torch.load(
                                            path,
                                            map_location=device,
                                            weights_only=False,
                                        )
                                    finally:
                                        try:
                                            maybe_ctx.__exit__(None, None, None)
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                            else:
                                return torch.load(
                                    path, map_location=device, weights_only=False
                                )
                        except Exception as ex_reg:
                            print(
                                "[warning] safe_globals call failed or behaved unexpectedly:",
                                ex_reg,
                            )
                            return torch.load(
                                path, map_location=device, weights_only=False
                            )
                    else:
                        # API not present — still try a non-weights-only load
                        return torch.load(path, map_location=device, weights_only=False)
                except TypeError:
                    # weights_only not supported by this torch version; try plain load again to propagate original error
                    return torch.load(path, map_location=device)
            except Exception as e2:
                print(f"[error] second torch.load attempt also failed: {e2}")
                raise e2
        else:
            # re-raise original
            raise e


# -----------------------
# Model loading logic (robust)
# -----------------------
def _extract_state_dict_and_config(obj):
    if isinstance(obj, torch.nn.Module):
        return None, None, obj
    if not isinstance(obj, dict):
        return None, None, None

    candidate_state_keys = [
        "state_dict",
        "model_state_dict",
        "model_state",
        "model",
        "state",
        "net",
    ]
    candidate_cfg_keys = ["model_config", "config", "cfg", "MODEL CONFIG", "args"]

    state = None
    cfg = None

    for k in candidate_state_keys:
        if k in obj:
            state = obj[k]
            break

    if state is None:
        # heuristic: state dict-like object
        if all(isinstance(k, str) for k in obj.keys()):
            sample_val = next(iter(obj.values()))
            if isinstance(sample_val, torch.Tensor) or hasattr(sample_val, "shape"):
                state = obj

    for k in candidate_cfg_keys:
        if k in obj:
            cfg = obj[k]
            break

    return state, cfg, None


def _infer_num_classes_from_state(state_dict):
    if not isinstance(state_dict, dict):
        return None
    for k, v in state_dict.items():
        lname = k.lower()
        if lname.endswith(".weight") and (
            "head" in lname
            or "classifier" in lname
            or "fc" in lname
            or "logits" in lname
        ):
            try:
                return int(v.shape[0])
            except Exception:
                continue
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] <= 1024:
            return int(v.shape[0])
    return None


def _strip_prefix_from_state_dict(state, prefix):
    new_state = {}
    for k, v in state.items():
        if k.startswith(prefix):
            new_state[k[len(prefix) :]] = v
        else:
            new_state[k] = v
    return new_state


def load_model(path: Path, model_kind_hint: str = None):
    """
    Robust loader with adapter fallback.
    - Tries torch.jit .pt first.
    - Tries robust_torch_load + construct original model if constructors present.
    - If constructors missing, builds a lightweight EncoderAdapter+Linear classifier and
      tries to transplant classifier weights from the checkpoint state_dict.
    """
    path = Path(path)
    entry = model_load_info.get(model_kind_hint, None)

    # prefer TorchScript artifact if available
    ts_path = path.with_suffix(".pt")
    if ts_path.exists():
        try:
            print(f"[info] Loading TorchScript module from {ts_path}")
            m = torch.jit.load(str(ts_path)).to(device).eval()
            if entry is not None:
                entry["loaded"] = True
                entry["path"] = str(ts_path)
            return m
        except Exception as e:
            print(f"[warning] torch.jit.load failed for {ts_path}: {e}")

    # load checkpoint robustly
    try:
        loaded = robust_torch_load(path)
    except Exception as e:
        err = f"torch.load failed: {e}"
        print(f"[error] load_model({path}): {err}")
        if entry is not None:
            entry["why_failed"] = err
        return None

    # If checkpoint itself is an nn.Module
    if isinstance(loaded, torch.nn.Module):
        mod = loaded.to(device).eval()
        print(f"[info] Loaded full nn.Module from {path}")
        if entry is not None:
            entry["loaded"] = True; entry["path"] = str(path)
        inspect_loaded_model(model_kind_hint, mod, path)
        return repair_single_stream_model(mod, model_kind_hint)

    # Extract state dict + config
    state_dict, cfg, module_obj = _extract_state_dict_and_config(loaded)
    if module_obj is not None:
        mdl = module_obj.to(device).eval()
        if entry is not None:
            entry["loaded"] = True; entry["path"] = str(path)
        inspect_loaded_model(model_kind_hint, mdl, path)
        return repair_single_stream_model(mdl, model_kind_hint)

    if state_dict is None:
        err = f"No recognizable state_dict or module found in {path}"
        print("[error]", err)
        if entry is not None:
            entry["why_failed"] = err
        return None

    # unwrap nested
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict) and len(state_dict) > 1:
        state_dict = state_dict["state_dict"]

    keys = list(state_dict.keys())
    # check if checkpoint is fusion-style (phone_enc/watch_enc)
    is_fusion_state = any(k.startswith("phone_enc.") or k.startswith("phone_enc") or k.startswith("phone_enc.encoder") for k in keys)

    # find config values
    model_cfg = None
    if isinstance(cfg, dict):
        model_cfg = cfg
    else:
        for candidate in ("model_config", "config", "cfg"):
            if isinstance(loaded, dict) and candidate in loaded and isinstance(loaded[candidate], dict):
                model_cfg = loaded[candidate]; break

    num_classes = None
    input_channels = None
    embedding_dim = None
    if isinstance(model_cfg, dict):
        num_classes = model_cfg.get("num_classes") or model_cfg.get("n_classes") or model_cfg.get("num_output")
        input_channels = model_cfg.get("input_channels") or model_cfg.get("in_channels")
        embedding_dim = model_cfg.get("embedding_dim") or model_cfg.get("embed_dim")

    if num_classes is None:
        inferred = _infer_num_classes_from_state(state_dict)
        if inferred is not None:
            num_classes = inferred

    # Attempt to import constructors if available
    global SingleStreamClassifier, FusionClassifier, load_pretrained_backbone
    if SingleStreamClassifier is None or FusionClassifier is None:
        try:
            from finetune.src.finetune.models import SingleStreamClassifier, FusionClassifier, load_pretrained_backbone  # type: ignore
            print("[info] Late-imported finetune constructors.")
        except Exception:
            try:
                from finetune.models import SingleStreamClassifier, FusionClassifier, load_pretrained_backbone  # type: ignore
                print("[info] Late-imported finetune.models constructors.")
            except Exception as e:
                print("[warning] Could not import constructors:", e)

    constructed_model = None
    # try to construct the original model if constructors exist
    if (is_fusion_state and FusionClassifier is not None) or (SingleStreamClassifier is not None):
        try:
            if is_fusion_state and FusionClassifier is not None:
                # try reasonable constructor signatures
                try:
                    constructed_model = FusionClassifier(num_classes=num_classes if num_classes else 5, input_channels=input_channels if input_channels else 6, embedding_dim=embedding_dim if embedding_dim else 256)
                except Exception:
                    constructed_model = FusionClassifier()
            else:
                try:
                    constructed_model = SingleStreamClassifier(num_classes=num_classes if num_classes else 5, input_channels=input_channels if input_channels else 6, embedding_dim=embedding_dim if embedding_dim else 256)
                except Exception:
                    constructed_model = SingleStreamClassifier()
        except Exception as e:
            print("[warning] Constructor attempt failed:", e)
            constructed_model = None

    # If constructed_model exists, try loading the state_dict into it (the original logic)
    if constructed_model is not None:
        try:
            sd = state_dict
            if any(k.startswith("module.") for k in sd.keys()):
                print("[info] Stripping 'module.' prefix from state_dict keys")
                sd = _strip_prefix_from_state_dict(sd, "module.")
            if is_fusion_state:
                model_keys = list(constructed_model.state_dict().keys())
                if not any(k.startswith("phone_enc") for k in model_keys) and any(k.startswith("phone_enc.") for k in sd.keys()):
                    sd = { (k[len("phone_enc."):] if k.startswith("phone_enc.") else (k[len("watch_enc."):] if k.startswith("watch_enc.") else k)): v for k,v in sd.items() }
            constructed_model.load_state_dict(sd, strict=False)
            constructed_model = constructed_model.to(device); constructed_model.eval()
            if entry is not None:
                entry["loaded"] = True; entry["path"] = str(path)
            print(f"[info] Loaded state_dict into constructed model from {path}")
            inspect_loaded_model(model_kind_hint, constructed_model, path)
            return repair_single_stream_model(constructed_model, name=str(path.name))
        except Exception as e:
            print("[warning] Failed to load state into constructed model:", e)
            traceback.print_exc()
            constructed_model = None

    # ---------- FALLBACK: build a small adapter model ----------
    # Determine sizes
    if num_classes is None:
        num_classes = 5
    if embedding_dim is None:
        # try to infer embedding dim from any 2D tensor in the state_dict
        embedding_dim = None
        for v in state_dict.values():
            if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] <= 1024:
                embedding_dim = int(v.shape[1]); break
        if embedding_dim is None:
            embedding_dim = 256

    in_ch = input_channels if input_channels is not None else 6

    class SimpleAdapterModel(nn.Module):
        def __init__(self, in_ch, embedding_dim, num_classes):
            super().__init__()
            self.encoder = EncoderAdapter(in_channels=in_ch, embedding_dim=embedding_dim)
            self.classifier = nn.Linear(embedding_dim, num_classes)
        def forward(self, x):
            emb = self.encoder(x)  # (batch, emb)
            logits = self.classifier(emb)
            return logits

    print(f"[fallback] Constructing SimpleAdapterModel(in_ch={in_ch}, emb={embedding_dim}, classes={num_classes}) because constructors were not usable.")
    adapter = SimpleAdapterModel(in_ch, embedding_dim, int(num_classes)).to(device).eval()

    # Try to transplant classifier weights from state_dict: find any 2D tensor of shape (num_classes, embedding_dim)
    found_w = None; found_b = None
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[0] == int(num_classes) and v.shape[1] == int(embedding_dim):
            found_w = v
            print(f"[fallback] Found candidate classifier weight in checkpoint: {k}")
            # look for corresponding bias key name possibility
            bias_candidates = [k.replace(".weight", ".bias"), k.replace("weight", "bias")]
            for bkey in bias_candidates:
                if bkey in state_dict and isinstance(state_dict[bkey], torch.Tensor) and state_dict[bkey].dim() == 1 and state_dict[bkey].shape[0] == int(num_classes):
                    found_b = state_dict[bkey]; print(f"[fallback] Found candidate bias: {bkey}")
                    break
            break

    if found_w is not None:
        try:
            adapter.classifier.weight.data.copy_(found_w.to(adapter.classifier.weight.device))
            if found_b is not None:
                adapter.classifier.bias.data.copy_(found_b.to(adapter.classifier.bias.device))
            else:
                # zero bias if none present
                adapter.classifier.bias.data.zero_()
            print("[fallback] Transplanted classifier weights into adapter classifier.")
        except Exception as e:
            print("[fallback] Failed to transplant classifier weights:", e)
    else:
        print("[fallback] No matching classifier weight found in checkpoint; adapter classifier is randomly initialized.")

    if entry is not None:
        entry["loaded"] = True
        entry["path"] = str(path)
        entry["why_failed"] = "Used adapter fallback (original constructors unavailable)"

    return adapter


# -----------------------
# Attempt to load models
# -----------------------
phone_model = load_model(
    MODEL_DIR / "phone_only_classifier.pth", model_kind_hint="phone"
)
watch_model = load_model(
    MODEL_DIR / "watch_only_classifier.pth", model_kind_hint="watch"
)
fusion_model = load_model(MODEL_DIR / "fusion_classifier.pth", model_kind_hint="fusion")


# -----------------------
# Utilities: pruning worker + identify device + normalization
# -----------------------
def auto_prune_buffers():
    max_age_seconds = 600
    cutoff_time = datetime.now() - timedelta(seconds=max_age_seconds)
    if MAX_DATA_POINTS is not None:
        return
    with buffer_lock:
        try:
            while len(phone_time_accel) > 0 and phone_time_accel[0] < cutoff_time:
                phone_time_accel.popleft()
                phone_accel_x.popleft()
                phone_accel_y.popleft()
                phone_accel_z.popleft()
            while len(phone_time_gyro) > 0 and phone_time_gyro[0] < cutoff_time:
                phone_time_gyro.popleft()
                phone_gyro_x.popleft()
                phone_gyro_y.popleft()
                phone_gyro_z.popleft()
            while len(watch_time_accel) > 0 and watch_time_accel[0] < cutoff_time:
                watch_time_accel.popleft()
                watch_accel_x.popleft()
                watch_accel_y.popleft()
                watch_accel_z.popleft()
            while len(watch_time_rot) > 0 and watch_time_rot[0] < cutoff_time:
                watch_time_rot.popleft()
                watch_rot_x.popleft()
                watch_rot_y.popleft()
                watch_rot_z.popleft()
            while len(phone_predictions) > 1000:
                phone_predictions.popleft()
                phone_probs.popleft()
            while len(watch_predictions) > 1000:
                watch_predictions.popleft()
                watch_probs.popleft()
            while len(fusion_predictions) > 1000:
                fusion_predictions.popleft()
                fusion_probs.popleft()
                prediction_times.popleft()
        except Exception as e:
            print(f"[WARNING] Pruning error: {e}")


def pruning_worker():
    while True:
        try:
            time_module.sleep(60)
            auto_prune_buffers()
            print(f"[pruning] Buffers pruned at {datetime.now().isoformat()}")
        except Exception as e:
            print(f"[ERROR] Pruning worker failed: {e}")


def identify_device(device_string: str):
    if not device_string:
        return "phone"
    s = device_string.lower()
    phone_keywords = [
        "phone",
        "mobile",
        "pixel",
        "galaxy",
        "iphone",
        "android",
        "oneplus",
        "sm-g",
        "redmi",
        "xiaomi",
        "mi",
    ]
    watch_keywords = [
        "watch",
        "wrist",
        "fitbit",
        "applewatch",
        "apple watch",
        "garmin",
        "mi band",
        "wear",
        "galaxywatch",
        "tizen",
        "sm-r",
        "pixel_watch",
        "pixel-watch",
    ]
    for kw in watch_keywords:
        if kw in s:
            return "watch"
    for kw in phone_keywords:
        if kw in s:
            return "phone"
    if "sm-r" in s:
        return "watch"
    if "sm-g" in s:
        return "phone"
    return "phone"


def normalize_window(window):
    """
    Keep numpy-based normalize_window for legacy uses (returns numpy array).
    But prediction will use tensor_normalize to avoid numpy dependency in torch.
    """
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-8
    return (window - mean) / std


def tensor_normalize(window):
    """
    Convert a window (numpy ndarray shape (6, T) or torch tensor shape (6, T))
    into a normalized torch.Tensor on `device` with dtype float32.
    """
    if isinstance(window, np.ndarray):
        # Use torch.tensor rather than from_numpy to avoid torch needing numpy C-API
        w = torch.tensor(window, dtype=torch.float32, device=device)
    elif torch.is_tensor(window):
        w = window.to(device).float()
    else:
        # fallback: try to create tensor
        w = torch.tensor(np.array(window), dtype=torch.float32, device=device)

    # Compute mean/std on channel axis (dim=1) -> keepdim for broadcasting
    mean = w.mean(dim=1, keepdim=True)
    std = w.std(dim=1, keepdim=True) + 1e-8
    return (w - mean) / std


def create_window_from_buffer(buffer):
    if len(buffer) < WINDOW_SIZE:
        return None
    recent = list(buffer)[-WINDOW_SIZE:]
    window = np.array(
        [
            [s["ax"] for s in recent],
            [s["ay"] for s in recent],
            [s["az"] for s in recent],
            [s["gx"] for s in recent],
            [s["gy"] for s in recent],
            [s["gz"] for s in recent],
        ],
        dtype=np.float32,
    )
    return window


# -----------------------
# Prediction functions
# -----------------------
def predict_phone(window):
    if window is None:
        return "---", np.zeros(len(ACTIVITY_LABELS))
    if not is_model_loaded(phone_model):
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        return ACTIVITY_LABELS[np.argmax(probs)], probs
    try:
        with torch.no_grad():
            # convert and normalize to torch (avoid torch.from_numpy)
            x = tensor_normalize(window).unsqueeze(0)  # shape (1,6,T)
            outputs = phone_model(x)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            # ensure logits is a tensor
            if not torch.is_tensor(logits):
                logits = torch.tensor(logits, device=device)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            return ACTIVITY_LABELS[np.argmax(probs)], probs
    except Exception as e:
        print("phone prediction error:", e)
        traceback.print_exc()
        return "Error", np.zeros(len(ACTIVITY_LABELS))


def predict_watch(window):
    if window is None:
        return "---", np.zeros(len(ACTIVITY_LABELS))
    if not is_model_loaded(watch_model):
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        return ACTIVITY_LABELS[np.argmax(probs)], probs
    try:
        with torch.no_grad():
            x = tensor_normalize(window).unsqueeze(0)
            outputs = watch_model(x)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if not torch.is_tensor(logits):
                logits = torch.tensor(logits, device=device)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            return ACTIVITY_LABELS[np.argmax(probs)], probs
    except Exception as e:
        print("watch prediction error:", e)
        traceback.print_exc()
        return "Error", np.zeros(len(ACTIVITY_LABELS))


def predict_fusion(phone_window, watch_window):
    if phone_window is None or watch_window is None:
        return "---", np.zeros(len(ACTIVITY_LABELS))
    if not is_model_loaded(fusion_model):
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        return ACTIVITY_LABELS[np.argmax(probs)], probs
    try:
        with torch.no_grad():
            xp = tensor_normalize(phone_window).unsqueeze(0)  # (1,6,T)
            xw = tensor_normalize(watch_window).unsqueeze(0)
            # prefer calling fusion_model(xp, xw) — but this will be caught if signature differs.
            try:
                outputs = fusion_model(xp, xw)
            except TypeError:
                # fallback: try single argument (concatenate channels)
                try:
                    cat = torch.cat([xp, xw], dim=1)  # concat channels
                    outputs = fusion_model(cat)
                except Exception:
                    # last resort: call model on phone only and watch only if it supports it,
                    # or average individual model probs below
                    raise

            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if not torch.is_tensor(logits):
                logits = torch.tensor(logits, device=device)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            return ACTIVITY_LABELS[np.argmax(probs)], probs
    except Exception as e:
        # Try fallback: average individual phone/watch model probs if possible
        try:
            print(
                "[warning] fusion model direct call failed, trying to average phone/watch predictions:",
                e,
            )
            _, p_p = predict_phone(phone_window)
            _, p_w = predict_watch(watch_window)
            p_p = np.asarray(p_p) if p_p is not None else np.zeros(len(ACTIVITY_LABELS))
            p_w = np.asarray(p_w) if p_w is not None else np.zeros(len(ACTIVITY_LABELS))
            probs = (p_p + p_w) / 2.0
            return ACTIVITY_LABELS[np.argmax(probs)], probs
        except Exception:
            print("fusion prediction error:", e)
            traceback.print_exc()
            return "Error", np.zeros(len(ACTIVITY_LABELS))


# -----------------------
# Prediction orchestration (non-blocking)
# -----------------------
def make_predictions():
    global both_ready_flag, last_both_ready_time
    with buffer_lock:
        phone_ready = len(phone_buffer) >= WINDOW_SIZE
        watch_ready = len(watch_buffer) >= WINDOW_SIZE
        phone_recent = list(phone_buffer)[-WINDOW_SIZE:] if phone_ready else None
        watch_recent = list(watch_buffer)[-WINDOW_SIZE:] if watch_ready else None

    phone_win = None
    watch_win = None
    if phone_recent is not None:
        phone_win = np.array(
            [
                [s["ax"] for s in phone_recent],
                [s["ay"] for s in phone_recent],
                [s["az"] for s in phone_recent],
                [s["gx"] for s in phone_recent],
                [s["gy"] for s in phone_recent],
                [s["gz"] for s in phone_recent],
            ],
            dtype=np.float32,
        )
    if watch_recent is not None:
        watch_win = np.array(
            [
                [s["ax"] for s in watch_recent],
                [s["ay"] for s in watch_recent],
                [s["az"] for s in watch_recent],
                [s["gx"] for s in watch_recent],
                [s["gy"] for s in watch_recent],
                [s["gz"] for s in watch_recent],
            ],
            dtype=np.float32,
        )

    now = datetime.now()
    phone_label, phone_p = ("---", np.zeros(len(ACTIVITY_LABELS)))
    watch_label, watch_p = ("---", np.zeros(len(ACTIVITY_LABELS)))
    fusion_label, fusion_p = ("---", np.zeros(len(ACTIVITY_LABELS)))
    made_fusion = False

    if phone_win is not None:
        phone_label, phone_p = predict_phone(phone_win)
        print(
            f"[phone_pred] {datetime.now().isoformat()} -> {phone_label} (phone_buf={len(phone_buffer)})"
        )
    if watch_win is not None:
        watch_label, watch_p = predict_watch(watch_win)
        print(
            f"[watch_pred] {datetime.now().isoformat()} -> {watch_label} (watch_buf={len(watch_buffer)})"
        )
    if phone_win is not None and watch_win is not None:
        fusion_label, fusion_p = predict_fusion(phone_win, watch_win)
        made_fusion = True
        print(
            f"[fusion] {now.isoformat()} fusion_pred={fusion_label} (phone_buf={len(phone_buffer)} watch_buf={len(watch_buffer)})"
        )

    with buffer_lock:
        if phone_win is not None:
            phone_predictions.append(phone_label)
            phone_probs.append(phone_p)
        if watch_win is not None:
            watch_predictions.append(watch_label)
            watch_probs.append(watch_p)
        if made_fusion:
            fusion_predictions.append(fusion_label)
            fusion_probs.append(fusion_p)
            prediction_times.append(now)
        if len(phone_buffer) >= WINDOW_SIZE and len(watch_buffer) >= WINDOW_SIZE:
            if not both_ready_flag:
                both_ready_flag = True
                last_both_ready_time = datetime.now()
                print(
                    f"[info] BOTH buffers reached WINDOW_SIZE at {last_both_ready_time.isoformat()}"
                )
        else:
            both_ready_flag = False


# -----------------------
# Plot helpers (unchanged)
# -----------------------
def create_sensor_graph(time_data, sensor_data, names, title, yaxis_label, colors):
    data = []
    x_times = [
        (t.isoformat() if hasattr(t, "isoformat") else t) for t in list(time_data)
    ]
    for d, name, color in zip(sensor_data, names, colors):
        data.append(
            go.Scatter(
                x=x_times,
                y=list(d),
                name=name,
                line=dict(color=color, width=2),
                mode="lines",
            )
        )
    if len(time_data) > 0:
        x_max = time_data[-1]
        x_min = x_max - timedelta(seconds=10)
        x_range = [x_min.isoformat(), x_max.isoformat()]
    else:
        x_range = None
    layout = go.Layout(
        title=dict(text=title, font=dict(size=14, color="#2c3e50")),
        xaxis=dict(
            title="Time", showgrid=True, gridcolor="#ecf0f1", type="date", range=x_range
        ),
        yaxis=dict(title=yaxis_label, showgrid=True, gridcolor="#ecf0f1"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=30, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig = {"data": data, "layout": layout}
    try:
        all_values = [item for sublist in sensor_data for item in sublist]
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_margin = 1.0 if y_max - y_min == 0 else (y_max - y_min) * 0.1
            fig["layout"]["yaxis"]["range"] = [y_min - y_margin, y_max + y_margin]
    except Exception:
        pass
    return fig


def create_prob_bars(probs, title):
    probs = np.asarray(probs)
    if probs.size != len(ACTIVITY_LABELS):
        probs = np.zeros(len(ACTIVITY_LABELS))
    fig = go.Figure(
        data=[
            go.Bar(
                x=ACTIVITY_LABELS,
                y=probs,
                marker=dict(color=ACTIVITY_COLORS),
                text=[f"{p:.1%}" for p in probs],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        yaxis=dict(
            title="Probability", range=[0, 1], showgrid=True, gridcolor="#ecf0f1"
        ),
        xaxis=dict(title="Activity"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def create_prediction_timeline():
    with buffer_lock:
        pred_times_copy = list(prediction_times)[-500:]
        phone_preds_copy = list(phone_predictions)[-500:]
        watch_preds_copy = list(watch_predictions)[-500:]
        fusion_preds_copy = list(fusion_predictions)[-500:]
    if len(pred_times_copy) == 0:
        return go.Figure()
    activity_to_num = {label: i for i, label in enumerate(ACTIVITY_LABELS)}
    phone_nums = [activity_to_num.get(p, -1) for p in phone_preds_copy]
    watch_nums = [activity_to_num.get(p, -1) for p in watch_preds_copy]
    fusion_nums = [activity_to_num.get(p, -1) for p in fusion_preds_copy]
    x_serial = [
        (t.isoformat() if hasattr(t, "isoformat") else t) for t in pred_times_copy
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_serial,
            y=phone_nums,
            name="Phone",
            mode="lines+markers",
            line=dict(color="#3498db", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_serial,
            y=watch_nums,
            name="Watch",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_serial,
            y=fusion_nums,
            name="Fusion",
            mode="lines+markers",
            line=dict(color="#27ae60", width=2),
        )
    )
    fig.update_layout(
        title="Prediction Timeline",
        xaxis=dict(title="Time", type="date"),
        yaxis=dict(
            title="Activity",
            tickmode="array",
            tickvals=list(range(len(ACTIVITY_LABELS))),
            ticktext=ACTIVITY_LABELS,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=30, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# -----------------------
# Data ingestion endpoint
# -----------------------
@server.route("/data", methods=["POST"])
def data():
    global frame_count
    if request.method != "POST":
        return "method not allowed", 405
    try:
        payload = json.loads(request.data)
    except Exception as e:
        print("Failed to parse JSON payload:", e)
        return "ok", 200
    samples = payload.get("payload", [])
    if not isinstance(samples, list):
        samples = [payload]
    time_tolerance = timedelta(milliseconds=200)
    call_predict = False
    if not buffer_lock.acquire(timeout=2.0):
        print(
            "[WARNING] /data endpoint: lock timeout - dropping frame to prevent deadlock"
        )
        return "ok", 200
    try:
        with buffer_lock:
            for d in samples:
                try:
                    recent_events.append(
                        {"received_at": datetime.now().isoformat(), "raw": d}
                    )
                    device_raw = d.get("device", None)
                    recent_device_strings.append(str(device_raw))
                    print("---------- INCOMING EVENT ----------")
                    print("Raw event:", d)
                    print("device:", device_raw)
                    print("name:", d.get("name"))
                    print("time:", d.get("time"))
                    print("values:", d.get("values"))
                    print("------------------------------------")
                    ts_ns = d.get("time", None)
                    if ts_ns is None:
                        ts = datetime.now()
                    else:
                        try:
                            if ts_ns > 1e12:
                                ts = datetime.fromtimestamp(ts_ns / 1_000_000_000)
                            else:
                                ts = datetime.fromtimestamp(ts_ns)
                        except Exception:
                            ts = datetime.now()
                    name_raw = d.get("name", "")
                    name = name_raw.lower() if isinstance(name_raw, str) else ""
                    if "wrist" in name:
                        role = "watch"
                    elif name in ("gyroscope", "accelerometer"):
                        role = "phone"
                    else:
                        role = identify_device(str(device_raw))
                    print(
                        f"[IDENTIFY] name='{name_raw}' device='{device_raw}' -> role='{role}'"
                    )
                    if "wrist" in name:
                        vals = d.get("values", {})
                        gx = float(
                            vals.get("rotationRateX", vals.get("rotationRate_x", 0.0))
                        )
                        gy = float(
                            vals.get("rotationRateY", vals.get("rotationRate_y", 0.0))
                        )
                        gz = float(
                            vals.get("rotationRateZ", vals.get("rotationRate_z", 0.0))
                        )
                        ax = float(
                            vals.get("accelerationX", vals.get("acceleration_x", 0.0))
                        )
                        ay = float(
                            vals.get("accelerationY", vals.get("acceleration_y", 0.0))
                        )
                        az = float(
                            vals.get("accelerationZ", vals.get("acceleration_z", 0.0))
                        )
                        watch_time_accel.append(ts)
                        watch_accel_x.append(ax)
                        watch_accel_y.append(ay)
                        watch_accel_z.append(az)
                        watch_time_rot.append(ts)
                        watch_rot_x.append(gx)
                        watch_rot_y.append(gy)
                        watch_rot_z.append(gz)
                        accel_cache["watch"].append(
                            {"ax": ax, "ay": ay, "az": az, "timestamp": ts}
                        )
                        gyro_cache["watch"].append(
                            {"gx": gx, "gy": gy, "gz": gz, "timestamp": ts}
                        )
                        if (
                            len(accel_cache["watch"]) > 0
                            and len(gyro_cache["watch"]) > 0
                        ):
                            paired_accel = accel_cache["watch"][-1]
                            paired_gyro = gyro_cache["watch"][-1]
                            sample = {
                                "ax": paired_accel["ax"],
                                "ay": paired_accel["ay"],
                                "az": paired_accel["az"],
                                "gx": paired_gyro["gx"],
                                "gy": paired_gyro["gy"],
                                "gz": paired_gyro["gz"],
                                "timestamp": ts,
                            }
                            watch_buffer.append(sample)
                            sample_counts["watch"] += 1
                            if sample_counts["watch"] % 50 == 0:
                                print(
                                    f"[append] watch_buffer size now {len(watch_buffer)}"
                                )
                            if sample_counts["watch"] % STEP_SIZE_RT == 0:
                                call_predict = True
                        else:
                            print(
                                "[debug] wrist-motion: updated caches; waiting for counterpart to pair."
                            )
                    elif name == "accelerometer":
                        vals = d.get("values", {})
                        ax = float(
                            vals.get(
                                "x",
                                vals.get(
                                    "accelerationX", vals.get("acceleration_x", 0.0)
                                ),
                            )
                        )
                        ay = float(
                            vals.get(
                                "y",
                                vals.get(
                                    "accelerationY", vals.get("acceleration_y", 0.0)
                                ),
                            )
                        )
                        az = float(
                            vals.get(
                                "z",
                                vals.get(
                                    "accelerationZ", vals.get("acceleration_z", 0.0)
                                ),
                            )
                        )
                        phone_time_accel.append(ts)
                        phone_accel_x.append(ax)
                        phone_accel_y.append(ay)
                        phone_accel_z.append(az)
                        accel_cache["phone"].append(
                            {"ax": ax, "ay": ay, "az": az, "timestamp": ts}
                        )
                        paired_gyro = None
                        for g in reversed(gyro_cache["phone"]):
                            if (
                                abs((ts - g["timestamp"]).total_seconds())
                                <= time_tolerance.total_seconds()
                            ):
                                paired_gyro = g
                                break
                        if paired_gyro is None and len(gyro_cache["phone"]) > 0:
                            paired_gyro = gyro_cache["phone"][-1]
                        if paired_gyro is not None:
                            sample = {
                                "ax": ax,
                                "ay": ay,
                                "az": az,
                                "gx": paired_gyro["gx"],
                                "gy": paired_gyro["gy"],
                                "gz": paired_gyro["gz"],
                                "timestamp": ts,
                            }
                            phone_buffer.append(sample)
                            sample_counts["phone"] += 1
                            if sample_counts["phone"] % 50 == 0:
                                print(
                                    f"[append] phone_buffer size now {len(phone_buffer)}"
                                )
                            if sample_counts["phone"] % STEP_SIZE_RT == 0:
                                call_predict = True
                        else:
                            print(
                                "[debug] accel(phone): cached accel; waiting for gyro to pair."
                            )
                    elif name == "gyroscope":
                        vals = d.get("values", {})
                        gx = float(
                            vals.get(
                                "x", vals.get("rotationRateX", vals.get("gx", 0.0))
                            )
                        )
                        gy = float(
                            vals.get(
                                "y", vals.get("rotationRateY", vals.get("gy", 0.0))
                            )
                        )
                        gz = float(
                            vals.get(
                                "z", vals.get("rotationRateZ", vals.get("gz", 0.0))
                            )
                        )
                        phone_time_gyro.append(ts)
                        phone_gyro_x.append(gx)
                        phone_gyro_y.append(gy)
                        phone_gyro_z.append(gz)
                        gyro_cache["phone"].append(
                            {"gx": gx, "gy": gy, "gz": gz, "timestamp": ts}
                        )
                        paired_accel = None
                        for a in reversed(accel_cache["phone"]):
                            if (
                                abs((ts - a["timestamp"]).total_seconds())
                                <= time_tolerance.total_seconds()
                            ):
                                paired_accel = a
                                break
                        if paired_accel is None and len(accel_cache["phone"]) > 0:
                            paired_accel = accel_cache["phone"][-1]
                        if paired_accel is not None:
                            sample = {
                                "ax": paired_accel["ax"],
                                "ay": paired_accel["ay"],
                                "az": paired_accel["az"],
                                "gx": gx,
                                "gy": gy,
                                "gz": gz,
                                "timestamp": ts,
                            }
                            phone_buffer.append(sample)
                            sample_counts["phone"] += 1
                            if sample_counts["phone"] % 50 == 0:
                                print(
                                    f"[append] phone_buffer size now {len(phone_buffer)}"
                                )
                            if sample_counts["phone"] % STEP_SIZE_RT == 0:
                                call_predict = True
                        else:
                            print(
                                "[debug] gyro(phone): cached gyro; waiting for accel to pair."
                            )
                except Exception as ex:
                    print("Error processing incoming sample:", ex)
                    traceback.print_exc()
                    continue
    finally:
        buffer_lock.release()
        frame_count += 1
        if frame_count % 200 == 0:
            print(
                f"[summary @ frame {frame_count}] phone_buffer={len(phone_buffer)} watch_buffer={len(watch_buffer)} accel_cache_p={len(accel_cache['phone'])} gyro_cache_p={len(gyro_cache['phone'])} accel_cache_w={len(accel_cache['watch'])} gyro_cache_w={len(gyro_cache['watch'])}"
            )
    if call_predict:
        try:
            make_predictions()
        except Exception as e:
            print("Error during make_predictions():", e)
            traceback.print_exc()
    return "ok", 200


# -----------------------
# Debug/status endpoints
# -----------------------
@server.route("/debug", methods=["GET"])
def debug_info():
    with buffer_lock:
        phone_len = len(phone_buffer)
        watch_len = len(watch_buffer)
        last_phone_ts = (
            phone_buffer[-1]["timestamp"].isoformat() if phone_len > 0 else None
        )
        last_watch_ts = (
            watch_buffer[-1]["timestamp"].isoformat() if watch_len > 0 else None
        )
        last_pred_time = (
            prediction_times[-1].isoformat() if len(prediction_times) > 0 else None
        )
        recent_dev_list = list(recent_device_strings)[-50:]
        recent_ev = list(recent_events)[-50:]
        model_status = {
            "phone_loaded": is_model_loaded(phone_model),
            "watch_loaded": is_model_loaded(watch_model),
            "fusion_loaded": is_model_loaded(fusion_model),
            "phone_model_var": str(type(phone_model)),
            "watch_model_var": str(type(watch_model)),
            "fusion_model_var": str(type(fusion_model)),
            "phone_load_info": model_load_info.get("phone"),
            "watch_load_info": model_load_info.get("watch"),
            "fusion_load_info": model_load_info.get("fusion"),
        }
        resp = {
            "timestamp": datetime.now().isoformat(),
            "phone_buffer_len": phone_len,
            "watch_buffer_len": watch_len,
            "last_phone_sample_ts": last_phone_ts,
            "last_watch_sample_ts": last_watch_ts,
            "last_prediction_time": last_pred_time,
            "sample_counts": sample_counts.copy(),
            "both_ready_flag": both_ready_flag,
            "last_both_ready_time": (
                last_both_ready_time.isoformat()
                if last_both_ready_time is not None
                else None
            ),
            "recent_device_strings": recent_dev_list,
            "recent_events_count": len(recent_ev),
            "recent_events_sample": recent_ev[-10:],
            "model_status": model_status,
        }
    return jsonify(resp)


@server.route("/status", methods=["GET"])
def status_compact():
    with buffer_lock:
        return jsonify(
            {
                "phone_buffer_len": len(phone_buffer),
                "watch_buffer_len": len(watch_buffer),
                "frames": frame_count,
                "both_ready": both_ready_flag,
            }
        )


@server.route("/clear", methods=["POST", "GET"])
def clear_buffers():
    with buffer_lock:
        phone_time_accel.clear()
        phone_accel_x.clear()
        phone_accel_y.clear()
        phone_accel_z.clear()
        phone_time_gyro.clear()
        phone_gyro_x.clear()
        phone_gyro_y.clear()
        phone_gyro_z.clear()
        watch_time_accel.clear()
        watch_accel_x.clear()
        watch_accel_y.clear()
        watch_accel_z.clear()
        watch_time_rot.clear()
        watch_rot_x.clear()
        watch_rot_y.clear()
        watch_rot_z.clear()
        phone_buffer.clear()
        watch_buffer.clear()
        accel_cache["phone"].clear()
        accel_cache["watch"].clear()
        gyro_cache["phone"].clear()
        gyro_cache["watch"].clear()
        phone_predictions.clear()
        watch_predictions.clear()
        fusion_predictions.clear()
        prediction_times.clear()
        phone_probs.clear()
        watch_probs.clear()
        fusion_probs.clear()
        recent_events.clear()
        recent_device_strings.clear()
    print("[action] cleared all buffers and caches via /clear")
    return jsonify({"status": "cleared"})


@server.route("/", methods=["GET"])
def index():
    return "IMU Activity Recognition Dashboard (unbounded) running. POST data to /data endpoint. Visit /debug or /status."


# -----------------------
# Dash layout (unchanged UI)
# -----------------------
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "🏃 Real-Time IMU Activity Recognition (unbounded)",
                    style={
                        "textAlign": "center",
                        "color": "#2c3e50",
                        "marginBottom": "10px",
                    },
                ),
                html.P(
                    "Streaming sensor data from phone and watch — debug enabled",
                    style={
                        "textAlign": "center",
                        "color": "#7f8c8d",
                        "fontSize": "14px",
                    },
                ),
            ],
            style={
                "padding": "20px",
                "backgroundColor": "#ecf0f1",
                "borderRadius": "10px",
                "margin": "10px",
            },
        ),
        html.Div(
            id="connection-status",
            style={
                "padding": "15px",
                "margin": "10px",
                "backgroundColor": "#fff",
                "border": "2px solid #3498db",
                "borderRadius": "8px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            },
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="phone_accel_graph", style={"height": "250px"})],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "padding": "5px",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    [dcc.Graph(id="watch_accel_graph", style={"height": "250px"})],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "padding": "5px",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    [dcc.Graph(id="phone_gyro_graph", style={"height": "250px"})],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "padding": "5px",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    [dcc.Graph(id="watch_rot_graph", style={"height": "250px"})],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "padding": "5px",
                        "verticalAlign": "top",
                    },
                ),
            ],
            style={"margin": "10px"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            "📱 Phone Model",
                            style={"textAlign": "center", "color": "#3498db"},
                        ),
                        html.Div(
                            id="phone-prediction",
                            style={
                                "fontSize": "32px",
                                "fontWeight": "bold",
                                "textAlign": "center",
                                "padding": "20px",
                                "backgroundColor": "#ecf0f1",
                                "borderRadius": "8px",
                                "margin": "10px",
                            },
                        ),
                        dcc.Graph(id="phone_probs", style={"height": "200px"}),
                    ],
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "5px",
                    },
                ),
                html.Div(
                    [
                        html.H3(
                            "⌚ Watch Model",
                            style={"textAlign": "center", "color": "#e74c3c"},
                        ),
                        html.Div(
                            id="watch-prediction",
                            style={
                                "fontSize": "32px",
                                "fontWeight": "bold",
                                "textAlign": "center",
                                "padding": "20px",
                                "backgroundColor": "#ecf0f1",
                                "borderRadius": "8px",
                                "margin": "10px",
                            },
                        ),
                        dcc.Graph(id="watch_probs", style={"height": "200px"}),
                    ],
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "5px",
                    },
                ),
                html.Div(
                    [
                        html.H3(
                            "🔗 Fusion Model",
                            style={"textAlign": "center", "color": "#27ae60"},
                        ),
                        html.Div(
                            id="fusion-prediction",
                            style={
                                "fontSize": "32px",
                                "fontWeight": "bold",
                                "textAlign": "center",
                                "padding": "20px",
                                "backgroundColor": "#ecf0f1",
                                "borderRadius": "8px",
                                "margin": "10px",
                            },
                        ),
                        dcc.Graph(id="fusion_probs", style={"height": "200px"}),
                    ],
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "padding": "5px",
                    },
                ),
            ],
            style={"margin": "10px"},
        ),
        html.Div(
            [dcc.Graph(id="prediction_timeline", style={"height": "250px"})],
            style={"margin": "10px"},
        ),
        dcc.Interval(id="counter", interval=500),
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f8f9fa",
        "minHeight": "100vh",
    },
)


# -----------------------
# Dash callback
# -----------------------
@app.callback(
    [
        Output("connection-status", "children"),
        Output("phone_accel_graph", "figure"),
        Output("phone_gyro_graph", "figure"),
        Output("watch_accel_graph", "figure"),
        Output("watch_rot_graph", "figure"),
        Output("phone-prediction", "children"),
        Output("watch-prediction", "children"),
        Output("fusion-prediction", "children"),
        Output("phone_probs", "figure"),
        Output("watch_probs", "figure"),
        Output("fusion_probs", "figure"),
        Output("prediction_timeline", "figure"),
    ],
    Input("counter", "n_intervals"),
)
def update_dashboard(_counter):
    try:
        with buffer_lock:
            phone_time_accel_list = list(phone_time_accel)
            phone_accel_x_list = list(phone_accel_x)
            phone_accel_y_list = list(phone_accel_y)
            phone_accel_z_list = list(phone_accel_z)
            phone_time_gyro_list = list(phone_time_gyro)
            phone_gyro_x_list = list(phone_gyro_x)
            phone_gyro_y_list = list(phone_gyro_y)
            phone_gyro_z_list = list(phone_gyro_z)
            watch_time_accel_list = list(watch_time_accel)
            watch_accel_x_list = list(watch_accel_x)
            watch_accel_y_list = list(watch_accel_y)
            watch_accel_z_list = list(watch_accel_z)
            watch_time_rot_list = list(watch_time_rot)
            watch_rot_x_list = list(watch_rot_x)
            watch_rot_y_list = list(watch_rot_y)
            watch_rot_z_list = list(watch_rot_z)
            phone_samples = len(phone_buffer)
            watch_samples = len(watch_buffer)
            last_phone_pred = (
                phone_predictions[-1] if len(phone_predictions) > 0 else "---"
            )
            last_watch_pred = (
                watch_predictions[-1] if len(watch_predictions) > 0 else "---"
            )
            last_fusion_pred = (
                fusion_predictions[-1] if len(fusion_predictions) > 0 else "---"
            )
            last_phone_probs = (
                phone_probs[-1]
                if len(phone_probs) > 0
                else np.zeros(len(ACTIVITY_LABELS))
            ).copy()
            last_watch_probs = (
                watch_probs[-1]
                if len(watch_probs) > 0
                else np.zeros(len(ACTIVITY_LABELS))
            ).copy()
            last_fusion_probs = (
                fusion_probs[-1]
                if len(fusion_probs) > 0
                else np.zeros(len(ACTIVITY_LABELS))
            ).copy()
            frame_count_copy = frame_count

        plot_limit = 2000
        ta_phone = phone_time_accel_list[-plot_limit:]
        ax_phone = phone_accel_x_list[-plot_limit:]
        ay_phone = phone_accel_y_list[-plot_limit:]
        az_phone = phone_accel_z_list[-plot_limit:]
        tg_phone = phone_time_gyro_list[-plot_limit:]
        gx_phone = phone_gyro_x_list[-plot_limit:]
        gy_phone = phone_gyro_y_list[-plot_limit:]
        gz_phone = phone_gyro_z_list[-plot_limit:]
        ta_watch = watch_time_accel_list[-plot_limit:]
        ax_watch = watch_accel_x_list[-plot_limit:]
        ay_watch = watch_accel_y_list[-plot_limit:]
        az_watch = watch_accel_z_list[-plot_limit:]
        tr_watch = watch_time_rot_list[-plot_limit:]
        rx_watch = watch_rot_x_list[-plot_limit:]
        ry_watch = watch_rot_y_list[-plot_limit:]
        rz_watch = watch_rot_z_list[-plot_limit:]

        status = html.Div(
            [
                html.Span("📱 Phone: ", style={"fontWeight": "bold"}),
                html.Span(
                    f"{phone_samples}/{WINDOW_SIZE} samples",
                    style={
                        "color": (
                            "#27ae60" if phone_samples >= WINDOW_SIZE else "#e74c3c"
                        )
                    },
                ),
                html.Span(" | ", style={"margin": "0 12px"}),
                html.Span("⌚ Watch: ", style={"fontWeight": "bold"}),
                html.Span(
                    f"{watch_samples}/{WINDOW_SIZE} samples",
                    style={
                        "color": (
                            "#27ae60" if watch_samples >= WINDOW_SIZE else "#e74c3c"
                        )
                    },
                ),
                html.Span(" | ", style={"margin": "0 12px"}),
                html.Span(f"Server: {local_ip}:8000", style={"fontWeight": "bold"}),
                html.Span(" | ", style={"margin": "0 12px"}),
                html.Span(f"Frames: {frame_count_copy}", style={"color": "#7f8c8d"}),
            ],
            style={"fontSize": "16px", "padding": "8px"},
        )

        phone_accel_fig = create_sensor_graph(
            ta_phone,
            [ax_phone, ay_phone, az_phone],
            ["Accel X", "Accel Y", "Accel Z"],
            "Phone Accelerometer",
            "Acceleration (m/s²)",
            ["#e74c3c", "#3498db", "#2ecc71"],
        )
        phone_gyro_fig = create_sensor_graph(
            tg_phone,
            [gx_phone, gy_phone, gz_phone],
            ["Gyro X", "Gyro Y", "Gyro Z"],
            "Phone Gyroscope",
            "Angular Velocity (rad/s)",
            ["#f39c12", "#9b59b6", "#1abc9c"],
        )
        watch_accel_fig = create_sensor_graph(
            ta_watch,
            [ax_watch, ay_watch, az_watch],
            ["Watch Accel X", "Watch Accel Y", "Watch Accel Z"],
            "Watch Accelerometer",
            "Acceleration (m/s²)",
            ["#c0392b", "#2980b9", "#27ae60"],
        )
        watch_rot_fig = create_sensor_graph(
            tr_watch,
            [rx_watch, ry_watch, rz_watch],
            ["Rot X", "Rot Y", "Rot Z"],
            "Watch Rotation Rate",
            "Rotation Rate (rad/s)",
            ["#8e44ad", "#d35400", "#16a085"],
        )

        phone_prob_fig = create_prob_bars(last_phone_probs, "Phone Model Confidence")
        watch_prob_fig = create_prob_bars(last_watch_probs, "Watch Model Confidence")
        fusion_prob_fig = create_prob_bars(last_fusion_probs, "Fusion Model Confidence")
        timeline_fig = create_prediction_timeline()

        return (
            status,
            phone_accel_fig,
            phone_gyro_fig,
            watch_accel_fig,
            watch_rot_fig,
            last_phone_pred,
            last_watch_pred,
            last_fusion_pred,
            phone_prob_fig,
            watch_prob_fig,
            fusion_prob_fig,
            timeline_fig,
        )
    except Exception as e:
        print(f"[ERROR] Dashboard callback: {e}")
        traceback.print_exc()
        empty_fig = go.Figure()
        return (
            html.Div("Error"),
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
            "---",
            "---",
            "---",
            empty_fig,
            empty_fig,
            empty_fig,
            empty_fig,
        )


# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING IMU ACTIVITY RECOGNITION DASHBOARD (unbounded storage)")
    print("=" * 70)
    print(f"Dashboard URL: http://{local_ip}:8000")
    print(f"Sensor Logger POST endpoint: http://{local_ip}:8000/data")
    print("=" * 70)
    print(
        "\nNOTE: MAX_DATA_POINTS is None -> buffers are unbounded but auto-pruning is ENABLED."
    )
    print("Old data is automatically removed to keep memory usage stable.")
    print("\nConfigure Sensor Logger:")
    print("  1. Set recording mode to 'Push to Server'")
    print(f"  2. Enter URL: http://{local_ip}:8000/data")
    print(
        "  3. Enable Accelerometer and Gyroscope sensors (watch may send 'wrist motion' events too)"
    )
    print("  4. Set recording frequency to 50 Hz")
    print("  5. Start recording on both phone and watch")
    print("=" * 70 + "\n")

    pruning_thread = threading.Thread(target=pruning_worker, daemon=True)
    pruning_thread.start()
    print("[info] Background pruning thread started\n")

    app.run(port=8000, host="0.0.0.0", debug=False, threaded=True)
