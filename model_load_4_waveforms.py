#!/usr/bin/env python3
# realtime_dashboard_unbounded_full.py
# Robust version: auto-adds local src paths, robust model loader, absolute model paths, input-window debug
from threading import RLock
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
import os
import typing

# -----------------------
# AUTO-SET LOCAL PROJECT PATHS (VERY IMPORTANT)
# This ensures `finetune.models` and `pretrain` are importable without requiring external PYTHONPATH.
# Put this before any attempt to import finetune.* modules.
ROOT = Path(__file__).resolve().parent  # repo folder where this file lives
finetune_src = ROOT / "finetune" / "src"
pretrain_src = ROOT / "pretrain" / "src"
for p in (finetune_src, pretrain_src):
    pstr = str(p)
    if p.exists() and pstr not in sys.path:
        sys.path.insert(0, pstr)

# Now it's safe to import finetune.models if present
try:
    import finetune.models as finetune_models  # type: ignore
except Exception:
    finetune_models = None

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
MAX_DATA_POINTS = None
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


# --- time series / buffers (same as your original) ---
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
# Models (user-provided or placeholders)
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_DIR = ROOT / "finetune" / "models" / "dashboard_models"


def load_model(path: Path) -> typing.Optional[torch.nn.Module]:
    """
    Try to load model robustly:
      - If file is a scripted model (.pt / TorchScript), use torch.jit.load
      - Try torch.load with weights_only=False (this will require finetune.models to be importable)
      - If loaded object is nn.Module, return it on device
      - Otherwise print diagnostics and return None
    """
    try:
        if not path.exists():
            print(f"[load_model] model file not found: {path}")
            return None
        print(f"[load_model] Loading: {path}")
        suffix = path.suffix.lower()
        # TorchScript fallback
        if suffix in (".pt", ".torchscript"):
            try:
                m = torch.jit.load(str(path), map_location=device)
                m.eval()
                print(f"[load_model] torch.jit.load succeeded for {path}")
                return m
            except Exception as e:
                print(f"[load_model] torch.jit.load failed: {e}")
        # Prefer loading with weights_only=False so pickled module objects can be recovered
        try:
            loaded = torch.load(str(path), map_location=device, weights_only=False)
        except TypeError:
            # some PyTorch versions may not accept weights_only kwarg; fallback
            loaded = torch.load(str(path), map_location=device)
        except Exception as e:
            print(f"[load_model] torch.load initial attempt failed: {e}")
            # If it fails with weights-only safe-loader, try without weights_only (only do if you trust the file)
            try:
                loaded = torch.load(str(path), map_location=device, weights_only=False)
            except Exception as e2:
                print(f"[load_model] torch.load second attempt failed: {e2}")
                return None

        # If it's an nn.Module
        if isinstance(loaded, torch.nn.Module):
            loaded.to(device)
            loaded.eval()
            print(f"[load_model] Loaded nn.Module from {path}")
            return loaded

        # If it's a dict, check common checkpoint keys
        if isinstance(loaded, dict):
            # common keys that indicate state_dict
            for k in ("model_state_dict", "state_dict", "model_state"):
                if k in loaded:
                    print(
                        f"[load_model] Found '{k}' in checkpoint from {path} (state dict)."
                    )
                    # To load this into a model class we need to instantiate the appropriate class
                    # We cannot reliably infer class here — return dict and print instruction
                    print(
                        "  -> This is a state_dict. To load it, instantiate the model class (SingleStreamClassifier or FusionClassifier) and call load_state_dict(...)."
                    )
                    return None
            # otherwise print keys for debugging
            print(
                f"[load_model] torch.load returned a dict with keys: {list(loaded.keys())[:10]}..."
            )
            return None

        print(
            f"[load_model] torch.load returned unexpected object of type {type(loaded)}"
        )
        return None
    except Exception as e:
        print(f"[load_model] Failed to load {path}: {e}")
        traceback.print_exc()
        return None


# Try to find the model files (supports both names - full or plain)
phone_candidates = [
    MODEL_DIR / "phone_classifier_full.pth",
    MODEL_DIR / "phone_classifier.pth",
    MODEL_DIR / "phone_classifier.pt",
    MODEL_DIR / "phone.pt",
]
watch_candidates = [
    MODEL_DIR / "watch_classifier_full.pth",
    MODEL_DIR / "watch_classifier.pth",
    MODEL_DIR / "watch_classifier.pt",
    MODEL_DIR / "watch.pt",
]
fusion_candidates = [
    MODEL_DIR / "fusion_classifier_full.pth",
    MODEL_DIR / "fusion_classifier.pth",
    MODEL_DIR / "fusion_classifier.pt",
    MODEL_DIR / "fusion.pt",
]


def try_candidates(cands):
    for p in cands:
        m = load_model(p)
        if m is not None:
            return m, p
    return None, None


phone_model, phone_model_path = try_candidates(phone_candidates)
watch_model, watch_model_path = try_candidates(watch_candidates)
fusion_model, fusion_model_path = try_candidates(fusion_candidates)

print("Model load results:")
print("  Phone:", bool(phone_model), phone_model_path)
print("  Watch:", bool(watch_model), watch_model_path)
print("  Fusion:", bool(fusion_model), fusion_model_path)


def is_model_loaded(m):
    return isinstance(m, torch.nn.Module)


# -----------------------
# Utilities (same as before, with small debug prints)
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


import threading, time as time_module


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
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-8
    return (window - mean) / std


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


def _ensure_prob_vector(p):
    """Return a 1D numpy float vector of length len(ACTIVITY_LABELS)."""
    try:
        arr = np.asarray(p, dtype=float).flatten()
    except Exception:
        arr = np.zeros(len(ACTIVITY_LABELS), dtype=float)
    if arr.size != len(ACTIVITY_LABELS):
        # If it's shorter/longer, fall back to zeros (or pad/truncate if you prefer)
        arr2 = np.zeros(len(ACTIVITY_LABELS), dtype=float)
        arr2[: min(arr.size, arr2.size)] = arr[: min(arr.size, arr2.size)]
        arr = arr2
    return arr


def predict_phone(window):
    if window is None:
        return "---", np.zeros(len(ACTIVITY_LABELS), dtype=float)
    if not is_model_loaded(phone_model):
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        probs = _ensure_prob_vector(probs)
        return ACTIVITY_LABELS[int(np.argmax(probs))], probs
    try:
        with torch.no_grad():
            x = (
                torch.from_numpy(normalize_window(window))
                .unsqueeze(0)
                .float()
                .to(device)
            )
            outputs = phone_model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            probs = _ensure_prob_vector(probs)
            return ACTIVITY_LABELS[int(np.argmax(probs))], probs
    except Exception as e:
        print("[ERROR] predict_phone:", e)
        traceback.print_exc()
        probs = np.zeros(len(ACTIVITY_LABELS), dtype=float)
        return "Error", probs


def predict_watch(window):
    if window is None:
        return "---", np.zeros(len(ACTIVITY_LABELS), dtype=float)
    if not is_model_loaded(watch_model):
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        probs = _ensure_prob_vector(probs)
        return ACTIVITY_LABELS[int(np.argmax(probs))], probs
    try:
        with torch.no_grad():
            x = (
                torch.from_numpy(normalize_window(window))
                .unsqueeze(0)
                .float()
                .to(device)
            )
            outputs = watch_model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            probs = _ensure_prob_vector(probs)
            return ACTIVITY_LABELS[int(np.argmax(probs))], probs
    except Exception as e:
        print("[ERROR] predict_watch:", e)
        traceback.print_exc()
        return "Error", np.zeros(len(ACTIVITY_LABELS), dtype=float)


def predict_fusion(phone_window, watch_window):
    if phone_window is None or watch_window is None:
        return "---", np.zeros(len(ACTIVITY_LABELS), dtype=float)
    if not is_model_loaded(fusion_model):
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        probs = _ensure_prob_vector(probs)
        return ACTIVITY_LABELS[int(np.argmax(probs))], probs
    try:
        with torch.no_grad():
            xp = (
                torch.from_numpy(normalize_window(phone_window))
                .unsqueeze(0)
                .float()
                .to(device)
            )
            xw = (
                torch.from_numpy(normalize_window(watch_window))
                .unsqueeze(0)
                .float()
                .to(device)
            )
            outputs = fusion_model(xp, xw)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            probs = _ensure_prob_vector(probs)
            return ACTIVITY_LABELS[int(np.argmax(probs))], probs
    except Exception as e:
        print("[ERROR] predict_fusion:", e)
        traceback.print_exc()
        return "Error", np.zeros(len(ACTIVITY_LABELS), dtype=float)


# make_predictions remains essentially the same as your original
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
            title="Time", type="date", range=x_range, showgrid=True, gridcolor="#ecf0f1"
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
    try:
        probs = np.asarray(probs, dtype=float).flatten()
    except Exception:
        probs = np.zeros(len(ACTIVITY_LABELS), dtype=float)
    if probs.size != len(ACTIVITY_LABELS):
        tmp = np.zeros(len(ACTIVITY_LABELS), dtype=float)
        tmp[: min(probs.size, tmp.size)] = probs[: min(probs.size, tmp.size)]
        probs = tmp
    # Safety clamp to [0,1]
    probs = np.clip(probs, 0.0, 1.0)
    # If they don't sum to 1, that's fine for visualization; show raw probs.
    try:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=ACTIVITY_LABELS,
                    y=probs.tolist(),
                    marker=dict(color=ACTIVITY_COLORS),
                    text=[f"{float(p):.1%}" for p in probs],
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
    except Exception as e:
        print("[ERROR] create_prob_bars:", e)
        traceback.print_exc()
        # Return an empty safe figure
        empty = go.Figure()
        return empty


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
        print("[WARNING] /data: lock timeout")
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
                    # debug print
                    print("Incoming:", d.get("name"), "device:", device_raw)
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
                    # handle watch wrist events
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
                            if sample_counts["watch"] % STEP_SIZE_RT == 0:
                                call_predict = True
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
                            if sample_counts["phone"] % STEP_SIZE_RT == 0:
                                call_predict = True
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
                            if sample_counts["phone"] % STEP_SIZE_RT == 0:
                                call_predict = True
                except Exception as ex:
                    print("Error processing sample:", ex)
                    traceback.print_exc()
                    continue
    finally:
        buffer_lock.release()
        frame_count += 1
        if frame_count % 200 == 0:
            print(
                f"[summary @ frame {frame_count}] phone_buffer={len(phone_buffer)} watch_buffer={len(watch_buffer)}"
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

import os
import platform
import psutil


@server.route("/debug", methods=["GET"])
def debug_info():
    """
    Diagnostic debug endpoint: returns process + model identity (type & id) plus
    model load paths. Helpful to detect multiple server processes or shadowed vars.
    """
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

        # model type & id diagnostics
        def model_info(m, path_var=None):
            return {
                "is_loaded": isinstance(m, torch.nn.Module),
                "type": str(type(m)),
                "id": id(m) if m is not None else None,
                "repr": repr(m)[:400] if m is not None else None,
                "path": str(path_var) if path_var is not None else None,
            }

        model_status = {
            "phone": model_info(
                phone_model,
                getattr(phone_model_path, "resolve", lambda: phone_model_path)(),
            ),
            "watch": model_info(
                watch_model,
                getattr(watch_model_path, "resolve", lambda: watch_model_path)(),
            ),
            "fusion": model_info(
                fusion_model,
                getattr(fusion_model_path, "resolve", lambda: fusion_model_path)(),
            ),
        }

        resp = {
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "models": model_status,
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
        }
    # Also print to server console (so you can correlate console logs)
    print(
        "[DEBUG ENDPOINT CALLED] pid:",
        resp["pid"],
        "models:",
        {k: (v["is_loaded"], v["type"]) for k, v in model_status.items()},
    )
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
    print("[action] cleared buffers")
    return jsonify({"status": "cleared"})


# -----------------------
# Dash layout (unchanged appearance)
# -----------------------
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "🏃 Real-Time IMU Activity Recognition (unbounded)",
                    style={"textAlign": "center"},
                ),
                html.P(
                    "Streaming sensor data from phone and watch — debug enabled",
                    style={"textAlign": "center"},
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
            },
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="phone_accel_graph", style={"height": "250px"})],
                    style={"width": "50%", "display": "inline-block"},
                ),
                html.Div(
                    [dcc.Graph(id="watch_accel_graph", style={"height": "250px"})],
                    style={"width": "50%", "display": "inline-block"},
                ),
                html.Div(
                    [dcc.Graph(id="phone_gyro_graph", style={"height": "250px"})],
                    style={"width": "50%", "display": "inline-block"},
                ),
                html.Div(
                    [dcc.Graph(id="watch_rot_graph", style={"height": "250px"})],
                    style={"width": "50%", "display": "inline-block"},
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
# Dashboard update callback
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

            # Coerce last probs to safe arrays
            last_phone_probs = (
                _ensure_prob_vector(phone_probs[-1])
                if len(phone_probs) > 0
                else np.zeros(len(ACTIVITY_LABELS))
            )
            last_watch_probs = (
                _ensure_prob_vector(watch_probs[-1])
                if len(watch_probs) > 0
                else np.zeros(len(ACTIVITY_LABELS))
            )
            last_fusion_probs = (
                _ensure_prob_vector(fusion_probs[-1])
                if len(fusion_probs) > 0
                else np.zeros(len(ACTIVITY_LABELS))
            )
            frame_count_copy = frame_count

        # Build figures (outside lock)
        ta_phone = phone_time_accel_list[-PLOT_MAX_POINTS:]
        ax_phone = phone_accel_x_list[-PLOT_MAX_POINTS:]
        ay_phone = phone_accel_y_list[-PLOT_MAX_POINTS:]
        az_phone = phone_accel_z_list[-PLOT_MAX_POINTS:]

        tg_phone = phone_time_gyro_list[-PLOT_MAX_POINTS:]
        gx_phone = phone_gyro_x_list[-PLOT_MAX_POINTS:]
        gy_phone = phone_gyro_y_list[-PLOT_MAX_POINTS:]
        gz_phone = phone_gyro_z_list[-PLOT_MAX_POINTS:]

        ta_watch = watch_time_accel_list[-PLOT_MAX_POINTS:]
        ax_watch = watch_accel_x_list[-PLOT_MAX_POINTS:]
        ay_watch = watch_accel_y_list[-PLOT_MAX_POINTS:]
        az_watch = watch_accel_z_list[-PLOT_MAX_POINTS:]

        tr_watch = watch_time_rot_list[-PLOT_MAX_POINTS:]
        rx_watch = watch_rot_x_list[-PLOT_MAX_POINTS:]
        ry_watch = watch_rot_y_list[-PLOT_MAX_POINTS:]
        rz_watch = watch_rot_z_list[-PLOT_MAX_POINTS:]

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
                html.Span(" | "),
                html.Span("⌚ Watch: ", style={"fontWeight": "bold"}),
                html.Span(
                    f"{watch_samples}/{WINDOW_SIZE} samples",
                    style={
                        "color": (
                            "#27ae60" if watch_samples >= WINDOW_SIZE else "#e74c3c"
                        )
                    },
                ),
                html.Span(" | "),
                html.Span(f"Server: {local_ip}:8000"),
                html.Span(" | "),
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
        # Print full stacktrace to server logs so you can inspect root cause
        print("[ERROR] Dashboard callback exception:", e)
        traceback.print_exc()
        # graceful fallbacks (non-blocking)
        empty_fig = go.Figure()
        fallback_status = html.Div(
            "Dashboard temporarily unavailable (see server logs)."
        )
        return (
            fallback_status,
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
import threading

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING IMU ACTIVITY RECOGNITION DASHBOARD (unbounded storage)")
    print("=" * 70)
    pruning_thread = threading.Thread(target=pruning_worker, daemon=True)
    pruning_thread.start()
    print("[info] Background pruning thread started\n")
    app.run(port=8000, host="0.0.0.0", debug=False, threaded=True)
