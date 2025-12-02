#!/usr/bin/env python3
# realtime_dashboard_monitor.py
"""
Real-time IMU dashboard + monitor.

Features:
- robust pairing of accel <-> gyro for phone + wrist (watch)
- unbounded buffers by default (set MAX_DATA_POINTS to cap)
- /data ingestion endpoint always returns 200 to keep Sensor Logger streaming
- Dash UI for real-time plotting and predictions (placeholder models)
- monitor thread that detects reasons why streaming may *appear* to stop
  (no requests, requests but no appends, slow handlers, buffer caps)
- additional detection for "stream keeps coming but plot is not moving"
  comparing last_append_time vs last_dashboard_plot_time and last_dashboard_update_time
- debug endpoints: /debug, /status, /errors, /clear
"""

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
from threading import Lock, Thread
import time
import traceback

# ========================
# Startup info
# ========================
HOSTNAME = socket.gethostname()
try:
    LOCAL_IP = socket.gethostbyname(HOSTNAME)
except Exception:
    LOCAL_IP = "127.0.0.1"

print(f"[INFO] Server running at: http://{LOCAL_IP}:8000")
print(f"[INFO] Configure Sensor Logger to POST to: http://{LOCAL_IP}:8000/data")

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# ------------------------
# Runtime configuration
# ------------------------
# None => unbounded buffers; set integer to cap memory usage
MAX_DATA_POINTS = None

# How many samples the UI will draw (keeps browser responsive)
PLOT_MAX_POINTS = 5000

# Timeline displayed points
PREDICTION_PLOT_MAX = 500

# Realtime / model settings
UPDATE_FREQ_MS = 100
WINDOW_SIZE = 150  # samples per window (3s @ 50Hz)
STEP_SIZE_RT = 75  # infer every 75 appended samples (match training)
TARGET_HZ = 50

# Health-monitor thresholds
NO_PAYLOAD_TIMEOUT_S = 10.0  # no incoming request for this duration => warn
NO_APPEND_TIMEOUT_S = (
    6.0  # requests arriving but no appended samples => pairing/parsing problem
)
SLOW_REQUEST_MS_THRESHOLD = 800.0  # avg request duration in ms to warn
MONITOR_INTERVAL_S = 3.0

# New thresholds for "plot not moving"
NO_DASHBOARD_UPDATE_TIMEOUT_S = 10.0  # dashboard callback hasn't run recently
PLOT_LAG_THRESHOLD_S = (
    2.0  # plot timestamp lags last append by this many seconds -> warn
)

# Activity labels
ACTIVITY_LABELS = ["Walk", "Run", "Sit", "Stand", "Lie"]
ACTIVITY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Verbose toggles
VERBOSE = True
LOG_EXCEPTION_TRACES = (
    False  # set True for full stack traces in console (error_log still stores them)
)


# ========================
# Shared state + buffers
# ========================
def make_deque(maxlen):
    return deque(maxlen=maxlen) if (maxlen is not None) else deque()


buffer_lock = Lock()

time_accel = make_deque(MAX_DATA_POINTS)
accel_x = make_deque(MAX_DATA_POINTS)
accel_y = make_deque(MAX_DATA_POINTS)
accel_z = make_deque(MAX_DATA_POINTS)

time_gyro = make_deque(MAX_DATA_POINTS)
gyro_x = make_deque(MAX_DATA_POINTS)
gyro_y = make_deque(MAX_DATA_POINTS)
gyro_z = make_deque(MAX_DATA_POINTS)

# main sample buffers (unbounded by default)
phone_buffer = make_deque(
    MAX_DATA_POINTS
)  # each element: dict(ax,ay,az,gx,gy,gz,timestamp)
watch_buffer = make_deque(MAX_DATA_POINTS)

# caches to pair accel <-> gyro in either order
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

# recent raw events & device strings for debugging
recent_events = deque(maxlen=500)
recent_device_strings = deque(maxlen=500)

# logs
error_log = deque(maxlen=1000)
event_log = deque(maxlen=2000)
request_durations_ms = deque(maxlen=500)

# timestamps used for diagnostics
last_received_time = None  # when /data was last called
last_append_time = None  # when we last appended any sample to phone/watch buffer
last_dashboard_update_time = None  # when Dash callback last executed (server-side)
last_dashboard_plot_time = (
    None  # newest sample timestamp that was included in the last dashboard response
)

frame_count = 0
both_ready_flag = False
last_both_ready_time = None

# ========================
# Models (placeholders)
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

MODEL_DIR = Path("finetune/models/dashboard_models")
# Replace strings with actual torch.nn.Module instances if you load them later.
phone_model = "phone_only_classifier.pth"
watch_model = "watch_only_classifier.pth"
fusion_model = "fusion_classifier.pth"


def is_model_loaded(m):
    return isinstance(m, torch.nn.Module)


# ========================
# Logging helpers
# ========================
def log_event(level, msg, extra=None):
    ts = datetime.now().isoformat()
    s = f"[{level}] {ts} - {msg}"
    if extra is not None:
        s += f" | {extra}"
    print(s)
    event_log.append({"ts": ts, "level": level, "msg": msg, "extra": extra})


def log_error(msg, tb_text=None):
    ts = datetime.now().isoformat()
    print(f"[ERROR] {ts} - {msg}")
    if LOG_EXCEPTION_TRACES and tb_text:
        print(tb_text)
    error_log.append({"ts": ts, "msg": msg, "trace": tb_text})


# ========================
# Parsing / prediction utils
# ========================
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


# safe prediction wrappers
def predict_phone(window):
    if window is None:
        return "---", np.zeros(len(ACTIVITY_LABELS))
    if not is_model_loaded(phone_model):
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        return ACTIVITY_LABELS[np.argmax(probs)], probs
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
            return ACTIVITY_LABELS[np.argmax(probs)], probs
    except Exception as e:
        tb = traceback.format_exc()
        log_error(f"phone prediction exception: {e}", tb)
        return "Error", np.zeros(len(ACTIVITY_LABELS))


def predict_watch(window):
    if window is None:
        return "---", np.zeros(len(ACTIVITY_LABELS))
    if not is_model_loaded(watch_model):
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        return ACTIVITY_LABELS[np.argmax(probs)], probs
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
            return ACTIVITY_LABELS[np.argmax(probs)], probs
    except Exception as e:
        tb = traceback.format_exc()
        log_error(f"watch prediction exception: {e}", tb)
        return "Error", np.zeros(len(ACTIVITY_LABELS))


def predict_fusion(phone_window, watch_window):
    if phone_window is None or watch_window is None:
        return "---", np.zeros(len(ACTIVITY_LABELS))
    if not is_model_loaded(fusion_model):
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        return ACTIVITY_LABELS[np.argmax(probs)], probs
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
            return ACTIVITY_LABELS[np.argmax(probs)], probs
    except Exception as e:
        tb = traceback.format_exc()
        log_error(f"fusion prediction exception: {e}", tb)
        return "Error", np.zeros(len(ACTIVITY_LABELS))


# central predictions flow
def make_predictions():
    global both_ready_flag, last_both_ready_time
    now = datetime.now()
    phone_win = create_window_from_buffer(phone_buffer)
    watch_win = create_window_from_buffer(watch_buffer)
    phone_ready = phone_win is not None
    watch_ready = watch_win is not None

    if phone_ready:
        p_label, p_probs = predict_phone(phone_win)
        phone_predictions.append(p_label)
        phone_probs.append(p_probs)
        log_event("INFO", f"phone_pred {p_label}", {"phone_buf": len(phone_buffer)})

    if watch_ready:
        w_label, w_probs = predict_watch(watch_win)
        watch_predictions.append(w_label)
        watch_probs.append(w_probs)
        log_event("INFO", f"watch_pred {w_label}", {"watch_buf": len(watch_buffer)})

    if phone_ready and watch_ready:
        f_label, f_probs = predict_fusion(phone_win, watch_win)
        fusion_predictions.append(f_label)
        fusion_probs.append(f_probs)
        prediction_times.append(now)
        log_event(
            "INFO",
            f"fusion_pred {f_label}",
            {"phone_buf": len(phone_buffer), "watch_buf": len(watch_buffer)},
        )

    if len(phone_buffer) >= WINDOW_SIZE and len(watch_buffer) >= WINDOW_SIZE:
        if not both_ready_flag:
            both_ready_flag = True
            last_both_ready_time = datetime.now()
            log_event(
                "INFO",
                "BOTH buffers reached WINDOW_SIZE",
                {"time": last_both_ready_time.isoformat()},
            )
    else:
        both_ready_flag = False


# ========================
# Plot helpers
# ========================
def create_sensor_graph(time_data, sensor_data, names, title, yaxis_label, colors):
    data = []
    for d, name, color in zip(sensor_data, names, colors):
        data.append(
            go.Scatter(
                x=list(time_data), y=list(d), name=name, line=dict(color=color, width=2)
            )
        )
    layout = go.Layout(
        title=dict(text=title, font=dict(size=14, color="#2c3e50")),
        xaxis=dict(title="Time", showgrid=True, gridcolor="#ecf0f1", type="date"),
        yaxis=dict(title=yaxis_label, showgrid=True, gridcolor="#ecf0f1"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=30, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig = {"data": data, "layout": layout}
    try:
        if len(time_data) > 0:
            fig["layout"]["xaxis"]["range"] = [min(time_data), max(time_data)]
        all_vals = [item for sublist in sensor_data for item in sublist]
        if all_vals:
            y_min = min(all_vals)
            y_max = max(all_vals)
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
    if len(prediction_times) == 0:
        return go.Figure()
    activity_to_num = {label: i for i, label in enumerate(ACTIVITY_LABELS)}
    last_times = list(prediction_times)[-PREDICTION_PLOT_MAX:]
    phone_nums = [
        activity_to_num.get(p, -1)
        for p in list(phone_predictions)[-PREDICTION_PLOT_MAX:]
    ]
    watch_nums = [
        activity_to_num.get(p, -1)
        for p in list(watch_predictions)[-PREDICTION_PLOT_MAX:]
    ]
    fusion_nums = [
        activity_to_num.get(p, -1)
        for p in list(fusion_predictions)[-PREDICTION_PLOT_MAX:]
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=last_times,
            y=phone_nums,
            name="Phone",
            mode="lines+markers",
            line=dict(color="#3498db", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=last_times,
            y=watch_nums,
            name="Watch",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=last_times,
            y=fusion_nums,
            name="Fusion",
            mode="lines+markers",
            line=dict(color="#27ae60", width=2),
        )
    )
    fig.update_layout(
        title="Prediction Timeline",
        xaxis=dict(title="Time"),
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


# ========================
# /data ingestion endpoint
# ========================
@server.route("/data", methods=["POST"])
def data():
    """
    Robust ingestion:
    - 'wrist' in name => watch (rotationRate*/acceleration*)
    - name == 'gyroscope'|'accelerometer' => phone
    - fallback: identify_device(device_string)
    Always returns 200 to keep Sensor Logger streaming.
    """
    global frame_count, last_received_time, last_append_time
    start = time.time()
    received_ok = False
    try:
        raw = request.data
        last_received_time = datetime.now()
        if VERBOSE:
            log_event("INFO", "Received /data request", {"size_bytes": len(raw)})
        try:
            payload = json.loads(raw)
        except Exception as e:
            tb = traceback.format_exc()
            log_error("JSON parse failure in /data", tb)
            request_durations_ms.append(int((time.time() - start) * 1000))
            return "ok", 200

        samples = payload.get("payload", [])
        if not isinstance(samples, list):
            samples = [payload]

        time_tolerance = timedelta(milliseconds=200)

        with buffer_lock:
            appended_any = False
            for d in samples:
                recent_events.append({"ts": datetime.now().isoformat(), "raw": d})
                device_raw = d.get("device", None)
                recent_device_strings.append(str(device_raw))
                if VERBOSE:
                    print("---- incoming sample ----")
                    print(
                        "device:",
                        device_raw,
                        "name:",
                        d.get("name"),
                        "time:",
                        d.get("time"),
                    )
                    print("values keys:", list(d.get("values", {}).keys()))
                    print("-------------------------")

                # parse timestamp
                ts_field = d.get("time", None)
                if ts_field is None:
                    ts = datetime.now()
                else:
                    try:
                        if isinstance(ts_field, (int, float)) and ts_field > 1e12:
                            ts = datetime.fromtimestamp(ts_field / 1_000_000_000)
                        else:
                            ts = datetime.fromtimestamp(ts_field)
                    except Exception:
                        ts = datetime.now()

                name_raw = d.get("name", "")
                name = str(name_raw).lower()

                # role by name-first rules
                if "wrist" in name:
                    role = "watch"
                elif name in ("gyroscope", "accelerometer"):
                    role = "phone"
                else:
                    role = identify_device(str(device_raw))

                if VERBOSE:
                    log_event(
                        "INFO",
                        "identify_device",
                        {
                            "raw_device": device_raw,
                            "name": name_raw,
                            "assigned_role": role,
                        },
                    )

                # ---------- watch wrist motion ----------
                if "wrist" in name:
                    vals = d.get("values", {})
                    try:
                        gx = float(
                            vals.get("rotationRateX", vals.get("rotationRate_x", 0.0))
                        )
                        gyv = float(
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
                    except Exception as e:
                        tb = traceback.format_exc()
                        log_error("Value parsing error for wrist event", tb)
                        continue

                    time_accel.append(ts)
                    accel_x.append(ax)
                    accel_y.append(ay)
                    accel_z.append(az)
                    time_gyro.append(ts)
                    gyro_x.append(gx)
                    gyro_y.append(gyv)
                    gyro_z.append(gz)
                    accel_cache["watch"].append(
                        {"ax": ax, "ay": ay, "az": az, "timestamp": ts}
                    )
                    gyro_cache["watch"].append(
                        {"gx": gx, "gy": gyv, "gz": gz, "timestamp": ts}
                    )

                    # create sample if caches have entries
                    if len(accel_cache["watch"]) > 0 and len(gyro_cache["watch"]) > 0:
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
                        appended_any = True
                        sample_counts["watch"] += 1
                        last_append_time = datetime.now()
                        if sample_counts["watch"] % 50 == 0:
                            log_event(
                                "INFO",
                                "appended watch sample",
                                {"watch_buffer_len": len(watch_buffer)},
                            )
                        if sample_counts["watch"] % STEP_SIZE_RT == 0:
                            make_predictions()
                    else:
                        log_event(
                            "WARN",
                            "wrist event cached but pairing not yet possible",
                            {
                                "accel_cache_watch": len(accel_cache["watch"]),
                                "gyro_cache_watch": len(gyro_cache["watch"]),
                            },
                        )

                # ---------- phone accelerometer ----------
                elif name == "accelerometer":
                    vals = d.get("values", {})
                    try:
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
                    except Exception as e:
                        tb = traceback.format_exc()
                        log_error("Value parsing error for phone accel", tb)
                        continue

                    time_accel.append(ts)
                    accel_x.append(ax)
                    accel_y.append(ay)
                    accel_z.append(az)
                    accel_cache["phone"].append(
                        {"ax": ax, "ay": ay, "az": az, "timestamp": ts}
                    )

                    # pair with nearby gyro or fallback to last known
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
                        appended_any = True
                        sample_counts["phone"] += 1
                        last_append_time = datetime.now()
                        if sample_counts["phone"] % 50 == 0:
                            log_event(
                                "INFO",
                                "appended phone sample",
                                {"phone_buffer_len": len(phone_buffer)},
                            )
                        if sample_counts["phone"] % STEP_SIZE_RT == 0:
                            make_predictions()
                    else:
                        log_event(
                            "WARN",
                            "phone accel cached; waiting for gyro pairing",
                            {
                                "accel_cache_phone": len(accel_cache["phone"]),
                                "gyro_cache_phone": len(gyro_cache["phone"]),
                            },
                        )

                # ---------- phone gyroscope ----------
                elif name == "gyroscope":
                    vals = d.get("values", {})
                    try:
                        gx = float(
                            vals.get(
                                "x", vals.get("rotationRateX", vals.get("gx", 0.0))
                            )
                        )
                        gyv = float(
                            vals.get(
                                "y", vals.get("rotationRateY", vals.get("gy", 0.0))
                            )
                        )
                        gz = float(
                            vals.get(
                                "z", vals.get("rotationRateZ", vals.get("gz", 0.0))
                            )
                        )
                    except Exception as e:
                        tb = traceback.format_exc()
                        log_error("Value parsing error for phone gyro", tb)
                        continue

                    time_gyro.append(ts)
                    gyro_x.append(gx)
                    gyro_y.append(gyv)
                    gyro_z.append(gz)
                    gyro_cache["phone"].append(
                        {"gx": gx, "gy": gyv, "gz": gz, "timestamp": ts}
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
                            "gy": gyv,
                            "gz": gz,
                            "timestamp": ts,
                        }
                        phone_buffer.append(sample)
                        appended_any = True
                        sample_counts["phone"] += 1
                        last_append_time = datetime.now()
                        if sample_counts["phone"] % 50 == 0:
                            log_event(
                                "INFO",
                                "appended phone sample (gyro event)",
                                {"phone_buffer_len": len(phone_buffer)},
                            )
                        if sample_counts["phone"] % STEP_SIZE_RT == 0:
                            make_predictions()
                    else:
                        log_event(
                            "WARN",
                            "phone gyro cached; waiting for accel pairing",
                            {
                                "accel_cache_phone": len(accel_cache["phone"]),
                                "gyro_cache_phone": len(gyro_cache["phone"]),
                            },
                        )

                # ---------- fallback heuristics ----------
                else:
                    vals = d.get("values", {})
                    if isinstance(vals, dict) and (
                        "x" in vals and "y" in vals and "z" in vals
                    ):
                        try:
                            ax = float(vals.get("x", 0.0))
                            ay = float(vals.get("y", 0.0))
                            az = float(vals.get("z", 0.0))
                            time_accel.append(ts)
                            accel_x.append(ax)
                            accel_y.append(ay)
                            accel_z.append(az)
                            accel_cache["phone"].append(
                                {"ax": ax, "ay": ay, "az": az, "timestamp": ts}
                            )
                            log_event(
                                "WARN",
                                "fallback: unknown event with x,y,z treated as phone accel",
                                {"name": name_raw},
                            )
                        except Exception:
                            log_event(
                                "WARN",
                                "fallback failed to parse x,y,z",
                                {"name": name_raw},
                            )
                    else:
                        log_event(
                            "INFO",
                            f"unknown sensor name skipped: {name_raw}",
                            {"device": device_raw},
                        )

            if appended_any:
                last_append_time = datetime.now()

        received_ok = True

    except Exception as ex:
        tb = traceback.format_exc()
        log_error(f"Exception while handling /data: {ex}", tb)

    finally:
        duration_ms = int((time.time() - start) * 1000)
        request_durations_ms.append(duration_ms)
        if not received_ok:
            log_error(
                "Failed to completely process /data request; check JSON format and payload keys. Consult /errors and logs."
            )
        else:
            if VERBOSE and duration_ms > SLOW_REQUEST_MS_THRESHOLD:
                log_event(
                    "WARN",
                    f"/data handler slow (duration_ms={duration_ms})",
                    {"threshold_ms": SLOW_REQUEST_MS_THRESHOLD},
                )
        try:
            # frame counter used for UI only
            global frame_count
            frame_count += 1
        except Exception:
            pass
        return "ok", 200


# ========================
# Monitor thread (enhanced detection)
# ========================
def monitor_loop():
    """
    Detect:
    - no /data requests (Sensor Logger stopped or wrong URL)
    - requests arriving but no appends (parsing mismatch)
    - slow handlers (may cause client timeouts)
    - buffers accidentally bounded to small maxlen
    - dashboard not updating (UI disconnected / JS throttled)
    - dashboard plot not including latest appended samples (plot lag)
    """
    while True:
        try:
            now = datetime.now()
            with buffer_lock:
                lr = last_received_time
                la = last_append_time
                ldu = last_dashboard_update_time
                ldp = last_dashboard_plot_time
                avg_req = (
                    (sum(request_durations_ms) / len(request_durations_ms))
                    if len(request_durations_ms) > 0
                    else 0.0
                )
                phone_buf_len = len(phone_buffer)
                watch_buf_len = len(watch_buffer)

                # 1) No /data requests at all recently
                if lr is None:
                    log_event(
                        "WARN",
                        "No /data requests received yet — check Sensor Logger URL and connectivity",
                        {"suggestion": f"POST to http://{LOCAL_IP}:8000/data"},
                    )
                else:
                    secs_since_recv = (now - lr).total_seconds()
                    if secs_since_recv > NO_PAYLOAD_TIMEOUT_S:
                        msg = f"No /data requests for {secs_since_recv:.1f}s."
                        suggestions = [
                            "1) Ensure Sensor Logger is configured to 'Push to Server' and the URL is correct.",
                            "2) Check that phone/watch are recording and network (Wi-Fi) is connected.",
                            "3) If multiple servers, confirm correct IP/port.",
                        ]
                        log_error(msg, "\n".join(suggestions))

                # 2) Requests arriving but no samples appended
                if la is not None:
                    secs_since_append = (now - la).total_seconds()
                    if secs_since_append > NO_APPEND_TIMEOUT_S:
                        if (
                            lr is not None
                            and (now - lr).total_seconds() < NO_PAYLOAD_TIMEOUT_S
                        ):
                            msg = f"Requests arriving but no samples appended for {secs_since_append:.1f}s."
                            suggestions = [
                                "Likely causes: payload field names changed ('name' / 'values' keys) or pairing logic failing.",
                                "Check /debug to inspect recent raw events and device strings.",
                                "Ensure 'accelerometer'/'gyroscope' and 'wrist' naming rules still match your Sensor Logger output.",
                            ]
                            log_error(msg, "\n".join(suggestions))
                else:
                    if (
                        lr is not None
                        and (now - lr).total_seconds() < NO_PAYLOAD_TIMEOUT_S
                    ):
                        log_event(
                            "WARN",
                            "Requests received but never appended a sample yet — inspect /debug for raw events",
                            None,
                        )

                # 3) Slow processing
                if avg_req > SLOW_REQUEST_MS_THRESHOLD:
                    log_event(
                        "WARN",
                        f"Avg /data processing time high: {avg_req:.1f} ms. This may cause client timeouts or dropped frames.",
                        {"suggest": "Reduce logs or simplify parsing"},
                    )

                # 4) Accidental bounded buffers smaller than WINDOW_SIZE
                if (
                    isinstance(phone_buffer, deque)
                    and getattr(phone_buffer, "maxlen", None) is not None
                ):
                    if (
                        phone_buffer.maxlen is not None
                        and phone_buffer.maxlen < WINDOW_SIZE
                    ):
                        log_error(
                            f"phone_buffer.maxlen={phone_buffer.maxlen} < WINDOW_SIZE ({WINDOW_SIZE}). This will cap samples.",
                            None,
                        )
                if (
                    isinstance(watch_buffer, deque)
                    and getattr(watch_buffer, "maxlen", None) is not None
                ):
                    if (
                        watch_buffer.maxlen is not None
                        and watch_buffer.maxlen < WINDOW_SIZE
                    ):
                        log_error(
                            f"watch_buffer.maxlen={watch_buffer.maxlen} < WINDOW_SIZE ({WINDOW_SIZE}). This will cap samples.",
                            None,
                        )

                # 5) Dashboard not updating (server-side callback hasn't run recently)
                if ldu is None:
                    # dashboard hasn't run yet
                    log_event(
                        "INFO",
                        "Dashboard callback has not run yet (no UI connection observed).",
                        None,
                    )
                else:
                    secs_since_dash = (now - ldu).total_seconds()
                    if secs_since_dash > NO_DASHBOARD_UPDATE_TIMEOUT_S:
                        # data incoming but UI not updating
                        if (
                            lr is not None
                            and (now - lr).total_seconds() < NO_PAYLOAD_TIMEOUT_S
                        ):
                            msg = f"Dashboard callback last ran {secs_since_dash:.1f}s ago while /data kept arriving."
                            suggestions = [
                                "1) Confirm browser tab is open and not heavily throttled (background tabs can pause timers).",
                                "2) Check browser console for WebSocket / network errors.",
                                "3) Ensure dcc.Interval component is present (it is) and not blocked by CSP or extensions.",
                                "4) Try opening the dashboard in another browser or machine to isolate client-side issues.",
                            ]
                            log_error(msg, "\n".join(suggestions))

                # 6) Dashboard plot lags latest appended samples
                if la and ldp:
                    lag_sec = (la - ldp).total_seconds()
                    if lag_sec > PLOT_LAG_THRESHOLD_S:
                        msg = f"Dashboard plot newest timestamp lags last appended sample by {lag_sec:.1f}s."
                        suggestions = [
                            "Possible causes:",
                            "- The Dash callback is running but returned a figure that doesn't include newest samples.",
                            "- Browser may be dropping updates because plots are very large (reduce PLOT_MAX_POINTS).",
                            "- Lock contention: ensure app isn't blocking for long in /data handler.",
                            "Check /debug and /status for last timestamps and consider lowering UPDATE_FREQ_MS or PLOT_MAX_POINTS.",
                        ]
                        log_error(msg, "\n".join(suggestions))

                # summary
                log_event(
                    "INFO",
                    "monitor summary",
                    {
                        "phone_buffer_len": phone_buf_len,
                        "watch_buffer_len": watch_buf_len,
                        "avg_req_ms": avg_req,
                        "last_received_age_s": (
                            (now - lr).total_seconds() if lr else None
                        ),
                        "last_append_age_s": (now - la).total_seconds() if la else None,
                        "last_dash_age_s": (now - ldu).total_seconds() if ldu else None,
                    },
                )

        except Exception as ex:
            tb = traceback.format_exc()
            log_error("monitor loop exception", tb)
        time.sleep(MONITOR_INTERVAL_S)


# start monitor thread
monitor_thread = Thread(target=monitor_loop, daemon=True)
monitor_thread.start()


# ========================
# Debug & admin endpoints
# ========================
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
        model_status = {
            "phone_loaded": is_model_loaded(phone_model),
            "watch_loaded": is_model_loaded(watch_model),
            "fusion_loaded": is_model_loaded(fusion_model),
        }
        resp = {
            "timestamp": datetime.now().isoformat(),
            "phone_buffer_len": phone_len,
            "watch_buffer_len": watch_len,
            "last_phone_sample_ts": last_phone_ts,
            "last_watch_sample_ts": last_watch_ts,
            "last_prediction_time": last_pred_time,
            "last_received_time": (
                last_received_time.isoformat() if last_received_time else None
            ),
            "last_append_time": (
                last_append_time.isoformat() if last_append_time else None
            ),
            "last_dashboard_update_time": (
                last_dashboard_update_time.isoformat()
                if last_dashboard_update_time
                else None
            ),
            "last_dashboard_plot_time": (
                last_dashboard_plot_time.isoformat()
                if last_dashboard_plot_time
                else None
            ),
            "sample_counts": sample_counts.copy(),
            "both_ready_flag": both_ready_flag,
            "last_both_ready_time": (
                last_both_ready_time.isoformat() if last_both_ready_time else None
            ),
            "recent_device_strings": list(recent_device_strings)[-50:],
            "recent_raw_events": list(recent_events)[-20:],
            "model_status": model_status,
            "monitor_interval_s": MONITOR_INTERVAL_S,
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
                "last_received_time": (
                    last_received_time.isoformat() if last_received_time else None
                ),
                "last_append_time": (
                    last_append_time.isoformat() if last_append_time else None
                ),
                "last_dashboard_update_time": (
                    last_dashboard_update_time.isoformat()
                    if last_dashboard_update_time
                    else None
                ),
                "last_dashboard_plot_time": (
                    last_dashboard_plot_time.isoformat()
                    if last_dashboard_plot_time
                    else None
                ),
                "avg_request_ms": (
                    (sum(request_durations_ms) / len(request_durations_ms))
                    if len(request_durations_ms)
                    else 0.0
                ),
            }
        )


@server.route("/errors", methods=["GET"])
def get_errors():
    return jsonify(list(error_log)[-200:])


@server.route("/clear", methods=["POST", "GET"])
def clear_buffers():
    with buffer_lock:
        time_accel.clear()
        accel_x.clear()
        accel_y.clear()
        accel_z.clear()
        time_gyro.clear()
        gyro_x.clear()
        gyro_y.clear()
        gyro_z.clear()
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
        error_log.clear()
        event_log.clear()
        request_durations_ms.clear()
    log_event("INFO", "Cleared buffers and logs via /clear", None)
    return jsonify({"status": "cleared"})


@server.route("/", methods=["GET"])
def index():
    return f"IMU dashboard running. POST to /data. Debug: /debug, Status: /status, Errors: /errors, Clear: /clear"


# ========================
# Dash UI layout & callbacks
# ========================
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "🏃 Real-Time IMU Activity Recognition (monitoring)",
                    style={"textAlign": "center"},
                ),
                html.P(
                    "Streaming sensor data from phone and watch — monitoring enabled",
                    style={"textAlign": "center"},
                ),
            ]
        ),
        html.Div(id="connection-status", style={"padding": "10px"}),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="accel_graph", style={"height": "300px"})],
                    style={"width": "50%", "display": "inline-block"},
                ),
                html.Div(
                    [dcc.Graph(id="gyro_graph", style={"height": "300px"})],
                    style={"width": "50%", "display": "inline-block"},
                ),
            ]
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3("📱 Phone Model"),
                        html.Div(id="phone-prediction"),
                        dcc.Graph(id="phone_probs", style={"height": "200px"}),
                    ],
                    style={"width": "33%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.H3("⌚ Watch Model"),
                        html.Div(id="watch-prediction"),
                        dcc.Graph(id="watch_probs", style={"height": "200px"}),
                    ],
                    style={"width": "33%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        html.H3("🔗 Fusion Model"),
                        html.Div(id="fusion-prediction"),
                        dcc.Graph(id="fusion_probs", style={"height": "200px"}),
                    ],
                    style={"width": "33%", "display": "inline-block"},
                ),
            ]
        ),
        dcc.Graph(id="prediction_timeline", style={"height": "250px"}),
        dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f8f9fa",
        "minHeight": "100vh",
    },
)


@app.callback(
    [
        Output("connection-status", "children"),
        Output("accel_graph", "figure"),
        Output("gyro_graph", "figure"),
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
def update_dashboard(_):
    """
    Called frequently by the client's dcc.Interval.
    We update last_dashboard_update_time (server-side) and compute the
    newest timestamp included in the data returned (last_dashboard_plot_time).
    The monitor compares these to last_append_time to detect "plot not moving".
    """
    global last_dashboard_update_time, last_dashboard_plot_time
    with buffer_lock:
        phone_samples = len(phone_buffer)
        watch_samples = len(watch_buffer)
        last_phone_pred = phone_predictions[-1] if len(phone_predictions) > 0 else "---"
        last_watch_pred = watch_predictions[-1] if len(watch_predictions) > 0 else "---"
        last_fusion_pred = (
            fusion_predictions[-1] if len(fusion_predictions) > 0 else "---"
        )
        last_phone_probs = (
            phone_probs[-1] if len(phone_probs) > 0 else np.zeros(len(ACTIVITY_LABELS))
        )
        last_watch_probs = (
            watch_probs[-1] if len(watch_probs) > 0 else np.zeros(len(ACTIVITY_LABELS))
        )
        last_fusion_probs = (
            fusion_probs[-1]
            if len(fusion_probs) > 0
            else np.zeros(len(ACTIVITY_LABELS))
        )

        slice_tail = None if PLOT_MAX_POINTS is None else -PLOT_MAX_POINTS
        ta = list(time_accel)[slice_tail:]
        ax = list(accel_x)[slice_tail:]
        ay = list(accel_y)[slice_tail:]
        az = list(accel_z)[slice_tail:]
        tg = list(time_gyro)[slice_tail:]
        gx = list(gyro_x)[slice_tail:]
        gy = list(gyro_y)[slice_tail:]
        gz = list(gyro_z)[slice_tail:]

        # compute newest timestamp included in this returned plot (if any)
        newest_ts = None
        if len(ta) > 0:
            try:
                newest_ts = ta[-1]
            except Exception:
                newest_ts = None
        elif len(tg) > 0:
            try:
                newest_ts = tg[-1]
            except Exception:
                newest_ts = None

        # update dashboard tracking timestamps
        last_dashboard_update_time = datetime.now()
        if newest_ts is not None:
            last_dashboard_plot_time = newest_ts

    status_div = html.Div(
        [
            html.Div(f"Server: {LOCAL_IP}:8000"),
            html.Div(
                f"phone_buffer_len: {phone_samples}, watch_buffer_len: {watch_samples}, frames: {frame_count}"
            ),
            html.Div(
                f"last_received: {last_received_time.isoformat() if last_received_time else 'never'}, last_append: {last_append_time.isoformat() if last_append_time else 'never'}"
            ),
            html.Div(
                f"last_dashboard_update: {last_dashboard_update_time.isoformat() if last_dashboard_update_time else 'never'}, last_dashboard_plot_time: {last_dashboard_plot_time.isoformat() if last_dashboard_plot_time else 'never'}"
            ),
        ]
    )

    accel_fig = create_sensor_graph(
        ta,
        [ax, ay, az],
        ["Accel X", "Accel Y", "Accel Z"],
        "Accelerometer",
        "m/s²",
        ["#e74c3c", "#3498db", "#2ecc71"],
    )
    gyro_fig = create_sensor_graph(
        tg,
        [gx, gy, gz],
        ["Gyro X", "Gyro Y", "Gyro Z"],
        "Gyroscope",
        "rad/s",
        ["#f39c12", "#9b59b6", "#1abc9c"],
    )
    phone_prob_fig = create_prob_bars(last_phone_probs, "Phone Model Confidence")
    watch_prob_fig = create_prob_bars(last_watch_probs, "Watch Model Confidence")
    fusion_prob_fig = create_prob_bars(last_fusion_probs, "Fusion Model Confidence")
    timeline_fig = create_prediction_timeline()

    return (
        status_div,
        accel_fig,
        gyro_fig,
        last_phone_pred,
        last_watch_pred,
        last_fusion_pred,
        phone_prob_fig,
        watch_prob_fig,
        fusion_prob_fig,
        timeline_fig,
    )


# ========================
# Run server
# ========================
if __name__ == "__main__":
    log_event(
        "INFO",
        "STARTING IMU ACTIVITY RECOGNITION DASHBOARD (monitoring enabled)",
        {"url": f"http://{LOCAL_IP}:8000"},
    )
    if MAX_DATA_POINTS is None:
        log_event(
            "WARN",
            "Buffers are UNBOUNDED (MAX_DATA_POINTS=None). Stop server to free memory when done.",
            None,
        )
    app.run(port=8000, host="0.0.0.0", debug=False, threaded=True)
