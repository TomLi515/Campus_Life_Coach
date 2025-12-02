#!/usr/bin/env python3
# realtime_dashboard_wrist_watch.py
import dash
from dash.dependencies import Output, Input
from dash import dcc, html
from datetime import datetime, timedelta
import json
import plotly.graph_objs as go
from collections import deque
from flask import Flask, request
import socket
import numpy as np
import torch
from pathlib import Path
from threading import Lock

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

# Configuration
MAX_DATA_POINTS = 5000
UPDATE_FREQ_MS = 100
WINDOW_SIZE = 150
STEP_SIZE_RT = 75
TARGET_HZ = 50

ACTIVITY_LABELS = ["Walk", "Run", "Sit", "Stand", "Lie"]
ACTIVITY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# -----------------------
# Shared buffers & locks
# -----------------------
buffer_lock = Lock()

time_accel = deque(maxlen=MAX_DATA_POINTS)
accel_x = deque(maxlen=MAX_DATA_POINTS)
accel_y = deque(maxlen=MAX_DATA_POINTS)
accel_z = deque(maxlen=MAX_DATA_POINTS)

time_gyro = deque(maxlen=MAX_DATA_POINTS)
gyro_x = deque(maxlen=MAX_DATA_POINTS)
gyro_y = deque(maxlen=MAX_DATA_POINTS)
gyro_z = deque(maxlen=MAX_DATA_POINTS)

phone_buffer = deque(maxlen=MAX_DATA_POINTS)  # dicts: ax,ay,az,gx,gy,gz,timestamp
watch_buffer = deque(maxlen=MAX_DATA_POINTS)

accel_cache = {"phone": deque(maxlen=2000), "watch": deque(maxlen=2000)}
gyro_cache = {"phone": deque(maxlen=2000), "watch": deque(maxlen=2000)}

sample_counts = {"phone": 0, "watch": 0}

phone_predictions = deque(maxlen=500)
watch_predictions = deque(maxlen=500)
fusion_predictions = deque(maxlen=500)
prediction_times = deque(maxlen=500)

phone_probs = deque(maxlen=500)
watch_probs = deque(maxlen=500)
fusion_probs = deque(maxlen=500)

recent_events = deque(maxlen=200)
recent_device_strings = deque(maxlen=200)

frame_count = 0
both_ready_flag = False
last_both_ready_time = None

# -----------------------
# Models (user may replace these with real objects)
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_DIR = Path("finetune/models/dashboard_models")
# user previously said they've replaced placeholders; we keep these variables and treat them as either filenames or nn.Modules
phone_model = "phone_only_classifier.pth"
watch_model = "watch_only_classifier.pth"
fusion_model = "fusion_classifier.pth"


def is_model_loaded(m):
    return isinstance(m, torch.nn.Module)


# -----------------------
# Utilities
# -----------------------
def identify_device(device_string: str):
    """Heuristics for device strings (fallback)."""
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


# Prediction functions (placeholders if models not actual nn.Modules)
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
        print("phone prediction error:", e)
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
        print("watch prediction error:", e)
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
        print("fusion prediction error:", e)
        return "Error", np.zeros(len(ACTIVITY_LABELS))


def make_predictions():
    """Run phone/watch/fusion predictions if windows available."""
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
        print(f"[phone_pred] {datetime.now().isoformat()} -> {p_label}")

    if watch_ready:
        w_label, w_probs = predict_watch(watch_win)
        watch_predictions.append(w_label)
        watch_probs.append(w_probs)
        print(f"[watch_pred] {datetime.now().isoformat()} -> {w_label}")

    if phone_ready and watch_ready:
        f_label, f_probs = predict_fusion(phone_win, watch_win)
        fusion_predictions.append(f_label)
        fusion_probs.append(f_probs)
        prediction_times.append(now)
        print(
            f"[fusion] {now.isoformat()} fusion_pred={f_label} (phone_buf={len(phone_buffer)} watch_buf={len(watch_buffer)})"
        )

    # both_ready flag notify once when buffers both reach WINDOW_SIZE
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
# Plot helpers
# -----------------------
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
    if len(prediction_times) == 0:
        return go.Figure()
    activity_to_num = {label: i for i, label in enumerate(ACTIVITY_LABELS)}
    phone_nums = [activity_to_num.get(p, -1) for p in phone_predictions]
    watch_nums = [activity_to_num.get(p, -1) for p in watch_predictions]
    fusion_nums = [activity_to_num.get(p, -1) for p in fusion_predictions]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(prediction_times),
            y=phone_nums,
            name="Phone",
            mode="lines+markers",
            line=dict(color="#3498db", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(prediction_times),
            y=watch_nums,
            name="Watch",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(prediction_times),
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


# -----------------------
# Data ingestion endpoint
# -----------------------
@server.route("/data", methods=["POST"])
def data():
    """
    Parsing rules:
    - If name contains 'wrist' (e.g. 'wrist motion') => treat as WATCH data.
      Extract:
        rotationRateX/Y/Z -> gyro x,y,z
        accelerationX/Y/Z -> accel x,y,z
    - If name equals 'gyroscope' or 'accelerometer' => treat as PHONE data (even if device string looks like watch).
      Handle as usual (x,y,z under d['values'])
    - Else: fallback to identify_device(device_string)
    """
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
                name = name_raw.lower()

                # Determine role by name rules first (explicit request)
                if "wrist" in name:  # treat wrist motion events as watch
                    role = "watch"
                elif name in ("gyroscope", "accelerometer"):
                    role = "phone"
                else:
                    # fallback to device string heuristics
                    role = identify_device(str(device_raw))

                print(
                    f"[IDENTIFY] name='{name_raw}' device='{device_raw}' -> role='{role}'"
                )

                # ----- Wrist-motion event (watch) -----
                if "wrist" in name:
                    vals = d.get("values", {})
                    # rotationRate* -> gyro, acceleration* -> accel
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

                    # append for plotting
                    time_accel.append(ts)
                    accel_x.append(ax)
                    accel_y.append(ay)
                    accel_z.append(az)

                    time_gyro.append(ts)
                    gyro_x.append(gx)
                    gyro_y.append(gy)
                    gyro_z.append(gz)

                    # push into caches for pairing
                    accel_cache["watch"].append(
                        {"ax": ax, "ay": ay, "az": az, "timestamp": ts}
                    )
                    gyro_cache["watch"].append(
                        {"gx": gx, "gy": gy, "gz": gz, "timestamp": ts}
                    )

                    # Try to immediately form sample (watch-side)
                    paired_accel = (
                        accel_cache["watch"][-1]
                        if len(accel_cache["watch"]) > 0
                        else None
                    )
                    paired_gyro = (
                        gyro_cache["watch"][-1]
                        if len(gyro_cache["watch"]) > 0
                        else None
                    )
                    if paired_accel is not None and paired_gyro is not None:
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
                            make_predictions()
                        print(
                            f"[debug] wrist-motion -> appended watch sample (total watch_buffer={len(watch_buffer)})"
                        )
                    else:
                        print(
                            "[debug] wrist-motion: cache updated but could not create paired sample yet."
                        )

                # ----- Phone accelerometer / gyroscope events -----
                elif name == "accelerometer":
                    vals = d.get("values", {})
                    ax = float(
                        vals.get(
                            "x",
                            vals.get("accelerationX", vals.get("acceleration_x", 0.0)),
                        )
                    )
                    ay = float(
                        vals.get(
                            "y",
                            vals.get("accelerationY", vals.get("acceleration_y", 0.0)),
                        )
                    )
                    az = float(
                        vals.get(
                            "z",
                            vals.get("accelerationZ", vals.get("acceleration_z", 0.0)),
                        )
                    )

                    time_accel.append(ts)
                    accel_x.append(ax)
                    accel_y.append(ay)
                    accel_z.append(az)

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
                            make_predictions()
                        print(
                            f"[debug] accel(phone) -> appended phone sample (total phone_buffer={len(phone_buffer)})"
                        )
                    else:
                        print(
                            "[debug] accel(phone): cached accel; waiting for gyro to pair."
                        )

                elif name == "gyroscope":
                    vals = d.get("values", {})
                    gx = float(
                        vals.get("x", vals.get("rotationRateX", vals.get("gx", 0.0)))
                    )
                    gy = float(
                        vals.get("y", vals.get("rotationRateY", vals.get("gy", 0.0)))
                    )
                    gz = float(
                        vals.get("z", vals.get("rotationRateZ", vals.get("gz", 0.0)))
                    )

                    time_gyro.append(ts)
                    gyro_x.append(gx)
                    gyro_y.append(gy)
                    gyro_z.append(gz)
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
                            make_predictions()
                        print(
                            f"[debug] gyro(phone) -> appended phone sample (total phone_buffer={len(phone_buffer)})"
                        )
                    else:
                        print(
                            "[debug] gyro(phone): cached gyro; waiting for accel to pair."
                        )

                else:
                    # unknown sensor name: fallback to previous general logic using identify_device
                    vals = d.get("values", {})
                    # attempt to extract x/y/z if present
                    gx = vy = vz = None
                    maybe_x = vals.get("x", None)
                    if maybe_x is not None:
                        # treat as phone by default
                        try:
                            ax = float(vals.get("x", 0.0))
                            ay = float(vals.get("y", 0.0))
                            az = float(vals.get("z", 0.0))
                            # append as phone accel
                            time_accel.append(ts)
                            accel_x.append(ax)
                            accel_y.append(ay)
                            accel_z.append(az)
                            accel_cache["phone"].append(
                                {"ax": ax, "ay": ay, "az": az, "timestamp": ts}
                            )
                            print(
                                "[debug] unknown event but had x,y,z -> treated as phone accel (fallback)."
                            )
                        except Exception:
                            print(
                                "[debug] unknown event fallback did not find numeric x,y,z."
                            )
                    else:
                        print(
                            f"[info] Unknown sensor name '{name_raw}' from device '{device_raw}' — skipping."
                        )

            except Exception as ex:
                print("Error processing incoming sample:", ex)
                continue

        frame_count += 1

        if frame_count % 50 == 0:
            print(
                f"[summary @ frame {frame_count}] phone_buffer={len(phone_buffer)} watch_buffer={len(watch_buffer)} "
                f"accel_cache(phone)={len(accel_cache['phone'])} gyro_cache(phone)={len(gyro_cache['phone'])} "
                f"accel_cache(watch)={len(accel_cache['watch'])} gyro_cache(watch)={len(gyro_cache['watch'])}"
            )

    return "ok", 200


# -----------------------
# Debug endpoint
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
    return resp


@server.route("/", methods=["GET"])
def index():
    return "IMU Activity Recognition Dashboard (wrist-watch mapping) running. POST data to /data endpoint. Visit /debug for state."


# -----------------------
# Dash layout
# -----------------------
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "🏃 Real-Time IMU Activity Recognition (wrist-watch mapping)",
                    style={
                        "textAlign": "center",
                        "color": "#2c3e50",
                        "marginBottom": "10px",
                    },
                ),
                html.P(
                    "Streaming sensor data from phone and watch with three-model prediction — debug enabled",
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
                    [dcc.Graph(id="accel_graph", style={"height": "300px"})],
                    style={"width": "50%", "display": "inline-block", "padding": "5px"},
                ),
                html.Div(
                    [dcc.Graph(id="gyro_graph", style={"height": "300px"})],
                    style={"width": "50%", "display": "inline-block", "padding": "5px"},
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
        dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
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
def update_dashboard(_counter):
    with buffer_lock:
        phone_samples = len(phone_buffer)
        watch_samples = len(watch_buffer)
        last_phone_pred = phone_predictions[-1] if phone_predictions else "---"
        last_watch_pred = watch_predictions[-1] if watch_predictions else "---"
        last_fusion_pred = fusion_predictions[-1] if fusion_predictions else "---"
        last_phone_probs = (
            phone_probs[-1] if phone_probs else np.zeros(len(ACTIVITY_LABELS))
        )
        last_watch_probs = (
            watch_probs[-1] if watch_probs else np.zeros(len(ACTIVITY_LABELS))
        )
        last_fusion_probs = (
            fusion_probs[-1] if fusion_probs else np.zeros(len(ACTIVITY_LABELS))
        )
        ta = list(time_accel)
        ax = list(accel_x)
        ay = list(accel_y)
        az = list(accel_z)
        tg = list(time_gyro)
        gx = list(gyro_x)
        gy = list(gyro_y)
        gz = list(gyro_z)
        pred_times = list(prediction_times)

    status = html.Div(
        [
            html.Span("📱 Phone: ", style={"fontWeight": "bold"}),
            html.Span(
                f"{phone_samples}/{WINDOW_SIZE} samples",
                style={
                    "color": "#27ae60" if phone_samples >= WINDOW_SIZE else "#e74c3c"
                },
            ),
            html.Span(" | ", style={"margin": "0 12px"}),
            html.Span("⌚ Watch: ", style={"fontWeight": "bold"}),
            html.Span(
                f"{watch_samples}/{WINDOW_SIZE} samples",
                style={
                    "color": "#27ae60" if watch_samples >= WINDOW_SIZE else "#e74c3c"
                },
            ),
            html.Span(" | ", style={"margin": "0 12px"}),
            html.Span(f"Server: {local_ip}:8000", style={"fontWeight": "bold"}),
            html.Span(" | ", style={"margin": "0 12px"}),
            html.Span(f"Frames: {frame_count}", style={"color": "#7f8c8d"}),
        ],
        style={"fontSize": "16px", "padding": "8px"},
    )

    accel_fig = create_sensor_graph(
        ta,
        [ax, ay, az],
        ["Accel X", "Accel Y", "Accel Z"],
        "Accelerometer",
        "Acceleration (m/s²)",
        ["#e74c3c", "#3498db", "#2ecc71"],
    )
    gyro_fig = create_sensor_graph(
        tg,
        [gx, gy, gz],
        ["Gyro X", "Gyro Y", "Gyro Z"],
        "Gyroscope",
        "Angular Velocity (rad/s)",
        ["#f39c12", "#9b59b6", "#1abc9c"],
    )

    phone_prob_fig = create_prob_bars(last_phone_probs, "Phone Model Confidence")
    watch_prob_fig = create_prob_bars(last_watch_probs, "Watch Model Confidence")
    fusion_prob_fig = create_prob_bars(last_fusion_probs, "Fusion Model Confidence")
    timeline_fig = create_prediction_timeline()

    return (
        status,
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


# -----------------------
# Run server
# -----------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING IMU ACTIVITY RECOGNITION DASHBOARD (wrist-watch mapping)")
    print("=" * 70)
    print(f"Dashboard URL: http://{local_ip}:8000")
    print(f"Sensor Logger POST endpoint: http://{local_ip}:8000/data")
    print("=" * 70)
    print("\nConfigure Sensor Logger:")
    print("  1. Set recording mode to 'Push to Server'")
    print(f"  2. Enter URL: http://{local_ip}:8000/data")
    print(
        "  3. Enable Accelerometer and Gyroscope sensors (watch sends 'wrist motion' events too)"
    )
    print("  4. Set recording frequency to 50 Hz")
    print("  5. Start recording on both phone and watch")
    print("=" * 70 + "\n")

    app.run(port=8000, host="0.0.0.0", debug=False, threaded=True)
