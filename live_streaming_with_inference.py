#!/usr/bin/env python3
# realtime_dashboard.py
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
# Get local IP (best-effort)
hostname = socket.gethostname()
try:
    local_ip = socket.gethostbyname(hostname)
except Exception:
    local_ip = "127.0.0.1"
print(f"Server running at: http://{local_ip}:8000")
print(f"Configure Sensor Logger to POST to: http://{local_ip}:8000/data")

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Configuration (adjust as needed)
MAX_DATA_POINTS = 5000  # keep many points so plotting continues long
UPDATE_FREQ_MS = 100  # dashboard refresh interval (ms)
WINDOW_SIZE = 150  # samples per window (3s @ 50Hz)
# For real-time step, I recommend matching training step_size (your finetune used 75).
STEP_SIZE_RT = (
    75  # run inference every STEP_SIZE_RT samples (set to 75 to match training)
)
TARGET_HZ = 50

# Activity labels/colors
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

# Sliding buffers keep full rolling history; we extract last WINDOW_SIZE for windows
phone_buffer = deque(
    maxlen=MAX_DATA_POINTS
)  # each element: dict with ax,ay,az,gx,gy,gz,timestamp
watch_buffer = deque(maxlen=MAX_DATA_POINTS)

# Per-role accel cache to pair gyro -> accel reliably (store recent accels for each role)
accel_cache = {"phone": deque(maxlen=2000), "watch": deque(maxlen=2000)}

# simple sample counters used to trigger inference cadence
sample_counts = {"phone": 0, "watch": 0}

# Prediction history
phone_predictions = deque(maxlen=500)
watch_predictions = deque(maxlen=500)
fusion_predictions = deque(maxlen=500)
prediction_times = deque(maxlen=500)

phone_probs = deque(maxlen=500)
watch_probs = deque(maxlen=500)
fusion_probs = deque(maxlen=500)

frame_count = 0

# -----------------------
# Models (placeholders / load)
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_DIR = Path("finetune/models/dashboard_models")
phone_model = "phone_only_classifier.pth"
watch_model = "watch_only_classifier.pth"
fusion_model = "fusion_classifier.pth"


def try_load_ckpt(path):
    try:
        if path.exists():
            ckpt = torch.load(path, map_location=device)
            return ckpt
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return None


# Try loading checkpoints ()
_ = try_load_ckpt(MODEL_DIR / "phone_only_classifier.pth")
_ = try_load_ckpt(MODEL_DIR / "watch_only_classifier.pth")
_ = try_load_ckpt(MODEL_DIR / "fusion_classifier.pth")


# -----------------------
# Utilities
# -----------------------
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
        "xs",
        "xr",
        "pro",
        "oneplus",
    ]
    watch_keywords = [
        "watch",
        "wrist",
        "fitbit",
        "applewatch",
        "garmin",
        "mi band",
        "wear",
        "galaxywatch",
        "tizen",
    ]
    for kw in watch_keywords:
        if kw in s:
            return "watch"
    for kw in phone_keywords:
        if kw in s:
            return "phone"
    # fallback
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


# Prediction functions (use placeholders if model not loaded)
def predict_phone(window):
    if window is None:
        return "---", np.zeros(len(ACTIVITY_LABELS))
    if phone_model is None:
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
    if watch_model is None:
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
    if fusion_model is None:
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
    """Make predictions for phone/watch/fusion if windows available.
    Called inside buffer_lock to avoid races.
    """
    now = datetime.now()
    phone_win = create_window_from_buffer(phone_buffer)
    watch_win = create_window_from_buffer(watch_buffer)

    if phone_win is not None:
        p_label, p_probs = predict_phone(phone_win)
        phone_predictions.append(p_label)
        phone_probs.append(p_probs)
    if watch_win is not None:
        w_label, w_probs = predict_watch(watch_win)
        watch_predictions.append(w_label)
        watch_probs.append(w_probs)
    if phone_win is not None and watch_win is not None:
        f_label, f_probs = predict_fusion(phone_win, watch_win)
        fusion_predictions.append(f_label)
        fusion_probs.append(f_probs)
        prediction_times.append(now)


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
    # dynamic ranges (safe)
    try:
        if len(time_data) > 0:
            fig["layout"]["xaxis"]["range"] = [min(time_data), max(time_data)]
        all_values = [item for sublist in sensor_data for item in sublist]
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            if y_max - y_min == 0:
                y_margin = 1.0
            else:
                y_margin = (y_max - y_min) * 0.1
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
gyro_cache = {"phone": deque(maxlen=2000), "watch": deque(maxlen=2000)}


@server.route("/data", methods=["POST"])
def data():
    """
    Robust ingestion that pairs accel <-> gyro in either order,
    creates samples on either event (using the most recent counterpart within tolerance),
    and always returns 200 to keep Sensor Logger streaming.
    """
    global frame_count
    if request.method != "POST":
        return "method not allowed", 405

    # parse body
    try:
        payload = json.loads(request.data)
    except Exception as e:
        print("Failed to parse JSON payload:", e)
        return "ok", 200

    samples = payload.get("payload", [])
    if not isinstance(samples, list):
        samples = [payload]

    time_tolerance = timedelta(milliseconds=200)  # pairing tolerance

    with buffer_lock:
        for d in samples:
            try:
                # Debug: print keys and device strings to help identify mis-named devices
                # (This will help you see what the watch actually sends.)
                if "device" in d or "name" in d:
                    print(
                        "[incoming sample keys] device_raw:",
                        d.get("device", None),
                        "name:",
                        d.get("name", None),
                        "keys:",
                        list(d.keys()),
                    )

                # timestamp handling (accept nanoseconds or seconds)
                ts_ns = d.get("time", None)
                if ts_ns is None:
                    ts = datetime.now()
                else:
                    # if value looks like nanoseconds
                    if ts_ns > 1e12:
                        ts = datetime.fromtimestamp(ts_ns / 1_000_000_000)
                    else:
                        # fallback treat as seconds
                        ts = datetime.fromtimestamp(ts_ns)

                # determine role (phone/watch)
                device_raw = str(d.get("device", "unknown"))
                role = identify_device(device_raw)  # 'phone' or 'watch'

                name = d.get("name", "").lower()

                # ----- Accelerometer event -----
                if name == "accelerometer":
                    ax = float(d["values"]["x"])
                    ay = float(d["values"]["y"])
                    az = float(d["values"]["z"])

                    # always append for plotting
                    time_accel.append(ts)
                    accel_x.append(ax)
                    accel_y.append(ay)
                    accel_z.append(az)

                    # push into accel cache
                    accel_cache[role].append(
                        {"ax": ax, "ay": ay, "az": az, "timestamp": ts}
                    )

                    # Try to form a sample immediately if there is a recent gyro nearby
                    paired_gyro = None
                    for g in reversed(gyro_cache[role]):
                        # prefer gyro <= accel_ts, but accept slightly newer if close enough
                        if (
                            abs((ts - g["timestamp"]).total_seconds())
                            <= time_tolerance.total_seconds()
                        ):
                            paired_gyro = g
                            break
                    # fallback: use last known gyro if exists
                    if paired_gyro is None and len(gyro_cache[role]) > 0:
                        paired_gyro = gyro_cache[role][-1]

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
                        if role == "phone":
                            phone_buffer.append(sample)
                            sample_counts["phone"] += 1
                            if sample_counts["phone"] % STEP_SIZE_RT == 0:
                                make_predictions()
                        else:
                            watch_buffer.append(sample)
                            sample_counts["watch"] += 1
                            if sample_counts["watch"] % STEP_SIZE_RT == 0:
                                make_predictions()
                    else:
                        # No gyro yet to pair — that's fine, an accel-only sample is still kept in accel_cache
                        pass

                # ----- Gyroscope event -----
                elif name == "gyroscope":
                    gx = float(d["values"]["x"])
                    gy = float(d["values"]["y"])
                    gz = float(d["values"]["z"])

                    # append for plotting
                    time_gyro.append(ts)
                    gyro_x.append(gx)
                    gyro_y.append(gy)
                    gyro_z.append(gz)

                    # push into gyro cache
                    gyro_cache[role].append(
                        {"gx": gx, "gy": gy, "gz": gz, "timestamp": ts}
                    )

                    # Try to pair with most recent accel in accel_cache for same role
                    paired_accel = None
                    for a in reversed(accel_cache[role]):
                        if (
                            abs((ts - a["timestamp"]).total_seconds())
                            <= time_tolerance.total_seconds()
                        ):
                            paired_accel = a
                            break
                    # fallback: use last accel if exists
                    if paired_accel is None and len(accel_cache[role]) > 0:
                        paired_accel = accel_cache[role][-1]

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
                        if role == "phone":
                            phone_buffer.append(sample)
                            sample_counts["phone"] += 1
                            if sample_counts["phone"] % STEP_SIZE_RT == 0:
                                make_predictions()
                        else:
                            watch_buffer.append(sample)
                            sample_counts["watch"] += 1
                            if sample_counts["watch"] % STEP_SIZE_RT == 0:
                                make_predictions()
                    else:
                        # no accel to pair with yet; we keep gyro in gyro_cache
                        pass

                else:
                    # Unknown sensor type — log for debugging, but don't crash.
                    print(
                        f"[info] Unknown sensor name '{name}' from device '{device_raw}' — keys: {list(d.keys())}"
                    )
                    continue

            except Exception as ex:
                print("Error processing incoming sample:", ex)
                continue

        frame_count += 1

    # Always return 200 OK to keep Sensor Logger from stopping the stream
    return "ok", 200


@server.route("/", methods=["GET"])
def index():
    return "IMU Activity Recognition Dashboard running. POST data to /data endpoint."


# -----------------------
# Dash layout (restored full original layout)
# -----------------------
app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(
                    "🏃 Real-Time IMU Activity Recognition",
                    style={
                        "textAlign": "center",
                        "color": "#2c3e50",
                        "marginBottom": "10px",
                    },
                ),
                html.P(
                    "Streaming sensor data from phone and watch with three-model prediction",
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
        # Connection Status
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
        # Sensor graphs row
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
        # Predictions row
        html.Div(
            [
                # Phone prediction
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
                # Watch prediction
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
                # Fusion prediction
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
        # Prediction timeline
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
# Dash update callback
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
    # Build status text and figures safely (no heavy locks for long)
    with buffer_lock:
        phone_samples = len(phone_buffer)
        watch_samples = len(watch_buffer)
        # copy last values safely
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
        # copy times and sensor arrays for plotting
        ta = list(time_accel)
        ax = list(accel_x)
        ay = list(accel_y)
        az = list(accel_z)
        tg = list(time_gyro)
        gx = list(gyro_x)
        gy = list(gyro_y)
        gz = list(gyro_z)
        pred_times = list(prediction_times)
        ph_preds = list(phone_predictions)
        wh_preds = list(watch_predictions)
        fu_preds = list(fusion_predictions)

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
    print("STARTING IMU ACTIVITY RECOGNITION DASHBOARD")
    print("=" * 70)
    print(f"Dashboard URL: http://{local_ip}:8000")
    print(f"Sensor Logger POST endpoint: http://{local_ip}:8000/data")
    print("=" * 70)
    print("\nConfigure Sensor Logger:")
    print("  1. Set recording mode to 'Push to Server'")
    print(f"  2. Enter URL: http://{local_ip}:8000/data")
    print("  3. Enable Accelerometer and Gyroscope sensors")
    print("  4. Set recording frequency to 50 Hz")
    print("  5. Start recording on both phone and watch")
    print("=" * 70 + "\n")

    # Important: set debug=False in production so the reloader doesn't restart processes.
    app.run(port=8000, host="0.0.0.0", debug=False, threaded=True)
