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
import torch.nn as nn
from pathlib import Path

# Get local IP
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print(f"Server running at: http://{local_ip}:8000")
print(f"Configure Sensor Logger to POST to: http://{local_ip}:8000/data")

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

# Configuration
MAX_DATA_POINTS = 1000
UPDATE_FREQ_MS = 100
WINDOW_SIZE = 150  # samples per window (3 seconds at 50Hz)
STEP_SIZE = 25  # step for sliding window (0.5 seconds)
TARGET_HZ = 50

# Activity labels
ACTIVITY_LABELS = ["Walk", "Run", "Sit", "Stand", "Lie"]
ACTIVITY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Data storage - raw sensor data (for visualization)
time_accel = deque(maxlen=MAX_DATA_POINTS)
accel_x = deque(maxlen=MAX_DATA_POINTS)
accel_y = deque(maxlen=MAX_DATA_POINTS)
accel_z = deque(maxlen=MAX_DATA_POINTS)

time_gyro = deque(maxlen=MAX_DATA_POINTS)
gyro_x = deque(maxlen=MAX_DATA_POINTS)
gyro_y = deque(maxlen=MAX_DATA_POINTS)
gyro_z = deque(maxlen=MAX_DATA_POINTS)

# Sliding window buffer for predictions (samples are dicts with ax,ay,az,gx,gy,gz,timestamp)
phone_buffer = deque(
    maxlen=MAX_DATA_POINTS
)  # we keep larger buffer; create_window_from_buffer extracts last WINDOW_SIZE
watch_buffer = deque(maxlen=MAX_DATA_POINTS)

# Per-device accel cache to correctly pair accel->gyro (stores recent accels for each device role)
accel_cache = {"phone": deque(maxlen=500), "watch": deque(maxlen=500)}

# Per-device sample counters (to trigger predictions at STEP_SIZE intervals)
sample_counts = {"phone": 0, "watch": 0}

# Prediction storage
phone_predictions = deque(maxlen=100)
watch_predictions = deque(maxlen=100)
fusion_predictions = deque(maxlen=100)
prediction_times = deque(maxlen=100)

phone_probs = deque(maxlen=100)
watch_probs = deque(maxlen=100)
fusion_probs = deque(maxlen=100)

# Frame counter
frame_count = 0

# ===========================
# Model Loading
# ===========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load your models here
MODEL_DIR = Path("finetune/models/dashboard_models")  # Adjust path as needed

phone_model = None
watch_model = None
fusion_model = None

try:
    # Load phone model
    phone_ckpt_path = MODEL_DIR / "phone_only_classifier.pth"
    if phone_ckpt_path.exists():
        # safer load
        phone_checkpoint = torch.load(phone_ckpt_path, map_location=device)
        # initialize and load your model appropriately here
        # phone_model = YourPhoneModel(...)
        # phone_model.load_state_dict(phone_checkpoint['model_state_dict'])
        # phone_model.to(device).eval()
        print("✓ Phone model loaded")
    else:
        print(f"⚠ Phone model not found at {phone_ckpt_path}")
except Exception as e:
    print(f"✗ Error loading phone model: {e}")

try:
    # Load watch model
    watch_ckpt_path = MODEL_DIR / "watch_only_classifier.pth"
    if watch_ckpt_path.exists():
        watch_checkpoint = torch.load(watch_ckpt_path, map_location=device)
        # watch_model = YourWatchModel(...)
        # watch_model.load_state_dict(watch_checkpoint['model_state_dict'])
        # watch_model.to(device).eval()
        print("✓ Watch model loaded")
    else:
        print(f"⚠ Watch model not found at {watch_ckpt_path}")
except Exception as e:
    print(f"✗ Error loading watch model: {e}")

try:
    # Load fusion model
    fusion_ckpt_path = MODEL_DIR / "fusion_classifier.pth"
    if fusion_ckpt_path.exists():
        fusion_checkpoint = torch.load(fusion_ckpt_path, map_location=device)
        # fusion_model = YourFusionModel(...)
        # fusion_model.load_state_dict(fusion_checkpoint['model_state_dict'])
        # fusion_model.to(device).eval()
        print("✓ Fusion model loaded")
    else:
        print(f"⚠ Fusion model not found at {fusion_ckpt_path}")
except Exception as e:
    print(f"✗ Error loading fusion model: {e}")

# ===========================
# Helpers
# ===========================


def identify_device(device_string: str):
    """
    Robust heuristic to determine whether a device string refers to phone or watch.
    Returns 'phone' or 'watch' (defaults to 'phone').
    """
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
        "galaxywatch",
        "fitbit",
        "applewatch",
        "wear",
        "mi band",
        "garmin",
        "tizen",
    ]
    for kw in watch_keywords:
        if kw in s:
            return "watch"
    for kw in phone_keywords:
        if kw in s:
            return "phone"
    # fallback: if string is short and contains digits, could be watch; otherwise phone
    if "watch" in s or "w" == s:
        return "watch"
    return "phone"


def normalize_window(window):
    """Normalize a single window (6, 150) using z-score."""
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-8
    return (window - mean) / std


def create_window_from_buffer(buffer):
    """
    Create a window array from buffer.
    Buffer contains dicts with keys: ax, ay, az, gx, gy, gz, timestamp
    Returns: numpy array of shape (6, window_size) or None if not enough samples
    """
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


# ===========================
# Prediction Functions
# ===========================


def predict_phone(window):
    """Make prediction using phone model."""
    if phone_model is None:
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        pred_idx = np.argmax(probs)
        return ACTIVITY_LABELS[pred_idx], probs
    try:
        with torch.no_grad():
            window_norm = normalize_window(window)
            x = torch.from_numpy(window_norm).unsqueeze(0).float().to(device)
            outputs = phone_model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            return ACTIVITY_LABELS[pred_idx], probs
    except Exception as e:
        print(f"Error in phone prediction: {e}")
        probs = np.zeros(len(ACTIVITY_LABELS))
        return "Error", probs


def predict_watch(window):
    """Make prediction using watch model."""
    if watch_model is None:
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        pred_idx = np.argmax(probs)
        return ACTIVITY_LABELS[pred_idx], probs
    try:
        with torch.no_grad():
            window_norm = normalize_window(window)
            x = torch.from_numpy(window_norm).unsqueeze(0).float().to(device)
            outputs = watch_model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            return ACTIVITY_LABELS[pred_idx], probs
    except Exception as e:
        print(f"Error in watch prediction: {e}")
        probs = np.zeros(len(ACTIVITY_LABELS))
        return "Error", probs


def predict_fusion(phone_window, watch_window):
    """Make prediction using fusion model."""
    if fusion_model is None:
        probs = np.random.dirichlet(np.ones(len(ACTIVITY_LABELS)))
        pred_idx = np.argmax(probs)
        return ACTIVITY_LABELS[pred_idx], probs
    try:
        with torch.no_grad():
            phone_norm = normalize_window(phone_window)
            watch_norm = normalize_window(watch_window)
            x_phone = torch.from_numpy(phone_norm).unsqueeze(0).float().to(device)
            x_watch = torch.from_numpy(watch_norm).unsqueeze(0).float().to(device)
            outputs = fusion_model(x_phone, x_watch)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            return ACTIVITY_LABELS[pred_idx], probs
    except Exception as e:
        print(f"Error in fusion prediction: {e}")
        probs = np.zeros(len(ACTIVITY_LABELS))
        return "Error", probs


def make_predictions():
    """Make predictions using all three models if windows are ready."""
    current_time = datetime.now()
    phone_window = create_window_from_buffer(phone_buffer)
    watch_window = create_window_from_buffer(watch_buffer)

    # Phone prediction
    if phone_window is not None:
        phone_pred, phone_prob = predict_phone(phone_window)
        phone_predictions.append(phone_pred)
        phone_probs.append(phone_prob)

    # Watch prediction
    if watch_window is not None:
        watch_pred, watch_prob = predict_watch(watch_window)
        watch_predictions.append(watch_pred)
        watch_probs.append(watch_prob)

    # Fusion prediction (needs both)
    if phone_window is not None and watch_window is not None:
        fusion_pred, fusion_prob = predict_fusion(phone_window, watch_window)
        fusion_predictions.append(fusion_pred)
        fusion_probs.append(fusion_prob)
        prediction_times.append(current_time)


# ===========================
# Dash Layout (unchanged)
# ===========================

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

# ===========================
# Plot helpers (unchanged, small tweaks)
# ===========================


def create_sensor_graph(time_data, sensor_data, names, title, yaxis_label, colors):
    data = [
        go.Scatter(x=list(time_data), y=list(d), name=name, line=dict(width=2))
        for d, name, color in zip(sensor_data, names, colors)
    ]
    for trace, color in zip(data, colors):
        trace["line"]["color"] = color

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

    if len(time_data) > 0:
        try:
            fig["layout"]["xaxis"]["range"] = [min(time_data), max(time_data)]
            all_values = [item for sublist in sensor_data for item in sublist]
            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
                y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) != 0 else 1.0
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
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(prediction_times),
            y=watch_nums,
            name="Watch",
            mode="lines+markers",
            line=dict(color="#e74c3c", width=2),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(prediction_times),
            y=fusion_nums,
            name="Fusion",
            mode="lines+markers",
            line=dict(color="#27ae60", width=2),
            marker=dict(size=8),
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


# ===========================
# Data Reception Endpoint
# ===========================


@server.route("/data", methods=["POST"])
def data():
    """Receive sensor data from Sensor Logger app."""
    global frame_count
    if request.method == "POST":
        try:
            payload = json.loads(request.data)
            # payload expected to contain 'payload': [ ... ] as in your original code
            for d in payload.get("payload", []):
                # timestamp in nanoseconds in your original code
                ts = datetime.fromtimestamp(d.get("time", 0) / 1_000_000_000)
                device_raw = str(d.get("device", "unknown"))
                role = identify_device(device_raw)  # 'phone' or 'watch'
                sensor_name = d.get("name", "").lower()

                # Process accelerometer measurements: cache per device role
                if sensor_name == "accelerometer":
                    ax = float(d["values"]["x"])
                    ay = float(d["values"]["y"])
                    az = float(d["values"]["z"])
                    # append to visualization arrays
                    if len(time_accel) == 0 or ts > time_accel[-1]:
                        time_accel.append(ts)
                        accel_x.append(ax)
                        accel_y.append(ay)
                        accel_z.append(az)
                    # store accel in per-role accel cache for later pairing with gyro
                    accel_cache[role].append(
                        {"ax": ax, "ay": ay, "az": az, "timestamp": ts}
                    )

                # Process gyroscope measurements: pair with most recent accel for same role
                elif sensor_name == "gyroscope":
                    gx = float(d["values"]["x"])
                    gy = float(d["values"]["y"])
                    gz = float(d["values"]["z"])
                    # append to visualization arrays
                    if len(time_gyro) == 0 or ts > time_gyro[-1]:
                        time_gyro.append(ts)
                        gyro_x.append(gx)
                        gyro_y.append(gy)
                        gyro_z.append(gz)

                    # find the latest accel in accel_cache[role] with timestamp <= gyro_ts (within small delta)
                    paired_accel = None
                    for a in reversed(accel_cache[role]):
                        if a["timestamp"] <= ts:
                            # require close in time (e.g., within 200ms)
                            if (ts - a["timestamp"]) <= timedelta(milliseconds=200):
                                paired_accel = a
                            else:
                                # still accept slightly older accel (fallback)
                                paired_accel = a
                            break

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
                        # append to the appropriate sliding buffer
                        if role == "phone":
                            phone_buffer.append(sample)
                            sample_counts["phone"] += 1
                            # trigger predictions every STEP_SIZE samples for phone
                            if sample_counts["phone"] % STEP_SIZE == 0:
                                make_predictions()
                        else:
                            watch_buffer.append(sample)
                            sample_counts["watch"] += 1
                            if sample_counts["watch"] % STEP_SIZE == 0:
                                make_predictions()
                    else:
                        # no accel for that device recently: ignore or log
                        # (keeps streaming stable instead of mixing devices)
                        print(
                            f"No matching accel found for gyro from device '{device_raw}' at {ts}"
                        )

            frame_count += 1
            return "success", 200

        except Exception as e:
            print(f"Error processing data: {e}")
            import traceback

            traceback.print_exc()
            return f"error: {e}", 400

    return "method not allowed", 405


@server.route("/", methods=["GET"])
def index():
    """Health check endpoint."""
    return f"IMU Activity Recognition Dashboard running. POST data to /data endpoint."


# ===========================
# Dash update callback
# ===========================


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
    # Connection status
    phone_samples = len(phone_buffer)
    watch_samples = len(watch_buffer)

    status = html.Div(
        [
            html.Div(
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
                    html.Span(" | ", style={"margin": "0 15px"}),
                    html.Span("⌚ Watch: ", style={"fontWeight": "bold"}),
                    html.Span(
                        f"{watch_samples}/{WINDOW_SIZE} samples",
                        style={
                            "color": (
                                "#27ae60" if watch_samples >= WINDOW_SIZE else "#e74c3c"
                            )
                        },
                    ),
                    html.Span(" | ", style={"margin": "0 15px"}),
                    html.Span(f"Server: {local_ip}:8000", style={"fontWeight": "bold"}),
                    html.Span(" | ", style={"margin": "0 15px"}),
                    html.Span(f"Frames: {frame_count}", style={"color": "#7f8c8d"}),
                ],
                style={"fontSize": "16px"},
            )
        ]
    )

    accel_fig = create_sensor_graph(
        time_accel,
        [accel_x, accel_y, accel_z],
        ["Accel X", "Accel Y", "Accel Z"],
        "Accelerometer",
        "Acceleration (m/s²)",
        ["#e74c3c", "#3498db", "#2ecc71"],
    )

    gyro_fig = create_sensor_graph(
        time_gyro,
        [gyro_x, gyro_y, gyro_z],
        ["Gyro X", "Gyro Y", "Gyro Z"],
        "Gyroscope",
        "Angular Velocity (rad/s)",
        ["#f39c12", "#9b59b6", "#1abc9c"],
    )

    phone_pred_text = phone_predictions[-1] if phone_predictions else "---"
    watch_pred_text = watch_predictions[-1] if watch_predictions else "---"
    fusion_pred_text = fusion_predictions[-1] if fusion_predictions else "---"

    phone_prob_fig = create_prob_bars(
        phone_probs[-1] if phone_probs else np.zeros(len(ACTIVITY_LABELS)),
        "Phone Model Confidence",
    )

    watch_prob_fig = create_prob_bars(
        watch_probs[-1] if watch_probs else np.zeros(len(ACTIVITY_LABELS)),
        "Watch Model Confidence",
    )

    fusion_prob_fig = create_prob_bars(
        fusion_probs[-1] if fusion_probs else np.zeros(len(ACTIVITY_LABELS)),
        "Fusion Model Confidence",
    )

    timeline_fig = create_prediction_timeline()

    return (
        status,
        accel_fig,
        gyro_fig,
        phone_pred_text,
        watch_pred_text,
        fusion_pred_text,
        phone_prob_fig,
        watch_prob_fig,
        fusion_prob_fig,
        timeline_fig,
    )


# ===========================
# Main
# ===========================

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

    app.run(port=8000, host="0.0.0.0", debug=True)
