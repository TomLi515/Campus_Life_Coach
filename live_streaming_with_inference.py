import dash
from dash.dependencies import Output, Input
from dash import dcc, html
from datetime import datetime
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
STEP_SIZE = 25     # step for sliding window (0.5 seconds)
TARGET_HZ = 50

# Activity labels
ACTIVITY_LABELS = ['Walk', 'Run', 'Sit', 'Stand', 'Lie']
ACTIVITY_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Data storage - raw sensor data
time_accel = deque(maxlen=MAX_DATA_POINTS)
accel_x = deque(maxlen=MAX_DATA_POINTS)
accel_y = deque(maxlen=MAX_DATA_POINTS)
accel_z = deque(maxlen=MAX_DATA_POINTS)

time_gyro = deque(maxlen=MAX_DATA_POINTS)
gyro_x = deque(maxlen=MAX_DATA_POINTS)
gyro_y = deque(maxlen=MAX_DATA_POINTS)
gyro_z = deque(maxlen=MAX_DATA_POINTS)

# Sliding window buffer for predictions
phone_buffer = deque(maxlen=WINDOW_SIZE)
watch_buffer = deque(maxlen=WINDOW_SIZE)

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
last_device = None  # Track which device we got data from

# ===========================
# Model Loading
# ===========================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        phone_checkpoint = torch.load(phone_ckpt_path, map_location=device, weights_only=False)
        # Initialize your model architecture here
        # phone_model = YourPhoneModel(...)
        # phone_model.load_state_dict(phone_checkpoint['model_state_dict'])
        # phone_model.eval()
        print("✓ Phone model loaded")
    else:
        print(f"⚠ Phone model not found at {phone_ckpt_path}")
except Exception as e:
    print(f"✗ Error loading phone model: {e}")

try:
    # Load watch model
    watch_ckpt_path = MODEL_DIR / "watch_only_classifier.pth"
    if watch_ckpt_path.exists():
        watch_checkpoint = torch.load(watch_ckpt_path, map_location=device, weights_only=False)
        # watch_model = YourWatchModel(...)
        # watch_model.load_state_dict(watch_checkpoint['model_state_dict'])
        # watch_model.eval()
        print("✓ Watch model loaded")
    else:
        print(f"⚠ Watch model not found at {watch_ckpt_path}")
except Exception as e:
    print(f"✗ Error loading watch model: {e}")

try:
    # Load fusion model
    fusion_ckpt_path = MODEL_DIR / "fusion_classifier.pth"
    if fusion_ckpt_path.exists():
        fusion_checkpoint = torch.load(fusion_ckpt_path, map_location=device, weights_only=False)
        # fusion_model = YourFusionModel(...)
        # fusion_model.load_state_dict(fusion_checkpoint['model_state_dict'])
        # fusion_model.eval()
        print("✓ Fusion model loaded")
    else:
        print(f"⚠ Fusion model not found at {fusion_ckpt_path}")
except Exception as e:
    print(f"✗ Error loading fusion model: {e}")

# ===========================
# Preprocessing Functions
# ===========================

def normalize_window(window):
    """Normalize a single window (6, 150) using z-score."""
    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True) + 1e-8
    return (window - mean) / std

def create_window_from_buffer(buffer):
    """
    Create a window array from buffer.
    Buffer contains dicts with keys: ax, ay, az, gx, gy, gz
    Returns: numpy array of shape (6, window_size)
    """
    if len(buffer) < WINDOW_SIZE:
        return None
    
    # Extract last WINDOW_SIZE samples
    recent = list(buffer)[-WINDOW_SIZE:]
    
    # Stack into (6, WINDOW_SIZE) array
    window = np.array([
        [s['ax'] for s in recent],
        [s['ay'] for s in recent],
        [s['az'] for s in recent],
        [s['gx'] for s in recent],
        [s['gy'] for s in recent],
        [s['gz'] for s in recent]
    ], dtype=np.float32)
    
    return window

# ===========================
# Prediction Functions
# ===========================

def predict_phone(window):
    """Make prediction using phone model."""
    if phone_model is None:
        # Placeholder prediction
        probs = np.random.dirichlet(np.ones(5))
        pred_idx = np.argmax(probs)
        return ACTIVITY_LABELS[pred_idx], probs
    
    try:
        with torch.no_grad():
            # Normalize
            window_norm = normalize_window(window)
            # Convert to tensor (1, 6, 150)
            x = torch.from_numpy(window_norm).unsqueeze(0).float().to(device)
            # Predict
            outputs = phone_model(x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            return ACTIVITY_LABELS[pred_idx], probs
    except Exception as e:
        print(f"Error in phone prediction: {e}")
        probs = np.zeros(5)
        return "Error", probs

def predict_watch(window):
    """Make prediction using watch model."""
    if watch_model is None:
        # Placeholder prediction
        probs = np.random.dirichlet(np.ones(5))
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
        probs = np.zeros(5)
        return "Error", probs

def predict_fusion(phone_window, watch_window):
    """Make prediction using fusion model."""
    if fusion_model is None:
        # Placeholder prediction
        probs = np.random.dirichlet(np.ones(5))
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
        probs = np.zeros(5)
        return "Error", probs

# ===========================
# Dash Layout
# ===========================

app.layout = html.Div([
    html.Div([
        html.H1("🏃 Real-Time IMU Activity Recognition", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
        html.P("Streaming sensor data from phone and watch with three-model prediction",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '14px'}),
    ], style={'padding': '20px', 'backgroundColor': '#ecf0f1', 'borderRadius': '10px', 'margin': '10px'}),
    
    # Connection Status
    html.Div(id="connection-status", style={
        'padding': '15px',
        'margin': '10px',
        'backgroundColor': '#fff',
        'border': '2px solid #3498db',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    }),
    
    # Sensor graphs row
    html.Div([
        html.Div([
            dcc.Graph(id="accel_graph", style={'height': '300px'})
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '5px'}),
        
        html.Div([
            dcc.Graph(id="gyro_graph", style={'height': '300px'})
        ], style={'width': '50%', 'display': 'inline-block', 'padding': '5px'}),
    ], style={'margin': '10px'}),
    
    # Predictions row
    html.Div([
        # Phone prediction
        html.Div([
            html.H3("📱 Phone Model", style={'textAlign': 'center', 'color': '#3498db'}),
            html.Div(id="phone-prediction", style={
                'fontSize': '32px', 
                'fontWeight': 'bold', 
                'textAlign': 'center',
                'padding': '20px',
                'backgroundColor': '#ecf0f1',
                'borderRadius': '8px',
                'margin': '10px'
            }),
            dcc.Graph(id="phone_probs", style={'height': '200px'})
        ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '5px'}),
        
        # Watch prediction
        html.Div([
            html.H3("⌚ Watch Model", style={'textAlign': 'center', 'color': '#e74c3c'}),
            html.Div(id="watch-prediction", style={
                'fontSize': '32px', 
                'fontWeight': 'bold', 
                'textAlign': 'center',
                'padding': '20px',
                'backgroundColor': '#ecf0f1',
                'borderRadius': '8px',
                'margin': '10px'
            }),
            dcc.Graph(id="watch_probs", style={'height': '200px'})
        ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '5px'}),
        
        # Fusion prediction
        html.Div([
            html.H3("🔗 Fusion Model", style={'textAlign': 'center', 'color': '#27ae60'}),
            html.Div(id="fusion-prediction", style={
                'fontSize': '32px', 
                'fontWeight': 'bold', 
                'textAlign': 'center',
                'padding': '20px',
                'backgroundColor': '#ecf0f1',
                'borderRadius': '8px',
                'margin': '10px'
            }),
            dcc.Graph(id="fusion_probs", style={'height': '200px'})
        ], style={'width': '33%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '5px'}),
    ], style={'margin': '10px'}),
    
    # Prediction timeline
    html.Div([
        dcc.Graph(id="prediction_timeline", style={'height': '250px'})
    ], style={'margin': '10px'}),
    
    dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f8f9fa', 'minHeight': '100vh'})

# ===========================
# Callbacks
# ===========================

@app.callback(
    [Output("connection-status", "children"),
     Output("accel_graph", "figure"),
     Output("gyro_graph", "figure"),
     Output("phone-prediction", "children"),
     Output("watch-prediction", "children"),
     Output("fusion-prediction", "children"),
     Output("phone_probs", "figure"),
     Output("watch_probs", "figure"),
     Output("fusion_probs", "figure"),
     Output("prediction_timeline", "figure")],
    Input("counter", "n_intervals")
)
def update_dashboard(_counter):
    global frame_count
    
    # Connection status
    phone_samples = len(phone_buffer)
    watch_samples = len(watch_buffer)
    
    status = html.Div([
        html.Div([
            html.Span("📱 Phone: ", style={'fontWeight': 'bold'}),
            html.Span(f"{phone_samples}/{WINDOW_SIZE} samples", 
                     style={'color': '#27ae60' if phone_samples >= WINDOW_SIZE else '#e74c3c'}),
            html.Span(" | ", style={'margin': '0 15px'}),
            html.Span("⌚ Watch: ", style={'fontWeight': 'bold'}),
            html.Span(f"{watch_samples}/{WINDOW_SIZE} samples",
                     style={'color': '#27ae60' if watch_samples >= WINDOW_SIZE else '#e74c3c'}),
            html.Span(" | ", style={'margin': '0 15px'}),
            html.Span(f"Server: {local_ip}:8000", style={'fontWeight': 'bold'}),
            html.Span(" | ", style={'margin': '0 15px'}),
            html.Span(f"Frames: {frame_count}", style={'color': '#7f8c8d'}),
        ], style={'fontSize': '16px'})
    ])
    
    # Accelerometer graph
    accel_fig = create_sensor_graph(
        time_accel, [accel_x, accel_y, accel_z],
        ["Accel X", "Accel Y", "Accel Z"],
        "Accelerometer", "Acceleration (m/s²)",
        ['#e74c3c', '#3498db', '#2ecc71']
    )
    
    # Gyroscope graph
    gyro_fig = create_sensor_graph(
        time_gyro, [gyro_x, gyro_y, gyro_z],
        ["Gyro X", "Gyro Y", "Gyro Z"],
        "Gyroscope", "Angular Velocity (rad/s)",
        ['#f39c12', '#9b59b6', '#1abc9c']
    )
    
    # Get latest predictions
    phone_pred_text = phone_predictions[-1] if phone_predictions else "---"
    watch_pred_text = watch_predictions[-1] if watch_predictions else "---"
    fusion_pred_text = fusion_predictions[-1] if fusion_predictions else "---"
    
    # Probability bar charts
    phone_prob_fig = create_prob_bars(
        phone_probs[-1] if phone_probs else np.zeros(5),
        "Phone Model Confidence"
    )
    
    watch_prob_fig = create_prob_bars(
        watch_probs[-1] if watch_probs else np.zeros(5),
        "Watch Model Confidence"
    )
    
    fusion_prob_fig = create_prob_bars(
        fusion_probs[-1] if fusion_probs else np.zeros(5),
        "Fusion Model Confidence"
    )
    
    # Prediction timeline
    timeline_fig = create_prediction_timeline()
    
    return (status, accel_fig, gyro_fig, 
            phone_pred_text, watch_pred_text, fusion_pred_text,
            phone_prob_fig, watch_prob_fig, fusion_prob_fig,
            timeline_fig)

def create_sensor_graph(time_data, sensor_data, names, title, yaxis_label, colors):
    """Helper to create sensor graphs."""
    data = [
        go.Scatter(
            x=list(time_data), 
            y=list(d), 
            name=name,
            line=dict(color=color, width=2)
        )
        for d, name, color in zip(sensor_data, names, colors)
    ]
    
    layout = go.Layout(
        title=dict(text=title, font=dict(size=14, color='#2c3e50')),
        xaxis=dict(title="Time", showgrid=True, gridcolor='#ecf0f1'),
        yaxis=dict(title=yaxis_label, showgrid=True, gridcolor='#ecf0f1'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=30, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig = {"data": data, "layout": layout}
    
    if len(time_data) > 0:
        fig["layout"]["xaxis"]["range"] = [min(time_data), max(time_data)]
        all_values = [item for sublist in sensor_data for item in sublist]
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_margin = (y_max - y_min) * 0.1
            fig["layout"]["yaxis"]["range"] = [y_min - y_margin, y_max + y_margin]
    
    return fig

def create_prob_bars(probs, title):
    """Create probability bar chart."""
    fig = go.Figure(data=[
        go.Bar(
            x=ACTIVITY_LABELS,
            y=probs,
            marker=dict(color=ACTIVITY_COLORS),
            text=[f"{p:.1%}" for p in probs],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=12)),
        yaxis=dict(title="Probability", range=[0, 1], showgrid=True, gridcolor='#ecf0f1'),
        xaxis=dict(title="Activity"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    return fig

def create_prediction_timeline():
    """Create timeline showing prediction history."""
    if len(prediction_times) == 0:
        return go.Figure()
    
    # Create numeric mapping for activities
    activity_to_num = {label: i for i, label in enumerate(ACTIVITY_LABELS)}
    
    phone_nums = [activity_to_num.get(p, -1) for p in phone_predictions]
    watch_nums = [activity_to_num.get(p, -1) for p in watch_predictions]
    fusion_nums = [activity_to_num.get(p, -1) for p in fusion_predictions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(prediction_times),
        y=phone_nums,
        name="Phone",
        mode='lines+markers',
        line=dict(color='#3498db', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(prediction_times),
        y=watch_nums,
        name="Watch",
        mode='lines+markers',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(prediction_times),
        y=fusion_nums,
        name="Fusion",
        mode='lines+markers',
        line=dict(color='#27ae60', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Prediction Timeline",
        xaxis=dict(title="Time"),
        yaxis=dict(
            title="Activity",
            tickmode='array',
            tickvals=list(range(len(ACTIVITY_LABELS))),
            ticktext=ACTIVITY_LABELS
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=30, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# ===========================
# Data Reception Endpoint
# ===========================

@server.route("/data", methods=["POST"])
def data():
    """Receive sensor data from Sensor Logger app."""
    global frame_count, last_device
    
    if request.method == "POST":
        try:
            data = json.loads(request.data)
            
            for d in data['payload']:
                ts = datetime.fromtimestamp(d["time"] / 1000000000)
                device = d.get("device", "unknown").lower()
                
                # Process accelerometer
                if d.get("name") == "accelerometer":
                    if len(time_accel) == 0 or ts > time_accel[-1]:
                        time_accel.append(ts)
                        ax = d["values"]["x"]
                        ay = d["values"]["y"]
                        az = d["values"]["z"]
                        accel_x.append(ax)
                        accel_y.append(ay)
                        accel_z.append(az)
                        
                        # Store in appropriate buffer (we'll get gyro separately)
                        # For now, just track we got accelerometer data
                        last_device = device
                
                # Process gyroscope
                elif d.get("name") == "gyroscope":
                    if len(time_gyro) == 0 or ts > time_gyro[-1]:
                        time_gyro.append(ts)
                        gx = d["values"]["x"]
                        gy = d["values"]["y"]
                        gz = d["values"]["z"]
                        gyro_x.append(gx)
                        gyro_y.append(gy)
                        gyro_z.append(gz)
                        
                        # Check if we have corresponding accelerometer data
                        # If so, create a sample and add to buffer
                        if len(accel_x) > 0:
                            sample = {
                                'ax': accel_x[-1],
                                'ay': accel_y[-1],
                                'az': accel_z[-1],
                                'gx': gx,
                                'gy': gy,
                                'gz': gz,
                                'timestamp': ts
                            }
                            
                            # Add to appropriate buffer based on device
                            if 'phone' in device:
                                phone_buffer.append(sample)
                            else:  # watch
                                watch_buffer.append(sample)
            
            # Try to make predictions if we have enough data
            if frame_count % (STEP_SIZE // 5) == 0:  # Run prediction every STEP_SIZE worth of data
                make_predictions()
            
            frame_count += 1
            return "success", 200
            
        except Exception as e:
            print(f"Error processing data: {e}")
            import traceback
            traceback.print_exc()
            return f"error: {e}", 400
    
    return "method not allowed", 405

def make_predictions():
    """Make predictions using all three models."""
    current_time = datetime.now()
    
    # Create windows from buffers
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

@server.route("/", methods=["GET"])
def index():
    """Health check endpoint."""
    return f"IMU Activity Recognition Dashboard running. POST data to /data endpoint."

# ===========================
# Main
# ===========================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING IMU ACTIVITY RECOGNITION DASHBOARD")
    print("="*70)
    print(f"Dashboard URL: http://{local_ip}:8000")
    print(f"Sensor Logger POST endpoint: http://{local_ip}:8000/data")
    print("="*70)
    print("\nConfigure Sensor Logger:")
    print("  1. Set recording mode to 'Push to Server'")
    print(f"  2. Enter URL: http://{local_ip}:8000/data")
    print("  3. Enable Accelerometer and Gyroscope sensors")
    print("  4. Set recording frequency to 50 Hz")
    print("  5. Start recording on both phone and watch")
    print("="*70 + "\n")
    
    app.run(port=8000, host="0.0.0.0", debug=True)