import dash
from dash.dependencies import Output, Input
from dash import dcc, html, dcc
from datetime import datetime
import json
import plotly.graph_objs as go
from collections import deque
from flask import Flask, request
import socket
hostname = socket.gethostname()
print(socket.gethostbyname(hostname))

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

MAX_DATA_POINTS = 1000
UPDATE_FREQ_MS = 100

time = deque(maxlen=MAX_DATA_POINTS)
avg_accel = deque(maxlen=MAX_DATA_POINTS)
avg_gyro = deque(maxlen=MAX_DATA_POINTS)

app.layout = html.Div(
	[
		dcc.Markdown(
			children="""
			# Live Sensor Readings
			Streamed from Sensor Logger: tszheichoi.com/sensorlogger
		"""
		),
		dcc.Graph(id="live_graph"),
		dcc.Interval(id="counter", interval=UPDATE_FREQ_MS),
	]
)


@app.callback(Output("live_graph", "figure"), Input("counter", "n_intervals"))
def update_graph(_counter):
	accel = list(avg_accel)
	gyro = list(avg_gyro)
	t = list(time)

	data = [
        go.Scatter(x=t, y=accel, name="accelerometer"),
        go.Scatter(x=t, y=gyro, name="gyroscope"),
    ]

	graph = {
        "data": data,
        "layout": go.Layout(
            xaxis={"type": "date"},
            yaxis={"title": "Sensor Value"},
        ),
    }

    # filter None values for y-range calc
	vals = [v for v in accel + gyro if v is not None]
	if vals:
		graph["layout"]["yaxis"]["range"] = [min(vals), max(vals)]

	if t:
		graph["layout"]["xaxis"]["range"] = [min(t), max(t)]

	return graph


@server.route("/data", methods=["POST"])
def data():  # listens to the data streamed from the sensor logger
	if str(request.method) == "POST":
		print(f'received data: {request.data}')
		data = json.loads(request.data)
		for d in data['payload']:
			if (
				d.get("name", None) == "accelerometer"
			):  #  modify to access different sensors
				print("accelerometer received")
				ts = datetime.fromtimestamp(d["time"] / 1000000000)
				if len(time) == 0 or ts > time[-1]:
					time.append(ts)
					# modify the following based on which sensor is accessed, log the raw json for guidance
					avg = (d["values"]["x"] + d["values"]["y"] + d["values"]["z"]) / 3
					avg_accel.append(avg)
					avg_gyro.append(None)
			elif (
				d.get("name", None) == "gyroscope"
			):
				print("gyroscope received")
				ts = datetime.fromtimestamp(d["time"] / 1000000000)
				if len(time) == 0 or ts > time[-1]:
					time.append(ts)
					# modify the following based on which sensor is accessed, log the raw json for guidance
					avg = (d["values"]["x"] + d["values"]["y"] + d["values"]["z"]) / 3
					avg_gyro.append(avg)
					avg_accel.append(None)
	return "success"


if __name__ == "__main__":
	app.run(port=8000, host="0.0.0.0")