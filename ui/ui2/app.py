from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

BASE_URL = "https://raspberrypi.tail059df.ts.net"


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/measurements/<room_id>")
def get_measurements(room_id):
    start_time = request.args.get("start_time", "")
    end_time = request.args.get("end_time", "")

    url = f"{BASE_URL}/rooms/{room_id}/measurements"
    params = {"start_time": start_time, "end_time": end_time}

    r = requests.get(url, params=params, verify=False)
    return jsonify(r.json())


@app.route("/api/facilities/<room_id>")
def get_facilities(room_id):
    url = f"{BASE_URL}/rooms/{room_id}/facilities"
    r = requests.get(url, verify=False)
    return jsonify(r.json())


@app.route("/api/calendar/<room_id>")
def get_calendar(room_id):
    url = f"{BASE_URL}/rooms/{room_id}/calendar"
    r = requests.get(url, verify=False)
    return jsonify(r.json())


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json
    url = f"{BASE_URL}/recommendations"
    r = requests.post(url, json=data, verify=False)
    return jsonify(r.json())


if __name__ == "__main__":
    app.run(debug=True, port=8000)
