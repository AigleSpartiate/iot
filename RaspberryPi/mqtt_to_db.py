#!/usr/bin/env python3
import json
import sqlite3
from paho.mqtt import client as mqtt

# MQTT config
MQTT_BROKER = "localhost"          # Pi itself
MQTT_PORT = 1883
MQTT_TOPIC = "test/room1/temperature"

# SQLite config
DB_PATH = "sensors.db"

# ---------- SQLite setup ----------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP,
    room_id INTEGER,
    room_name TEXT,
    measurement_id INTEGER,
    uptime_ms INTEGER,
    temperature REAL,
    humidity REAL,
    light INTEGER,
    sound INTEGER,
    air_quality INTEGER,
    air_quality_level TEXT
)
""")
conn.commit()


# ---------- MQTT callbacks ----------
def on_connect(client, userdata, flags, reason_code, properties=None):
    print("Connected to MQTT broker with code:", reason_code)
    client.subscribe(MQTT_TOPIC)
    print("Subscribed to:", MQTT_TOPIC)


def on_message(client, userdata, msg):
    payload = msg.payload.decode("utf-8")
    print("Received:", payload)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        return

    # Extract fields with .get() (safe if some are missing)
    room_id          = data.get("room_id")
    room_name        = data.get("name")
    measurement_id   = data.get("measurement_id")
    uptime_ms        = data.get("uptime")
    temperature      = data.get("temperature")
    humidity         = data.get("humidity")
    light            = data.get("light")
    sound            = data.get("sound")
    air_quality      = data.get("airQuality")
    air_quality_lvl  = data.get("airQuality-Level")

    # Insert into SQLite
    cur.execute("""
        INSERT INTO measurements (
            room_id,
            room_name,
            measurement_id,
            uptime_ms,
            temperature,
            humidity,
            light,
            sound,
            air_quality,
            air_quality_level
           
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        room_id,
        room_name,
        measurement_id,
        uptime_ms,
        temperature,
        humidity,
        light,
        sound,
        air_quality,
        air_quality_lvl,
        
    ))
    conn.commit()

    print("Saved to DB (measurement_id = {})".format(measurement_id))


# ---------- Main ----------
def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

    print("Listening for MQTT messages...")
    client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        conn.close()
