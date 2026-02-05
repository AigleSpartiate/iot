#!/usr/bin/env python3
"""
Reads Arduino serial output and saves to structured JSON file
"""

import serial
import serial.tools.list_ports
import json
import os
from datetime import datetime
import argparse
import time

# ============== CONFIGURATION ==============
DEFAULT_PORT = "COM3"  # windows default
BAUD_RATE = 9600
OUTPUT_FILE = "sensor_data.json"


def create_empty_data_structure():
    """Create the base JSON structure for multi-room data"""
    return {
        "project": "temp",
        "created": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "rooms": [],
    }


def find_or_create_room(data, room_id, room_name):
    """Find existing room or create new one"""
    for room in data["rooms"]:
        if room["room_id"] == room_id:
            return room

    # create new room
    new_room = {
        "room_id": room_id,
        "name": room_name,
        "measurements": {
            "temperature": [],
            "humidity": [],
            "light": [],
            "sound": [],
            "air_quality": [],
        },
    }
    data["rooms"].append(new_room)
    return new_room


def load_existing_data(filepath):
    """Load existing JSON data or create new structure"""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filepath}, creating new file")
    return create_empty_data_structure()


def save_data(filepath, data):
    """Save data to JSON file"""
    data["last_updated"] = datetime.now().isoformat()
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def process_sensor_reading(data, reading):
    """Process a single sensor reading and add to data structure"""
    try:
        sensor_data = json.loads(reading)

        # skip status messages
        if "status" in sensor_data:
            print(f"Arduino: {sensor_data.get('message', sensor_data['status'])}")
            return False

        # Get room info
        room_id = sensor_data.get("room_id", 1)
        room_name = sensor_data.get("room_name", "Unknown")

        room = find_or_create_room(data, room_id, room_name)

        # current timestamp
        timestamp = datetime.now().isoformat()
        sensors = sensor_data.get("sensors", {})

        # Add temperature reading
        if "temperature" in sensors and "value" in sensors["temperature"]:
            room["measurements"]["temperature"].append(
                {
                    "timestamp": timestamp,
                    "value": sensors["temperature"]["value"],
                    "unit": sensors["temperature"].get("unit", "celsius"),
                }
            )

        # Add humidity reading
        if "humidity" in sensors and "value" in sensors["humidity"]:
            room["measurements"]["humidity"].append(
                {
                    "timestamp": timestamp,
                    "value": sensors["humidity"]["value"],
                    "unit": sensors["humidity"].get("unit", "percent"),
                }
            )

        # Add light reading
        if "light" in sensors:
            room["measurements"]["light"].append(
                {
                    "timestamp": timestamp,
                    "raw": sensors["light"].get("raw", 0),
                    "percent": sensors["light"].get("percent", 0),
                }
            )

        # Add sound reading
        if "sound" in sensors:
            room["measurements"]["sound"].append(
                {
                    "timestamp": timestamp,
                    "raw": sensors["sound"].get("raw", 0),
                    "percent": sensors["sound"].get("percent", 0),
                }
            )

        # Add air quality reading
        if "air_quality" in sensors:
            room["measurements"]["air_quality"].append(
                {
                    "timestamp": timestamp,
                    "raw": sensors["air_quality"].get("raw", 0),
                    "level": sensors["air_quality"].get("level", "unknown"),
                }
            )

        return True

    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse reading: {reading[:50]}...")
        return False


def list_available_ports():
    """List all available serial ports"""
    ports = serial.tools.list_ports.comports()
    if ports:
        print("\nAvailable ports:")
        for port in ports:
            print(f"- {port.device}: {port.description}")
    else:
        print("\nNo serial ports found.")


def main():
    parser = argparse.ArgumentParser(description="Collect sensor data from Arduino")
    parser.add_argument("-p", "--port", default=DEFAULT_PORT, help="Serial port")
    parser.add_argument("-o", "--output", default=OUTPUT_FILE, help="Output JSON file")
    parser.add_argument("-b", "--baud", type=int, default=BAUD_RATE, help="Baud rate")
    parser.add_argument(
        "-l", "--list-ports", action="store_true", help="List available ports and exit"
    )
    args = parser.parse_args()

    if args.list_ports:
        list_available_ports()
        return

    print(
        f"""
╔════════════════════════════════════════════════════════╗
║         Room data collector                            ║
╠════════════════════════════════════════════════════════╣
║  Port: {args.port:<47} ║
║  Output: {args.output:<45} ║
║  Baud Rate: {args.baud:<42} ║
╚════════════════════════════════════════════════════════╝
    """
    )

    data = load_existing_data(args.output)
    print(f"Loaded data with {len(data['rooms'])} room(s)")

    # connect to Arduino
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        print(f"Connected to {args.port}")
        time.sleep(2)  # wait for Arduino reset
    except serial.SerialException as e:
        print(f"Error: Could not connect to {args.port}: {e}")
        list_available_ports()
        return

    print("Listening for sensor data...\n")

    reading_count = 0
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode("utf-8").strip()
                if line:
                    if process_sensor_reading(data, line):
                        reading_count += 1
                        save_data(args.output, data)
                        print(
                            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                            f"Reading #{reading_count} saved"
                        )

    except KeyboardInterrupt:
        print(f"\n\nStopped. Total readings collected: {reading_count}")
        print(f"Data saved to: {args.output}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
