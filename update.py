# mqtt_to_csv.py
import json
import csv
import os
from datetime import datetime
import paho.mqtt.client as mqtt

# MQTT Broker configuration
MQTT_BROKER = "your_broker_ip_or_hostname"  # Change to your broker's address
MQTT_PORT = 1883
MQTT_TOPIC = "renewable/energy"  # The topic your Raspberry Pi is publishing to

# CSV file path
CSV_FILE = "data/dataset.csv"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
    else:
        print("Failed to connect, return code %d\n", rc)

def on_message(client, userdata, msg):
    try:
        # Decode the incoming message
        payload = json.loads(msg.payload.decode())
        
        # Extract required fields
        timestamp = payload.get("timestamp", datetime.now().isoformat())
        voltage = payload.get("voltage")
        current = payload.get("current")
        temperature = payload.get("temperature")
        
        # Check for completeness of data
        if voltage is None or current is None or temperature is None:
            print("Incomplete data received:", payload)
            return

        # Check if CSV file exists to write headers if not
        file_exists = os.path.isfile(CSV_FILE)
        with open(CSV_FILE, mode='a', newline='') as csv_file:
            fieldnames = ["timestamp", "voltage", "current", "temperature"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": timestamp,
                "voltage": voltage,
                "current": current,
                "temperature": temperature
            })
        print(f"Data logged: {timestamp}, Voltage: {voltage}, Current: {current}, Temp: {temperature}")
    except Exception as e:
        print("Error processing message:", e)

def main():
    # Ensure the directory exists for the CSV file
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_forever()

if __name__ == "__main__":
    main()
