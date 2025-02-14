import paho.mqtt.client as mqtt
import csv
import datetime
import os

# Path to the CSV file which will be updated with sensor data
CSV_FILE = "sensor_data.csv"

# MQTT broker configuration (update with your broker's details)
BROKER = "mqtt.example.com"  # Replace with your broker address
PORT = 1883
TOPIC = "sensor/data"        # Replace with your desired topic

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    # Decode the payload (assuming it's in string format)
    sensor_payload = msg.payload.decode()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if CSV file exists to write header if necessary
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # Write header if the file is new
            writer.writerow(["timestamp", "sensor_value"])
        writer.writerow([timestamp, sensor_payload])
    print(f"Data logged at {timestamp}: {sensor_payload}")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)
    client.loop_forever()

if __name__ == '__main__':
    main()