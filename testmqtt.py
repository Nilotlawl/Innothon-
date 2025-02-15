import paho.mqtt.client as mqtt
import json

BROKER = "172.20.10.2"  # Replace with your Raspberry Pi's IP
TOPIC = "renewable_energy_data"

def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    print(f"Received Data: {data}")
    # Send this data to your ML model for analysis

client = mqtt.Client()
client.connect(BROKER, 1883, 60)
client.subscribe(TOPIC)

client.on_message = on_message
client.loop_forever()  # Keep listening for incoming data