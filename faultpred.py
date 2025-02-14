#!/usr/bin/env python3
"""
Fault Management System: MQTT Data Acquisition, Fault Detection, and ML-based Fault Prediction.
"""

import time
import json
import csv
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText

import paho.mqtt.client as mqtt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch



# ------------------------------
# Section 1: MQTT Data Acquisition
# ------------------------------

# MQTT Broker Configuration
BROKER = "mqtt.example.com"  # Replace with your broker address
PORT = 1883
TOPIC = "energy/sensors"

# Global list to store sensor data
data_list = []

def on_message(client, userdata, message):
    """
    Callback function for MQTT message arrival.
    Parses the incoming JSON and stores the data with a timestamp.
    """
    global data_list
    try:
        payload = json.loads(message.payload.decode())
        voltage = payload.get("voltage")
        current = payload.get("current")
        timestamp = time.time()

        # Append the sensor reading to our data list
        data_list.append({"timestamp": timestamp, "voltage": voltage, "current": current})
        print(f"Received: Voltage={voltage}V, Current={current}A")

    except Exception as e:
        print("Error processing message:", e)

def start_mqtt():
    """
    Initializes the MQTT client and starts listening for messages.
    """
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.subscribe(TOPIC)
    print("Starting MQTT loop. Listening to MQTT Broker...")
    client.loop_start()  # Use loop_start() so that the rest of the script can run concurrently

# ------------------------------
# Section 2: Data Storage and Fault Detection
# ------------------------------

# Thresholds for fault detection (example values)
VOLTAGE_THRESHOLD = (210, 250)  # For a 230V system
CURRENT_THRESHOLD = (0, 10)     # For a 10A max system

def save_data():
    """
    Saves the current data_list to a CSV file and clears the list.
    """
    if data_list:
        df = pd.DataFrame(data_list)
        # Append mode: create header only if file doesn't exist
        df.to_csv("sensor_data.csv", index=False, mode='a', header=not file_exists("sensor_data.csv"))
        print("Data saved to sensor_data.csv.")
        data_list.clear()

def file_exists(filename):
    """
    Checks if a file exists.
    """
    try:
        with open(filename, 'r'):
            return True
    except FileNotFoundError:
        return False

def detect_fault(voltage, current):
    """
    Basic fault detection logic based on voltage and current thresholds.
    """
    if voltage < VOLTAGE_THRESHOLD[0] or voltage > VOLTAGE_THRESHOLD[1]:
        return "Voltage Fault"
    if current < CURRENT_THRESHOLD[0] or current > CURRENT_THRESHOLD[1]:
        return "Current Fault"
    return "Normal"

def process_saved_data():
    """
    Reads saved sensor data, applies fault detection, and writes a fault report.
    """
    try:
        # Read CSV with header since sensor_data.csv was written with header on first write.
        df = pd.read_csv("sensor_data.csv", header=0)
        df["fault_status"] = df.apply(lambda row: detect_fault(row["voltage"], row["current"]), axis=1)
        df.to_csv("fault_report.csv", index=False)
        print("Fault detection completed. Report saved to fault_report.csv.")
    except Exception as e:
        print("Error processing saved data:", e)

# ------------------------------
# Section 3: Machine Learning-Based Fault Prediction
# ------------------------------

def train_fault_model():
    """
    Trains a Random Forest model on the fault report data and saves the model.
    """
    try:
        df = pd.read_csv("fault_report.csv")
        # Map fault statuses to numeric labels
        fault_mapping = {"Normal": 0, "Voltage Fault": 1, "Current Fault": 2}
        df["fault_label"] = df["fault_status"].map(fault_mapping)

        # Feature columns: voltage and current
        X = df[["voltage", "current"]]
        y = df["fault_label"]

        # Split the data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the Random Forest Classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")

        # Save the trained model
        joblib.dump(model, "fault_detection_model.pkl")
        print("Trained model saved to fault_detection_model.pkl.")

    except Exception as e:
        print("Error during model training:", e)

def predict_fault(voltage, current):
    """
    Uses the trained ML model to predict the fault type for given sensor readings.
    """
    try:
        model = joblib.load("fault_detection_model.pkl")
        prediction = model.predict([[voltage, current]])
        fault_mapping = {0: "Normal", 1: "Voltage Fault", 2: "Current Fault"}
        return fault_mapping.get(prediction[0], "Unknown")
    except Exception as e:
        print("Error during prediction:", e)
        return "Prediction Error"

# ------------------------------
# Section 4: Alert System (Email Example)
# ------------------------------

def send_alert(fault_type):
    """
    Sends an email alert when a fault is detected.
    """
    sender = "your_email@example.com"
    receiver = "technician@example.com"
    subject = f"Fault Alert: {fault_type} Detected"
    body = f"A {fault_type} has been detected in the system. Immediate attention is required."

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    smtp_server = "smtp.example.com"
    smtp_port = 587
    smtp_user = "your_email@example.com"
    smtp_password = "your_password"

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print("Alert sent successfully!")
    except Exception as e:
        print("Failed to send alert:", e)

# ------------------------------
# Section 5: CUDA Check Function
# ------------------------------

def check_cuda():
    """
    Checks if CUDA is available on the system using PyTorch.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"CUDA is available. Device: {device_name}")
        else:
            print("CUDA is not available.")
    except ImportError:
        print("PyTorch is not installed. Cannot check CUDA availability.")

# ------------------------------
# Main Execution Flow
# ------------------------------

if __name__ == '__main__':
    # Check for CUDA availability
    check_cuda()
    
    # Start the MQTT client in a separate thread
    start_mqtt()

    try:
        # Loop to periodically save data and process it.
        while True:
            # Save data every 60 seconds (adjust as needed)
            time.sleep(60)
            save_data()

            # Optionally, process the saved data to generate fault reports
            process_saved_data()

            # Optionally, retrain the ML model periodically (or trigger it based on conditions)
            train_fault_model()

            # Example: Use the ML model to predict fault for a sample reading
            sample_voltage = 220  # Example value
            sample_current = 5    # Example value
            prediction = predict_fault(sample_voltage, sample_current)
            print(f"Predicted Fault for Voltage={sample_voltage}, Current={sample_current}: {prediction}")

            # Optionally, send an alert if a fault is detected
            if prediction != "Normal":
                send_alert(prediction)

    except KeyboardInterrupt:
        print("Stopping the fault management system.")
