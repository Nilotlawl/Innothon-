import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch

# Using device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_data(filepath):
    """
    Load data from a CSV file updated by MQTT.
    Generates a 'fault' column based on sensor_value threshold.
    """
    data = pd.read_csv(filepath)
    # Convert sensor_value to float (if needed)
    data['sensor_value'] = data['sensor_value'].astype(float)
    
    # Define a threshold to determine fault status (adjust as needed)
    threshold = 0.8  # Example threshold (in normalized scale)
    data['fault'] = (data['sensor_value'] > threshold).astype(int)
    return data

def preprocess_data(data):
    """
    Separate features and target.
    Exclude 'timestamp' and 'fault' are used for labeling.
    """
    X = data.drop(['timestamp', 'fault'], axis=1)
    y = data['fault']
    return X, y

def train_model(X, y):
    """
    Split the data, train an SVM model, and print evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return model

if __name__ == '__main__':
    data_filepath = 'D:\Innothon\sensor_data.csv'
    data = load_data(data_filepath)
    X, y = preprocess_data(data)
    model = train_model(X, y)