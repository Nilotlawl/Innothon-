import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import torch


# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_data(filepath):
    """
    Load data from a CSV file.
    Ensure the CSV has a 'fault' column for labeling.
    """
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    """
    Separate features and target.
    Assumes the target column is named 'fault'.
    """
    X = data.drop('fault', axis=1)
    y = data['fault']
    return X, y

def train_model(X, y):
    """
    Split the data, train an SVM model, and print evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)  # Using SVM with RBF kernel
    model.fit(X_train, y_train)
    
    # Predict and evaluate the model on the test set
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return model

if __name__ == '__main__':
    data_filepath = 'data.csv'
    data = load_data(data_filepath)
    X, y = preprocess_data(data)
    model = train_model(X, y)