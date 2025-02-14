import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Using device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define the LSTM model
class LSTMFaultPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMFaultPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Get the output of the last time step
        out = self.fc(lstm_out[:, -1, :])
        return out

def load_time_series_data(filepath):
    """
    Load time series sensor data from CSV.
    Assumes a column 'timestamp' and other sensor readings.
    """
    df = pd.read_csv(filepath)
    df.sort_values('timestamp', inplace=True)
    return df

def create_sequences(data, window_size):
    sequences = []
    targets = []
    for i in range(len(data) - window_size):
        sequences.append(data[i:i+window_size])
        targets.append(data[i+window_size])
    return np.array(sequences), np.array(targets)

if __name__ == "__main__":
    # Load and preprocess data
    data_filepath = 'sensor_data.csv'  # replace with your time series file
    df = load_time_series_data(data_filepath)
    
    # For simplicity, we use one sensor column, e.g., 'temperature'
    sensor_data = df['temperature'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    sensor_data = scaler.fit_transform(sensor_data)
    
    window_size = 10  # Number of time steps
    X, y = create_sequences(sensor_data, window_size)
    
    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model parameters
    input_size = 1
    hidden_size = 50
    num_layers = 2
    output_size = 1
    
    model = LSTMFaultPredictor(input_size, hidden_size, num_layers, output_size).to(device)
    
    criterion = nn.MSELoss()  # For regression forecasting
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 20
    for epoch in range(epochs):
        for sequences, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # After training, use the model to predict future sensor readings.
    # These predictions can serve as an early warning for faults.