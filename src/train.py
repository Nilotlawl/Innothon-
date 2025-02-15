# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import TimeSeriesDataset
from model import LSTMModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_model(model, train_loader, val_loader, device, num_epochs=50, lr=0.001):
    criterion = nn.MSELoss()  # For regression tasks, MSE loss is common
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * sequences.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Evaluate on the validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * sequences.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    print("Training complete. Best validation loss: {:.4f}".format(best_val_loss))

def main():
    # Hyperparameters and file paths (adjust these as needed)
    csv_file = 'data/dataset.csv'  # Path to your dataset CSV file
    seq_length = 10
    target_column = 'solar_power_output'
    # Specify feature columns available in your CSV
    feature_columns = ['temperature', 'humidity', 'solar_irradiance', 'voltage', 'current']
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    hidden_size = 64
    num_layers = 2
    dropout = 0.2
    
    # Load dataset
    dataset = TimeSeriesDataset(csv_file, seq_length, target_column, feature_columns)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Determine input size based on the number of feature columns
    input_size = len(feature_columns)
    output_size = 1  # Single value prediction (regression)
    
    # Configure device for CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    
    # Initialize the LSTM model
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout).to(device)
    
    # Train the model
    train_model(model, train_loader, val_loader, device, num_epochs, learning_rate)

if __name__ == '__main__':
    main()
