# dataset.py
import pandas as pd
import torch
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, seq_length=10, target_column='solar_power_output', feature_columns=None):
        """
        Args:
            csv_file (string): Path to the csv file with data.
            seq_length (int): Number of time steps in the input sequence.
            target_column (string): Column name for the target variable.
            feature_columns (list): List of columns to be used as features. If None, all columns except target.
        """
        self.data = pd.read_csv(csv_file)
        self.seq_length = seq_length
        self.target_column = target_column
        
        # If feature_columns is not provided, use all columns except the target
        if feature_columns is None:
            self.feature_columns = [col for col in self.data.columns if col != target_column]
        else:
            self.feature_columns = feature_columns
            
        # Extract features and target
        self.features = self.data[self.feature_columns].values.astype('float32')
        self.targets = self.data[target_column].values.astype('float32')
        
    def __len__(self):
        # Minus seq_length because each sample is a sequence of seq_length time steps.
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Get a sequence of features and the target for the next time step.
        sequence = self.features[idx:idx+self.seq_length]
        target = self.targets[idx+self.seq_length]  # Predict next time step
        
        # Convert to torch tensors
        sequence = torch.tensor(sequence, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        return sequence, target
    