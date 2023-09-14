from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import os
import pandas as pd
import torch
import numpy as np
from Bio import SeqIO
import torch.nn.functional as F

class SangerSequencingDataset(Dataset):
    def __init__(self, fasta_folder, csv_folder, seq_length, start_bp=40):
        csv_files = [file for file in os.listdir(csv_folder)]
        csv_files = [file for file in csv_files if len(pd.read_csv(os.path.join(csv_folder, file)).values) > 0] # Don't include empty CSV files
        fasta_files = [file.replace('.csv', '.fasta') for file in csv_files]

        self.fasta_files = [os.path.join(fasta_folder, file) for file in fasta_files]
        self.csv_files = [os.path.join(csv_folder, file) for file in csv_files]
        self.seq_length = seq_length
        self.start_bp = start_bp

        self.encoder = OneHotEncoder(sparse_output=False, categories=[['A', 'C', 'G', 'T', 'N']])

        # Filter fasta_files and csv_files to only include sequences of length self.start_bp+self.seq_length
        self.fasta_files = [file for file in self.fasta_files if len(str(list(SeqIO.parse(file, "fasta"))[0].seq)) >= self.start_bp+self.seq_length]
        self.csv_files = [file for file in self.csv_files if len(pd.read_csv(file).values) >= self.start_bp+self.seq_length]

    def __len__(self):
        return len(self.fasta_files)

    def __getitem__(self, idx):
        # Read FASTA file
        fasta_sequence = str(list(SeqIO.parse(self.fasta_files[idx], "fasta"))[0].seq)

        # Convert FASTA sequence to one-hot encoding
        fasta_sequence = self.encoder.fit_transform(np.array(list(fasta_sequence)[self.start_bp:self.start_bp+self.seq_length]).reshape(-1, 1))

        # Read CSV file
        csv_sequence = pd.read_csv(self.csv_files[idx]).values[self.start_bp:self.start_bp+self.seq_length]

        scaler = MinMaxScaler()
        csv_sequence = scaler.fit_transform(csv_sequence)

        return torch.tensor(fasta_sequence, dtype=torch.float32), torch.tensor(csv_sequence, dtype=torch.float32)

class SequenceToSangerMLP(nn.Module):
    def __init__(self, input_size=1500, hidden_sizes=[1024, 512, 256], output_size=1200):
        super(SequenceToSangerMLP, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        
        self.fc1 = nn.Linear(input_size, self.hidden_sizes[0])
        self.bn1 = nn.BatchNorm1d(self.hidden_sizes[0])
        
        self.fc2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(self.hidden_sizes[1])
        
        self.fc3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.bn3 = nn.BatchNorm1d(self.hidden_sizes[2])
        
        self.fc4 = nn.Linear(self.hidden_sizes[2], output_size)
    
    def forward(self, x):
        shape = x.shape
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        
        x = self.fc4(x)
        return x.view(shape[0], shape[1], -1)

