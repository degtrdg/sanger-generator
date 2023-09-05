from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
import os
import pandas as pd
import torch
import numpy as np
from Bio import SeqIO

class SangerSequencingDataset(Dataset):
    def __init__(self, fasta_folder, csv_folder, seq_length):
        csv_files = [file for file in os.listdir(csv_folder)]
        csv_files = [file for file in csv_files if len(pd.read_csv(os.path.join(csv_folder, file)).values) > 0] # Don't include empty CSV files
        fasta_files = [file.replace('.csv', '.fasta') for file in csv_files]

        self.fasta_files = [os.path.join(fasta_folder, file) for file in fasta_files]
        self.csv_files = [os.path.join(csv_folder, file) for file in csv_files]
        self.seq_length = seq_length

        self.encoder = OneHotEncoder(sparse_output=False, categories=[['A', 'C', 'G', 'T', 'N']])

    def __len__(self):
        return len(self.fasta_files)

    def __getitem__(self, idx):
        # Read FASTA file
        fasta_sequence = str(list(SeqIO.parse(self.fasta_files[idx], "fasta"))[0].seq)

        # Convert FASTA sequence to one-hot encoding
        fasta_sequence = self.encoder.fit_transform(np.array(list(fasta_sequence)[:self.seq_length]).reshape(-1, 1))

        # Read CSV file
        csv_sequence = pd.read_csv(self.csv_files[idx]).values[:self.seq_length]

        return torch.tensor(fasta_sequence, dtype=torch.float32), torch.tensor(csv_sequence, dtype=torch.float32)

# Model architecture
class SequenceToSangerModel(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length):
        super(SequenceToSangerModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_size, 4)  # 4 outputs for A, C, G, T

    def forward(self, x):
        x = self.conv1(x).transpose(-1, -2)
        x = self.fc(x)
        return nn.functional.softmax(x, dim=-1)


