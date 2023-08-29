import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Define data folders
fasta_folder = 'fasta_files'
csv_folder = 'traces'
# Modifying the SangerSequencingDataset class to handle 'N' as a nucleotide

class SangerSequencingDataset(Dataset):
    def __init__(self, fasta_folder, csv_folder):
        csv_files = [file for file in os.listdir(csv_folder)]
        fasta_files = [file.replace('.csv', '.fasta') for file in csv_files]

        self.fasta_files = [os.path.join(fasta_folder, file) for file in fasta_files]
        self.csv_files = [os.path.join(csv_folder, file) for file in csv_files]

        self.encoder = OneHotEncoder(sparse_output=False, categories=[['A', 'C', 'G', 'T', 'N']])

    def __len__(self):
        return len(self.fasta_files)

    def __getitem__(self, idx):
        # Read FASTA file
        with open(self.fasta_files[idx], 'r') as file:
            fasta_sequence = ''.join(file.readlines()[1:]).replace('\n', '')

        # Convert FASTA sequence to one-hot encoding
        fasta_sequence = self.encoder.fit_transform(np.array(list(fasta_sequence)).reshape(-1, 1))

        # Read CSV file
        csv_sequence = pd.read_csv(self.csv_files[idx]).values

        return torch.tensor(fasta_sequence, dtype=torch.float32), torch.tensor(csv_sequence, dtype=torch.float32)


# Create dataset and dataloader with updated class
dataset = SangerSequencingDataset(fasta_folder, csv_folder)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size of 1 for the given example