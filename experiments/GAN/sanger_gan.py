import torch
import torch.nn as nn
import pandas as pd
from Bio import SeqIO
import os

# Hyperparameters
batch_size = 16
hidden_size = 16
learning_rate = 1e-3
training_steps = 500
input_length_gen = 5  # One-hot encoding size for A, T, C, G, N
input_length_dis = 4
seq_length = 300

# Directories containing the CSV files and corresponding FASTA files
traces_dir = 'traces'
fasta_dir = 'fasta_files'

# Lists to store the data
chromatogram_list = []
sequence_list = []

# One-hot encoding mapping
nucleotide_mapping = {'A': [1, 0, 0, 0, 0], 'T': [0, 1, 0, 0, 0], 'C': [0, 0, 1, 0, 0], 'G': [0, 0, 0, 1, 0], 'N': [0, 0, 0, 0, 1]}

# Iterate through the CSV files in the traces folder
for chromatogram_file in os.listdir(traces_dir):
    chromatogram_path = os.path.join(traces_dir, chromatogram_file)
    chromatogram_data = pd.read_csv(chromatogram_path).values[:seq_length]

    # Check if chromatogram data is empty and continue to the next file if so
    if chromatogram_data.size == 0:
        print(f"Ignoring empty chromatogram file {chromatogram_path}")
        continue

    normalized_chromatogram_data = torch.tensor(chromatogram_data / chromatogram_data.max()).float()

    # Construct the corresponding FASTA file path
    sequence_name = os.path.splitext(chromatogram_file)[0]
    fasta_path = os.path.join(fasta_dir, sequence_name + '.fasta')

    # Read the sequence from the corresponding FASTA file
    matching_sequence = str(list(SeqIO.parse(fasta_path, "fasta"))[0].seq)[:seq_length]

    # Check if sequence is empty and continue to the next file if so
    if not matching_sequence:
        print(f"Ignoring empty sequence in file {fasta_path}")
        continue

    # One-hot encode the sequence
    one_hot_sequence = torch.tensor([nucleotide_mapping[n] for n in matching_sequence]).float()

    # Add data to lists
    chromatogram_list.append(normalized_chromatogram_data)
    sequence_list.append(one_hot_sequence)

# Concatenate the data and split into batches
normalized_chromatogram_data = torch.stack(chromatogram_list).split(batch_size)
one_hot_sequences = torch.stack(sequence_list).split(batch_size)

# Generator definition
class Generator(nn.Module):
    def __init__(self, input_length_gen, hidden_size, seq_length):
        super().__init__()
        self.lstm = nn.LSTM(input_length_gen, hidden_size, batch_first=True)
        self.dense_layer = nn.Linear(hidden_size, 4)
        self.activation = nn.Softmax(dim=2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.dense_layer(lstm_out)
        return self.activation(output)

# Discriminator definition
class Discriminator(nn.Module):
    def __init__(self, input_length_dis, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_length_dis, hidden_size, batch_first=True)
        self.dense_layer = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.dense_layer(lstm_out) # Using the last hidden state
        return self.activation(output)

# Initialize models
g = Generator(input_length_gen, hidden_size, seq_length)
d = Discriminator(input_length_dis, hidden_size)

# Optimizers
g_optim = torch.optim.Adam(g.parameters(), lr=learning_rate)
d_optim = torch.optim.Adam(d.parameters(), lr=learning_rate)

# Loss function
loss = nn.MSELoss()

# Training loop
for i in range(training_steps):
    for chromatogram_batch, sequence_batch in zip(normalized_chromatogram_data, one_hot_sequences):
        # Reset gradients
        g_optim.zero_grad()
        d_optim.zero_grad()

        # Generate data with the generator
        g_data = g(sequence_batch)

        # Loss for the generator
        g_d_out = d(g_data)
        g_loss = loss(g_d_out, torch.ones_like(g_d_out))
        g_loss.backward()
        g_optim.step()

        # Discriminator loss for real data
        d_true_data = d(chromatogram_batch)
        d_true_loss = loss(d_true_data, torch.ones_like(d_true_data))

        # Discriminator loss for generated data
        d_gen_loss = loss(g_d_out.detach(), torch.zeros_like(g_d_out))
        d_loss = (d_true_loss + d_gen_loss) / 2
        d_loss.backward()
        d_optim.step()

    # Print stats every 50 steps
    if i % 2 == 0:
        print(f'Step {i}:')
        print(f'  Generator Loss: {g_loss.item()}')
        print(f'  Discriminator Loss: {d_loss.item()}')

    # Save the generator's state after the zeroth and eighth steps
    if i == 0 or i == 8:
        torch.save(g.state_dict(), f'generator_step_{i}.pth')
