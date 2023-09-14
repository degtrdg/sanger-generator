import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from algorithms.lstm_cnn import SangerSequencingDataset, SequenceToSangerModel
import os

# WandB – Initialize a new run
wandb.init(entity="degtrdg", project="sanger-generator")

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config
config = wandb.config
config.input_size = 5  # A, C, G, T, N
config.hidden_size = 64
config.seq_length = 300
config.learning_rate = 1e-3
config.epochs = 10

# Define data folders
fasta_folder = 'data/fasta_files'
csv_folder = 'data/traces'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and dataloader with updated class
dataset = SangerSequencingDataset(fasta_folder, csv_folder, config.seq_length)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the model, loss, and optimizer
model = SequenceToSangerModel(config.input_size, config.hidden_size, config.seq_length).to(device)

loss_function = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
# Using log="all" log histograms of parameter values in addition to gradients
wandb.watch(model, log="all")

# Create a directory for model checkpoints
if not os.path.exists('model_checkpoints'):
    os.makedirs('model_checkpoints')

# Training loop
for epoch in range(config.epochs):
    for batch_fasta, batch_csv in dataloader:

        batch_fasta = batch_fasta.to(device)
        batch_csv = batch_csv.to(device)

        # Forward pass
        predictions = model(batch_fasta)

        # Compute loss
        loss = loss_function(predictions, batch_csv)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Save model checkpoint after each epoch
    torch.save(model.state_dict(), f'model_checkpoints/{type(model).__name__}_epoch_{epoch+1}.pth')

    print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {loss.item()}")
    wandb.log({"Epoch": epoch + 1, "Loss": loss.item()})

print("Training complete.")
wandb.save('model.h5')
