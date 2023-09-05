from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from algorithms.baseline_mlp import SangerSequencingDataset, SequenceToSangerMLP

# WandB – Initialize a new run
wandb.init(entity="degtrdg", project="sanger-generator")

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config
config.input_size = 1500  # Updated input size. seq_length * 5. [A, C, G, T, N] 
config.hidden_sizes = [1024, 512, 256]  # Updated hidden sizes
config.seq_length = 300
config.learning_rate = 1e-3
config.epochs = 10
config.output_size = 1200  # Added output size

# Define data folders
fasta_folder = 'data/fasta_files'
csv_folder = 'data/traces'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset and dataloader with updated class
dataset = SangerSequencingDataset(fasta_folder, csv_folder, config.seq_length)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize the model, loss, and optimizer
model = SequenceToSangerMLP(config.input_size, config.hidden_sizes, config.output_size).to(device)  # Updated model parameters

loss_function = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

# WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
# Using log="all" log histograms of parameter values in addition to gradients
wandb.watch(model, log="all")

# Training loop
for epoch in range(config.epochs):
    for batch_fasta, batch_csv in dataloader:
        # batch_fasta = batch_fasta.transpose(2, 1).to(device)  for CNN
        batch_fasta = batch_fasta.to(device)  
        shape = batch_csv.shape
        # Reshape the tensor
        batch_csv_reshaped = batch_csv.view(-1, shape[-1])
        
        # Convert to numpy for scaling
        batch_csv_np = batch_csv_reshaped.cpu().numpy()
        
        # Apply Min-Max scaling
        scaler = MinMaxScaler()
        batch_csv_scaled = scaler.fit_transform(batch_csv_np)
        
        # Convert back to tensor
        batch_csv_tensor = torch.tensor(batch_csv_scaled, device=device)
        
        # Reshape to the original shape
        batch_csv = batch_csv_tensor.view(shape).to(device)

        # Forward pass
        predictions = model(batch_fasta)

        # Compute loss
        loss = loss_function(predictions, batch_csv)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{config.epochs}, Loss: {loss.item()}")
    wandb.log({"Epoch": epoch + 1, "Loss": loss.item()})

print("Training complete.")
wandb.save('model.h5')
