from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn

max_int = 128
batch_size = 16
training_steps = 500
input_length = int(np.log2(max_int)) + 1


def number_binary(number: int) -> list[int]:
    return [int(bit) for bit in bin(number)[2:]]


def generate_even_data(max_int: int, batch_size: int = 16) -> tuple[list[int], list[list[int]]]:
    # Sample batch_size number of integers in range 0 to max_int.
    sampled_integers = np.random.randint(0, max_int, batch_size)

    # Create a list of labels, all ones, because all numbers are even.
    labels = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [number_binary(num) for num in sampled_integers]

    # Get the number of binary places needed to represent the maximum number.
    max_length = int(np.log2(max_int)) + 1

    # Add padding in front of each number to regularize it
    data = [[0]*(max_length - len(num)) + num for num in data]

    return data, labels


class Generator(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.dense_layer = nn.Linear(input_length, input_length)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))


class Discriminator(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.dense_layer = nn.Linear(input_length, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.dense_layer(x))


g = Generator(input_length)
d = Discriminator(input_length)

loss = nn.BCELoss()

g_optim = torch.optim.Adam(g.parameters(), lr=1e-3)
d_optim = torch.optim.Adam(d.parameters(), lr=1e-3)

writer = SummaryWriter()

for i in range(training_steps):
    g_optim.zero_grad()

    # I want to generate data with the generator, so I need to put something in it first. I'll be putting in noise.
    noise = torch.randint(0, 2, (batch_size, input_length)).float()
    g_data = g(noise)

    # I want real data to put inside the discriminator
    true_data, true_labels = generate_even_data(max_int=max_int, batch_size=batch_size)
    true_labels = torch.tensor(true_labels).float()
    true_data = torch.tensor(true_data).float()

    # Use what the discriminator thinks as a way to evaluate the performance of the generator.
    g_d_out = d(g_data).view(-1)
    g_loss = loss(g_d_out, true_labels)
    g_loss.backward()
    g_optim.step()

    # Clear the gradients from the filled-up expression graph that included the discriminator parameters
    d_optim.zero_grad()

    # Get the loss from the discriminator looking at the true data
    d_true_data = d(true_data).view(-1)
    # true_labels was not the result of any of the parameters in any of the neural networks so no need to detach
    d_true_loss = loss(d_true_data, true_labels)

    if i % 50 == 0:  # N is the frequency with which you want to print the stats
        with torch.no_grad():  # Avoid tracking computations
            true_predictions = torch.round(d_true_data).int()
            generated_predictions = torch.round(g_d_out.detach()).int()
            print(f'Step {i}:')
            print(f'  Real predictions: {true_predictions.tolist()}')
            print(f'  Generated predictions: {int("".join(list(map(lambda x: str(x), generated_predictions.tolist()))),2)}')
            print(f'  True labels: {true_labels.int().tolist()}')

    # Get the loss from the discriminator looking at the generated data.
    g_d_out = d(g_data.detach()).view(-1)
    d_gen_loss = loss(g_d_out, torch.zeros(batch_size))
    d_loss = (d_true_loss + d_gen_loss) / 2
    d_loss.backward()
    d_optim.step()

    writer.add_scalar('Generator Loss', g_loss.item(), i)
    writer.add_scalar('Discriminator Loss', d_loss.item(), i)

writer.close()
