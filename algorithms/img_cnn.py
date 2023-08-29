import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # In our constructor, we define our neural network architecture that we'll use in the forward pass.
        # Conv2d() adds a convolution layer that generates 2 dimensional feature maps to learn different aspects of our image
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Linear(x,y) creates dense, fully connected layers with x inputs and y outputs
        # Linear layers simply output the dot product of our inputs and weights.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Here we feed the feature maps from the convolutional layers into a max_pool2d layer.
        # The max_pool2d layer reduces the size of the image representation our convolutional layers learnt,
        # and in doing so it reduces the number of parameters and computations the network needs to perform.
        # Finally we apply the relu activation function which gives us max(0, max_pool2d_output)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        
        # Reshapes x into size (-1, 16 * 5 * 5) so we can feed the convolution layer outputs into our fully connected layer
        x = x.view(-1, 16 * 5 * 5)
        
        # We apply the relu activation function and dropout to the output of our fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Finally we apply the softmax function to squash the probabilities of each class (0-9) and ensure they add to 1.
        return F.log_softmax(x, dim=1)