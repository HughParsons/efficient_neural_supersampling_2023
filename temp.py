import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            
        )
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu3 = nn.ReLU()            

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        x = self.relu3(x)
        return x

class ConvolutionalMLP(nn.Module):
    def __init__(self, input_channels, hidden_size, output_channels, kernel_size):
        super(ConvolutionalMLP, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        # Define the MLP to predict the convolutional kernel (1 x 1 x kernel_size x kernel_size)
        self.mlp = MLP(input_channels, hidden_size, kernel_size * kernel_size)
        
        # Define the convolutional layer
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, padding=1, stride=1)
        self.relu = nn.ReLU()
    

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Predict the convolutional kernel
        kernel_weights = self.mlp(x[:, :, 0, 0])
        kernel = kernel_weights.view(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(self.output_channels, self.input_channels, 1, 1)
        self.kernel = kernel
        
        
        
        # Apply convolution with the predicted kernel
        x = F.conv2d(x, weight=kernel, padding=1, stride=1)
        
        # Apply the convolutional layer
        x = self.conv(x)
        x = self.relu(x)
        
        return x

# Example usage:
B = 1
input_channels = 3  # Number of input channels
hidden_size = 64    # Size of the hidden layer in the MLP
output_channels = 3  # Number of output channels for the convolutional layer
kernel_size = 3      # Size of the convolutional kernel

# Create an instance of the ConvolutionalMLP model
model = ConvolutionalMLP(input_channels, hidden_size, output_channels, kernel_size)

# Define a loss function and optimizer for training
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Forward pass
input_data = torch.randn(B, input_channels, 64, 64)  # Example input data
input_data.requires_grad = True
labels = torch.randn(B, output_channels, 64, 64)  # Example labels


# # Calculate loss and perform backpropagation
# loss = criterion(output, labels)
# loss.backward()
# optimizer.step()

mlp = model.mlp
conv = model.conv

# Perform a few training iterations
for i in range(5):
    
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, labels)
    print(f"Loss: {loss}")
    loss.backward()
    # print(conv.weight.grad)
    print(model.kernel)
    optimizer.step()


# # Print MLP weights after training
# print("MLP weights after training:")
# print(mlp_weights_before_training[0:10])
# mlp_weights_after_training = model.mlp.net[0].weight
# conv_weights_after_training = model.net[0].weight

# print(f"MLP weights have been updated: {not torch.equal(mlp_weights_before_training, mlp_weights_after_training)}")
# print(f"Conv weights have been updated: {not torch.equal(conv_weights_before_training, conv_weights_after_training)}")
