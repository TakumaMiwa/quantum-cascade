import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, layer_num: int, input_size: int, output_size: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)   
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=-1)  # Apply softmax to the output
        return x
