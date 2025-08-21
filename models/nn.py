import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, layer_num: int, input_size: int, output_size: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, input_size))
        for _ in range(3):  
            self.layers.append(nn.Linear(input_size, input_size))
        self.layers.append(nn.Linear(input_size, output_size)) 

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nn.functional.relu(x)
        x = self.layers[-1](x)
        x = nn.functional.softmax(x, dim=-1) 
        return x
