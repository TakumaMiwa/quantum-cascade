import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
class QuantumNeuralNetwork(nn.Module):
    """Simple QNN for word classification."""

    def __init__(self, num_qubits: int, num_layers: int, num_classes: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers

        dev = qml.device("default.qubit", wires=num_qubits)

        weight_shapes = {"weights": (num_layers, num_qubits, 3)}

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.templates.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
            
            for i in range(num_layers):
                
                for j in range(num_qubits):
                    qml.RX(weights[i, j, 0], wires=j)
                    qml.RY(weights[i, j, 1], wires=j)
                    qml.RZ(weights[i, j, 2], wires=j)
                for j in range(num_qubits):
                    qml.CNOT(wires=[j, (j + 1) % num_qubits])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.classifier = nn.Linear(num_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qlayer(x)
        return self.classifier(x)
