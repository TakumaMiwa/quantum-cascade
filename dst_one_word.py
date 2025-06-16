import argparse
from typing import Dict, List

import datasets
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

        weight_shapes = {"weights": (num_layers, num_qubits)}

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.classifier = nn.Linear(num_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qlayer(x)
        return self.classifier(x)


def prepare_features(num_qubits: int, text_column: str):
    """Create a preprocessing function for the dataset."""

    def _preprocess(batch: Dict) -> Dict:
        audio = batch["audio"]
        array = np.asarray(audio["array"], dtype=np.float32)
        if len(array) < num_qubits:
            padded = np.zeros(num_qubits, dtype=np.float32)
            padded[: len(array)] = array
            array = padded
        else:
            idxs = np.linspace(0, len(array) - 1, num_qubits).astype(int)
            array = array[idxs]
        batch["input_features"] = array
        batch["labels"] = batch[text_column]
        return batch

    return _preprocess


def encode_labels(dataset: datasets.Dataset, label_column: str) -> Dict[str, int]:
    """Encode string labels to ids and apply the mapping."""

    labels = sorted(set(dataset[label_column]))
    label2id = {l: i for i, l in enumerate(labels)}

    def _map(batch: Dict) -> Dict:
        batch["labels"] = label2id[batch[label_column]]
        return batch

    return dataset.map(_map), label2id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple QNN on one-word audio data")
    parser.add_argument("--dataset_path", default="one_word_dataset", help="Path of the dataset on disk")
    parser.add_argument("--audio_column", default="audio", help="Column containing audio data")
    parser.add_argument("--num_qubits", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--model_output", default="qnn_model.pt", help="Where to save the trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = datasets.load_from_disk(args.dataset_path)

    # identify text column
    text_column = None
    for col in ["sentence", "text", "transcript", "labels"]:
        if col in dataset.column_names:
            text_column = col
            break
    if text_column is None:
        raise ValueError("No transcription column found in dataset")

    dataset = dataset.cast_column(args.audio_column, datasets.Audio(sampling_rate=16000))
    dataset = dataset.map(prepare_features(args.num_qubits, text_column))
    dataset, label2id = encode_labels(dataset, "labels")
    num_classes = len(label2id)

    dataset.set_format(type="torch", columns=["input_features", "labels"])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = QuantumNeuralNetwork(args.num_qubits, args.num_layers, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.num_epochs):
        epoch_loss: List[float] = []
        for batch in dataloader:
            inputs = batch["input_features"]
            labels = batch["labels"]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        print(f"Epoch {epoch + 1}/{args.num_epochs} - Loss: {np.mean(epoch_loss):.4f}")

    torch.save({"model_state_dict": model.state_dict(), "label2id": label2id}, args.model_output)
    print(f"Model saved to {args.model_output}")


if __name__ == "__main__":
    main()
