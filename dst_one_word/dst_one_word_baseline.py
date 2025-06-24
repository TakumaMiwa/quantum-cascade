import argparse
from typing import Dict, List

import datasets
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import os

## TO DO
# スロットが無いデータの対応を決める
# 訓練データとテストデータの重複率を調べる

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
            qml.templates.AmplitudeEmbedding(inputs, wires=range(num_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
        self.classifier = nn.Linear(num_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qlayer(x)
        return self.classifier(x)


def prepare_features(num_qubits: int, text_column: str, label_column: str):
    """Create a preprocessing function for the dataset."""

    word2index: Dict[str, int] = {}
    label2index: Dict[str, int] = {}

    def _preprocess(batch: Dict) -> Dict:
        """Create one-hot encoded features for the transcription."""
        # Use the slot data as the label
        label = batch[label_column][0]
        if label not in label2index:
            label2index[label] = len(label2index)
        batch["labels"] = label2index[label]
        text = batch[text_column]

        # Map each unique word to an index for the one-hot feature vector
        if text not in word2index:
            word2index[text] = len(word2index)

        index = word2index[text]
        features = np.zeros(2**num_qubits, dtype=np.float32)
        if index < 2**num_qubits:
            features[index] = 1.0
        batch["input_features"] = features
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
    parser.add_argument("--train_dataset_path", default="one_word_dataset/traindev", help="Path of the dataset on disk")
    parser.add_argument("--test_dataset_path", default="one_word_dataset/test", help="Path of the dataset on disk")
    parser.add_argument("--audio_column", default="audio", help="Column containing audio data")
    parser.add_argument("--num_qubits", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--model_output", default="models", help="Where to save the trained model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dataset = datasets.load_from_disk(args.train_dataset_path)
    test_dataset = datasets.load_from_disk(args.test_dataset_path)

    # identify text column
    text_column = None
    for col in ["sentence", "text", "transcript", "labels"]:
        if col in train_dataset.column_names:
            text_column = col
            break
    if text_column is None:
        raise ValueError("No transcription column found in dataset")

    preprocess_fn = prepare_features(args.num_qubits, text_column, "slots")

    train_dataset = train_dataset.cast_column(
        args.audio_column, datasets.Audio(sampling_rate=16000)
    )
    test_dataset = test_dataset.cast_column(
        args.audio_column, datasets.Audio(sampling_rate=16000)
    )

    train_dataset = train_dataset.map(preprocess_fn)
    test_dataset = test_dataset.map(preprocess_fn)

    train_dataset, label2id = encode_labels(train_dataset, "labels")

    def _map_test_labels(batch: Dict) -> Dict:
        batch["labels"] = label2id[batch["labels"]]
        return batch

    test_dataset = test_dataset.map(_map_test_labels)
    num_classes = len(label2id)

    train_dataset.set_format(type="torch", columns=["input_features", "labels"])
    test_dataset.set_format(type="torch", columns=["input_features", "labels"])

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

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

        test_loss: List[float] = []
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch["input_features"]
                labels = batch["labels"]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss.append(loss.item())
                if (epoch + 1) % 5 == 0:
                    print(outputs)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

        log = (
            f"Epoch {epoch + 1}/{args.num_epochs} - Loss: {np.mean(epoch_loss):.4f}"
            f" - Test Loss: {np.mean(test_loss):.4f}"
        )
        if (epoch + 1) % 5 == 0 and total > 0:
            accuracy = correct / total
            log += f" - Test Acc: {accuracy:.4f}"
        print(log)
        if (epoch + 1) % 10 == 0:
            print(f"Saving model at epoch {epoch + 1}")
            torch.save({"model_state_dict": model.state_dict(), "label2id": label2id}, os.path.join(args.model_output, f"model_epoch_{epoch + 1}.pt"))
      


if __name__ == "__main__":
    main()
