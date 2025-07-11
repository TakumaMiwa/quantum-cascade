import argparse
from typing import Dict, List

import datasets
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import os
import sys
sys.path.append("quantum-cascade")
from models.nn import NeuralNetwork
import json
import csv
import matplotlib.pyplot as plt

## TO DO
# スロットが無いデータの対応を決める
# 訓練データとテストデータの重複率を調べる




def prepare_features(num_qubits: int, text_column: str, label_column: str):
    """Create a preprocessing function for the dataset."""

    word2index: Dict[str, int] = {}

    def _preprocess(batch: Dict) -> Dict:
        """Create one-hot encoded features for the transcription."""
        # Use the slot data as the label
        batch["labels"] = batch[label_column][0]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple QNN on one-word audio data")
    parser.add_argument("--train_dataset_path", default="one_word_dataset/traindev", help="Path of the dataset on disk")
    parser.add_argument("--test_dataset_path", default="one_word_dataset/test", help="Path of the dataset on disk")
    parser.add_argument("--audio_column", default="audio", help="Column containing audio data")
    parser.add_argument("--num_qubits", type=int, default=7)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--model_output", default="models/nn", help="Where to save the trained model")
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

    train_dataset = train_dataset.map(preprocess_fn, load_from_cache_file=False)
    test_dataset = test_dataset.map(preprocess_fn, load_from_cache_file=False)

    def _map_test_labels(batch: Dict) -> Dict:
        batch["labels"] = int(label2id[batch["labels"]])
        return batch
    
    with open("one_word_dataset/slot_list.json", "r") as f:
        label2id = json.load(f)
    train_dataset = train_dataset.map(_map_test_labels, load_from_cache_file=False)
    test_dataset = test_dataset.map(_map_test_labels, load_from_cache_file=False)
    num_classes = len(label2id)

    train_dataset.set_format(type="torch", columns=["input_features", "labels"])
    test_dataset.set_format(type="torch", columns=["input_features", "labels"])

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = NeuralNetwork(args.num_layers, input_size=128, output_size=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_loss_history: List[float] = []
    test_loss_history: List[float] = []
    accuracy_history: List[float] = []

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
                # print(inputs)
                labels = batch["labels"]
                # print(labels)
                
                outputs = model(inputs)
                # print(outputs)
                # sys.exit()
                loss = criterion(outputs, labels)
                test_loss.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        log = (
            f"Epoch {epoch + 1}/{args.num_epochs} - Loss: {np.mean(epoch_loss):.4f}"
            f" - Test Loss: {np.mean(test_loss):.4f}"
        )
        accuracy = correct / total if total > 0 else 0.0
        accuracy_history.append(accuracy)
        train_loss_history.append(np.mean(epoch_loss))
        test_loss_history.append(np.mean(test_loss))
        log += f" - Test Acc: {accuracy:.4f}"
        print(log)
        if (epoch + 1) % 10 == 0:
            print(f"Saving model at epoch {epoch + 1}")
            torch.save({"model_state_dict": model.state_dict(), "label2id": label2id}, os.path.join(args.model_output, f"model_epoch_{epoch + 1}.pt"))

    os.makedirs(args.model_output, exist_ok=True)

    metrics_path = os.path.join(args.model_output, "gold", "metrics_nn_gold.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "accuracy"])
        for i, (tr, te, ac) in enumerate(zip(train_loss_history, test_loss_history, accuracy_history)):
            writer.writerow([i + 1, tr, te, ac])

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), train_loss_history, label="train_loss")
    plt.plot(range(1, args.num_epochs + 1), test_loss_history, label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(args.model_output, "loss_history_nn_gold.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), accuracy_history, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(args.model_output, "accuracy_history_nn_gold.png"))
    plt.close()

    id2label = {v: k for k, v in label2id.items()}
    results_path = os.path.join(args.model_output, "test_results_nn_gold.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_features", "output_features", "true_label", "pred_label"])
        with torch.no_grad():
            for batch in test_dataloader:
                inputs = batch["input_features"]
                labels = batch["labels"]
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                for inp, lab, pred in zip(inputs, labels, preds):
                    writer.writerow([
                        json.dumps(inp.tolist()),
                        json.dumps(outputs.tolist()),
                        id2label[int(lab)],
                        id2label[int(pred)],
                    ])
      


if __name__ == "__main__":
    main()
