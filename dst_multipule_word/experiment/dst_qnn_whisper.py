import sys
import os
sys.path.append("quantum-cascade")
from models.qnn import QuantumNeuralNetwork
from dst_multipule_word.utils.prepare_feature import prepare_feature
import torch
import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import List
import numpy as np
import csv
import json
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a quantum neural network for DST on multiple-word data."
    )
    parser.add_argument(
        "--train_dataset_path",
        default="multiple_word_dataset/traindev",
        help="Path of the dataset on disk",
    )
    parser.add_argument(
        "--num_qubits", type=int, default=10, help="Number of qubits used for feature preparation"
    )
    parser.add_argument(
        "--num_layers", type=int, default=5, help="Number of layers in the quantum circuit"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument(
        "--model_path",
        default="whisper_finetuned/checkpoint-3903",
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--processor_path",
        default="openai/whisper-small",
        help="Path or name of the processor to use",
    )
    parser.add_argument("--model_output", default="multiple_word_output/qnn", help="Where to save the trained model")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument(
        "--experiment_name", default="amplitude", help="Feature generation method (amplitude or 1-best)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(args.processor_path)
    classical_model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)
    if hasattr(classical_model, "generation_config"):
        classical_model.generation_config.forced_decoder_ids = None
    else:
        classical_model.config.forced_decoder_ids = None

    train_dataset = prepare_feature(
        args.train_dataset_path,
        model=classical_model,
        processor=processor,
        num_qubits=args.num_qubits,
        experiment_name=args.experiment_name,
    )

    dataset_split = train_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = dataset_split["train"]
    val_dataset = dataset_split["test"]

    with open(os.path.join("multiple_word_dataset", "dictionary", "slot_list.json"), "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)

    model = QuantumNeuralNetwork(
        num_qubits=args.num_qubits, num_layers=args.num_layers, num_classes=num_classes
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []
    accuracy_history: List[float] = []

    if args.experiment_name == "amplitude":
        save_dir = "whisper_amplitude"
    elif args.experiment_name == "1-best":
        save_dir = "whisper_1_best"
    else:
        save_dir = args.experiment_name

    for epoch in range(args.num_epochs):
        epoch_loss: List[float] = []
        for batch in train_dataloader:
            inputs = batch["input_features"]
            labels = batch["labels"]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        val_loss: List[float] = []
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs = batch["input_features"]
                labels = batch["labels"]
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss.append(loss.item())
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        log = (
            f"Epoch {epoch + 1}/{args.num_epochs} - Loss: {np.mean(epoch_loss):.4f}"
            f" - Val Loss: {np.mean(val_loss):.4f}"
        )
        accuracy = correct / total if total > 0 else 0.0
        accuracy_history.append(accuracy)
        train_loss_history.append(np.mean(epoch_loss))
        val_loss_history.append(np.mean(val_loss))
        log += f" - Val Acc: {accuracy:.4f}"
        print(log)
        if (epoch + 1) % 10 == 0:
            os.makedirs(os.path.join(args.model_output, save_dir, "models"), exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(
                    args.model_output,
                    save_dir,
                    f"quantum_cascade_epoch_{epoch + 1}.pt",
                ),
            )

    metrics_path = os.path.join(args.model_output, save_dir)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(os.path.join(metrics_path, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "accuracy"])
        for i in range(len(train_loss_history)):
            writer.writerow([i + 1, train_loss_history[i], val_loss_history[i], accuracy_history[i]])

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), train_loss_history, label="train_loss")
    plt.plot(range(1, args.num_epochs + 1), val_loss_history, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(metrics_path, "loss_history_qnn_whisper.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), accuracy_history, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(metrics_path, "accuracy_history_qnn_whisper.png"))
    plt.close()

    results_path = os.path.join(metrics_path, "val_results_qnn_whisper.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_features", "output_features", "true_label", "pred_label"])
        with torch.no_grad():
            for batch in val_dataloader:
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

    print("Training complete.")


if __name__ == "__main__":
    main()
