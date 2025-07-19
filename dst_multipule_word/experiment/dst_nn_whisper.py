import sys
import os
sys.path.append("quantum-cascade")
from models.nn import NeuralNetwork
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
    parser = argparse.ArgumentParser(description="Train a neural network for DST on multiple-word data.")
    parser.add_argument(
        "--train_dataset_path",
        default="multiple_word_dataset/traindev",
        help="Path of the dataset on disk",
    )
    parser.add_argument(
        "--test_dataset_path",
        default="multiple_word_dataset/test",
        help="Path of the dataset on disk",
    )
    parser.add_argument("--num_qubits", type=int, default=10, help="Number of qubits used for feature preparation")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of layers in the neural network")
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
    parser.add_argument("--model_output", default="models/nn", help="Where to save the trained model")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--experiment_name", default="amplitude", help="Feature generation method")
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
    test_dataset = prepare_feature(
        args.test_dataset_path,
        model=classical_model,
        processor=processor,
        num_qubits=args.num_qubits,
        experiment_name=args.experiment_name,
    )

    with open(os.path.join("multiple_word_dataset", "slot_list.json"), "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    num_classes = len(label2id)

    model = NeuralNetwork(args.num_layers, input_size=2 * args.num_qubits, output_size=num_classes)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

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
                labels = batch["labels"]
                outputs = model(inputs)
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
            os.makedirs(os.path.join(args.model_output, "whisper_amplitude"), exist_ok=True)
            torch.save(
                {"model_state_dict": model.state_dict()},
                os.path.join(
                    args.model_output,
                    "whisper_amplitude",
                    f"quantum_cascade_epoch_{epoch + 1}.pt",
                ),
            )

    metrics_path = os.path.join(args.model_output, "whisper_amplitude", "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "accuracy"])
        for i in range(len(train_loss_history)):
            writer.writerow([i + 1, train_loss_history[i], test_loss_history[i], accuracy_history[i]])

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), train_loss_history, label="train_loss")
    plt.plot(range(1, args.num_epochs + 1), test_loss_history, label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(args.model_output, "whisper_amplitude", "loss_history_nn_whisper.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), accuracy_history, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(args.model_output, "whisper_amplitude", "accuracy_history_nn_whisper.png"))
    plt.close()

    results_path = os.path.join(args.model_output, "whisper_amplitude", "test_results_nn_whisper.csv")
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

    print("Training complete.")


if __name__ == "__main__":
    main()
