import sys
sys.path.append("quantum-cascade")
from models.nn import NeuralNetwork
from utils.prepare_features import prepare_features
import torch
import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Dict, List
import os
import numpy as np
import datasets
import csv
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Train a quantum neural network for DST.")
    parser.add_argument("--train_dataset_path", default="one_word_dataset/traindev", help="Path of the dataset on disk")
    parser.add_argument("--test_dataset_path", default="one_word_dataset/test", help="Path of the dataset on disk")
    parser.add_argument("--num_qubits", type=int, default=7, help="Number of qubits in the quantum circuit")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of layers in the quantum circuit")
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
        help="Path or name of the processor to use (defaults to base Whisper model)",
    )
    parser.add_argument("--model_output", default="models/nn", help="Where to save the trained model")
    parser.add_argument("--lr", type=float, default=1e-3)
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model
    processor = WhisperProcessor.from_pretrained(args.processor_path)
    classical_model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)
    if hasattr(classical_model, "generation_config"):
        classical_model.generation_config.forced_decoder_ids = None
    else:
        classical_model.config.forced_decoder_ids = None



    # Load datasets
    train_dataset = prepare_features(args.train_dataset_path, model=classical_model, processor=processor)
    # train_dataset.save_to_disk("one_word_dataset/intermidiate_data/traindev_1_best")
    # train_dataset = datasets.load_from_disk("one_word_dataset/traindev_prepared")
    test_dataset = prepare_features(args.test_dataset_path, model=classical_model, processor=processor)
    # test_dataset.save_to_disk("one_word_dataset/intermidiate_data/test_1_best")
    # test_dataset = datasets.load_from_disk("one_word_dataset/test_prepared")
    
    
    # Initialize model
    model = NeuralNetwork(args.num_layers, input_size=128, output_size=71)
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )


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
        accuracy = correct / total
        accuracy_history.append(accuracy)
        train_loss_history.append(np.mean(epoch_loss))
        test_loss_history.append(np.mean(test_loss))
        log += f" - Test Acc: {accuracy:.4f}"
        print(log)
        if (epoch + 1) % 10 == 0:
            print(f"Saving model at epoch {epoch + 1}")
            torch.save({"model_state_dict": model.state_dict()}, os.path.join(args.model_output, "whisper_amplitude", f"quantum_cascade_epoch_{epoch + 1}.pt"))
        
    metrics_path = os.path.join(args.model_output, "whisper_amplitude", "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer =csv.writer(f)
        writer.writerow(["epoch", "train_loss", "test_loss", "accuracy"])
        for i in range(len(train_loss_history)):
            writer.writerow([i + 1, train_loss_history[i], test_loss_history[i], accuracy_history[i]])

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), train_loss_history, label="train_loss")
    plt.plot(range(1, args.num_epochs + 1), test_loss_history, label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(args.model_output, "whisper_amplitude", "loss_history_qnn_whisper.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, args.num_epochs + 1), accuracy_history, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(args.model_output, "whisper_amplitude", "accuracy_history_qnn_whisper.png"))
    plt.close()

    with open(os.path.join("one_word_dataset", "slot_list.json"), "r") as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    results_path = os.path.join(args.model_output, "whisper_amplitude", "test_results_qnn_whisper.csv")
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