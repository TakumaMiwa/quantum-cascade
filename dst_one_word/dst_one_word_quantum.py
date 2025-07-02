import sys
sys.path.append("quantum-cascade")
from models.qnn import QuantumNeuralNetwork
from utils.prepare_features import prepare_features
import torch
import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Dict, List
import os
import numpy as np
import datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Train a quantum neural network for DST.")
    parser.add_argument("--train_dataset_path", default="one_word_dataset/traindev", help="Path of the dataset on disk")
    parser.add_argument("--test_dataset_path", default="one_word_dataset/test", help="Path of the dataset on disk")
    parser.add_argument("--num_qubits", type=int, default=10, help="Number of qubits in the quantum circuit")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of layers in the quantum circuit")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
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
    parser.add_argument("--model_output", default="models/quantum_cascade", help="Where to save the trained model")
    parser.add_argument("--lr", type=float, default=1e-2)
    
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
    train_dataset.save_to_disk("one_word_dataset/intermidiate_data/traindev_1_best")
    # train_dataset = datasets.load_from_disk("one_word_dataset/traindev_prepared")
    test_dataset = prepare_features(args.test_dataset_path, model=classical_model, processor=processor)
    test_dataset.save_to_disk("one_word_dataset/intermidiate_data/test_1_best")
    # test_dataset = datasets.load_from_disk("one_word_dataset/test_prepared")
    
    
    # Initialize model
    model = QuantumNeuralNetwork(num_qubits=args.num_qubits, num_layers=args.num_layers, num_classes=len(train_dataset['labels'].unique()))

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    model = QuantumNeuralNetwork(args.num_qubits, args.num_layers, num_classes=71)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

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
            torch.save({"model_state_dict": model.state_dict()}, os.path.join(args.model_output, f"quantum_cascade_epoch_{epoch + 1}.pt"))

    print("Training complete.")
if __name__ == "__main__":
    main()