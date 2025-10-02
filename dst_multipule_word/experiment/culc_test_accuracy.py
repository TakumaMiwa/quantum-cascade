import argparse
import json
import os
import sys
from typing import Tuple

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.append("quantum-cascade")

from models.qnn import QuantumNeuralNetwork  # noqa: E402
from dst_multipule_word.utils.prepare_feature import prepare_feature  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate test accuracy for a trained quantum neural network."
    )
    parser.add_argument(
        "--dataset_path",
        default="multiple_word_dataset/test",
        help="Path to the dataset on disk.",
    )
    parser.add_argument(
        "--processor_path",
        default="openai/whisper-small",
        help="Identifier or path for the Whisper processor.",
    )
    parser.add_argument(
        "--model_path",
        default="whisper_finetuned/checkpoint-3903",
        help="Path to the fine-tuned Whisper model used for feature extraction.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/mnt/home/takuma-m/multiple_word_output2/qnn/whisper_amplitude/models/quantum_cascade_epoch_100.pt",
        help="Path to the trained QNN checkpoint.",
    )
    parser.add_argument(
        "--num_qubits",
        type=int,
        default=10,
        help="Number of qubits used for feature preparation and the QNN.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=5,
        help="Number of layers in the quantum circuit.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--experiment_name",
        default="amplitude",
        help="Feature generation method (e.g., amplitude or 1-best).",
    )
    return parser.parse_args()


def load_label_mapping() -> Tuple[dict, dict]:
    with open(
        os.path.join("multiple_word_dataset", "dictionary", "slot_list.json"), "r"
    ) as f:
        label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = WhisperProcessor.from_pretrained(args.processor_path)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(
        device
    )
    whisper_model.eval()

    if hasattr(whisper_model, "generation_config"):
        whisper_model.generation_config.forced_decoder_ids = None
    else:
        whisper_model.config.forced_decoder_ids = None

    test_dataset = prepare_feature(
        args.dataset_path,
        model=whisper_model,
        processor=processor,
        num_qubits=args.num_qubits,
        experiment_name=args.experiment_name,
    )

    dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    label2id, _ = load_label_mapping()
    num_classes = len(label2id)

    qnn_model = QuantumNeuralNetwork(
        num_qubits=args.num_qubits,
        num_layers=args.num_layers,
        num_classes=num_classes,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    qnn_model.load_state_dict(state_dict)
    qnn_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input_features"]
            labels = batch["labels"]
            outputs = qnn_model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
