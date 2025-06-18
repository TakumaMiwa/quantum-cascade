import argparse
import os
from typing import Dict, List

import datasets
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from datasets import load_from_disk, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a QNN on one-word data using Whisper predictions"
    )
    parser.add_argument("--dataset_name", default="one_word_dataset/test", help="Dataset name")
    parser.add_argument("--language", default="default", help="Dataset configuration")
    parser.add_argument("--train_split", default="traindev", help="Dataset split for training")
    parser.add_argument("--test_split", default="test", help="Dataset split for evaluation")
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
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument(
        "--n_best",
        type=int,
        default=1,
        help="Number of beams to generate (overwritten by dictionary size)",
    )
   
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    dataset = load_from_disk(
        args.dataset_name,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    # Keep only the audio column to avoid collate errors when using a DataLoader
    dataset = dataset.remove_columns(
        [c for c in dataset.column_names if c != "audio"]
    )
    dataset.set_format(type="torch", columns=["audio"])

    processor = WhisperProcessor.from_pretrained(args.processor_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
    else:
        model.config.forced_decoder_ids = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )
    for batch in dataloader:
        inputs = processor(
            audio=batch["audio"],
            return_tensors="pt",
            sampling_rate=16000,
        ).to(device)
        beam_outputs = model.generate(
            inputs.input_features,
            num_beams=max(1, args.n_best),
            num_return_sequences=args.n_best,
            max_length=5,
            output_scores=True,
            return_dict_in_generate=True,
        )
        # 出力列とそれぞれのスコア
        sequences = beam_outputs.sequences  # (num_return_sequences, seq_len)
        sequence_scores = beam_outputs.sequences_scores  # (num_return_sequences,)

        # トークン列 → テキスト列に変換
        decoded_texts = processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        # スコアを確率に変換（logスコア → softmax）
        probs = torch.softmax(sequence_scores, dim=0)

        # 表示
        for i, (text, score, prob) in enumerate(
            zip(decoded_texts, sequence_scores, probs)
        ):
            print(f"Candidate {i+1}:")
            print(f"  Text: {text}")
            print(f"  Log score: {score.item():.4f}")
            print(f"  Normalized prob: {prob.item():.4f}")


if __name__ == "__main__":
    main()
