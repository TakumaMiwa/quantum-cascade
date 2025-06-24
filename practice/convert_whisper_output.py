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
    parser.add_argument("--dataset_name", default="marcel-gohsen/dstc2", help="Dataset name")
    parser.add_argument("--language", default="default", help="Dataset configuration")
    parser.add_argument("--train_split", default="traindev", help="Dataset split for training")
    parser.add_argument("--test_split", default="test", help="Dataset split for evaluation")
    parser.add_argument(
        "--model_path",
        default="openai/whisper-small",
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
        default=5,
        help="Number of beams to generate (overwritten by dictionary size)",
    )
   
    return parser.parse_args()
def main():
    args = parse_args()
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    dataset = datasets.load_dataset(
        args.dataset_name,
        split="test"
    )
    processor = WhisperProcessor.from_pretrained(args.processor_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
    ).to(device)
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
    else:
        model.config.forced_decoder_ids = None

    inputs = processor(
        audio=dataset[0]["audio"]["array"],
        return_tensors="pt",
        sampling_rate=16000,
    ).to(device)
    
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
    with torch.no_grad():
        outputs = model.generate(
            input_features=inputs["input_features"],
            # do_sample=True,
            # top_k=50,
            # temperature=1.2,
            # num_return_sequences=5,
            return_dict_in_generate=True,
            output_scores=True
        )
    # 各時刻のlogits
    logits_per_step = outputs.scores  # list of [batch_size, vocab_size]

    # 各生成系列
    sequences = outputs.sequences  # [batch_size, seq_len]

    log_probs_per_sequence = []
    for i, seq in enumerate(sequences):
        log_prob = 0.0
        for t, logits in enumerate(logits_per_step[:len(seq)-1]):  # eosは除く
            token_id = seq[t+1]  # decoderのt+1時刻に出力されたトークン
            log_prob += torch.nn.functional.softmax(logits[i], dim=-1)[token_id]
        log_probs_per_sequence.append(log_prob.item())
    print(log_probs_per_sequence)
    # 生成された系列のトークンをデコード
    decoded_sequences = processor.batch_decode(sequences, skip_special_tokens=True)
    print(decoded_sequences)

    # 各系列のlogitsを取得
    logits = torch.stack(logits_per_step, dim=1)  # [batch_size, seq_len, vocab_size]
    print(logits.shape)


if __name__ == "__main__":
    main()