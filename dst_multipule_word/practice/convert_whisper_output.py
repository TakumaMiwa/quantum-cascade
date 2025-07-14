import argparse
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import json
import sys
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a QNN on one-word data using Whisper predictions"
    )
    parser.add_argument("--dataset_name", default="./multiple_word_dataset/test", help="Dataset name")
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
        default=5,
        help="Number of beams to generate (overwritten by dictionary size)",
    )
   
    return parser.parse_args()
def main():
 
    args = parse_args()
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    dataset = load_from_disk(
        args.dataset_name,
    )
    processor = WhisperProcessor.from_pretrained(args.processor_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
    )
    model.to(device)
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
    else:
        model.config.forced_decoder_ids = None
    with open("multiple_word_dataset/dictionary/whisper_idlist.json", "r") as f:
        id2word = json.load(f)
    for i in range(10):
        audio = dataset[i]["audio"]
        text = dataset[i]["transcript"]
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features
        input_features = torch.tensor(input_features).to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_features=input_features,
                # do_sample=True,
                # top_k=50,
                # temperature=1.2,
                # num_return_sequences=5,
                return_dict_in_generate=True,
                output_scores=True
            )
        sequences = outputs.sequences           # shape: [n_seq, seq_len]
        logits_per_step = outputs.scores       # list of tensors [n_seq, vocab_size] per step
        logits_per_step = torch.nn.functional.softmax(torch.stack(logits_per_step, dim=1), dim=-1)
        text_pred = []
        for j in range(len(logits_per_step[0])):
            text_pred.append(id2word[str(logits_per_step[0][j].argmax().item())])
        print(f"dataset[{i}]")
        print(text)
        print(text_pred)
        print(logits_per_step.shape)  # shape: [n_seq, seq_len, vocab_size]
    sys.exit()
    ## 対応表をもとにlogits_per_stepを変換
    # cor_tableを読み込む
    with open("multiple_word_dataset/dictionary/correspondence_table.json", "r") as f:
        cor_table = json.load(f)
    # 対応表を使ってlogits_per_stepを変換
    new_logits_per_step = np.zeros((4, 2**8), dtype=np.float32)
    for i in range(len(new_logits_per_step)):
        for j in range(len(cor_table)):
            if j in cor_table:
                new_logits_per_step[i][j] = logits_per_step[i][cor_table[j]]
        # 正規化
        new_logits_per_step[i] /= np.sum(new_logits_per_step[i]) if np.sum(new_logits_per_step[i]) > 0 else 1.0
    print(new_logits_per_step)
    # 全文字列の確率を計算
    with open("multiple_word_dataset/tokenizer_cache/tokenizer_cache_traindev.json", "r") as f:
        cache_dic = json.load(f)
    features = np.zeros(2**10, dtype=np.float32)
    for key, value in cache_dic.items():
        idxs = [int(i) for i in key.split("_")]
        features[int(value)] = np.prod(new_logits_per_step[:, idxs])
    # 正規化
    features /= np.sum(features) if np.sum(features) > 0 else 1.0
    print(text)
    print(features.shape)
    print(features)


if __name__ == "__main__":
    main()