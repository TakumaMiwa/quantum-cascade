import argparse
from typing import List

import numpy as np
import torch
from datasets import load_from_disk, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
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
        "--target_words",
        nargs="+",
        default=[],
        help="Target words/phrases to calculate probabilities for",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4,
        help="Maximum number of tokens considered (padding shorter targets)",
    )
   
    return parser.parse_args()
def main():
 
    args = parse_args()
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    dataset = load_from_disk(args.dataset_name)
    processor = WhisperProcessor.from_pretrained(args.processor_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path)
    model.to(device)

    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
    else:
        model.config.forced_decoder_ids = None

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    target_token_seqs: List[List[int]] = []
    for w in args.target_words:
        ids = processor.tokenizer.encode(w, add_special_tokens=False)
        if len(ids) < args.max_length:
            ids += [pad_id] * (args.max_length - len(ids))
        else:
            ids = ids[: args.max_length]
        target_token_seqs.append(ids)

    for i in range(len(dataset)):
        audio = dataset[i]["audio"]
        inputs = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features
        input_features = torch.tensor(inputs).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_features=input_features,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=args.max_length,
            )

        logits = torch.stack(outputs.scores, dim=1)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        print(f"sample {i}: probability distribution shape {probs.shape}")

        for phrase, ids in zip(args.target_words, target_token_seqs):
            prob = 1.0
            for step, token_id in enumerate(ids):
                if step >= probs.shape[1]:
                    break
                prob *= probs[0, step, token_id].item()
            print(f"P('{phrase}') = {prob}")


if __name__ == "__main__":
    main()
