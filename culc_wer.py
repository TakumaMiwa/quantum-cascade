import argparse
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from jiwer import wer
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate WER for a fine-tuned Whisper model")
    parser.add_argument("--dataset_name", default="librispeech_asr", help="Dataset name")
    parser.add_argument("--language", default="clean", help="Dataset configuration")
    parser.add_argument("--split", default="test", help="Dataset split for evaluation")
    parser.add_argument(
        "--dataset_cache_dir",
        default="librispeech_asr_cache",
        help="Where the dataset cache is stored",
    )
    parser.add_argument(
        "--model_path",
        default="whisper_finetuned/checkpoint-3903",
        help="Path to the fine-tuned model",
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"

    dataset = load_dataset(
        args.dataset_name,
        args.language,
        split=args.split,
        cache_dir=args.dataset_cache_dir,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(args.model_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)

    text_column = None
    for col in ["sentence", "text", "transcript"]:
        if col in dataset.column_names:
            text_column = col
            break
    if text_column is None:
        raise ValueError("No transcription column found in dataset")

    def generate(batch):
        audio = batch["audio"]
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features
        input_features = torch.tensor(input_features).to(device)
        with torch.no_grad():
            predicted_ids = model.generate(input_features)[0]
        batch["prediction"] = processor.decode(predicted_ids, skip_special_tokens=True)
        return batch

    results = dataset.map(generate)
    references = results[text_column]
    predictions = results["prediction"]
    score = wer(references, predictions)
    print(f"WER: {score:.4f}")


if __name__ == "__main__":
    main()
