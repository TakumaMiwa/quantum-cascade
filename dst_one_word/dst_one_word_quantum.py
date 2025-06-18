import argparse
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate WER for a fine-tuned Whisper model")
    parser.add_argument("--dataset_name", default="marcel-gohsen/dstc2", help="Dataset name")
    parser.add_argument("--language", default="default", help="Dataset configuration")
    parser.add_argument("--split", default="test", help="Dataset split for evaluation")
    parser.add_argument(
        "--dataset_cache_dir",
        default="dstc2_asr_cache",
        help="Where the dataset cache is stored",
    )
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
    parser.add_argument("--n_best", type=int, default=1, help="Number of beams to generate")
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

    processor = WhisperProcessor.from_pretrained(args.processor_path)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device)

    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None
    else:
        model.config.forced_decoder_ids = None

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
            predicted_ids = model.generate(
                input_features,
                num_beams=max(1, args.n_best),
                num_return_sequences=args.n_best,
            )
        predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        batch["prediction"] = predictions if args.n_best > 1 else predictions[0]
        return batch

    dataset = dataset.map(generate)
    

    

if __name__ == "__main__":
    main()