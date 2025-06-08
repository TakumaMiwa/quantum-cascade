"""
Finetune OpenAI Whisper on a speech dataset.
The dataset must contain an 'audio' column and a 'sentence' column with the transcription.
Each example is automatically resampled to 16kHz.
"""
import argparse

import datasets
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from dataclasses import dataclass
from typing import Any
import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator that dynamically pads the inputs and labels for Whisper."""

    processor: Any
    decoder_start_token_id: int = 50257

    def __call__(self, features):
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: f[model_input_name]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune OpenAI Whisper on a speech dataset")
    parser.add_argument("--dataset_name", default="mozilla-foundation/common_voice_11_0", help="Name of the dataset on the Hugging Face Hub")
    parser.add_argument("--language", default="en", help="Language id for Common Voice or similar datasets")
    parser.add_argument("--model_name", default="openai/whisper-small", help="Pretrained model name")
    parser.add_argument("--output_dir", default="whisper_finetuned", help="Where to store the finetuned model")
    parser.add_argument(
        "--dataset_cache_dir",
        default=None,
        help="Directory to store the downloaded dataset cache",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset. For Common Voice the transcription is in the 'sentence' column.
    dataset = load_dataset(
        args.dataset_name,
        args.language,
        cache_dir=args.dataset_cache_dir,
    )

    # Resample audio and cast column to correct sampling rate expected by Whisper
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(args.model_name)

    def prepare_batch(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    dataset = dataset.map(prepare_batch, remove_columns=dataset["train"].column_names, num_proc=1)

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="epoch",
        predict_with_generate=True,
        fp16=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
