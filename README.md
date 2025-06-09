# quantum-cascade

This repository contains an example script for fine-tuning [OpenAI Whisper](https://github.com/openai/whisper) on a speech dataset. See `finetune_whisper.py`.

## Dataset format

The script expects a dataset with two columns:

- `audio`: audio data or file path. When loaded with the Hugging Face `datasets` library, this should be an `Audio` feature so that the audio array and sampling rate are available.
- `sentence`: the text transcription for the corresponding audio.

[Mozilla Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0) is an example dataset with this structure.

## Usage

Install the required libraries and run:

```bash
pip install datasets transformers
python finetune_whisper.py --dataset_name mozilla-foundation/common_voice_11_0 --language en
```

To store the downloaded dataset in a specific directory, add the `--dataset_cache_dir` option.

The finetuned model will be saved to `whisper_finetuned` by default.

## Evaluating word error rate

After fine-tuning, you can evaluate the model using `culc_wer.py`:

```bash
python culc_wer.py --model_path path/to/checkpoint \
                   --processor_path openai/whisper-small
```

The `--processor_path` argument should point to the base Whisper model
whose tokenizer was used during training. If not specified, it defaults
to `openai/whisper-small`.
