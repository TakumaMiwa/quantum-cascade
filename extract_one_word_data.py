import argparse
import datasets
import torch
def main():
    parser = argparse.ArgumentParser(description="Extract one word from audio")
    parser.add_argument("--dataset_name", default="marcel-gohsen/dstc2", help="Name of the dataset")
    parser.add_argument("--audio_column", default="audio", help="Name of the audio column")
    parser.add_argument(
        "--dataset_cache_dir",
        default="dstc2_asr_cache",
        help="Where the dataset cache is stored",
    )
    parser.add_argument(
        "--new_dataset_path",
        default="one_word_dataset",
        help="Path to save the new dataset",
    )
    args = parser.parse_args()

    dataset = datasets.load_dataset(
        args.dataset_name,
        cache_dir=args.dataset_cache_dir,
    )
    text_column = None
    for key in dataset.keys():
        for col in ["sentence", "text", "transcript"]:
            if col in dataset[key].column_names:
                text_column = col
                break
        break
    if text_column is None:
        raise ValueError("No transcription column found in dataset")
    new_dataset = datasets.DatasetDict()
    for data_key in ["traindev", "test"]:
        new_datalist = []
        for item in dataset[data_key]:
            if len(item[text_column].split()) == 1:
                slots = item.get("slots", [])
                for slot in slots:
                    key, value = slot.split("=")
                    if key == "food":
                        new_item = {
                            "audio": item[args.audio_column],
                            "transcript": item[text_column],
                            "slots": item.get("slots", []),  # Assuming 'slots' is optional
                        }
                        new_datalist.append(new_item)
        new_dataset[data_key] = datasets.Dataset.from_list(new_datalist)
    ## save the new dataset
    new_dataset.save_to_disk(args.new_dataset_path)
    print(f"New dataset with one-word transcriptions saved to {args.new_dataset_path}")

if __name__ == "__main__":
    main()