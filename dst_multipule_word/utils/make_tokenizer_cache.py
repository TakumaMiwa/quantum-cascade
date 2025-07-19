from datasets import load_from_disk, Audio
from transformers import WhisperProcessor
import json
def main():
    dataset = {"traindev":load_from_disk("./multiple_word_dataset/traindev"),
                "test":load_from_disk("./multiple_word_dataset/test")}
    tokenizer = WhisperProcessor.from_pretrained("openai/whisper-small")
    target_token_list = []
    count = 0
    for split in ["traindev", "test"]:

        for i, item in enumerate(dataset[split]):
            ids = tokenizer.tokenizer.encode(
                item["transcript"], add_special_tokens=False, max_length=10, truncation=True
            )
            if len(ids) < 10:
                ids += [tokenizer.tokenizer.pad_token_id] * (10 - len(ids))
            if ids[:10] not in target_token_list:
                target_token_list.append(ids[:10])
        
        if split == "traindev":
            target_token_seqs = {str(i): ids for i, ids in enumerate(target_token_list)}
            with open("multiple_word_dataset/tokenizer_cache/tokenizer_cache_traindev.json", "w") as f:
                json.dump(target_token_seqs, f, indent=4)
        else:
            target_token_seqs = {str(i): ids for i, ids in enumerate(target_token_list)}
            with open("multiple_word_dataset/tokenizer_cache/tokenizer_cache_all.json", "w") as f:
                json.dump(target_token_seqs, f, indent=4)
if __name__ == "__main__":
    main()