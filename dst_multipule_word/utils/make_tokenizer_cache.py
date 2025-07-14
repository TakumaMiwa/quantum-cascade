from datasets import load_from_disk, Audio
import json
def main():
    dataset = {"traindev":load_from_disk("./multiple_word_dataset/traindev"),
                "test":load_from_disk("./multiple_word_dataset/test")}

    # 辞書の読み込み
    with open("multiple_word_dataset/dictionary/dstc2_wordlist.json", "r") as f:
        word_dic = json.load(f)
    text_list = set()
    for split in ["traindev", "test"]:
        for item in dataset[split]:
            text = item["transcript"]
            # 単語を分割してリストに追加
            text_ids = [str(word_dic[word]) for word in text.split() if word in word_dic]
            if len(text_ids) < 4:
                text_ids += ["1"] * (4 - len(text_ids))
            key = "_".join(text_ids)
            text_list.add(key)
        if split == "traindev":
            text_dic = {word: idx for idx, word in enumerate(text_list)}
            with open("multiple_word_dataset/tokenizer_cache/tokenizer_cache_traindev.json", "w") as f:
                json.dump(text_dic, f)
        elif split == "test":
            text_dic = {word: idx for idx, word in enumerate(text_list)}
            with open("multiple_word_dataset/tokenizer_cache/tokenizer_cache_all.json", "w") as f:
                json.dump(text_dic, f)
    print(f"Number of unique words in the dataset: {len(text_list)}")

if __name__ == "__main__":
    main()