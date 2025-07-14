from datasets import load_from_disk
import json
def main():
    dataset = load_from_disk("./multiple_word_dataset/traindev")

    text_column = None
    for col in ["sentence", "text", "transcript"]:
        if col in dataset.column_names:
            text_column = col
            break
    if text_column is None:
        raise ValueError("No suitable text column found in the dataset.")
    word_dic = set()
    for item in dataset:
        if text_column in item:
            text = item[text_column]
            words = text.split()
            for word in words:
                word_dic.add(word.lower())
    word_dic = {"<unk>": 0, "<blank>": 1, **{word: i + 2 for i, word in enumerate(sorted(word_dic))}}
    id_dic = {v: k for k, v in word_dic.items()}
    with open("multiple_word_dataset/dictionary/dstc2_wordlist.json", "w") as f:
        json.dump(word_dic, f, indent=4)
    with open("multiple_word_dataset/dictionary/dstc2_idlist.json", "w") as f:
        json.dump(id_dic, f, indent=4)

if __name__ == "__main__":
    main()