import argparse
import os
from collections import defaultdict
from typing import Dict

import datasets
import pandas as pd

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple QNN on one-word audio data")
    parser.add_argument("--dataset_name", default="marcel-gohsen/dstc2", help="Name of the dataset")
    return parser.parse_args()

def main():
    """
    datasets.load_datasetを用いて指定のデータセットを読み込み，各単語数とスロットにおけるデータ数をそれぞれ保存してください．
    保存先はmultiple_word/dstc2_data_num_per_word.csvとします．
    縦軸をスロット，横軸を単語数とし，スロットごとに各単語数のデータ数をカウントしてCSVファイルに保存します。
    データセットの形式は以下のようになっています：
    {
        "traindev":
        [
            {
                "audio": "path/to/audio1.wav",
                "transcript": "word1",
                "slots": ["slot1=value1", "slot2=value2"]
            },
            {
                "audio": "path/to/audio2.wav",
                "transcript": "word2",
                "slots": ["slot1=value3"]
            }
        ],
        test:
        [
            {
                "audio": "path/to/audio3.wav",
                "transcript": "word3",
                "slots": ["slot1=value4"]
            }
        ]
    }
    traindevとtestのデータセットを結合し，スロットごとに単語数をカウントしてCSVファイルに保存します．
    """

    args = parse_args()
    dataset = datasets.load_dataset(args.dataset_name)
    combined = datasets.concatenate_datasets([dataset["traindev"], dataset["test"]])

    slot_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    max_words = 0

    for item in combined:
        num_words = len(str(item["transcript"]).split())
        max_words = max(max_words, num_words)
        for slot in item["slots"]:
            slot_counts[slot][num_words] += 1

    word_range = list(range(1, max_words + 1))
    df = pd.DataFrame(0, index=sorted(slot_counts.keys()), columns=word_range)

    for slot, counts in slot_counts.items():
        for n, c in counts.items():
            df.loc[slot, n] = c

    os.makedirs("multiple_word", exist_ok=True)
    df.index.name = "slot"
    df.to_csv("multiple_word/dstc2_data_num_per_word.csv")


if __name__ == "__main__":
    main()
