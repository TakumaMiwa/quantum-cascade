import argparse

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