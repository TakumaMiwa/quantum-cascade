

def main() -> None:
    """dst_one_word/utils/draw_learning_process.pyをもとに，
    各モデルの訓練過程を可視化するスクリプトを記述してください．
    ただし，モデルのパスは以下のように変更してください．"""

    metric_files: Dict[str, str] = {
        "nn_gold": "multiple_word_output/nn/gold/metrics.csv",
        "nn_whisper_1_best": "multiple_word_output/nn/whisper_1_best/metrics.csv",
        "nn_whisper_amplitude": "multiple_word_output/nn/whisper_amplitude/metrics.csv",
        "qnn_gold": "multiple_word_output/qnn/gold/metrics.csv",
        "qnn_whisper_1_best": "multiple_word_output/qnn/whisper_1_best/metrics.csv",
        "qnn_whisper_amplitude": "multiple_word_output/qnn/whisper_amplitude/metrics.csv"
    }