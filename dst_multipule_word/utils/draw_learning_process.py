"""Draw learning curves for multiple word models.

This script is based on ``dst_one_word/utils/draw_learning_process.py`` and
visualises the training process of the models trained on the multiple word
dataset.  The only difference is the location of the metrics CSV files, which
are adjusted for this experiment.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


def _load_metrics(path: str) -> Optional[pd.DataFrame]:
    """Load a metrics CSV if it exists.

    Parameters
    ----------
    path:
        Path to the CSV file containing the recorded metrics for each epoch.

    Returns
    -------
    Optional[pd.DataFrame]
        The loaded data frame or ``None`` if the file does not exist.
    """

    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    return pd.read_csv(path)


def main() -> None:
    """Draw learning curves from saved metrics for the multiple word task."""

    metric_files: Dict[str, str] = {
        "nn_gold": "multiple_word_output/nn/gold/metrics.csv",
        "nn_whisper_1_best": "multiple_word_output/nn/whisper_1_best/metrics.csv",
        "nn_whisper_amplitude": "multiple_word_output/nn/whisper_amplitude/metrics.csv",
        "nn_binary_gold": "multiple_word_output/nn/binary_gold/metrics.csv",
        "nn_binary_1_best": "multiple_word_output/nn/binary_1-best/metrics.csv",
        "qnn_gold": "multiple_word_output/qnn/gold/metrics.csv",
        "qnn_1_best": "multiple_word_output/qnn/whisper_1_best/metrics.csv",
        "qnn_amplitude": "multiple_word_output/qnn/whisper_amplitude/metrics.csv"
    }

    # Load each metrics file if it exists
    metrics: Dict[str, pd.DataFrame] = {}
    for name, path in metric_files.items():
        df = _load_metrics(path)
        if df is not None:
            metrics[name] = df

    if not metrics:
        print("No metrics files found.")
        return

    # Plot learning curves for each metric to enable model comparison
    fig, (ax_train_loss, ax_test_loss, ax_acc) = plt.subplots(1, 3, figsize=(18, 5))

    for name, df in metrics.items():
        epochs = df["epoch"]
        ax_train_loss.plot(epochs, df["train_loss"], label=name)
        ax_test_loss.plot(epochs, df["val_loss"], label=name)
        ax_acc.plot(epochs, df["accuracy"], label=name)

    ax_train_loss.set_xlabel("Epoch")
    ax_train_loss.set_ylabel("Train Loss")
    ax_train_loss.set_title("Train Loss")
    ax_train_loss.legend()

    ax_test_loss.set_xlabel("Epoch")
    ax_test_loss.set_ylabel("Test Loss")
    ax_test_loss.set_title("Test Loss")
    ax_test_loss.legend()

    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    # Summarise final metrics for quick comparison
    summary = {
        name: {
            "train_loss": df["train_loss"].iloc[-1],
            "val_loss": df["val_loss"].iloc[-1],
            "accuracy": df["accuracy"].iloc[-1],
        }
        for name, df in metrics.items()
    }
    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    print("Final metrics:")
    print(summary_df)

    output_dir = "multiple_word_output/output_summary"
    date = "20250816"
    os.makedirs(output_dir, exist_ok=True)
    summary_df.to_csv(os.path.join(output_dir, f"metrics_summary_{date}.csv"))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"learning_process_{date}.png"))


if __name__ == "__main__":
    main()

