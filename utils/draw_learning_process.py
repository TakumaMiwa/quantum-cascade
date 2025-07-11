
import os
from typing import Dict, Optional

import pandas as pd
import matplotlib.pyplot as plt


def _load_metrics(path: str) -> Optional[pd.DataFrame]:
    """Load a metrics CSV if it exists."""

    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    return pd.read_csv(path)


def main() -> None:
    """Draw learning curves from saved metrics."""

    metric_files: Dict[str, str] = {
        "nn_gold": "models/nn/gold/metrics.csv",
        "nn_whisper_all_dic": "models/nn/whisper_all_dic/metrics.csv",
        "nn_whisper_within_dstc_dic": "models/nn/whisper_within_dstc_dic/metrics.csv",
        "nn_whisper_amplitude": "models/nn/whisper_amplitude/metrics.csv",
        "qnn_gold": "models/qnn/gold/metrics.csv",
        "qnn_whisper_all_dic": "models/qnn/whisper_all_dic/metrics.csv",
        "qnn_whisper_within_dstc_dic": "models/qnn/whisper_within_dstc_dic/metrics.csv",
        "qnn_whisper_amplitude": "models/qnn/whisper_amplitude/metrics.csv"
    }

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
        ax_test_loss.plot(epochs, df["test_loss"], label=name)
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

    # Summarize final metrics for quick comparison
    summary = {
        name: {
            "train_loss": df["train_loss"].iloc[-1],
            "test_loss": df["test_loss"].iloc[-1],
            "accuracy": df["accuracy"].iloc[-1],
        }
        for name, df in metrics.items()
    }
    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    print("Final metrics:")
    print(summary_df)
    summary_df.to_csv("quantum-cascade/metrics_summary.csv")

    plt.tight_layout()
    plt.savefig("quantum-cascade/learning_process.png")

if __name__ == "__main__":
    main()
