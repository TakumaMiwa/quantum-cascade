from datasets import load_from_disk, Audio
from typing import Dict
import torch
import datasets
import json
import numpy as np
def prepare_feature(
    dataset_path: str, 
        model,
        processor,
        num_qubits: int = 10,
        experiment_name: str = "amplitude",
        max_length: int = 10
    ) -> Dict:

    
    ## load dataset
    dataset = load_from_disk(dataset_path)

    """
    experiment_name=="gold"の場合の処理を追加してください．
    この時datasetの"transcript"をそのまま使い，slotを予測します．
    入力されたテキストをトークナイザでエンコーディングし，target_token_seqsを用いてone-hotベクトルを生成します．
    ラベルの取得は他と同様に行います．
    """
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

    def _generate(batch: Dict) -> Dict:
        audio = batch["audio"]
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features
        input_features = torch.tensor(input_features).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                input_features=input_features,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_length,
            )
        logits = torch.stack(outputs.scores, dim=1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        batch["probs"] = probs
        return batch

    if experiment_name != "gold":
        dataset = dataset.map(
            _generate,
            load_from_cache_file=False,
        )
    ## target_token_seqsの定義
    # キャッシュから読み込む
    with open("multiple_word_dataset/tokenizer_cache/tokenizer_cache_traindev.json", "r") as f:
        target_token_seqs = json.load(f)
    target_token_seqs = {int(k): v for k, v in target_token_seqs.items()}
    with open("multiple_word_dataset/dictionary/slot_list.json", 'r') as f:
        slots_dic = json.load(f)
    def _culc_text_prob(batch: Dict) -> Dict:
        # シーケンスごとの確率を求めるまたはゴールドテキストから特徴量を生成する
        feature = np.zeros(2 ** num_qubits, dtype=np.float32)
        if experiment_name in ["amplitude", "1-best"]:
            probs = batch["probs"]
            if experiment_name == "amplitude":
                for i, ids in target_token_seqs.items():
                    prob = 1.0
                    for step, token_id in enumerate(ids):
                        if step >= len(probs[0]):
                            break
                        prob *= probs[0][step][token_id]
                    feature[i] = prob
            else:  # 1-best
                max_value, max_index = 0, 0
                for i, ids in target_token_seqs.items():
                    prob = 1.0
                    for step, token_id in enumerate(ids):
                        if step >= len(probs[0]):
                            break
                        prob *= probs[0][step][token_id]
                    if prob > max_value:
                        max_value = prob
                        max_index = i
                feature[max_index] = 1.0
        elif experiment_name == "gold":
            token_ids = processor.tokenizer.encode(
                batch["transcript"], add_special_tokens=False,
                max_length=max_length, truncation=True
            )
            if len(token_ids) < max_length:
                token_ids += [processor.tokenizer.pad_token_id] * (max_length - len(token_ids))
            found = False
            for i, ids in target_token_seqs.items():
                if token_ids[:len(ids)] == ids:
                    feature[i] = 1.0
                    found = True
                    break
            if not found:
                feature[0] = 1.0
        else:
            raise ValueError(f"Unknown experiment_name: {experiment_name}")
        # When all probabilities are zero, the feature vector becomes all zeros
        # which results in NaN during AmplitudeEmbedding normalization. Avoid
        # this by setting a small value to the first index so that the norm is
        # non-zero and the vector can be normalized safely.
        if sum(feature) == 0:
            feature[0] = 1.0
        # Normalize the feature vector
        feature = feature / np.sum(feature)
        batch["input_features"] = feature

        # ラベルの取得と変換
        label = batch["slots"][0]
        batch["labels"] = slots_dic.get(label, 0)
        return batch
    dataset = dataset.map(_culc_text_prob, remove_columns=dataset.column_names, load_from_cache_file=False)
    dataset.set_format(type="torch", columns=["input_features", "labels"])
    return dataset