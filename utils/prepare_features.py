from typing import Dict, List
import datasets
import torch
import json
import pennylane as qml
from pennylane import numpy as np
def prepare_features(
            dataset_path: str, 
            model,
            processor,
            local: bool = True,
            split: str = "train",
            cache_dir: str = "cache",
            cor_table_path: str = "quantum-cascade/correspondence_table.json",
            slots_dic_path: str = "one_word_dataset/slot_list.json",
            n_best: int = 5,
            num_qubits: int = 10,
        ) -> Dict:
    ## load dataset
    if not local:
        dataset = datasets.load_dataset(
            dataset_path,
            split=split,
            cache_dir=cache_dir,
        )
    else:
        dataset = datasets.load_from_disk(
            dataset_path,
        )
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
                # do_sample=True,
                # top_k=50,
                # temperature=1.2,
                # num_return_sequences=5,
                return_dict_in_generate=True,
                output_scores=True
            )
        sequences = outputs.sequences           # shape: [n_seq, seq_len]
        logits_per_step = outputs.scores       # list of tensors [n_seq, vocab_size] per step
        logits_per_step = torch.nn.functional.softmax(torch.stack(logits_per_step, dim=1), dim=-1)

        # シーケンスごとの確率を求める
        sequence_probs = []

        # for i, seq in enumerate(sequences):
        #     log_prob = 0.0
        #     for t, logits in enumerate(logits_per_step[:len(seq)-1]):  # [:-1]でeos分を除く
        #         token_id = seq[t + 1]  # decoderの t+1 ステップ目で出力されたトークン
        #         token_log_prob = torch.nn.functional.log_softmax(logits[i], dim=-1)[token_id]
        #         log_prob += token_log_prob
        #     prob = torch.exp(log_prob)  # log-prob → prob に変換
        #     sequence_probs.append(prob.item())
        batch["logits"] = logits_per_step  # [n_seq, seq_len, vocab_size]
        return batch

    
    with open(cor_table_path, 'r') as f:
        cor_table = json.load(f)
    # cor_table = {v: k for k, v in cor_table.items()}
    dataset = dataset.map(_generate)
    with open(slots_dic_path, 'r') as f:
        slots_dic = json.load(f)
    def _preprocess(batch: Dict) -> Dict:
        ## convert logits to dstc2 format
        logits = batch["logits"]
        features = np.zeros(2**num_qubits, dtype=np.float32)
        max_key = 0
        max_value = 0.0
        # if max_key in cor_table:
        #     features[cor_table[max_key]] = 1.0
        # else:
        #     features[0] = 1.0
        for key, value in cor_table.items():
            if max_value < logits[0][-1][value]:
                max_value = logits[0][-1][value]
                max_key = int(key)
        features[max_key] = 1.0
        
    
        batch["input_features"] = features
    
        label = batch["slots"][0]
        batch["labels"] = slots_dic.get(label, 0)  # デフォルト値は0
        return batch
    dataset = dataset.map(_preprocess, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["input_features", "labels"])
    return dataset
        

        

