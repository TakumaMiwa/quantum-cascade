from datasets import load_from_disk, Audio

def prepare_feature(
    dataset_path: str, 
        model,
        processor,
        num_qubits: int = 10,
        experiment_name: str = "amplitude",
    ) -> Dict:
    ## load dataset
    dataset = load_from_disk(dataset_path)
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
                max_new_tokens=args.max_length,
            )
        logits = torch.stack(outputs.scores, dim=1)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        batch["probs"] = probs

    ## target_token_seqsの定義
    # キャッシュから読み込む
    with open("multiple_word_dataset/tokenizer_cache/tokenizer_cache_traindev.json", "r") as f:
        target_token_seqs = json.load(f)
    
    with open(slots_dic_path, 'r') as f:
        slots_dic = json.load(f)
    def _culc_text_prob(batch: Dict) -> Dict:
        # シーケンスごとの確率を求める
        feature = np.zeros(2 * num_qubits, dtype=np.float32)
        if experiment_name == "amplitude":
            for i, ids in target_token_seqs.items():
                prob = 1.0
                for step, token_id in enumerate(ids):
                    if step >= probs.shape[1]:
                        break
                    prob *= probs[0, step, token_id].item()
                feature[i] = prob
        elif experiment_name == "1-best":
            max_value, max_index = 0, 0
            for i, ids in target_token_seqs.items():
                prob = 1.0
                for step, token_id in enumerate(ids):
                    if step >= probs.shape[1]:
                        break
                    prob *= probs[0, step, token_id].item()
                if prob > max_value:
                    max_value = prob
                    max_index = i
            feature[max_index] = 1.0
        batch["feature"] = feature

        # ラベルの取得と変換
        label = batch["slots"][0]
        batch["labels"] = slots_dic.get(label, 0)
        return batch
    dataset = dataset.map(_culc_text_prob, remove_columns=dataset.column_names, load_from_cache_file=False)
    dataset.set_format(type="torch", columns=["input_features", "labels"])
    return dataset