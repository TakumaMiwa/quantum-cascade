from transformers import WhisperProcessor
import json

def main():
    # Whisperモデルのプロセッサをロード（例：small）
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    tokenizer = processor.tokenizer

    # トークンIDと対応するトークンの一覧を格納
    id_dic = {}
    word_dic = {}
    for token_id in range(tokenizer.vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id)
        id_dic[token_id] = token.lower()
        word_dic[token.lower()] = token_id
    with open("one_word_dataset/whisper_idlist.json", "w") as f:
        json.dump(id_dic, f, indent=4)
    with open("one_word_dataset/whisper_wordlist.json", "w") as f:
        json.dump(word_dic, f, indent=4)
if __name__ == "__main__":
    main()
