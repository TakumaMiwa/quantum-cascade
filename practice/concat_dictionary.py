import json
def main():
    with open("quantum-cascade/whisper_wordlist.json", "r") as f:
        word_dic_whisper = json.load(f)
    with open("dstc2_asr_cache/dstc2_wordlist.json", "r") as f:
        word_dic_dstc2 = json.load(f)

    ## Check if all words in dstc2 are present in whisper wordlist
    ## words not found in whisper wordlist will be removed from dstc2 wordlist
    unk_words = set()
    for word in word_dic_dstc2:
        if "\u0120"+word not in word_dic_whisper and word not in word_dic_whisper and "\u0121" + word not in word_dic_whisper:
            unk_words.add(word)
            print(f"Word '{word}' from dstc2 not found in whisper wordlist.")
    for word in unk_words:
        del word_dic_dstc2[word]
    with open("dstc2_asr_cache/dstc2_wordlist_clean.json", "w") as f:
        json.dump(word_dic_dstc2, f, indent=4)
    
    ## make correspendence dictionary
    
    cor_dic = {}
    for word in word_dic_dstc2:
        if "\u0120"+word in word_dic_whisper:
            cor_dic[word_dic_dstc2[word]] = word_dic_whisper["\u0120"+word]
        elif word in word_dic_whisper:
            cor_dic[word_dic_dstc2[word]] = word_dic_whisper[word]
        elif "\u0121" + word in word_dic_whisper:
            cor_dic[word_dic_dstc2[word]] = word_dic_whisper["\u0121" + word]
    with open("quantum-cascade/correspondence_table.json", "w") as f:
        json.dump(cor_dic, f, indent=4)
if __name__ == "__main__": 
    main()