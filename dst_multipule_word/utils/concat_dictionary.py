import json
def main():
    with open("multiple_word_dataset/dictionary/whisper_wordlist.json", "r") as f:
        word_dic_whisper = json.load(f)
    with open("multiple_word_dataset/dictionary/dstc2_wordlist.json", "r") as f:
        word_dic_dstc2 = json.load(f)

    
    ## make correspendence dictionary
    
    cor_dic = {}
    for word in word_dic_dstc2:
        if "\u0120"+word in word_dic_whisper:
            cor_dic[word_dic_dstc2[word]] = word_dic_whisper["\u0120"+word]
        elif word in word_dic_whisper:
            cor_dic[word_dic_dstc2[word]] = word_dic_whisper[word]
        elif "\u0121" + word in word_dic_whisper:
            cor_dic[word_dic_dstc2[word]] = word_dic_whisper["\u0121" + word]
    with open("multiple_word_dataset/dictionary/correspondence_table.json", "w") as f:
        json.dump(cor_dic, f, indent=4)
if __name__ == "__main__": 
    main()