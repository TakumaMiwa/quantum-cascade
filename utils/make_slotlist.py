import datasets
from typing import Dict, List
import json

def main() -> None:
    dataset = [
         datasets.load_from_disk("one_word_dataset/traindev"),
         datasets.load_from_disk("one_word_dataset/test")
    ]
    slot_list: Dict[str, int] = {}
    for data in dataset:
        for item in data:
            for slot in item["slots"]:
                if slot not in slot_list:
                    slot_list[slot] = len(slot_list)
    
    with open("one_word_dataset/slot_list.json", "w") as f:
        json.dump(slot_list, f, indent=4)
if __name__ == "__main__":
    main()