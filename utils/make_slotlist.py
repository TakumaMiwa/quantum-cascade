import datasets
from typing import Dict, List
import json

def main() -> None:
    dataset = datasets.load_from_disk("one_word_dataset/traindev")
    slot_list: Dict[str, int] = {}
    for data in dataset:
        for slot in data["slots"]:
            key, value = slot.split("=")
            if key == "food" and value not in slot_list:
                    slot_list[value] = len(slot_list)
    
    with open("one_word_dataset/slot_list.json", "w") as f:
        json.dump(slot_list, f, indent=4)
if __name__ == "__main__":
    main()