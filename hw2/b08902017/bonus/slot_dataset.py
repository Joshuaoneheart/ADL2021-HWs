from typing import List, Dict

import torch
from torch.utils.data import Dataset
import re


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        label_mapping: Dict[str, int],
        max_len: int,
        tokenizer,
        mode: str
    ):
        self.data = data
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.mode = mode

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        if self.mode == "train":
            return {"id": instance["id"], "text": self.tokenizer.convert_tokens_to_ids(instance["tokens"]), "intent": [self.label_mapping[j] for j in instance["tags"]]}
        else:
            return {"id": instance["id"], "text": self.tokenizer.convert_tokens_to_ids(instance["tokens"])}

    def collate_fn(self, samples):
        ids = []
        input_ids = []
        token_type_ids = []
        attention_mask = []
        intents = []
        for data in samples:
            input_id = [101] + data["text"] + [102]
            token_type_id = [0] + [0]*len(data["text"]) + [0]
            attention_mas = [1] + [1]*len(data["text"]) + [1]
            if "intent" in data:
                intent = [9] + data["intent"] + [9]
            while len(input_id) < self.max_len:
                data["text"].append(0)
                input_id.append(0)
                token_type_id.append(1)
                attention_mas.append(0)
                if "intent" in data:
                    intent.append(9)
            ids.append(data["id"])
            input_ids.append(input_id)
            token_type_ids.append(token_type_id)
            attention_mask.append(attention_mas)
            if "intent" in data:
                intents.append(intent)
        return ids, input_ids, token_type_ids, attention_mask, intents

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
