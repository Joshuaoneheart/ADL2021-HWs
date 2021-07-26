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
            return {"id": instance["id"], "text": self.tokenizer(instance["text"], add_special_tokens=False), "intent": self.label_mapping[instance["intent"]]}
        else:
            return {"id": instance["id"], "text": self.tokenizer(instance["text"], add_special_tokens=False)}

    def collate_fn(self, samples):
        ids = []
        input_ids = []
        token_type_ids = []
        attention_mask = []
        intents = []
        for data in samples:
            data["text"]["input_ids"] = [101] + data["text"]["input_ids"] + [102]
            data["text"]["token_type_ids"] = [0] + data["text"]["token_type_ids"] + [0]
            data["text"]["attention_mask"] = [1] + data["text"]["attention_mask"] + [1]
            while len(data["text"]["input_ids"]) < self.max_len:
                data["text"]["input_ids"].append(0)
                data["text"]["token_type_ids"].append(0)
                data["text"]["attention_mask"].append(0)
            ids.append(data["id"])
            input_ids.append(data["text"]["input_ids"])
            token_type_ids.append(data["text"]["token_type_ids"])
            attention_mask.append(data["text"]["attention_mask"])
            if "intent" in data:
                intents.append(data["intent"])
        return ids, input_ids, token_type_ids, attention_mask, intents

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
