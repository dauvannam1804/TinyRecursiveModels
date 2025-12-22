import torch
from torch.utils.data import Dataset
import pandas as pd
from tokenizers import Tokenizer
from typing import Dict
import os
import json

class MathDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_path: str, max_seq_len: int = 512):
        if data_path.endswith(".csv"):
            self.data = pd.read_csv(data_path)
            # Parse conversations column if it's a string representation of list
            if isinstance(self.data["conversations"].iloc[0], str):
                import json
            # Parse conversations column if it's a string representation of list
            if isinstance(self.data["conversations"].iloc[0], str):
                self.data["conversations"] = self.data["conversations"].apply(json.loads)
        else:
            self.data = pd.read_parquet(data_path)
            
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        
        # Special token IDs
        # Special token IDs (Unigram/SentencePiece standard)
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.eos_token_id = self.tokenizer.token_to_id("</s>")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.data.iloc[idx]
        conversations = row["conversations"]
        
        # Extract problem (human) and solution (gpt)
        problem = ""
        solution = ""
        
        for turn in conversations:
            if turn["from"] == "human":
                problem = turn["value"]
            elif turn["from"] == "gpt":
                solution = turn["value"]
        
        # Format: <BOS> Problem: ... \n Solution: ... <EOS>
        text = f"Problem: {problem}\nSolution: {solution}"
        
        encoded = self.tokenizer.encode(text)
        ids = [self.bos_token_id] + encoded.ids + [self.eos_token_id]
        
        # Truncate if needed
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
            
        # Padding
        padding_len = self.max_seq_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * padding_len
        ids = ids + [self.pad_token_id] * padding_len
        
        # Labels
        labels = ids.copy()
        for i in range(len(labels)):
            if attention_mask[i] == 0:
                labels[i] = -100
                
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
