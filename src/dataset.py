import torch
from torch.utils.data import Dataset
import pandas as pd
from tokenizers import Tokenizer
from typing import Dict
import os
import json

class ToolCallDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_path: str, max_seq_len: int = 512):
        self.data_path = data_path
        # Only support JSON format (XLAM/Swift)
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_seq_len = max_seq_len
        
        # Special token IDs
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.eos_token_id = self.tokenizer.token_to_id("</s>")
        # ChatML tokens
        self.im_start_id = self.tokenizer.token_to_id("<|im_start|>")
        self.im_end_id = self.tokenizer.token_to_id("<|im_end|>")
        
        # Fallback if ChatML tokens are not in tokenizer
        if self.im_start_id is None:
            self.im_start_id = self.bos_token_id
        if self.im_end_id is None:
            self.im_end_id = self.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        tools_str = item['tools'] # JSON string
        messages = item['messages']
        
        # Construct ChatML text with Hermes template
        # 1. System Prompt
        text = "<|im_start|>system\n"
        text += "You are a helpful assistant.\n"
        text += "# Tools\n"
        text += "You may call one or more functions to assist with the user query.\n"
        text += "You are provided with function signatures within <tools></tools> XML tags:\n"
        text += f"<tools>\n{tools_str}\n</tools>\n"
        text += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        text += "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
        
        # 2. Messages
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'tool_call':
                # Map tool_call role to assistant role with <tool_call> tags
                role = 'assistant'
                content = f"<tool_call>\n{content}\n</tool_call>"
            
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            
        encoded = self.tokenizer.encode(text)
        # We don't need BOS/EOS if we use ChatML tags, but let's keep BOS at start for consistency with model expectation if any
        ids = [self.bos_token_id] + encoded.ids + [self.eos_token_id]
        
        return self._process_ids(ids)

    def _process_ids(self, ids):
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
