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
        
        all_input_ids = []
        all_labels = []
        
        # 1. System Prompt
        system_text = "<|im_start|>system\n"
        system_text += "You are a helpful assistant.\n"
        system_text += "# Tools\n"
        system_text += "You may call one or more functions to assist with the user query.\n"
        system_text += "You are provided with function signatures within <tools></tools> XML tags:\n"
        system_text += f"<tools>\n{tools_str}\n</tools>\n"
        system_text += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        system_text += "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
        
        system_ids = self.tokenizer.encode(system_text).ids
        all_input_ids.extend(system_ids)
        all_labels.extend([-100] * len(system_ids))
        
        # 2. Messages
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'tool_call':
                # Map tool_call role to assistant role with <tool_call> tags
                # Split into header and content for masking
                
                # Header: <|im_start|>assistant\n
                header_text = "<|im_start|>assistant\n"
                header_ids = self.tokenizer.encode(header_text).ids
                all_input_ids.extend(header_ids)
                all_labels.extend([-100] * len(header_ids))
                
                # Content: <tool_call>...<|im_end|>\n
                content_text = f"<tool_call>\n{content}\n</tool_call><|im_end|>\n"
                content_ids = self.tokenizer.encode(content_text).ids
                all_input_ids.extend(content_ids)
                all_labels.extend(content_ids) # Train on this
                
            else:
                # User or other roles: Mask everything
                text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
                ids = self.tokenizer.encode(text).ids
                all_input_ids.extend(ids)
                all_labels.extend([-100] * len(ids))
            
        # Add BOS at start and EOS at end
        # BOS
        all_input_ids = [self.bos_token_id] + all_input_ids
        all_labels = [-100] + all_labels
        
        # EOS
        all_input_ids = all_input_ids + [self.eos_token_id]
        all_labels = all_labels + [self.eos_token_id] # Train on EOS
        
        return self._process_ids(all_input_ids, all_labels)

    def _process_ids(self, ids, labels):
        # Truncate if needed
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            
        # Padding
        padding_len = self.max_seq_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * padding_len
        
        ids = ids + [self.pad_token_id] * padding_len
        labels = labels + [-100] * padding_len # Pad with -100
        
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
