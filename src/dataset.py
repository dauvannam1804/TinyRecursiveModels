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
        system_text = "<|im_start|>system "
        system_text += "You are a helpful assistant. "
        system_text += "# Tools "
        system_text += "You may call one or more functions to assist with the user query. "
        system_text += "You are provided with function signatures within <tools></tools> XML tags: "
        system_text += f"<tools> {tools_str} </tools> "
        system_text += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags: "
        system_text += "<tool_call> {\"name\": <function-name>, \"arguments\": <args-json-object>} </tool_call><|im_end|> "
        
        system_ids = self.tokenizer.encode(system_text).ids
        all_input_ids.extend(system_ids)
        all_labels.extend([-100] * len(system_ids))
        
        # 2. Extract Labels (Prompts) from Tools
        import json
        try:
            tools_list = json.loads(tools_str)
            label_names = []
            for t in tools_list:
                # Support multiple formats if needed, assuming standard format per sample
                if "function" in t:
                    func = t["function"]
                    if "parameters" in func and "properties" in func["parameters"]: # Standard JSON schema
                         label_names.extend(func["parameters"]["properties"].keys())
                    elif "parameters" in func: # Simplified schema from sample
                         label_names.extend(func["parameters"].keys())
            # Remove duplicates, keep order
            label_names = list(dict.fromkeys(label_names))
        except:
            label_names = ["unknown"]
            
        # Tokenize prompts (Label names)
        # We will pad/truncate this in collator or here? 
        # For simplicity, let's keep it simple: List of IDs.
        # Trainer will handle batching of prompts (or we make fixed size).
        # Let's pad prompts to a fixed length or handle in collator.
        # But wait, num_classes is fixed in config? Config claims 25.
        # Let's truncate/pad label_names to num_classes=25? 
        # Or better, this is per-sample dynamic classes?
        # GLiNER usually supports dynamic.
        # Let's return the raw IDs and let Trainer/Collator handle.
        # IMPORTANT: For now, assuming batch_size=1 or wepad dynamically.
        # Let's just tokenize them.
        prompts_ids_list = [self.tokenizer.encode(name).ids for name in label_names]
        
        span_indices = []
        span_label_ids = []
        
        # 2. Messages
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'tool_call':
                # Map tool_call role to assistant role with <tool_call> tags
                # Header: <|im_start|>assistant 
                header_text = "<|im_start|>assistant "
                header_ids = self.tokenizer.encode(header_text).ids
                all_input_ids.extend(header_ids)
                all_labels.extend([-100] * len(header_ids))
                
                # Content: <tool_call> {JSON} </tool_call><|im_end|> 
                # We need to find spans inside THIS content.
                prefix_text = "<tool_call> "
                suffix_text = " </tool_call><|im_end|> "
                
                # We tokenize the middle content (JSON) separately to match?
                # Or just tokenize the whole string.
                full_content_text = prefix_text + content + suffix_text
                
                # Get Encoding object to map chars to tokens
                content_encoding = self.tokenizer.encode(full_content_text)
                content_ids = content_encoding.ids
                
                # Offset for spans: Current length of input_ids
                base_offset = len(all_input_ids)
                
                all_input_ids.extend(content_ids)
                all_labels.extend(content_ids)
                
                # Parse JSON content to find values and their spans
                try:
                    tool_call_json = json.loads(content)
                    if "arguments" in tool_call_json:
                        args = tool_call_json["arguments"]
                        for arg_name, arg_value in args.items():
                            if arg_name in label_names:
                                label_idx = label_names.index(arg_name)
                                val_str = str(arg_value)
                                
                                # Use robust character finding in the full text
                                # Note: full_content_text contains the value.
                                # BE CAREFUL: find() matches first occurrence.
                                # If there are duplicates (e.g. 789012 appears twice), this might be wrong.
                                # However, constructing regex or smarter find is safer.
                                # For now, let's find the value searching from the beginning of JSON part.
                                # The JSON part starts after prefix_text.
                                
                                search_start = len(prefix_text)
                                char_start = full_content_text.find(val_str, search_start)
                                
                                if char_start != -1:
                                    char_end = char_start + len(val_str) - 1
                                    
                                    # Map to token indices
                                    # char_to_token returns None if char is in special token or padding?
                                    # or if it doesn't align? Usually aligns.
                                    token_start = content_encoding.char_to_token(char_start)
                                    token_end = content_encoding.char_to_token(char_end)
                                    
                                    # Look out for None (if char is part of a token but not start? No, char_to_token maps any char to its token)
                                    # But sometimes split tokens...
                                    # Try to expand range if None? 
                                    # If token_start is None, maybe it's whitespace?
                                    
                                    if token_start is not None and token_end is not None:
                                        # Adjust to global input_ids
                                        abs_start = base_offset + token_start
                                        abs_end = base_offset + token_end
                                        
                                        span_indices.append([abs_start, abs_end])
                                        span_label_ids.append(label_idx)
                                    else:
                                        # Fallback or debug?
                                        pass

                except Exception as e:
                    # Ignore parsing errors for now
                    pass
                
            else:
                # User or other roles: Mask everything
                text = f"<|im_start|>{role} {content}<|im_end|> "
                ids = self.tokenizer.encode(text).ids
                all_input_ids.extend(ids)
                all_labels.extend([-100] * len(ids))
            
        # Add BOS at start and EOS at end
        # BOS
        all_input_ids = [self.bos_token_id] + all_input_ids
        all_labels = [-100] + all_labels
        # Shift spans by 1 due to BOS
        span_indices = [[s[0]+1, s[1]+1] for s in span_indices]
        
        # EOS
        all_input_ids = all_input_ids + [self.eos_token_id]
        all_labels = all_labels + [self.eos_token_id]
        
        return self._process_ids(all_input_ids, all_labels, span_indices, span_label_ids, prompts_ids_list)

    def _process_ids(self, ids, labels, span_indices, span_labels, prompts_ids_list):
        # Truncate if needed
        if len(ids) > self.max_seq_len:
            ids = ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            # Filter spans that are out of bounds
            valid_spans = []
            valid_labels = []
            for i, span in enumerate(span_indices):
                if span[1] < self.max_seq_len:
                    valid_spans.append(span)
                    valid_labels.append(span_labels[i])
            span_indices = valid_spans
            span_labels = valid_labels
            
        # Padding
        padding_len = self.max_seq_len - len(ids)
        attention_mask = [1] * len(ids) + [0] * padding_len
        
        ids = ids + [self.pad_token_id] * padding_len
        labels = labels + [-100] * padding_len
        
        # Pad prompts?
        # To batch prompts, we need fixed size [NumClasses, Hidden].
        # But we verify passing IDs. 
        # We need to PAD the prompts list to max_prompts and max_prompt_len?
        # Complex. For now, let's just output the first 25 classes (config based) 
        # and pad each prompt to say 10 tokens.
        
        MAX_CLASSES = 25
        MAX_PROMPT_LEN = 10
        
        padded_prompts = torch.zeros((MAX_CLASSES, MAX_PROMPT_LEN), dtype=torch.long)
        # Using pad_token_id (or 0)
        
        for i, p_ids in enumerate(prompts_ids_list[:MAX_CLASSES]):
            l = min(len(p_ids), MAX_PROMPT_LEN)
            padded_prompts[i, :l] = torch.tensor(p_ids[:l], dtype=torch.long)
            
        # Spans
        # We need to pad spans to fixed size? Or Trainer handles collation?
        # Standard DataLoader collate_fn stacks tensors.
        # If variable length, we need custom collate.
        # Let's pad spans to MAX_SPANS = 20 for simplicity?
        MAX_SPANS = 20
        padded_spans = torch.zeros((MAX_SPANS, 2), dtype=torch.long)
        padded_span_labels = torch.zeros((MAX_SPANS), dtype=torch.long) - 100 # -100 ignore
        
        for i in range(min(len(span_indices), MAX_SPANS)):
            padded_spans[i] = torch.tensor(span_indices[i], dtype=torch.long)
            padded_span_labels[i] = span_labels[i]

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "span_idx": padded_spans,
            "span_labels": padded_span_labels,
            "prompts_ids": padded_prompts
        }
