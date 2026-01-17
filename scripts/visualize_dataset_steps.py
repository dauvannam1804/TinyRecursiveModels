import sys
import os
import torch
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from tokenizers import Tokenizer

def visualize_dataset_steps():
    config = Config()
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    
    # Mirroring init logic
    pad_token_id = tokenizer.token_to_id("<pad>")
    bos_token_id = tokenizer.token_to_id("<s>")
    eos_token_id = tokenizer.token_to_id("</s>")
    im_start_id = tokenizer.token_to_id("<|im_start|>")
    im_end_id = tokenizer.token_to_id("<|im_end|>")
    
    if im_start_id is None: im_start_id = bos_token_id
    if im_end_id is None: im_end_id = eos_token_id

    print("\n" + "="*80)
    print("VISUALIZING DATASET PROCESSING PIPELINE (Matches src/dataset.py)")
    print("="*80)

    # Load 1 sample
    with open(config.data_path, 'r') as f:
        data = json.load(f)
    print(f"\n[Data Source]: {config.data_path}")
    print(f"Loaded {len(data)} samples. Using sample index 0.")
    
    item = data[0]
    print("item:", item)
    tools_str = item['tools'] 
    messages = item['messages']
    
    all_input_ids = []
    all_labels = []
    
    print("\n" + "-"*40)
    print("STEP 1: SYSTEM PROMPT")
    print("-"*40)
    
    system_text = "<|im_start|>system "
    system_text += "You are a helpful assistant. "
    system_text += "# Tools "
    system_text += "You may call one or more functions to assist with the user query. "
    system_text += "You are provided with function signatures within <tools></tools> XML tags: "
    system_text += f"<tools> {tools_str} </tools> "
    system_text += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags: "
    system_text += "<tool_call> {\"name\": <function-name>, \"arguments\": <args-json-object>} </tool_call><|im_end|> "
    
    system_ids = tokenizer.encode(system_text).ids
    all_input_ids.extend(system_ids)
    all_labels.extend([-100] * len(system_ids))
    
    print(f"System Text Length: {len(system_text)} chars")
    print(f"System IDs Length: {len(system_ids)} tokens")
    print(f"Current Global Input IDs Length: {len(all_input_ids)}")

    print("\n" + "-"*40)
    print("STEP 2: EXTRACT LABELS (PROMPTS) FROM TOOLS")
    print("-"*40)
    
    try:
        tools_list = json.loads(tools_str)
        label_names = []
        for t in tools_list:
            if "function" in t:
                func = t["function"]
                if "parameters" in func and "properties" in func["parameters"]: 
                     label_names.extend(func["parameters"]["properties"].keys())
                elif "parameters" in func: 
                     label_names.extend(func["parameters"].keys())
        label_names = list(dict.fromkeys(label_names))
    except:
        label_names = ["unknown"]

    print(f"Extracted Label Names (Classes): {label_names}")
    
    prompts_ids_list = [tokenizer.encode(name).ids for name in label_names]
    print("Encoded Prompts (Token IDs):")
    for name, p_ids in zip(label_names, prompts_ids_list):
        print(f"  '{name}' -> {p_ids}")

    print("\n" + "-"*40)
    print("STEP 3: PROCESS MESSAGES & FIND SPANS")
    print("-"*40)
    
    span_indices = []
    span_label_ids = []
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        print(f"\nProcessing Message [{role.upper()}]...")
        
        if role == 'tool_call':
            # Header
            header_text = "<|im_start|>assistant "
            header_ids = tokenizer.encode(header_text).ids
            all_input_ids.extend(header_ids)
            all_labels.extend([-100] * len(header_ids))
            print(f"  Appended Header IDs ({len(header_ids)} tokens) -> Masked in Labels")
            
            # Content construction
            prefix_text = "<tool_call> "
            suffix_text = " </tool_call><|im_end|> "
            full_content_text = prefix_text + content + suffix_text
            
            # Tokenize Content
            content_encoding = tokenizer.encode(full_content_text)
            content_ids = content_encoding.ids
            
            # Base offset before adding content
            base_offset = len(all_input_ids)
            
            all_input_ids.extend(content_ids)
            all_labels.extend(content_ids) # Train on THIS
            print(f"  Appended Content IDs ({len(content_ids)} tokens) -> INCLUDED in Labels (Training Target)")
            print(f"  Current Global Input IDs Length: {len(all_input_ids)}")
            
            print(f"  > Scanning for Spans in this message...")
            
            try:
                tool_call_json = json.loads(content)
                if "arguments" in tool_call_json:
                    args = tool_call_json["arguments"]
                    for arg_name, arg_value in args.items():
                        if arg_name in label_names:
                            label_idx = label_names.index(arg_name)
                            val_str = str(arg_value)
                            
                            # Character finding
                            search_start = len(prefix_text)
                            char_start = full_content_text.find(val_str, search_start)
                            
                            if char_start != -1:
                                char_end = char_start + len(val_str) - 1
                                print(f"    Found Argument '{arg_name}' = '{val_str}' at char [{char_start}:{char_end}]")
                                
                                # Token mapping
                                token_start = content_encoding.char_to_token(char_start)
                                token_end = content_encoding.char_to_token(char_end)
                                
                                print(f"    Mapped to Local Token Indices: [{token_start}:{token_end}]")
                                
                                if token_start is not None and token_end is not None:
                                    abs_start = base_offset + token_start
                                    abs_end = base_offset + token_end
                                    
                                    span_indices.append([abs_start, abs_end])
                                    span_label_ids.append(label_idx)
                                    print(f"    -> Added Absolute Span: [{abs_start}, {abs_end}] (Label ID: {label_idx})")
                                else:
                                    print("    [WARNING] Could not map chars to tokens.")
                            else:
                                print(f"    [WARNING] Value '{val_str}' not found in content text.")
            except Exception as e:
                print(f"    [Error parsing JSON]: {e}")
                
        else:
            # User or other roles
            text = f"<|im_start|>{role} {content}<|im_end|> "
            ids = tokenizer.encode(text).ids
            all_input_ids.extend(ids)
            all_labels.extend([-100] * len(ids))
            print(f"  Appended Message IDs ({len(ids)} tokens) -> Masked in Labels")

    print("\n" + "-"*40)
    print("STEP 4: ADD BOS/EOS & SHIFT SPANS")
    print("-"*40)
    
    # LOGIC FROM DATASET.PY
    # BOS at start
    all_input_ids = [bos_token_id] + all_input_ids
    all_labels = [-100] + all_labels # Mask BOS? Probably. usually -100.
    
    print(f"Prepend BOS (ID {bos_token_id}) -> Total Length: {len(all_input_ids)}")
    
    # SHIFT SPANS
    old_spans = list(span_indices)
    span_indices = [[s[0]+1, s[1]+1] for s in span_indices]
    print(f"Shifted Spans by +1 due to BOS:")
    for old, new in zip(old_spans, span_indices):
        print(f"  {old} -> {new}")
        
    # EOS at end
    all_input_ids = all_input_ids + [eos_token_id]
    all_labels = all_labels + [eos_token_id]
    print(f"Append EOS (ID {eos_token_id}) -> Total Length: {len(all_input_ids)}")

    print("\n" + "-"*40)
    print("STEP 5: PADDING & BATCHING (_process_ids)")
    print("-"*40)
    
    # Simulating _process_ids
    max_seq_len = config.model.max_seq_len
    ids = all_input_ids
    labels = all_labels
    
    # Truncate
    if len(ids) > max_seq_len:
        print(f"Truncating from {len(ids)} to {max_seq_len}...")
        ids = ids[:max_seq_len]
        labels = labels[:max_seq_len]
        # Filter spans logic
        valid_spans = []
        valid_labels_ids = []
        for i, span in enumerate(span_indices):
            if span[1] < max_seq_len:
                valid_spans.append(span)
                valid_labels_ids.append(span_label_ids[i])
            else:
                 print(f"  Dropping out-of-bounds span: {span}")
        span_indices = valid_spans
        span_label_ids = valid_labels_ids
        
    # Padding
    padding_len = max_seq_len - len(ids)
    print(f"Padding Length: {padding_len}")
    
    ids = ids + [pad_token_id] * padding_len
    labels = labels + [-100] * padding_len
    
    # Pad Prompts & Spans
    MAX_CLASSES = 25
    MAX_PROMPT_LEN = 10
    MAX_SPANS = 20
    
    print(f"Padding Prompts to [25, 10]...")
    padded_prompts = torch.zeros((MAX_CLASSES, MAX_PROMPT_LEN), dtype=torch.long)
    for i, p_ids in enumerate(prompts_ids_list[:MAX_CLASSES]):
        l = min(len(p_ids), MAX_PROMPT_LEN)
        padded_prompts[i, :l] = torch.tensor(p_ids[:l], dtype=torch.long)
        
    print(f"Padding Spans to [20, 2]...")
    padded_spans = torch.zeros((MAX_SPANS, 2), dtype=torch.long)
    padded_span_labels = torch.zeros((MAX_SPANS), dtype=torch.long) - 100
    
    for i in range(min(len(span_indices), MAX_SPANS)):
        padded_spans[i] = torch.tensor(span_indices[i], dtype=torch.long)
        padded_span_labels[i] = span_label_ids[i]

    print("\n" + "="*80)
    print("FINAL OUTPUT TENSORS (Ready for Model)")
    print("="*80)
    print(f"input_ids shape: {torch.tensor(ids).shape}")
    print(f"labels shape: {torch.tensor(labels).shape}")
    print(f"span_idx: \n{padded_spans}")
    print(f"span_labels: \n{padded_span_labels}")
    print(f"prompts_ids shape: {padded_prompts.shape}")
    
    print("\nDone.")

if __name__ == "__main__":
    visualize_dataset_steps()
