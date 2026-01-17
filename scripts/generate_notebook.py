import json
import os

def generate_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_cell(source, cell_type="code"):
        notebook["cells"].append({
            "cell_type": cell_type,
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
        })

    # Cell 1: Description
    add_cell([
        "# GLiNER Dataset Integration Visualization",
        "",
        "This notebook breaks down the dataset processing logic for GLiNER integration step-by-step.",
        "We use a specific example provided to visualize how `tools`, `messages` are converted into `input_ids`, `span_idx`, and `labels`."
    ], cell_type="markdown")

    # Cell 2: Imports & Setup
    add_cell("""import sys
import os
import torch
import json
from tokenizers import Tokenizer

# Fix paths: Notebook is in 'notebooks/', strictly move to project root
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
    
# Add project root to path (current dir after chdir)
sys.path.append(os.getcwd())

from src.config import Config
config = Config()

# Verify paths
print(f"Current Working Directory: {os.getcwd()}")
print(f"Tokenizer Path: {config.tokenizer_path}")

try:
    tokenizer = Tokenizer.from_file(config.tokenizer_path)
    print("Config and Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    # Fallback/Debug info
    print(f"Files in data/processed: {os.listdir('data/processed') if os.path.exists('data/processed') else 'Not Found'}")
""")

    # Cell 3: Raw Data
    add_cell("""# The sample item provided by the user
item = {
  "tools": "[{\\\"type\\\": \\\"function\\\", \\\"function\\\": {\\\"name\\\": \\\"peers\\\", \\\"description\\\": \\\"Retrieves a list of company peers given a stock symbol.\\\", \\\"parameters\\\": {\\\"symbol\\\": {\\\"description\\\": \\\"The stock symbol for the company.\\\", \\\"type\\\": \\\"str\\\", \\\"default\\\": \\\"\\\"}}}}, {\\\"type\\\": \\\"function\\\", \\\"function\\\": {\\\"name\\\": \\\"web_chain_details\\\", \\\"description\\\": \\\"python\\\", \\\"parameters\\\": {\\\"chain_slug\\\": {\\\"description\\\": \\\"The slug identifier for the blockchain (e.g., 'ethereum' for Ethereum mainnet).\\\", \\\"type\\\": \\\"str\\\", \\\"default\\\": \\\"ethereum\\\"}}}}]",
  "messages": [
    {"role": "user", "content": "I need to understand the details of the Ethereum blockchain for my cryptocurrency project. Can you fetch the details for 'ethereum'?"},
    {"role": "tool_call", "content": "{\\\"name\\\": \\\"web_chain_details\\\", \\\"arguments\\\": {\\\"chain_slug\\\": \\\"ethereum\\\"}}"}
  ]
}

tools_str = item['tools']
messages = item['messages']

print("Loaded sample item.")""")

    # Cell 4: Step 1 - Parsing Prompts
    add_cell("""# Step 1: Parsing Prompts (Classes)
print("-" * 40 + "\\nSTEP 1: PARSING PROMPTS\\n" + "-" * 40)

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

print(f"Extracted Label Names (Classes for GLiNER): {label_names}")

prompts_ids_list = [tokenizer.encode(name).ids for name in label_names]
print("\\nEncoded Prompts (Token IDs):")
for name, p_ids in zip(label_names, prompts_ids_list):
    print(f"  '{name}' -> {p_ids}")""")

    # Cell 5: Step 2 - System Prompt
    add_cell("""# Step 2: System Prompt & Init
print("-" * 40 + "\\nSTEP 2: SYSTEM PROMPT\\n" + "-" * 40)

all_input_ids = []
all_labels = []

system_text = "<|im_start|>system "
system_text += "You are a helpful assistant. "
system_text += "# Tools "
system_text += "You may call one or more functions to assist with the user query. "
system_text += "You are provided with function signatures within <tools></tools> XML tags: "
system_text += f"<tools> {tools_str} </tools> "
system_text += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags: "
system_text += "<tool_call> {\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>} </tool_call><|im_end|> "

system_ids = tokenizer.encode(system_text).ids
all_input_ids.extend(system_ids)
all_labels.extend([-100] * len(system_ids))

print(f"System IDs Length: {len(system_ids)}")""")

    # Cell 6: Step 3 - Messages & Spans
    add_cell("""# Step 3: Process Messages and Find Spans
print("-" * 40 + "\\nSTEP 3: MESSAGES & SPANS\\n" + "-" * 40)

span_indices = []
span_label_ids = []

for msg in messages:
    role = msg['role']
    content = msg['content']
    print(f"\\nProcessing Role: {role}")
    
    if role == 'tool_call':
        # Header <|im_start|>assistant
        header_text = "<|im_start|>assistant "
        header_ids = tokenizer.encode(header_text).ids
        all_input_ids.extend(header_ids)
        all_labels.extend([-100] * len(header_ids))
        
        # Content <tool_call> ... </tool_call>
        prefix_text = "<tool_call> "
        suffix_text = " </tool_call><|im_end|> "
        full_content_text = prefix_text + content + suffix_text
        
        content_encoding = tokenizer.encode(full_content_text)
        content_ids = content_encoding.ids
        
        base_offset = len(all_input_ids)
        all_input_ids.extend(content_ids)
        all_labels.extend(content_ids) # Train on this content
        
        print(f"  > Full Content Text: {full_content_text}")
        print(f"  > Content IDs Length: {len(content_ids)}")
        
        # FIND SPANS
        tool_call_json = json.loads(content)
        if "arguments" in tool_call_json:
            args = tool_call_json["arguments"]
            for arg_name, arg_value in args.items():
                if arg_name in label_names:
                    label_idx = label_names.index(arg_name)
                    val_str = str(arg_value)
                    
                    # Search
                    search_start = len(prefix_text)
                    char_start = full_content_text.find(val_str, search_start)
                    
                    if char_start != -1:
                        char_end = char_start + len(val_str) - 1
                        print(f"    FOUND SPAN: '{val_str}' at Chars [{char_start}:{char_end}] for Label '{arg_name}'")
                        
                        token_start = content_encoding.char_to_token(char_start)
                        token_end = content_encoding.char_to_token(char_end)
                        
                        print(f"    -> MAPPED TO TOKENS (Local): [{token_start}:{token_end}]")
                        
                        if token_start is not None and token_end is not None:
                            abs_start = base_offset + token_start
                            abs_end = base_offset + token_end
                            span_indices.append([abs_start, abs_end])
                            span_label_ids.append(label_idx)
                            print(f"    -> ABSOLUTE SPAN INDICES: [{abs_start}, {abs_end}]")
    else:
        # User/System
        text = f"<|im_start|>{role} {content}<|im_end|> "
        ids = tokenizer.encode(text).ids
        all_input_ids.extend(ids)
        all_labels.extend([-100] * len(ids))""")
        
    # Cell 7: Step 4 - BOS/EOS & Shift
    add_cell("""# Step 4: Add BOS/EOS and Shift Spans
print("-" * 40 + "\\nSTEP 4: BOS/EOS & SHIFT\\n" + "-" * 40)

bos_token_id = tokenizer.token_to_id("<s>")
eos_token_id = tokenizer.token_to_id("</s>")

# Prepend BOS
all_input_ids = [bos_token_id] + all_input_ids
all_labels = [-100] + all_labels

# SHIFT SPANS BY +1
old_spans = list(span_indices)
span_indices = [[s[0]+1, s[1]+1] for s in span_indices]

print(f"Span Indices Shifted (+1 for BOS):")
for old, new in zip(old_spans, span_indices):
    print(f"  {old} -> {new}")

# Append EOS
all_input_ids = all_input_ids + [eos_token_id]
all_labels = all_labels + [eos_token_id]""")

    # Cell 8: Final Padding Visualization
    add_cell("""# Step 5: Final Tensors (Padding)
print("-" * 40 + "\\nSTEP 5: FINAL OUTPUT\\n" + "-" * 40)

MAX_SEQ_LEN = config.model.max_seq_len
print(f"Using Max Sequence Length from Config: {MAX_SEQ_LEN}")

ids = all_input_ids
labels = all_labels

# Pad
padding_len = MAX_SEQ_LEN - len(ids)
if padding_len > 0:
    ids = ids + [tokenizer.token_to_id("<pad>")] * padding_len
    labels = labels + [-100] * padding_len
else:
    # Truncate if needed
    ids = ids[:MAX_SEQ_LEN]
    labels = labels[:MAX_SEQ_LEN]

input_tensor = torch.tensor(ids)
labels_tensor = torch.tensor(labels)

print(f"Final Input IDs Shape: {input_tensor.shape}")
print(f"Final Labels Shape: {labels_tensor.shape}")
print(f"Final Spans: {span_indices}")
print(f"Final Span Labels: {span_label_ids}")""")

    output_path = os.path.join(os.path.dirname(__file__), "../notebooks/dataset_visualization.ipynb")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Notebook generated at: {output_path}")

if __name__ == "__main__":
    generate_notebook()
