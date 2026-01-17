import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import Config
from src.dataset import ToolCallDataset
from tokenizers import Tokenizer

def test_dataset():
    config = Config()
    
    # Ensure paths exist
    if not os.path.exists(config.data_path):
        print(f"Data path not found: {config.data_path}")
        return
    if not os.path.exists(config.tokenizer_path):
        print(f"Tokenizer path not found: {config.tokenizer_path}")
        return

    print(f"Loading dataset from {config.data_path}...")
    dataset = ToolCallDataset(config.data_path, config.tokenizer_path, config.model.max_seq_len)
    print(f"Dataset size: {len(dataset)}")
    
    tokenizer = Tokenizer.from_file(config.tokenizer_path)

    # Inspect a few samples
    for i in range(min(1, len(dataset))):
        print(f"\n{'='*50}")
        print(f"Sample {i}")
        print(f"{'='*50}")
        
        item = dataset[i]
        input_ids = item["input_ids"]
        labels = item["labels"]
        print("Input IDs:", input_ids)
        # Decode Input
        decoded_input = tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
        print(f"[Decoded Input]:\n{decoded_input}")
        print("Input IDs len:", len(input_ids))

        print(f"\n[Labels Token IDs (Training Target)]: {labels[:].tolist()}")

        valid_labels = labels[labels != -100]
        decoded_labels = tokenizer.decode(valid_labels.tolist(), skip_special_tokens=False)
        print(f"\n[Decoded Labels (Training Target)]:\n{decoded_labels}")
        print("Label len:", len(labels))
        
        # New GLiNER fields
        if "span_idx" in item:
            print(f"\n[Span Indices]:\n{item['span_idx']}")
            print(f"Span Indices Shape: {item['span_idx'].shape}")
            
        if "span_labels" in item:
            print(f"\n[Span Labels]:\n{item['span_labels']}")
            print(f"Span Labels Shape: {item['span_labels'].shape}")
            
        if "prompts_ids" in item:
            print(f"\n[{'='*20} Decoded Prompts (Classes) {'='*20}]")
            prompts_ids = item["prompts_ids"]
            for cls_idx, p_ids in enumerate(prompts_ids):
                # Filter padding
                valid_ids = [pid for pid in p_ids.tolist() if pid != 0]
                if not valid_ids:
                    continue # Skip empty/padded rows
                
                decoded_prompt = tokenizer.decode(valid_ids)
                print(f"  Class ID {cls_idx}: '{decoded_prompt}'")

        # Decode Spans for visualization
        if "span_idx" in item and "span_labels" in item and "prompts_ids" in item:
            print(f"\n[{'='*20} Decoded Spans {'='*20}]")
            span_idx = item["span_idx"]
            span_labels = item["span_labels"]
            prompts_ids = item["prompts_ids"]
            
            found_any = False
            for idx, (start, end) in enumerate(span_idx):
                # Filter padding (simple check)
                if span_labels[idx] == -100:
                    continue
                    
                found_any = True
                
                # Decode Span Text (Argument Value)
                # Note: span indices are inclusive in dataset.py logic
                span_token_ids = input_ids[start:end+1].tolist()
                span_text = tokenizer.decode(span_token_ids)
                
                # Decode Label Name (Argument Name / Prompt)
                label_idx = span_labels[idx]
                if label_idx < len(prompts_ids):
                    prompt_token_ids = prompts_ids[label_idx]
                    # Filter trailing zeros (padding) assuming 0 is pad
                    valid_prompt_ids = [pid for pid in prompt_token_ids.tolist() if pid != 0] # 0 might be pad or not? 
                    # Dataset uses whatever tokenizer pad id is. But generated prompts_ids uses 0 or pad_token_id?
                    # In dataset.py: `padded_prompts = torch.zeros(...)`. So 0 is pad.
                    prompt_text = tokenizer.decode(valid_prompt_ids)
                else:
                    prompt_text = "Unknown"
                
                print(f"  Span [{start}:{end}] -> '{span_text}' | Label: '{prompt_text}' (ID: {label_idx})")
            
            if not found_any:
                print("  No valid spans found in this sample.")


if __name__ == "__main__":
    test_dataset()
