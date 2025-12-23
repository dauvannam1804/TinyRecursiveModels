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


if __name__ == "__main__":
    test_dataset()
