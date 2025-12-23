import os
import json
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers
from src.config import Config

def train_unigram_tokenizer(data_path: str, save_path: str, vocab_size: int = 32000):
    print(f"Loading data from {data_path}...")
    
    # Load data based on extension
    if data_path.endswith(".json"):
        with open(data_path, 'r') as f:
            data = json.load(f)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        # Legacy support or conversion if needed, but primarily for JSON now
        if "conversations" in df.columns and isinstance(df["conversations"].iloc[0], str):
             df["conversations"] = df["conversations"].apply(json.loads)
        data = df.to_dict('records')
    else:
        # Fallback for parquet or other
        df = pd.read_parquet(data_path)
        data = df.to_dict('records')
    
    # Prepare iterator for tokenizer training
    def batch_iterator(batch_size=1000):
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            texts = []
            for item in batch:
                # Extract text from XLAM/Swift format
                if "tools" in item and "messages" in item:
                    # 1. Tools definition
                    texts.append(item["tools"])
                    
                    # 2. Messages content
                    for msg in item["messages"]:
                        if "content" in msg:
                            texts.append(msg["content"])
                            
                # Fallback for legacy format (conversations)
                elif "conversations" in item:
                    for turn in item["conversations"]:
                        if "value" in turn:
                            texts.append(turn["value"])
                            
            yield texts

    print("Initializing Unigram tokenizer...")
    tokenizer = Tokenizer(models.Unigram())
    
    # Normalization (NFKC is standard for SentencePiece)
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ])
    
    # Pre-tokenization (Metaspace is standard for SentencePiece)
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size, 
        special_tokens=[
            # 1. Standard
            "<pad>", "<s>", "</s>", "<unk>",
            # 2. ChatML
            "<|im_start|>", "<|im_end|>",
            # 3. Tool Calling
            "<tools>", "</tools>", 
            "<tool_call>", "</tool_call>", 
            # # 4. Thinking
            # "<think>", "</think>"
        ],
        unk_token="<unk>",
        show_progress=True
    )

    print("Training Unigram tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Post-processing
    tokenizer.decoder = decoders.Metaspace()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Unigram Tokenizer saved to {save_path}")
    return tokenizer

def train_bpe_tokenizer(data_path: str, save_path: str, vocab_size: int = 32000):
    print(f"Loading data from {data_path}...")
    
    # Load data based on extension
    if data_path.endswith(".json"):
        with open(data_path, 'r') as f:
            data = json.load(f)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        if "conversations" in df.columns and isinstance(df["conversations"].iloc[0], str):
             df["conversations"] = df["conversations"].apply(json.loads)
        data = df.to_dict('records')
    else:
        df = pd.read_parquet(data_path)
        data = df.to_dict('records')
    
    def batch_iterator(batch_size=1000):
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            texts = []
            for item in batch:
                if "tools" in item and "messages" in item:
                    texts.append(item["tools"])
                    for msg in item["messages"]:
                        if "content" in msg:
                            texts.append(msg["content"])
                elif "conversations" in item:
                    for turn in item["conversations"]:
                        if "value" in turn:
                            texts.append(turn["value"])
            yield texts

    print("Initializing BPE tokenizer (SentencePiece-like)...")
    tokenizer = Tokenizer(models.BPE())
    
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=[
            # 1. Standard
            "<pad>", "<s>", "</s>", "<unk>",
            # 2. ChatML
            "<|im_start|>", "<|im_end|>",
            # 3. Tool Calling
            "<tools>", "</tools>", 
            "<tool_call>", "</tool_call>", 
            # # 4. Thinking
            # "<think>", "</think>"
        ],
        show_progress=True
    )

    print("Training BPE tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    tokenizer.decoder = decoders.Metaspace()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"BPE Tokenizer saved to {save_path}")
    return tokenizer

def train_wordlevel_tokenizer(data_path: str, save_path: str, vocab_size: int = 32000):
    print(f"Loading data from {data_path}...")
    
    # Load data based on extension
    if data_path.endswith(".json"):
        with open(data_path, 'r') as f:
            data = json.load(f)
    elif data_path.endswith(".csv"):
        df = pd.read_csv(data_path)
        if "conversations" in df.columns and isinstance(df["conversations"].iloc[0], str):
             df["conversations"] = df["conversations"].apply(json.loads)
        data = df.to_dict('records')
    else:
        df = pd.read_parquet(data_path)
        data = df.to_dict('records')
    
    def batch_iterator(batch_size=1000):
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            texts = []
            for item in batch:
                if "tools" in item and "messages" in item:
                    texts.append(item["tools"])
                    for msg in item["messages"]:
                        if "content" in msg:
                            texts.append(msg["content"])
                elif "conversations" in item:
                    for turn in item["conversations"]:
                        if "value" in turn:
                            texts.append(turn["value"])
            yield texts

    print("Initializing WordLevel tokenizer...")
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ])
    
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    
    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size, 
        special_tokens=[
            # 1. Standard
            "<pad>", "<s>", "</s>", "<unk>",
            # 2. ChatML
            "<|im_start|>", "<|im_end|>",
            # 3. Tool Calling
            "<tools>", "</tools>", 
            "<tool_call>", "</tool_call>", 
            # # 4. Thinking
            # "<think>", "</think>"
        ],
        show_progress=True
    )

    print("Training WordLevel tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    tokenizer.decoder = decoders.Metaspace()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"WordLevel Tokenizer saved to {save_path}")
    return tokenizer

if __name__ == "__main__":
    import argparse

    config = Config()
    
    parser = argparse.ArgumentParser(description="Train a tokenizer (Unigram, BPE, or WordLevel)")
    parser.add_argument("--type", type=str, default="unigram", choices=["unigram", "bpe", "wordlevel"], help="Type of tokenizer to train")
    parser.add_argument("--vocab_size", type=int, default=config.model.vocab_size, help="Vocabulary size")
    args = parser.parse_args()

    # Ensure data path is correct relative to where we run
    if not os.path.exists(config.data_path):
        # Fallback if running from root and config has relative path
        if os.path.exists(os.path.join("TinyRecursiveModels", config.data_path)):
             config.data_path = os.path.join("TinyRecursiveModels", config.data_path)
        else:
             print(f"Warning: Data file not found at {config.data_path}")

    vocab_size = args.vocab_size if args.vocab_size else config.model.vocab_size

    if args.type == "unigram":
        train_unigram_tokenizer(config.data_path, config.tokenizer_path, vocab_size)
    elif args.type == "bpe":
        train_bpe_tokenizer(config.data_path, config.tokenizer_path, vocab_size)
    else:
        train_wordlevel_tokenizer(config.data_path, config.tokenizer_path, vocab_size)
