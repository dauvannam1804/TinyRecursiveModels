import os
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from src.config import Config

def train_tokenizer(data_path: str, save_path: str, vocab_size: int = 32000):
    print(f"Loading data from {data_path}...")
    if data_path.endswith(".csv"):
        import json
        df = pd.read_csv(data_path)
        # Parse conversations if needed
        if isinstance(df["conversations"].iloc[0], str):
            df["conversations"] = df["conversations"].apply(json.loads)
    else:
        df = pd.read_parquet(data_path)
    
    # Prepare iterator for tokenizer training
    # We extract text from the 'conversations' column
    def batch_iterator(batch_size=1000):
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            texts = []
            for conversations in batch["conversations"]:
                # conversations is a numpy array of dictionaries or list of dictionaries
                for turn in conversations:
                    if "value" in turn:
                        texts.append(turn["value"])
            yield texts

    print("Initializing tokenizer (BPE)...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        show_progress=True
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Post-processing
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    return tokenizer

if __name__ == "__main__":
    config = Config()
    # Ensure data path is correct relative to where we run
    if not os.path.exists(config.data_path):
        # Fallback if running from root and config has relative path
        if os.path.exists(os.path.join("TinyRecursiveModels", config.data_path)):
             config.data_path = os.path.join("TinyRecursiveModels", config.data_path)
        else:
             # Try root
             if os.path.exists("train-00000-of-00001.parquet"):
                 config.data_path = "train-00000-of-00001.parquet"
             else:
                 print(f"Warning: Data file not found at {config.data_path}")

    train_tokenizer(config.data_path, config.tokenizer_path, config.model.vocab_size)
