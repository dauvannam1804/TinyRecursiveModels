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

    print("Initializing tokenizer (Unigram)...")
    tokenizer = Tokenizer(models.Unigram())
    
    # Normalization (NFKC is standard for SentencePiece)
    from tokenizers import normalizers
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ])
    
    # Pre-tokenization (Metaspace is standard for SentencePiece)
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
    
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size, 
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
        unk_token="<unk>",
        show_progress=True
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Post-processing
    # Unigram with Metaspace usually doesn't need ByteLevel post-processor
    # But we need a decoder
    tokenizer.decoder = decoders.Metaspace()
    
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
