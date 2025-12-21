import pandas as pd
import os

def create_samples(input_path, output_dir):
    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Define splits
    train_size = 20000
    val_size = 5000
    
    # Ensure we have enough data
    if len(df) < train_size + val_size:
        print(f"Warning: Not enough data for requested split. Using available data.")
        train_df = df.iloc[:int(len(df)*0.8)]
        val_df = df.iloc[int(len(df)*0.8):]
    else:
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:train_size+val_size]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Helper to save
    def save_split(df, name):
        path = os.path.join(output_dir, name)
        print(f"Saving {len(df)} rows to {path}...")
        # Serialize conversations
        import json
        df = df.copy()
        df["conversations"] = df["conversations"].apply(lambda x: json.dumps(x.tolist() if hasattr(x, "tolist") else x))
        df.to_csv(path, index=False)
        
    save_split(train_df, "train_20k.csv")
    save_split(val_df, "val_5k.csv")
    print("Done.")

if __name__ == "__main__":
    # Parquet file is in root based on list_dir output
    create_samples("train-00000-of-00001.parquet", "data/processed")
