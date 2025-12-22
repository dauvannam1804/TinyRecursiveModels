import pandas as pd
import os

def create_sample_csv(input_path, output_path, n=1000, offset=0):
    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    
    print(f"Sampling {n} rows starting from {offset}...")
    # Sample n rows starting from offset
    if offset + n > len(df):
        print(f"Warning: Requested {n} rows from offset {offset}, but only {len(df) - offset} available.")
        sample_df = df.iloc[offset:]
    else:
        sample_df = df.iloc[offset : offset + n]
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Serialize conversations to JSON string to avoid parsing issues
    import json
    # Ensure we work on a copy to avoid SettingWithCopyWarning
    sample_df = sample_df.copy()
    # Convert numpy array/list to list then to json string
    sample_df["conversations"] = sample_df["conversations"].apply(lambda x: json.dumps(x.tolist() if hasattr(x, "tolist") else x))
    
    print(f"Saving to {output_path}...")
    sample_df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    # Parquet file is in root based on list_dir output
    input_file = "train-00000-of-00001.parquet"
    
    # Generate Train (20k)
    create_sample_csv(input_file, "data/processed/train_20k.csv", n=20000, offset=0)
    
    # Generate Val (5k) - take from 20000 to 25000
    create_sample_csv(input_file, "data/processed/val_5k.csv", n=5000, offset=20000)
