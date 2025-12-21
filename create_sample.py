import pandas as pd
import os

def create_sample_csv(input_path, output_path, n=1000):
    print(f"Reading {input_path}...")
    df = pd.read_parquet(input_path)
    
    print(f"Sampling {n} rows...")
    # Sample n rows, or less if dataframe is smaller
    sample_df = df.head(n) if len(df) > n else df
    
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
    create_sample_csv("train-00000-of-00001.parquet", "data/processed/sample_1k.csv", n=1000)
