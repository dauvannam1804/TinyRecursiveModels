import json
import os

def filter_dataset(input_file, output_file):
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    print(f"Total samples: {len(data)}")

    filtered_data = []
    for sample in data:
        answers_raw = sample.get('answers')
        
        if answers_raw is None:
            continue

        try:
            # Check if answers is already a list or a string that needs parsing
            if isinstance(answers_raw, list):
                answers = answers_raw
            elif isinstance(answers_raw, str):
                answers = json.loads(answers_raw)
            else:
                print(f"Warning: Unexpected type for answers in sample {sample.get('id')}: {type(answers_raw)}")
                continue

            if isinstance(answers, list) and len(answers) == 1:
                filtered_data.append(sample)
        except json.JSONDecodeError:
            print(f"Error decoding JSON for sample {sample.get('id')}")
            continue

    print(f"Filtered samples: {len(filtered_data)}")
    
    print(f"Saving filtered dataset to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    input_path = "/home/namdv/workspace/TinyRecursiveModels/data/xlam_function_calling_60k.json"
    output_path = "/home/namdv/workspace/TinyRecursiveModels/data/xlam_function_calling_60k_filtered.json"
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
    else:
        filter_dataset(input_path, output_path)
