import json
import os
from tqdm import tqdm

def format_xlam_data(input_path, output_path, start_idx=0, end_idx=1000):
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Slice data
    data_slice = data[start_idx:end_idx]
    
    formatted_data = []
    
    print(f"Formatting data from index {start_idx} to {end_idx}...")
    for item in tqdm(data_slice):
        # 1. Format Tools
        # XLAM tools is a JSON string representing a list of tool definitions
        tools_list = json.loads(item['tools'])
        formatted_tools = []
        for tool in tools_list:
            # Wrap in OpenAI function format
            formatted_tool = {
                "type": "function",
                "function": tool
            }
            formatted_tools.append(formatted_tool)
        
        # Convert back to JSON string as per Swift requirement
        tools_str = json.dumps(formatted_tools)
        
        # 2. Format Messages
        messages = []
        
        # User Message
        messages.append({
            "role": "user",
            "content": item['query']
        })
        
        # Tool Calls (from answers)
        # XLAM answers is a JSON string representing a list of calls
        answers_list = json.loads(item['answers'])
        for answer in answers_list:
            # Swift expects tool_call content to be a JSON string with name and arguments
            tool_call_content = json.dumps({
                "name": answer['name'],
                "arguments": answer['arguments']
            })
            
            messages.append({
                "role": "tool_call",
                "content": tool_call_content
            })
            
        # Add to formatted list
        formatted_data.append({
            "tools": tools_str,
            "messages": messages
        })
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Saved {len(formatted_data)} samples to {output_path}")

if __name__ == "__main__":
    input_file = "xlam_function_calling_60k.json"
    
    if os.path.exists(input_file):
        # Train: 0-1000
        format_xlam_data(input_file, "data/processed/xlam_1k_swift.json", start_idx=0, end_idx=1000)
        # Val: 1000-1200
        format_xlam_data(input_file, "data/processed/xlam_val_200_swift.json", start_idx=1000, end_idx=1200)
    else:
        print(f"Input file {input_file} not found.")
