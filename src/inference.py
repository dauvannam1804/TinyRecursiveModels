import torch
from src.model import TinyRecursiveModel
from src.config import Config
from tokenizers import Tokenizer
import os

class InferenceEngine:
    def __init__(self, checkpoint_path: str = None, config: Config = None):
        self.config = config if config else Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = TinyRecursiveModel(self.config.model)
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Warning: No checkpoint found, using random weights.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Load Tokenizer
        if os.path.exists(self.config.tokenizer_path):
            self.tokenizer = Tokenizer.from_file(self.config.tokenizer_path)
        else:
            raise FileNotFoundError(f"Tokenizer not found at {self.config.tokenizer_path}")
            
        self.pad_token_id = self.tokenizer.token_to_id("<pad>")
        self.bos_token_id = self.tokenizer.token_to_id("<s>")
        self.eos_token_id = self.tokenizer.token_to_id("</s>")

    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0) -> str:
        # Encode
        # Encode
        encoded = self.tokenizer.encode(prompt)
        ids = [self.bos_token_id] + encoded.ids
        
        # Truncate to ensure we have space for generation
        # We need to ensure len(ids) + max_new_tokens <= self.config.model.max_seq_len
        max_input_len = self.config.model.max_seq_len - max_new_tokens
        if len(ids) > max_input_len:
            print(f"Warning: Prompt too long ({len(ids)} tokens). Truncating to {max_input_len} tokens to fit max_seq_len ({self.config.model.max_seq_len}).")
            ids = ids[:max_input_len]
            
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        
        # Generation Loop
        for _ in range(max_new_tokens):
            # Prepare inputs
            seq_len = input_ids.size(1)
            attention_mask = torch.ones((1, seq_len), device=self.device) # Simple mask for now
            
            # Initialize y and z (Learnable init)
            y, z = None, None
            
            # Deep Supervision Loop (Simulate "Thinking")
            logits = None
            for step in range(self.config.model.n_supervision_steps):
                y, z, logits, q_hat = self.model(input_ids, attention_mask, y_init=y, z_init=z)
                # We only care about the last step's logits for generation
                # But we must loop to let y and z evolve
            
            # Get next token from the last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Greedy or Sampling (Here Greedy for math)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            # Stop condition
            if next_token_id.item() == self.eos_token_id:
                break
                
        # Decode
        output_ids = input_ids[0].tolist()
        decoded = self.tokenizer.decode(output_ids)
        return decoded

    def generate_tool_call(self, tools: str, query: str, max_new_tokens: int = 200) -> str:
        # Construct ChatML text with Hermes template
        # 1. System Prompt
        text = "<|im_start|>system\n"
        text += "You are a helpful assistant.\n"
        text += "# Tools\n"
        text += "You may call one or more functions to assist with the user query.\n"
        text += "You are provided with function signatures within <tools></tools> XML tags:\n"
        text += f"<tools>\n{tools}\n</tools>\n"
        text += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        text += "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
        
        # 2. User Query
        text += f"<|im_start|>user\n{query}<|im_end|>\n"
        
        # 3. Assistant Start
        text += "<|im_start|>assistant\n"
        
        # Generate
        return self.generate(text, max_new_tokens=max_new_tokens)

    def _parse_tool_call(prediction_raw: str, gt_name: str, gt_args: dict):
    """
    Parse prediction_raw để trích pred_name và pred_args
    Ưu tiên parse theo gt_name & gt_args
    """
    # -----------------------------
    # 1. Thử parse JSON hoàn chỉnh
    # -----------------------------
    json_pattern = r'\{[\s\S]*?"name"\s*:\s*"' + re.escape(gt_name) + r'"[\s\S]*?\}'
    matches = re.findall(json_pattern, prediction_raw)

    for m in matches:
        try:
            parsed = json.loads(m)
            return {
                "name": parsed.get("name", ""),
                "arguments": parsed.get("arguments", {})
            }
        except json.JSONDecodeError:
            pass

    # -----------------------------
    # 2. Fallback: heuristic parse
    # -----------------------------
    pred_name = ""
    pred_args = {}

    # 2.1 Tool name
    if gt_name in prediction_raw:
        pred_name = gt_name
    else:
        return {"name": "", "arguments": {}}

    # 2.2 Arguments (dựa vào gt_args keys)
    for arg_key, arg_val in gt_args.items():
        # Nếu GT là số → tìm số trong text
        if isinstance(arg_val, int):
            num_matches = re.findall(r'\b\d+\b', prediction_raw)
            if num_matches:
                pred_args[arg_key] = int(num_matches[-1])  # thường số cuối là đúng

        # Nếu GT là string
        elif isinstance(arg_val, str):
            str_pattern = rf'"{arg_key}"\s*:\s*"([^"]+)"'
            m = re.search(str_pattern, prediction_raw)
            if m:
                pred_args[arg_key] = m.group(1)

    return {
        "name": pred_name,
        "arguments": pred_args
    }

    def evaluate_dataset(self, data_path: str, n_samples: int = None) -> float:
        import json
        import csv
        from tqdm import tqdm
        
        if not os.path.exists(data_path):
            print(f"Dataset not found at {data_path}")
            return 0.0
            
        # Load data (JSON only)
        if data_path.endswith(".json"):
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            print("Unsupported file format. Only .json is supported.")
            return 0.0
            
        if n_samples:
            data = data[:n_samples]
            
        correct_name = 0
        correct_full = 0
        total = 0
        results = []
        
        print(f"Evaluating on {len(data)} samples...")
        for item in tqdm(data):
            # XLAM / Swift Format
            if "tools" in item and "messages" in item:
                tools = item["tools"]
                query = ""
                solution_str = ""
                
                for msg in item["messages"]:
                    if msg["role"] == "user":
                        query = msg["content"]
                    elif msg["role"] == "tool_call":
                        solution_str = msg["content"] # Keep raw content for parsing
                
                if not query or not solution_str:
                    continue
                    
                # Generate
                prediction_raw = self.generate_tool_call(tools, query)
                
                # Parse Ground Truth
                try:
                    gt_json = json.loads(solution_str)
                except json.JSONDecodeError:
                    gt_json = {}
                
                gt_name = gt_json.get("name", "")
                gt_args = gt_json.get("arguments", {})
                
                # Parse Prediction with Hint
                pred_json = self._parse_tool_call(prediction_raw, hint_name=gt_name)
                pred_name = pred_json.get("name", "")
                pred_args = pred_json.get("arguments", {})
                
                # Check Name Match
                name_match = (gt_name.lower() == pred_name.lower()) and (gt_name != "")
                if name_match:
                    correct_name += 1
                    
                # Check Full Match (Name + Args)
                # Note: Arguments comparison can be tricky (order, types). 
                # Here we do a simple dict comparison.
                full_match = name_match and (gt_args == pred_args)
                if full_match:
                    correct_full += 1
                
                total += 1
                
                results.append({
                    "query": query,
                    "gt_name": gt_name,
                    "gt_args": json.dumps(gt_args),
                    "pred_name": pred_name,
                    "pred_args": json.dumps(pred_args),
                    "name_match": name_match,
                    "full_match": full_match,
                    "raw_prediction": prediction_raw
                })
            
        acc_name = correct_name / total if total > 0 else 0
        acc_full = correct_full / total if total > 0 else 0
        
        print(f"Tool Name Accuracy: {acc_name:.2%} ({correct_name}/{total})")
        print(f"Full Accuracy: {acc_full:.2%} ({correct_full}/{total})")
        
        # Save results to CSV
        output_csv = "inference_results.csv"
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["query", "gt_name", "gt_args", "pred_name", "pred_args", "name_match", "full_match", "raw_prediction"])
            writer.writeheader()
            writer.writerows(results)
            
        print(f"Results saved to {output_csv}")
        
        return acc_full

if __name__ == "__main__":
    # Example usage
    # Assuming a checkpoint exists at checkpoints/checkpoint_epoch_0.pt (or similar)
    # You might need to adjust the path based on your training output
    checkpoint_dir = "checkpoints"
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if checkpoints:
            checkpoints.sort() # Simple sort, ideally sort by epoch number
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])
    
    engine = InferenceEngine(checkpoint_path=latest_checkpoint)
    
    prompt = "Problem: 1+1=\nSolution:"
    print(f"Prompt: {prompt}")
    result = engine.generate(prompt)
    print(f"Result: {result}")
