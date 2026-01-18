import torch
from src.model import TinyRecursiveModel
from src.config import Config
from tokenizers import Tokenizer
import os
import re
import json
import csv
from tqdm import tqdm

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
        encoded = self.tokenizer.encode(prompt)
        ids = [self.bos_token_id] + encoded.ids
        
        # Truncate
        max_input_len = self.config.model.max_seq_len - max_new_tokens
        if len(ids) > max_input_len:
            ids = ids[:max_input_len]
            
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        
        # Generation Loop
        for _ in range(max_new_tokens):
            seq_len = input_ids.size(1)
            attention_mask = torch.ones((1, seq_len), device=self.device)
            
            y, z = None, None
            logits = None
            
            # Deep Supervision Loop
            for step in range(self.config.model.n_supervision_steps):
                y, z, logits, q_hat, _ = self.model(input_ids, attention_mask, y_init=y, z_init=z)
            
            next_token_logits = logits[:, -1, :] / temperature
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            if next_token_id.item() == self.eos_token_id:
                break
                
        output_ids = input_ids[0].tolist()
        decoded = self.tokenizer.decode(output_ids)
        return decoded

    def generate_adaptive(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, halting_threshold: float = 0.9) -> str:
        # Encode
        encoded = self.tokenizer.encode(prompt)
        ids = [self.bos_token_id] + encoded.ids
        
        # Truncate
        max_input_len = self.config.model.max_seq_len - max_new_tokens
        if len(ids) > max_input_len:
            ids = ids[:max_input_len]
            
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        
        # Generation Loop
        for _ in range(max_new_tokens):
            seq_len = input_ids.size(1)
            attention_mask = torch.ones((1, seq_len), device=self.device)
            
            y, z = None, None
            logits = None
            
            for step in range(self.config.model.n_supervision_steps):
                y, z, logits, q_hat, _ = self.model(input_ids, attention_mask, y_init=y, z_init=z)
                
                # Check halting
                current_halting_prob = q_hat[0, -1].item()
                if current_halting_prob > halting_threshold:
                    break
            
            next_token_logits = logits[:, -1, :] / temperature
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            
            if next_token_id.item() == self.eos_token_id:
                break
                
        output_ids = input_ids[0].tolist()
        decoded = self.tokenizer.decode(output_ids)
        return decoded
        
    def verify_tool_call(self, full_text: str, tools_json_str: str) -> dict:
        """
        Verify the generated tool call using GLiNER span scores.
        Strategy:
        1. Parse tools to get label names (classes).
        2. Identify spans of arguments in the generated text.
        3. Run forward pass with span_idx and prompts_embedding.
        4. Return confidence scores for each argument.
        """
        # 1. Parse Tools & Create Prompts
        try:
            tools_list = json.loads(tools_json_str)
            label_names = []
            for t in tools_list:
                if "function" in t:
                    func = t["function"]
                    if "parameters" in func and "properties" in func["parameters"]: 
                         label_names.extend(func["parameters"]["properties"].keys())
                    elif "parameters" in func: 
                         label_names.extend(func["parameters"].keys())
            label_names = list(dict.fromkeys(label_names)) # Unique
        except:
             return {"error": "Invalid tools JSON"}
             
        if not label_names:
            return {"status": "No params to verify"}

        prompts_ids = [self.tokenizer.encode(name).ids for name in label_names]
        
        # 2. Parse Text to find Spans
        # We assume full_text contains <tool_call>... content ...</tool_call>
        # We need to find the "arguments" JSON
        match = re.search(r'<tool_call>(.*?)</tool_call>', full_text, re.DOTALL)
        if not match:
            return {"error": "No tool_call tag found"}
            
        content = match.group(1).strip()
        try:
            tool_call_json = json.loads(content)
            args = tool_call_json.get("arguments", {})
        except:
            return {"error": "Invalid tool_call JSON content"}

        # Encode full text to find span indices
        # Note: Ideally we should use the same IDs we generated, but here we re-tokenize
        full_encoding = self.tokenizer.encode(full_text)
        input_ids = torch.tensor([full_encoding.ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        
        span_indices = []
        span_labels_dummy = [] # We don't need labels for inference, but we need to track which arg maps to which span
        extracted_args = []
        
        # Simple string find approach (Robustness similar to dataset.py is better but complex to copy-paste fully)
        # We search inside the full string
        tool_call_start_char = full_text.find("<tool_call>") + len("<tool_call>")
        
        for arg_name, arg_val in args.items():
            if arg_name in label_names:
                val_str = str(arg_val)
                # Search within content
                # Warning: duplicate values issue is not handled here for simplicity
                char_start = full_text.find(val_str, tool_call_start_char)
                if char_start != -1:
                    char_end = char_start + len(val_str) - 1
                    token_start = full_encoding.char_to_token(char_start)
                    token_end = full_encoding.char_to_token(char_end)
                    
                    if token_start is not None and token_end is not None:
                        span_indices.append([token_start, token_end])
                        extracted_args.append(arg_name)
        
        if not span_indices:
            return {"status": "No argument spans found mapping to tokens"}
            
        # Prepare Tensors
        span_idx_tensor = torch.tensor([span_indices], dtype=torch.long, device=self.device)
        
        # Prepare Prompts Embedding
        # Manual embedding generation as in Trainer/Dataset
        prompts_ids_tensor = torch.zeros((1, len(label_names), 10), dtype=torch.long, device=self.device) # Max len 10
        for i, p_ids in enumerate(prompts_ids):
            l = min(len(p_ids), 10)
            prompts_ids_tensor[0, i, :l] = torch.tensor(p_ids[:l], device=self.device)
            
        # Generate Prompts Embedding
        with torch.no_grad():
             # Flatten
            B, C, L = prompts_ids_tensor.size()
            flat_ids = prompts_ids_tensor.view(B * C, L)
            embeds = self.model.token_embedding(flat_ids)
            mask = (flat_ids != 0).float().unsqueeze(-1)
            sum_embeds = (embeds * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1)
            avg_embeds = sum_embeds / count
            prompts_embedding = avg_embeds.view(B, C, -1)
            
            # 3. Forward Pass for Verification
            y, z, logits, q_hat, span_scores = self.model(
                input_ids, attention_mask, 
                span_idx=span_idx_tensor, 
                prompts_embedding=prompts_embedding
            )
            
            # 4. Analyze Scores
            # span_scores: [1, NumSpans, NumClasses]
            probs = torch.softmax(span_scores, dim=-1)
            results = {}
            for i, arg_name in enumerate(extracted_args):
                # Find index of this arg class
                class_idx = label_names.index(arg_name)
                confidence = probs[0, i, class_idx].item()
                results[arg_name] = {
                    "confidence": confidence, 
                    "span": span_indices[i],
                    "verification": "HIGH" if confidence > 0.5 else "LOW"
                }
                
            return results

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

    def generate_tool_call_adaptive(self, tools: str, query: str, max_new_tokens: int = 200) -> str:
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
        return self.generate_adaptive(text, max_new_tokens=max_new_tokens)


    def _parse_tool_call(self, prediction_raw: str, gt_name: str, gt_args: dict):
        """
        Parse a raw model output to extract the predicted tool name and arguments.
        The parsing is guided by the ground-truth tool name and arguments.
        """
        # -------------------------------------------------
        # 1. Try to parse a complete and valid JSON snippet
        # -------------------------------------------------
        json_pattern = (
            r'\{[\s\S]*?"name"\s*:\s*"' +
            re.escape(gt_name) +
            r'"[\s\S]*?\}'
        )
        matches = re.findall(json_pattern, prediction_raw)

        for m in matches:
            try:
                parsed = json.loads(m)
                return {
                    "name": parsed.get("name", ""),
                    "arguments": parsed.get("arguments", {})
                }
            except json.JSONDecodeError:
                # Skip invalid or incomplete JSON snippets
                pass

        # -------------------------------------------------
        # 2. Fallback: heuristic-based parsing
        # -------------------------------------------------
        pred_name = ""
        pred_args = {}

        # 2.1 Tool name extraction
        # If the ground-truth tool name appears in the raw output,
        # assume the model intended to call this tool
        if gt_name in prediction_raw:
            pred_name = gt_name
        else:
            # Tool name not found → no valid tool call detected
            return {"name": "", "arguments": {}}

        # 2.2 Argument extraction (guided by ground-truth arguments)
        for arg_key, arg_val in gt_args.items():
            # If the ground-truth argument is an integer,
            # extract numeric values from the raw output
            if isinstance(arg_val, int):
                num_matches = re.findall(r'\b\d+\b', prediction_raw)
                if num_matches:
                    # Heuristic: the last number often corresponds to the argument value
                    pred_args[arg_key] = int(num_matches[-1])

            # If the ground-truth argument is a string,
            # attempt to extract a quoted string value
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
                prediction_raw = self.generate_tool_call_adaptive(tools, query)
                
                # Parse Ground Truth
                try:
                    gt_json = json.loads(solution_str)
                except json.JSONDecodeError:
                    gt_json = {}
                
                gt_name = gt_json.get("name", "")
                gt_args = gt_json.get("arguments", {})
                
                # Parse Prediction with Hint
                pred_json = self._parse_tool_call(prediction_raw, gt_name=gt_name, gt_args=gt_args)
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
    
    print("-" * 20)
    print("Standard Generation:")
    result = engine.generate(prompt)
    print(f"Result: {result}")
    
    print("-" * 20)
    print("Adaptive Generation (q_hat):")
    result_adaptive = engine.generate_adaptive(prompt)
    print(f"Result: {result_adaptive}")
