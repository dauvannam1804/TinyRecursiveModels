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
        encoded = self.tokenizer.encode(prompt)
        input_ids = [self.bos_token_id] + encoded.ids
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
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

    def evaluate_dataset(self, data_path: str, n_samples: int = None) -> float:
        import json
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
            
        correct = 0
        total = 0
        
        print(f"Evaluating on {len(data)} samples...")
        for item in tqdm(data):
            # XLAM / Swift Format
            if "tools" in item and "messages" in item:
                tools = item["tools"]
                query = ""
                solution = ""
                
                for msg in item["messages"]:
                    if msg["role"] == "user":
                        query = msg["content"]
                    elif msg["role"] == "tool_call":
                        solution = f"<tool_call>\n{msg['content']}\n</tool_call>"
                
                if not query or not solution:
                    continue
                    
                # Generate
                prediction = self.generate_tool_call(tools, query)
                
                # Re-construct prompt to strip it
                prompt_text = "<|im_start|>system\n"
                prompt_text += "You are a helpful assistant.\n"
                prompt_text += "# Tools\n"
                prompt_text += "You may call one or more functions to assist with the user query.\n"
                prompt_text += "You are provided with function signatures within <tools></tools> XML tags:\n"
                prompt_text += f"<tools>\n{tools}\n</tools>\n"
                prompt_text += "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                prompt_text += "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"
                prompt_text += f"<|im_start|>user\n{query}<|im_end|>\n"
                prompt_text += "<|im_start|>assistant\n"
                
                if prediction.startswith(prompt_text):
                    prediction = prediction[len(prompt_text):].strip()
                else:
                    # Fallback: try to find the last <|im_start|>assistant\n
                    idx = prediction.rfind("<|im_start|>assistant\n")
                    if idx != -1:
                        prediction = prediction[idx + len("<|im_start|>assistant\n"):].strip()
                
                # Normalize for comparison (simple strip)
                solution = solution.strip()
                prediction = prediction.strip()
                
                # Remove <|im_end|> if present in prediction
                prediction = prediction.replace("<|im_end|>", "").strip()
                
                if prediction == solution:
                    correct += 1
                total += 1
            
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        return accuracy

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
