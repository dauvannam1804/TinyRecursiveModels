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

    def evaluate_dataset(self, csv_path: str, n_samples: int = None) -> float:
        import pandas as pd
        import json
        from tqdm import tqdm
        
        if not os.path.exists(csv_path):
            print(f"Dataset not found at {csv_path}")
            return 0.0
            
        df = pd.read_csv(csv_path)
        if n_samples:
            df = df.head(n_samples)
            
        correct = 0
        total = 0
        
        print(f"Evaluating on {len(df)} samples...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            conversations = json.loads(row["conversations"])
            problem = ""
            solution = ""
            for turn in conversations:
                if turn["from"] == "human":
                    problem = turn["value"]
                elif turn["from"] == "gpt":
                    solution = turn["value"]
            
            prompt = f"Problem: {problem}\nSolution:"
            
            # Generate
            # We want to stop at EOS or newline to avoid generating extra text
            # But for now, let's just generate and strip
            prediction = self.generate(prompt, max_new_tokens=50)
            
            # Post-processing for comparison
            # Prediction might contain the prompt if not handled carefully, 
            # but our generate appends to input_ids. 
            # Wait, self.generate returns decoded text. 
            # Does it return the FULL text or just the new tokens?
            # Looking at generate: 
            # output_ids = input_ids[0].tolist() -> input_ids includes prompt.
            # So decoded includes prompt.
            # We need to strip the prompt from prediction.
            
            if prediction.startswith(prompt):
                prediction = prediction[len(prompt):].strip()
            
            # Also strip solution just in case
            solution = solution.strip()
            
            # Simple Exact Match
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
