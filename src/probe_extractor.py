import urllib.request
import pandas as pd
from pathlib import Path
import torch

def get_truthful_qa_pairs(num_samples=100):
    """
    Reads EleutherAI/truthful_qa_mc from the local data/ directory.
    Automatically downloads it if missing (e.g., on a fresh remote clone).
    """
    # Dynamically resolve paths relative to src/probe_extractor.py
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    local_file = data_dir / "truthfulqa.parquet"
    
    # Auto-download fallback for remote servers
    if not local_file.exists():
        print(f"Data not found locally. Downloading to {local_file}...")
        data_dir.mkdir(parents=True, exist_ok=True)
        url = "https://huggingface.co/datasets/EleutherAI/truthful_qa_mc/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
        urllib.request.urlretrieve(url, local_file)
        
    # Read the local parquet file
    df = pd.read_parquet(local_file)
    
    pairs = []
    for _, row in df.iterrows():
        question = row['question']
        choices = row['choices']
        label_idx = row['label']
        
        # 1. Truthful text is the choice at the label index
        truth_text = choices[label_idx]
        
        # 2. Hallucinated text is just the first incorrect choice
        hallu_text = None
        for i, choice in enumerate(choices):
            if i != label_idx:
                hallu_text = choice
                break
                
        if truth_text and hallu_text:
            pairs.append((f"Q: {question}\nA: {truth_text}", f"Q: {question}\nA: {hallu_text}"))
            
        if len(pairs) >= num_samples:
            break
            
    return pairs

class ProbeExtractor:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _get_single_activation(self, layer_idx, prompt, key):
        layer = self.model.model.layers[layer_idx]
        captured_val = []

        def hook_fn(module, input_tensor, output_tensor):
            val = input_tensor[0] if key in ['attn_in', 'mlp_in'] else (output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor)
            captured_val.append(val[0, -1, :].detach().cpu())

        if key == 'attn_in':
            target = layer.input_layernorm
        elif key == 'post_attn':
            target = layer.self_attn.o_proj
        elif key == 'mlp_in':
            target = layer.post_attention_layernorm
        elif key == 'bottleneck':
            target = layer.mlp.act_fn

        handle = target.register_forward_hook(hook_fn)
        
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            self.model(**inputs)

        handle.remove()
        return captured_val[0] if len(captured_val) > 0 else None

    def extract_4_probes(self, layer_idx, truthful_prompts, hallucinated_prompts):
        probes = {}
        for key in ['attn_in', 'post_attn', 'mlp_in', 'bottleneck']:
            truth_acts = []
            hallu_acts = []

            for t_p, h_p in zip(truthful_prompts, hallucinated_prompts):
                act_t = self._get_single_activation(layer_idx, t_p, key)
                act_h = self._get_single_activation(layer_idx, h_p, key)
                
                if act_t is not None and act_h is not None:
                    truth_acts.append(act_t)
                    hallu_acts.append(act_h)

            t_tensor = torch.stack(truth_acts).float()
            h_tensor = torch.stack(hallu_acts).float()

            direction = t_tensor.mean(dim=0) - h_tensor.mean(dim=0)
            norm = torch.norm(direction, p=2)
            
            probes[key] = direction / norm if norm > 0 else direction

        return probes