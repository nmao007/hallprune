import urllib.request
import pandas as pd
from pathlib import Path
import torch

def get_truthful_qa_pairs(num_samples=100):
    # __file__ is src/lib/probe_extractor.py
    # .parent (1) is lib/
    # .parent (2) is src/
    # .parent (3) is hallprune/ (the root)
    root_dir = Path(__file__).resolve().parent.parent.parent
    
    # This now points to /content/hallprune/data/truthfulqa.parquet
    local_file = root_dir / "data" / "truthfulqa.parquet"
    
    if not local_file.exists():
        # Fallback if running directly from the root
        local_file = Path("data/truthfulqa.parquet")
        
    if not local_file.exists():
        raise FileNotFoundError(f"Could not find local dataset at: {local_file.resolve()}")

    print(f"Loading local dataset from {local_file.resolve()}...")
    
    # Read the parquet file into a pandas DataFrame
    df = pd.read_parquet(local_file)
    
    pairs = []
    for _, row in df.iterrows():
        question = row['question']
        choices = row['choices']
        label_idx = row['label']
        
        # The correct answer is at the label index
        truth_text = choices[label_idx]
        
        # Grab the first incorrect answer to act as the hallucinated baseline
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

    def extract_all_layers_probes(self, truthful_prompts, hallucinated_prompts):
        """
        Extracts probes for ALL layers in a single batched pass to prevent 
        model state corruption during sequential layer pruning.
        """
        num_layers = len(self.model.model.layers)
        keys = ['attn_in', 'post_attn', 'mlp_in', 'bottleneck']
        
        # Storage: layer_idx -> key -> list of activations
        truth_storage = {l: {k: [] for k in keys} for l in range(num_layers)}
        hallu_storage = {l: {k: [] for k in keys} for l in range(num_layers)}

        def make_hook(layer_idx, key):
            def hook_fn(module, input_tensor, output_tensor):
                val = input_tensor[0] if key in ['attn_in', 'mlp_in'] else (output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor)
                # Store activation for this specific layer and key
                # We use a mutable container or close over lists
                active_collection.append((layer_idx, key, val[0, -1, :].detach().cpu()))
            return hook_fn

        self.model.eval()
        for t_p, h_p in zip(truthful_prompts, hallucinated_prompts):
            for prompt, storage in [(t_p, truth_storage), (h_p, hallu_storage)]:
                active_collection = []
                handles = []
                
                # Register hooks across ALL layers simultaneously
                for l_idx, layer in enumerate(self.model.model.layers):
                    targets = {
                        'attn_in': layer.input_layernorm,
                        'post_attn': layer.self_attn.o_proj,
                        'mlp_in': layer.post_attention_layernorm,
                        'bottleneck': layer.mlp.act_fn
                    }
                    for k, target in targets.items():
                        handles.append(target.register_forward_hook(make_hook(l_idx, k)))

                with torch.no_grad():
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    self.model(**inputs)

                for h in handles:
                    h.remove()

                # Distribute captured activations into storage
                for l_idx, k, act in active_collection:
                    storage[l_idx][k].append(act)

        # Compute final normalized difference vectors (probes) for every layer
        all_probes = {}
        for l_idx in range(num_layers):
            layer_probes = {}
            for key in keys:
                t_list = truth_storage[l_idx][key]
                h_list = hallu_storage[l_idx][key]
                
                if len(t_list) > 0 and len(h_list) > 0:
                    t_tensor = torch.stack(t_list).float()
                    h_tensor = torch.stack(h_list).float()
                    direction = t_tensor.mean(dim=0) - h_tensor.mean(dim=0)
                    norm = torch.norm(direction, p=2)
                    layer_probes[key] = direction / norm if norm > 0 else direction
                else:
                    layer_probes[key] = None
            all_probes[l_idx] = layer_probes

        return all_probes

    # Keep legacy method signature intact just in case external code calls it directly
    def extract_4_probes(self, layer_idx, truthful_prompts, hallucinated_prompts):
        all_probes = self.extract_all_layers_probes(truthful_prompts, hallucinated_prompts)
        return all_probes.get(layer_idx, {})