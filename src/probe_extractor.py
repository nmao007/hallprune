import torch
import json
import urllib.request
from datasets import load_dataset
import mlcroissant as mlc

def get_truthful_qa_pairs(num_samples=100):
    """
    Downloads EleutherAI/truthful_qa_mc using the mlcroissant library.
    Reads strictly from Hugging Face's auto-generated Croissant API endpoint.
    """
    # HF's automated Croissant metadata endpoint for the dataset
    croissant_url = "https://huggingface.co/api/datasets/EleutherAI/truthful_qa_mc/croissant"
    
    print("Loading dataset via mlcroissant...")
    dataset = mlc.Dataset(croissant_url)
    
    # HF Croissant record sets are typically named "<config>_<split>"
    # We want the default config, validation split.
    try:
        records = dataset.records("default_validation")
    except ValueError:
        # Fallback dynamic lookup just in case the Croissant API naming changes
        record_sets = [rs.name for rs in dataset.metadata.record_sets]
        target_rs = next(rs for rs in record_sets if "validation" in rs.lower())
        records = dataset.records(target_rs)

    pairs = []
    for item in records:
        question = item['question']
        choices = item['choices']
        labels = item['labels']
        
        truth_text = None
        hallu_text = None
        
        for choice, label in zip(choices, labels):
            if label == 1 and truth_text is None:
                truth_text = choice
            elif label == 0 and hallu_text is None:
                hallu_text = choice
            if truth_text and hallu_text:
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