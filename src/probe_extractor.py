import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

class ProbeExtractor:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.activations = {}

    def _get_hook(self, key):
        def hook(module, input, output):
            # Capture output tensor
            if isinstance(output, tuple):
                val = output[0]
            else:
                val = output
            self.activations[key] = val.detach().cpu()
        return hook

    def extract_4_probes(self, layer_idx, truthful_prompts, hallucinated_prompts):
        """
        Extracts 4 directional vectors for layer `layer_idx`
        Returns: dict with keys ['attn_in', 'post_attn', 'mlp_in', 'bottleneck']
        """
        layer = self.model.model.layers[layer_idx]
        
        # Register PyTorch Hooks
        h1 = layer.input_layernorm.register_forward_hook(self._get_hook('attn_in'))
        h2 = layer.self_attn.o_proj.register_forward_hook(self._get_hook('post_attn'))
        h3 = layer.post_attention_layernorm.register_forward_hook(self._get_hook('mlp_in'))
        h4 = layer.mlp.act_fn.register_forward_hook(self._get_hook('bottleneck'))

        # Collect activations for both conditions
        acts_truth = self._get_batch_activations(truthful_prompts)
        acts_hallu = self._get_batch_activations(hallucinated_prompts)

        # Remove hooks
        for h in [h1, h2, h3, h4]: h.remove()

        # Compute Mass-Mean Directions (Mean_Truth - Mean_Hallu)
        probes = {}
        for key in ['attn_in', 'post_attn', 'mlp_in', 'bottleneck']:
            vec_truth = acts_truth[key].mean(dim=(0, 1))
            vec_hallu = acts_hallu[key].mean(dim=(0, 1))
            
            direction = vec_truth - vec_hallu
            # L2 Normalize
            probes[key] = direction / torch.norm(direction, p=2)

        return probes

    def _get_batch_activations(self, prompts):
        # Helper to pass text and store hook dictionary
        # ...
        return self.activations


def get_truthful_qa_pairs(num_samples=100):
        """
        Loads TruthfulQA from Hugging Face and returns formatted 
        (truthful_prompt, hallucinated_prompt) tuples.
        """
        dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
        pairs = []
        
        for item in dataset:
            question = item['question']
            targets = item['mc1_targets']
            
            truth_text = None
            hallu_text = None
            
            # Extract one true and one false choice
            for choice, label in zip(targets['choices'], targets['labels']):
                if label == 1 and truth_text is None:
                    truth_text = choice
                elif label == 0 and hallu_text is None:
                    hallu_text = choice
                    
                if truth_text and hallu_text:
                    break
                    
            if truth_text and hallu_text:
                prompt_truth = f"Q: {question}\nA: {truth_text}"
                prompt_hallu = f"Q: {question}\nA: {hallu_text}"
                pairs.append((prompt_truth, prompt_hallu))
                
            if len(pairs) >= num_samples:
                break
                
        return pairs