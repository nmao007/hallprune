import json
import pandas as pd
from pathlib import Path
import torch

# __file__ is src/lib/probe_extractor.py -> parents[2] is the repo root
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT_DIR / "data" / "truthfulqa.parquet"
SPLIT_FILE = ROOT_DIR / "data" / "probe_split.json"

# Probe keys, each named after the linear layers whose *input* it lives in
PROBE_KEYS = ['qkv_in', 'o_in', 'gate_up_in', 'down_in']


def load_probe_questions(split_file=SPLIT_FILE):
    """Questions reserved for probe extraction. These are excluded from the held-out TruthfulQA eval."""
    with open(split_file) as f:
        return json.load(f)["probe_questions"]


def get_truthful_qa_pairs(num_samples=100, split_file=SPLIT_FILE):
    local_file = DATA_FILE
    if not local_file.exists():
        # Fallback if running directly from the root
        local_file = Path("data/truthfulqa.parquet")
    if not local_file.exists():
        raise FileNotFoundError(f"Could not find local dataset at: {local_file.resolve()}")

    probe_questions = load_probe_questions(split_file)
    if num_samples > len(probe_questions):
        raise ValueError(f"Requested {num_samples} probe pairs but the split only reserves {len(probe_questions)} questions")
    probe_questions = probe_questions[:num_samples]

    print(f"Loading local dataset from {local_file.resolve()}...")
    df = pd.read_parquet(local_file)
    df = df.set_index(df['question'].str.strip())

    pairs = []
    for question in probe_questions:
        row = df.loc[question]
        choices = row['choices']
        label_idx = row['label']

        # The correct answer is at the label index
        truth_text = choices[label_idx]

        # Grab the first incorrect answer to act as the hallucinated baseline
        hallu_text = next(c for i, c in enumerate(choices) if i != label_idx)

        pairs.append((f"Q: {question}\nA: {truth_text}", f"Q: {question}\nA: {hallu_text}"))

    return pairs


def random_probes(model, seed=0):
    """Random unit vectors with the same shapes as the real probes (random-direction control)."""
    gen = torch.Generator().manual_seed(seed)
    all_probes = {}
    for l_idx, layer in enumerate(model.model.layers):
        dims = {
            'qkv_in': layer.self_attn.q_proj.in_features,
            'o_in': layer.self_attn.o_proj.in_features,
            'gate_up_in': layer.mlp.gate_proj.in_features,
            'down_in': layer.mlp.down_proj.in_features,
        }
        layer_probes = {}
        for key in PROBE_KEYS:
            v = torch.randn(dims[key], generator=gen)
            layer_probes[key] = v / torch.norm(v, p=2)
        all_probes[l_idx] = layer_probes
    return all_probes


class ProbeExtractor:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract_all_layers_probes(self, truthful_prompts, hallucinated_prompts):
        """
        Extracts probes for ALL layers in a single batched pass to prevent
        model state corruption during sequential layer pruning.

        Activations are captured with forward *pre*-hooks on the linear layers being
        pruned, so each probe lives in the same space as the Hessian columns it penalizes
        (post-RMSNorm for q/k/v and gate/up, o_proj input, act_fn(gate) * up for down_proj).
        """
        num_layers = len(self.model.model.layers)
        keys = PROBE_KEYS

        # Storage: layer_idx -> key -> list of activations
        truth_storage = {l: {k: [] for k in keys} for l in range(num_layers)}
        hallu_storage = {l: {k: [] for k in keys} for l in range(num_layers)}

        def make_hook(layer_idx, key):
            def hook_fn(module, args):
                val = args[0]
                # Store the last-token input to this linear layer
                active_collection.append((layer_idx, key, val[0, -1, :].detach().float().cpu()))
            return hook_fn

        self.model.eval()
        for t_p, h_p in zip(truthful_prompts, hallucinated_prompts):
            for prompt, storage in [(t_p, truth_storage), (h_p, hallu_storage)]:
                active_collection = []
                handles = []

                # Register hooks across ALL layers simultaneously.
                # k_proj/v_proj share q_proj's input and up_proj shares gate_proj's input.
                for l_idx, layer in enumerate(self.model.model.layers):
                    targets = {
                        'qkv_in': layer.self_attn.q_proj,
                        'o_in': layer.self_attn.o_proj,
                        'gate_up_in': layer.mlp.gate_proj,
                        'down_in': layer.mlp.down_proj,
                    }
                    for k, target in targets.items():
                        handles.append(target.register_forward_pre_hook(make_hook(l_idx, k)))

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
                    # Degenerate probe (e.g. layer-0 qkv_in when both answers end in the same token):
                    # return None so the layer is pruned without a penalty instead of a NaN Hessian.
                    layer_probes[key] = direction / norm if norm > 1e-6 else None
                else:
                    layer_probes[key] = None
            all_probes[l_idx] = layer_probes

        return all_probes

    # Keep legacy method signature intact just in case external code calls it directly
    def extract_4_probes(self, layer_idx, truthful_prompts, hallucinated_prompts):
        all_probes = self.extract_all_layers_probes(truthful_prompts, hallucinated_prompts)
        return all_probes.get(layer_idx, {})
