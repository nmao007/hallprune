import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.probe_extractor import ProbeExtractor, get_truthful_qa_pairs
from src.modified_sparsegpt import Hall_SparseGPT

def main():
    model_id = "meta-llama/Llama-3.1-8B"
    alpha = 50.0  # Hyperparameter: Strength of ITI constraint
    sparsity = 0.5

    print("Fetching TruthfulQA Prompts...")
    prompt_pairs = get_truthful_qa_pairs(num_samples=100)
    truthful_prompts = [p[0] for p in prompt_pairs]
    hallucinated_prompts = [p[1] for p in prompt_pairs] 

    print("Loading Model & Tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    extractor = ProbeExtractor(model, tokenizer)

    for layer_idx, layer in enumerate(model.model.layers):
        print(f"\n--- Processing Layer {layer_idx} ---")
        
        # 1. Extract 4 probes for current layer
        probes = extractor.extract_4_probes(layer_idx, truthful_prompts, hallucinated_prompts)
        
        # 2. Prune Attention Projections using θ_attn_in
        for name, proj in [("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj), ("v", layer.self_attn.v_proj)]:
            solver = Hall_SparseGPT(proj)
            # ... Pass standard calibration samples (SparseGPT add_batch) ...
            solver.add_iti_penalty(probes['attn_in'], alpha=alpha)
            solver.fasterprune(sparsity=sparsity, blocksize=128, percdamp=0.01)

        # 3. Prune Attention Output using θ_post_attn
        solver = Hall_SparseGPT(layer.self_attn.o_proj)
        solver.add_iti_penalty(probes['post_attn'], alpha=alpha)
        solver.fasterprune(sparsity=sparsity, blocksize=128, percdamp=0.01)

        # 4. Prune MLP Gate/Up using θ_mlp_in
        for name, proj in [("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj)]:
            solver = Hall_SparseGPT(proj)
            solver.add_iti_penalty(probes['mlp_in'], alpha=alpha)
            solver.fasterprune(sparsity=sparsity, blocksize=128, percdamp=0.01)

        # 5. Prune MLP Down using θ_bottleneck
        solver = Hall_SparseGPT(layer.mlp.down_proj)
        solver.add_iti_penalty(probes['bottleneck'], alpha=alpha)
        solver.fasterprune(sparsity=sparsity, blocksize=128, percdamp=0.01)

    print("\nPruning complete! Saving model...")
    model.save_pretrained("./pruned_llama3_iti_guided")

if __name__ == "__main__":
    main()