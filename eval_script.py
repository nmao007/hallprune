import json
import os
import sys
import subprocess
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure 'src' is in the Python search path for importing lib modules
sys.path.append(os.path.abspath("src"))
from lib.eval import eval_ppl

# Load environment variables from .env
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment or .env file.")

os.environ["HF_TOKEN"] = HF_TOKEN

# Models to evaluate
MODELS = [
    "nmao7/hallprune-llama-3.1-8b",
    "nmao7/magnitude_pruned-llama-3.1-8b",
    "nmao7/wanda_pruned-llama-3.1-8b",
    "nmao7/sparsegpt_pruned-llama-3.1-8b"
]

for model_id in MODELS:
    model_name = model_id.split("/")[-1]
    base_output_dir = f"./eval_results/{model_name}"
    
    os.makedirs(base_output_dir, exist_ok=True)

    print("=" * 58)
    print(f"🚀 STARTING EVALUATION FOR: {model_name}")
    print("=" * 58)

    # ----------------------------------------------------
    # Phase 0: WikiText-2 Perplexity Evaluation
    # ----------------------------------------------------
    print("\n📉 Phase 0: Evaluating WikiText-2 Perplexity...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Set model sequence length required by eval_ppl (standard context length is 2048)
    if not hasattr(model, "seqlen"):
        seq_len = getattr(model.config, "max_position_embeddings", 2048)
        model.seqlen = min(seq_len, 2048)

    ppl_score = eval_ppl(args=None, model=model, tokenizer=tokenizer, device=device)
    print(f"✨ WikiText-2 Perplexity: {ppl_score:.4f}")

    # Save perplexity results
    ppl_output_dir = os.path.join(base_output_dir, "perplexity_results")
    os.makedirs(ppl_output_dir, exist_ok=True)
    with open(os.path.join(ppl_output_dir, "perplexity.json"), "w") as f:
        json.dump({"wikitext2_perplexity": ppl_score}, f, indent=4)

    # Clear model from GPU memory before running lm_eval CLI tasks
    del model
    del tokenizer
    torch.cuda.empty_cache()

    # ----------------------------------------------------
    # Helper Function for lm_eval Tasks
    # ----------------------------------------------------
    def run_eval(tasks, num_fewshot, output_subdir):
        output_path = os.path.join(base_output_dir, output_subdir)
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_id}",
            "--tasks", tasks,
            "--device", "cuda:0",
            "--batch_size", "auto",
            "--output_path", output_path
        ]
        if num_fewshot is not None:
            cmd.extend(["--num_fewshot", str(num_fewshot)])
        
        print(f"\n▶️ Running {tasks} ({f'{num_fewshot}-shot' if num_fewshot else '0-shot'})...")
        subprocess.run(cmd, check=True)

    # 1. TruthfulQA (0-shot)
    run_eval("truthfulqa_mc2", num_fewshot=None, output_subdir="hallucination_results")

    # 2. MMLU (5-shot)
    run_eval("mmlu", num_fewshot=5, output_subdir="mmlu_results")

    # 3. HellaSwag (10-shot)
    run_eval("hellaswag", num_fewshot=10, output_subdir="hellaswag_results")

    # 4. ARC-Challenge (25-shot)
    run_eval("arc_challenge", num_fewshot=25, output_subdir="arc_c_results")

    # 5. Winogrande (5-shot)
    run_eval("winogrande", num_fewshot=5, output_subdir="winogrande_results")

    print(f"\n✅ FINISHED EVALUATING: {model_name}")
    print(f"Results saved to {base_output_dir}/")
    print("=" * 58 + "\n")

print("🎉 ALL EVALUATIONS COMPLETED SUCCESSFULLY!")