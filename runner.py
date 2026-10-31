"""Run HallPrune (or a baseline) on Llama 3.1 8B on a Modal H100.

    modal secret create huggingface HF_TOKEN=hf_...
    modal run runner.py::main                             # probe, alpha=1.0
    modal run runner.py::main --hall-alpha 10 --hall-direction random
    modal run runner.py::main --prune-method sparsegpt    # baseline
    modal run --detach runner.py::main ...                # keep running if you close the terminal

The pruned model goes to a private HF repo <your-user>/<run_name>. Logs and the
model also stay on the "hallprune-out" volume.

Evaluate a pruned model (same tasks and few-shot settings as eval_script.py).
Results are copied back to ./results/<model_name>/:

    modal run runner.py::evaluate --model-id nmao7/hallprune-probe-a1.0-mean_diag-llama-3.1-8b
    modal run runner.py::evaluate --model-id ... --truthfulqa-only   # quick check
"""
import subprocess
from pathlib import Path

import modal

REPO = "/root/hallprune"
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("sentencepiece", "protobuf", "tiktoken", "pandas", "pyarrow", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
    .add_local_dir("src", f"{REPO}/src", ignore=["**/.DS_Store", "**/__pycache__", "out/**", "llm_weights/**"])
    .add_local_dir("data", f"{REPO}/data")
    .add_local_dir("tasks", f"{REPO}/tasks", ignore=["**/__pycache__"])
)

app = modal.App("hallprune", image=image)
hf_cache = modal.Volume.from_name("hallprune-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("hallprune-out", create_if_missing=True)


@app.function(
    gpu="H100",
    timeout=4 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": hf_cache, "/out": out_vol},
)
def prune(
    prune_method: str = "hallprune",
    sparsity_ratio: float = 0.5,
    percdamp: float = 0.01,
    hall_alpha: float = 1.0,
    hall_alpha_scale: str = "mean_diag",
    hall_direction: str = "probe",
    hall_direction_seed: int = 0,
    upload: bool = True,
):
    from huggingface_hub import HfApi

    if prune_method == "hallprune":
        run_name = f"hallprune-{hall_direction}-a{hall_alpha}-{hall_alpha_scale}-llama-3.1-8b"
        if hall_direction == "random":
            run_name = run_name.replace("-llama", f"-s{hall_direction_seed}-llama")
    else:
        # Distinct from the pre-audit *_pruned-llama-3.1-8b repos, so they aren't overwritten
        run_name = f"{prune_method}-pd{percdamp}-llama-3.1-8b"
    out_dir = f"/out/{run_name}"

    cmd = [
        "python", "src/main.py",
        "--model", MODEL_ID,
        "--cache_dir", "/cache/llm_weights",
        "--prune_method", prune_method,
        "--sparsity_ratio", str(sparsity_ratio),
        "--sparsity_type", "unstructured",
        "--percdamp", str(percdamp),
        "--hall_alpha", str(hall_alpha),
        "--hall_alpha_scale", hall_alpha_scale,
        "--hall_direction", hall_direction,
        "--hall_direction_seed", str(hall_direction_seed),
        "--save", out_dir,
        "--save_model", out_dir,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)
    hf_cache.commit()
    out_vol.commit()

    if upload:
        api = HfApi()
        repo_id = f"{api.whoami()['name']}/{run_name}"
        api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="model")
        print(f"Uploaded to https://huggingface.co/{repo_id}")

    return Path(out_dir, f"log_{prune_method}.txt").read_text()


# Keep in sync with eval_script.py: (tasks, num_fewshot, output_subdir)
EVAL_SUITE = [
    ("truthfulqa_mc2,truthfulqa_mc2_heldout,truthfulqa_mc1_heldout", None, "hallucination_results"),
    ("mmlu", 5, "mmlu_results"),
    ("hellaswag", 10, "hellaswag_results"),
    ("arc_challenge", 25, "arc_c_results"),
    ("winogrande", 5, "winogrande_results"),
]


@app.function(
    gpu="H100",
    timeout=6 * 60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": hf_cache, "/out": out_vol},
)
def evaluate_model(model_id: str, truthfulqa_only: bool = False) -> dict[str, bytes]:
    import json
    import sys

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.append(f"{REPO}/src")
    from lib.eval import eval_ppl

    model_name = model_id.split("/")[-1]
    base_output_dir = Path(f"/out/results/{model_name}")
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: WikiText-2 perplexity
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model.seqlen = min(getattr(model.config, "max_position_embeddings", 2048), 2048)
    ppl_score = eval_ppl(args=None, model=model, tokenizer=tokenizer, device=torch.device("cuda:0"))
    print(f"WikiText-2 perplexity: {ppl_score:.4f}")
    ppl_dir = base_output_dir / "perplexity_results"
    ppl_dir.mkdir(exist_ok=True)
    (ppl_dir / "perplexity.json").write_text(json.dumps({"wikitext2_perplexity": ppl_score}, indent=4))
    del model, tokenizer
    torch.cuda.empty_cache()

    # lm_eval tasks
    suite = EVAL_SUITE[:1] if truthfulqa_only else EVAL_SUITE
    for tasks, num_fewshot, output_subdir in suite:
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_id}",
            "--tasks", tasks,
            "--device", "cuda:0",
            "--batch_size", "auto",
            "--output_path", str(base_output_dir / output_subdir),
            "--include_path", f"{REPO}/tasks",
        ]
        if num_fewshot is not None:
            cmd.extend(["--num_fewshot", str(num_fewshot)])
        print(f"\nRunning {tasks} ({f'{num_fewshot}-shot' if num_fewshot else '0-shot'})...")
        subprocess.run(cmd, cwd=REPO, check=True)
        out_vol.commit()

    hf_cache.commit()
    return {
        str(p.relative_to(base_output_dir)): p.read_bytes()
        for p in base_output_dir.rglob("*") if p.is_file()
    }


@app.local_entrypoint()
def main(
    prune_method: str = "hallprune",
    hall_alpha: float = 1.0,
    hall_alpha_scale: str = "mean_diag",
    hall_direction: str = "probe",
    hall_direction_seed: int = 0,
    percdamp: float = 0.01,
    upload: bool = True,
):
    print(prune.remote(
        prune_method=prune_method,
        percdamp=percdamp,
        hall_alpha=hall_alpha,
        hall_alpha_scale=hall_alpha_scale,
        hall_direction=hall_direction,
        hall_direction_seed=hall_direction_seed,
        upload=upload,
    ))


@app.local_entrypoint()
def evaluate(model_id: str, truthfulqa_only: bool = False):
    files = evaluate_model.remote(model_id, truthfulqa_only=truthfulqa_only)
    local_dir = Path("results") / model_id.split("/")[-1]
    for rel_path, data in files.items():
        dest = local_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    print(f"Saved {len(files)} files to {local_dir}/")
