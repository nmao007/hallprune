# HallPrune work log

## 2026-10-25: audit fixes

These changes respond to an external review of the HallPrune method (see `hallprune_method.pdf`).
Old artifacts: `pruning.ipynb` is obsolete, and everything under `evaluation_results/` is stale.
In particular, `evaluation_results/hallprune-llama-3.1-8b/` is known to be wrong. Don't compare against those numbers.
Every model must be re-pruned and re-evaluated with the code below.

### What the audit confirmed (before these fixes)

| # | Issue | Where | Status |
|---|---|---|---|
| 1 | Probes captured in the wrong space: pre-norm residual for q/k/v and gate/up, o_proj **output**, `act_fn(gate)` instead of `act_fn(gate)*up` for down_proj | `src/lib/probe_extractor.py` | Fixed |
| 2 | α scaled by trace(H): α=50 → about 2e5× mean diag (d=4096), about 7e5× (d=14336). In practice a hard constraint `δW·v = 0` | `src/lib/modified_sparsegpt.py` | Fixed (configurable) |
| 3 | Damping 0.05 vs the baseline's 0.01, **and** computed after the penalty. Effective damping was 0.05 × 51 ≈ 2.55× mean diag, about **255×** the SparseGPT baseline | `src/lib/prune.py`, `src/lib/sparsegpt.py` | Fixed |
| 4 | Leakage: the probe questions (first 100 rows of `data/truthfulqa.parquet`) are all in `truthfulqa_mc2`. Checked against the official 817-question split: 100/100 overlap (all 684 parquet rows are in it) | `probe_extractor.py`, `eval_script.py` | Fixed (held-out task) |
| 5 | No random-direction control | — | Added |
| 6 | Headline MMLU compares mismatched setups: HallPrune 0.578 is **5-shot** (lm_eval 0.4.13, Oct 2), SparseGPT 0.503 is **0-shot** (0.4.12, Jul 29). The 0-shot HallPrune run (Aug 18) was 0.558 | result JSONs | Needs re-eval |
| 7 | `prune_sparsegpt` and `prune_ablate` reference `position_embeddings` without defining it → `NameError`. The SparseGPT baseline came from older code | `src/lib/prune.py` | Fixed |
| 8 | TruthfulQA mc2 doesn't track quality: magnitude pruning (MMLU 0.338) scored the *highest* mc2 (0.472) | result JSONs | Note for analysis |

### Changes

**Probe extraction** (`src/lib/probe_extractor.py`)
- Activations now come from `register_forward_pre_hook` on `q_proj`, `o_proj`, `gate_proj` and `down_proj`. That's exactly what each pruned linear (and its Hessian) sees. k/v share q's input, and up shares gate's.
- Keys renamed to their spaces: `qkv_in`, `o_in`, `gate_up_in`, `down_in` (were `attn_in`, `post_attn`, `mlp_in`, `bottleneck`).
- Probe questions now come from a fixed split file, `data/probe_split.json`: 100 questions sampled with `numpy.random.default_rng(0)`. Previously it was the first 100 rows.
- Degenerate probes (norm < 1e-6) return `None`, so that layer is pruned without a penalty. Before, a zero probe was divided by its norm and gave a NaN Hessian. This happens in practice: **all 100 probe pairs end in "."**, so at layer 0 the `qkv_in` input at the last token is identical for the truthful and hallucinated prompts.
- Added `random_probes(model, seed)`: random unit vectors with the same shapes, for the control.

**Penalty and damping** (`src/lib/modified_sparsegpt.py`, `src/lib/sparsegpt.py`)
- `add_truthfulness_penalty(theta, alpha, scale_mode)`:
  - `scale_mode="mean_diag"` (new default): penalty = `alpha · mean(diag H) · vvᵀ`, so α means the same thing for every layer width.
  - `scale_mode="trace"`: the old behavior. `--hall_alpha 50 --hall_alpha_scale trace` reproduces the original penalty.
- The penalty records `mean(diag H)` *before* it is added, and `fasterprune` damps relative to that (`damp_base`). With α=0, HallPrune now produces exactly the same weights as SparseGPT (checked in the sanity test).

**CLI** (`src/main.py`)
- `--percdamp` (default 0.01): used by both `sparsegpt` and `hallprune`.
- `--hall_alpha` (default 1.0), `--hall_alpha_scale {mean_diag,trace}`
- `--hall_direction {probe,random}`, `--hall_direction_seed`
- `--probe_nsamples` (default 100, at most the size of the split)

**Bug fix** (`src/lib/prune.py`): `prune_sparsegpt` and `prune_ablate` now read `position_embeddings` from the cache.

**Held-out eval** (`tasks/truthfulqa_heldout/`, `eval_script.py`)
- New lm_eval tasks `truthfulqa_mc1_heldout` and `truthfulqa_mc2_heldout`: same prompt and scoring as upstream, with the 100 probe questions filtered out (817 → 717). The filter asserts that it dropped exactly the split size.
- `eval_script.py` passes `--include_path tasks` and runs `truthfulqa_mc2`, `truthfulqa_mc2_heldout` and `truthfulqa_mc1_heldout`. **Report the held-out numbers.**

**Run script** (`hallprune_script.py`)
- Passes `--percdamp`, `--hall_alpha`, `--hall_alpha_scale` and `--hall_direction` explicitly.
- The run name and HF repo now include the config (e.g. `hallprune-probe-a1.0-mean_diag-llama-3.1-8b`), so sweep runs no longer overwrite `hallprune-llama-3.1-8b`. Update `MODELS` in `eval_script.py` to match.

### Verification
- A CPU sanity test on a tiny random Llama (2 layers, non-trivial norm weights) checked that:
  - the probe pre-hooks capture the same tensors as the inputs to `q_proj` and `down_proj`;
  - the shapes are right and the probes have unit norm;
  - the random probes are seeded and reproducible;
  - `damp_base` equals the pre-penalty mean diagonal;
  - α=0 reproduces SparseGPT exactly at 50% sparsity.
- Dose-response on one layer, as ‖δW·v‖ (SparseGPT baseline 2.63): α = 1e-3 → 2.63, 1e-1 → 2.47, 1 → 1.62, 10 → 0.53, 1000 → 0.01.
- `truthfulqa_mc2_heldout` filtering was checked against the real HF validation split (817 → 717).
- **Not run:** anything on the real 8B model or a GPU. lm_eval isn't installed locally, so the task YAMLs haven't been loaded by lm_eval itself.

### To do: experiments
1. Re-prune SparseGPT and Wanda baselines (with the `NameError` fixed) at `--percdamp 0.01`.
2. α sweep with `mean_diag` scaling: e.g. 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4. For reference, the old setting (trace α=50) is mean_diag α ≈ 2e5 at d=4096.
3. Random-direction control at each α (`--hall_direction random`, a few seeds).
4. Evaluate everything with the same `eval_script.py` (same lm_eval version and few-shot settings), reporting held-out TruthfulQA.
5. Consider an out-of-distribution truthfulness eval (e.g. HaluEval, or a TruthfulQA generation task), given issue 8.
6. Consider averaging the probe over the answer tokens instead of using only the last token. The last token is "." in every probe pair.
