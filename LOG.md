# HallPrune work log

## 2026-10-25: audit fixes

These changes respond to an external review of the HallPrune method (see `hallprune_method.pdf`).
Old artifacts: `pruning.ipynb` is obsolete. All pre-audit eval results now live in `results/legacy/` (see the 2026-10-26 entry). Don't compare against those numbers.
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

## 2026-10-26: results reorganized

- Tagged `74ff45e` as `pre-audit` (local tag, not pushed).
- Moved every pre-audit eval result into `results/legacy/`, grouped by run date. `results/legacy/README.md` explains why each run is invalid and keeps the numbers for reference.
  - `eval_results/hallprune-llama-3.1-8b` (Oct 2) → `results/legacy/2026-10-02_hallprune-v0/`
  - `evaluation_results/evaluation_results 2/hallprune-llama-3.1-8b` (Aug 18) → `results/legacy/2026-08-18_hallprune-v0/`
  - `evaluation_results/{magnitude,wanda,sparsegpt}_pruned-llama-3.1-8b` (Jul 29) → `results/legacy/2026-07-29_baselines/`
  - Deleted `evaluation_results/hallprune-llama-3.1-8b` (Jul 29): wrong even for the v0 method. It's still in git history.
- `eval_script.py` now writes to `results/<model_name>/`.
- Pinned `lm-eval==0.4.13` in `requirements.txt` (the old runs mixed 0.4.12 and 0.4.13).
- To do, by hand on Hugging Face: rename `nmao7/hallprune-llama-3.1-8b` to `-legacy` (keep it private), and replace the magnitude, Wanda and SparseGPT repos once they've been re-pruned with the current code.

## 2026-11-07: contrastive truth subspace (rank-k) from external data

### Why
- The rank-1 TruthfulQA probe gave +0.007 mc2_heldout over the random-direction control on one seed. That's within noise.
- The gain saturates between α=10 and α=100. At α=10, ‖δW·v‖ is already about 5× smaller, so pushing harder on one direction can't help, and adding more directions is the only lever left.
- The closest prior work, TPLO (EMNLP 2025 Findings), only re-allocates layer-wise sparsity in Wanda, using true/false separability. It changes no weights and adds no subspace constraint, so a truth constraint inside SparseGPT's solver is still open.
- TruthfulQA mc2 doesn't track quality well (see audit issue 8). The goal is now a lower *hallucination rate* than SparseGPT and Wanda, at some cost in perplexity.

### Archived
- All post-audit runs (SparseGPT, Wanda, probe α=1/10/100, random α=10) moved to `results/legacy/2026-11_rank1-probe/`. The numbers are in `results/legacy/README.md`.

### Data (`data/truth/`, `scripts/build_truth_data.py`; see `data/truth/README.md`)
- **Construction:** 15,486 CounterFact 2×2 quads (true/false object × affirmative/"It is not true that ..."). Subjects that appear in the eval sets are removed.
- **Held-out eval:** the Bürger et al. True/False sets (6 topics + negations, TPLO's probe-accuracy protocol), plus larger_than and smaller_than.

### Method (`src/lib/truth_data.py`, `Hall_SparseGPT.add_subspace_penalty`, `prune_hall`)
- **Contrasts:** at the last token of each statement, `t = d_aff + d_neg` (truth) and `o = d_aff − d_neg` (object identity).
- **Penalty:** P = top-k eigenvectors of the positive part of `E[ttᵀ] − β·E[ooᵀ]`, normalized to unit trace. H += α·mean_diag(H)·P.
  - P is PSD, so H stays positive definite and Cholesky is unaffected.
  - With k = 1 this is the rank-1 penalty, and α means the same thing as before.
- **Sequential measurement:** contrast statements go through the same partially pruned prefix as the C4 calibration data, so P for layer i is measured in the model actually being pruned. The rank-1 probe was extracted once, from the dense model.
- **Knowledge filter:** only quads the dense model gets right are used.
- **Matched control:** `contrast_random` keeps the same eigenvalues but uses a random orthonormal basis.
- **Diagnostics:** written to `<save>/contrast_diagnostics.json`:
  - `pos_mass`: share of the spectrum where the truth contrast beats the control
  - `control_ratio`: tr E[ooᵀ] / tr E[ttᵀ]
  - `coverage`: tr(PH) / mean diag H; ≫1 means plain SparseGPT already protects this subspace

### CLI
`--hall_direction {probe,random,contrast,contrast_random}`, `--contrast_rank` (64; 0 = all positive), `--contrast_nquads` (1024), `--contrast_control_weight` (1.0), `--contrast_quads_per_batch` (16), `--contrast_no_filter`, `--contrast_file`. `runner.py` passes these through and puts them in the run name (e.g. `hallprune-contrast-k64-n1024-a10.0-mean_diag-llama-3.1-8b`).

### Verification (CPU, tiny random Llama: 2 layers, hidden 64)
- Right-padded batched statements give the same last-token linear inputs as unpadded single forwards.
- Every P is PSD with unit trace. All pruned linears are at 50% sparsity.
- tr(δW P δWᵀ) drops monotonically with α. Example, layer 1 down_proj: 2.4e-3 at α=0, 9.0e-4 at α=10, 8.1e-5 at α=1000.
- α=0 reproduces `prune_sparsegpt` exactly (max |ΔW| = 0).
- `contrast_random`, β=0, and full rank all run. The old rank-1 probe and random paths still run.
- Layer-0 `qkv_in` has no signal (every statement ends in "."), so it's pruned without a penalty, as before.
- **Not run:** anything on the 8B model or a GPU. The eigendecomposition of a 14336×14336 matrix (down_proj) per layer is new compute; expect it to take minutes in total on an H100.

### Next experiments
1. Dense-model eval, and 2–3 more seeds of the rank-1 random control, to calibrate the noise.
2. Rank sweep k ∈ {1, 8, 64, 256} at α ∈ {10, 100}, each with a `contrast_random` control at the same k.
3. Ablations: β=0 (no object control) vs β=1; knowledge filter on vs off; number of quads.
4. Build the evals: Bürger probe accuracy (to compare with TPLO), and generation-level hallucination rate plus attempt rate (SimpleQA/PopQA-style, split by entity from CounterFact).
