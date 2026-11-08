# Legacy results (pre-audit)

These are kept for the record only. **None of them is evidence for or against HallPrune**, and they should not be compared with runs in `results/`.
The code that produced them is at git tag `pre-audit` (and earlier commits). See `LOG.md` for the full audit.

| Folder | Models | Setup | Why it's not valid |
|---|---|---|---|
| `2026-07-29_baselines/` | magnitude, Wanda, SparseGPT | lm_eval 0.4.12, all 0-shot | Pruned with older code than HallPrune (current `prune_sparsegpt` crashed until it was fixed). SparseGPT used `percdamp=0.01`, and HallPrune effectively ~2.55 |
| `2026-08-18_hallprune-v0/` | HallPrune v0 | lm_eval 0.4.12, all 0-shot | Probes in the wrong space; trace-scaled α=50 (hard constraint); ~255× SparseGPT's damping; TruthfulQA includes the 100 probe questions |
| `2026-10-02_hallprune-v0/` | HallPrune v0 (same weights as 08-18: identical mc2) | lm_eval 0.4.13; mc2 0-shot, **MMLU 5-shot**; WikiText-2 PPL | Same as above. MMLU is 5-shot, so it isn't comparable to the 0-shot baselines |

The original 2026-07-29 HallPrune run was deleted: it wasn't a valid record even of the v0 method. It's still in git history before the `pre-audit` tag.

## Numbers, for reference only

| Model | Date | TruthfulQA mc2 | MMLU | ARC-C | HellaSwag | Winogrande | PPL |
|---|---|---|---|---|---|---|---|
| HallPrune v0 | 08-18 | 0.453 | 0.558 | 0.448 | 0.528 | 0.709 | — |
| HallPrune v0 | 10-02 | 0.453 | 0.578 (5-shot) | — | — | — | 10.80 |
| SparseGPT | 07-29 | 0.395 | 0.503 | 0.451 | 0.539 | 0.717 | — |
| Wanda | 07-29 | 0.370 | 0.473 | 0.426 | 0.516 | 0.693 | — |
| Magnitude | 07-29 | 0.472 | 0.338 | 0.300 | 0.355 | 0.557 | — |

For context, uniform guessing scores 0.448 on TruthfulQA mc2 and 0.226 on mc1.

## 2026-11: rank-1 probe, post-audit (`2026-11_rank1-probe/`)

These runs use the post-audit code (`74ff45e` onward): probes in the right space, `mean_diag` α scaling, damping fixed, held-out TruthfulQA. They're valid for what they measured, but they were archived when the project moved from a rank-1 TruthfulQA probe to a contrastive truth subspace built from external true/false data, with generation-level hallucination evals (see LOG.md, 2026-11-07). Re-evaluate the baselines with the new eval suite rather than reusing these numbers.

| Model | mc2 (all) | mc2_heldout | mc1_heldout | WikiText-2 PPL |
|---|---|---|---|---|
| SparseGPT (pd 0.01) | 0.457 | 0.458 | 0.283 | — |
| Wanda | 0.462 | 0.463 | 0.299 | — |
| HallPrune probe α=1 | 0.457 | 0.457 | 0.292 | 10.42 |
| HallPrune probe α=10 | 0.466 | 0.468 | 0.299 | 10.41 |
| HallPrune probe α=100 | 0.468 | 0.467 | 0.301 | 10.45 |
| HallPrune random α=10 (seed 0) | 0.459 | 0.460 | 0.289 | 10.40 |

Probe vs. random at α=10 is +0.007 mc2_heldout on a single seed, which is within noise (unpaired stderr is about ±0.015 on 717 questions). The gain saturates between α=10 and α=100, consistent with a single direction already being close to fully protected.
