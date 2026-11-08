# Truth data

Rebuild everything here with `python scripts/build_truth_data.py`. Raw downloads go to `raw/` (gitignored).

The data is split by role. Data used to **build** the protected truth subspace is never used to **evaluate** it.

## Construction: `counterfact_quads.jsonl` (used by pruning)

Source: **CounterFact** (Meng et al., 2022, *Locating and Editing Factual Associations in GPT*, the ROME paper). It has 21,919 Wikidata facts, each with the true object and a counterfactual one, across 34 relations. MIT license.

Each record becomes a 2×2 quad over (truth, object):

| | true object (A) | counterfactual object (B) |
|---|---|---|
| affirmative | `aff_true`: "X is located in A." | `aff_false`: "X is located in B." |
| negated | `neg_false`: "It is not true that X is located in A." | `neg_true`: "It is not true that X is located in B." |

At the input of each pruned linear, using the last token of each statement:

- `t = (aff_true − aff_false) + (neg_true − neg_false)`: truth, with the object identity cancelled out
- `o = (aff_true − aff_false) − (neg_true − neg_false)`: object identity only (the control)

HallPrune penalizes the top-k eigenvectors of the positive part of `E[ttᵀ] − β·E[ooᵀ]`. See `src/lib/truth_data.py`.

Filtering done by the build script:

- **Templates:** only templates that read as a full clause are kept. Questions and `"{}, the"`-style fragments are dropped (5,657 records).
- **Overlap with the eval sets:** records whose subject appears in any of the evaluation statements are dropped (457).
- **Duplicates:** duplicate (subject, relation) pairs are dropped (319).

That leaves **15,486 quads**, shuffled with seed 0.

At prune time, `select_known_quads` also keeps only facts the **dense** model gets right (mean log-prob of the true object > the false object). Otherwise the "truth" contrast is noise. Disable it with `--contrast_no_filter`.

## Evaluation: `eval/*.csv` (held out from construction)

Source: **Truth_is_Universal** (Bürger et al., NeurIPS 2024). MIT license.

- **Topics:** `cities`, `sp_en_trans`, `inventors`, `animal_class`, `element_symb`, `facts`, each with a `neg_` version. These are the 12 datasets TPLO (Fu et al., EMNLP 2025 Findings) uses for probe accuracy, so results can be compared with it directly.
- **Also included:** `larger_than` and `smaller_than` (numerical comparisons, from Marks & Tegmark).
- **Format:** `statement,label` (plus metadata columns for some sets).

**Planned (not built yet):** generation-level hallucination evals are the main metric for the paper. They should be split from the CounterFact subjects by entity:

- **Short-form factual QA** (SimpleQA-style or PopQA): hallucination rate = incorrect ÷ attempted, reported together with the attempt rate.
- **Long-form factual precision** (FActScore-style biographies).
- **TruthfulQA generation:** %True and %Informative, for comparison with TPLO.

## Sources

- CounterFact: https://rome.baulab.info/data/dsets/counterfact.json
- Truth_is_Universal: https://github.com/sciai-lab/Truth_is_Universal/tree/main/datasets
