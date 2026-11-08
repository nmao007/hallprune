"""Truth-probe accuracy on the held-out Bürger et al. true/false sets (data/truth/eval).

Measures whether truth is still linearly decodable from a (pruned) model's residual stream, the
quantity TPLO reports as "hallucination detection" accuracy. Probes are retrained on every model.

Protocols, all on the residual stream after decoder layer L at the last token of each statement:
  - lofo (main metric): leave-one-topic-out. For each of the 6 topics, train on the other 5 topics
    (affirmative + negated) and test on the held-out topic's affirmative and negated sets, giving
    12 test accuracies. Their mean is the headline number.
  - comparisons: train on all 12 topic sets, test on larger_than / smaller_than (a different
    kind of statement altogether).
  - iid: 80/20 random split within each of the 12 sets (an easy ceiling, for reference).

Probes:
  - lr: logistic regression on standardized activations.
  - mm: mass-mean direction (mean_true - mean_false), thresholded halfway between the class means.

Note: probe accuracy measures what the model represents, not what it says. It complements, and
does not replace, generation-level hallucination evals.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT_DIR / "data" / "truth" / "eval"
TOPICS = ["cities", "sp_en_trans", "inventors", "animal_class", "element_symb", "facts"]
COMPARISONS = ["larger_than", "smaller_than"]
DEFAULT_LAYERS = [4, 8, 12, 16, 20, 24, 28]


def load_sets(eval_dir=EVAL_DIR):
    sets = {}
    for t in TOPICS:
        for name in (t, f"neg_{t}"):
            sets[name] = pd.read_csv(Path(eval_dir) / f"{name}.csv")[["statement", "label"]]
    for name in COMPARISONS:
        sets[name] = pd.read_csv(Path(eval_dir) / f"{name}.csv")[["statement", "label"]]
    return sets


@torch.no_grad()
def extract_acts(model, tokenizer, statements, layers, dev, batch_size=64):
    """{layer: float32 array (n, hidden)} at the last real token of each statement."""
    out = {l: [] for l in layers}
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for i in range(0, len(statements), batch_size):
        ids = [tokenizer(s, add_special_tokens=True)["input_ids"] for s in statements[i:i + batch_size]]
        L = max(len(x) for x in ids)
        batch = torch.full((len(ids), L), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), L), dtype=torch.long)
        for j, x in enumerate(ids):
            batch[j, : len(x)] = torch.tensor(x)
            mask[j, : len(x)] = 1
        last = torch.tensor([len(x) - 1 for x in ids])
        hs = model(input_ids=batch.to(dev), attention_mask=mask.to(dev), output_hidden_states=True).hidden_states
        rows = torch.arange(len(ids))
        for l in layers:
            # hidden_states[0] is the embedding output, hidden_states[l] the output of decoder layer l-1
            out[l].append(hs[l][rows, last.to(hs[l].device)].float().cpu().numpy())
    return {l: np.concatenate(v) for l, v in out.items()}


def _fit_predict(kind, Xtr, ytr, Xte):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
        return clf.predict(sc.transform(Xte))
    if kind == "mm":
        mu_t, mu_f = Xtr[ytr == 1].mean(0), Xtr[ytr == 0].mean(0)
        w = mu_t - mu_f
        thr = 0.5 * (mu_t + mu_f) @ w
        return (Xte @ w > thr).astype(int)
    raise ValueError(kind)


def probe_accuracies(acts, labels, kinds=("lr", "mm"), seed=0):
    """acts/labels: {set_name: array}. Returns nested dict of accuracies for one layer."""
    res = {}
    for kind in kinds:
        lofo = {}
        for t in TOPICS:
            train = [n for n in acts if n not in (t, f"neg_{t}") and n not in COMPARISONS]
            Xtr = np.concatenate([acts[n] for n in train])
            ytr = np.concatenate([labels[n] for n in train])
            for n in (t, f"neg_{t}"):
                lofo[n] = float((_fit_predict(kind, Xtr, ytr, acts[n]) == labels[n]).mean())

        train = [n for n in acts if n not in COMPARISONS]
        Xtr = np.concatenate([acts[n] for n in train])
        ytr = np.concatenate([labels[n] for n in train])
        comp = {n: float((_fit_predict(kind, Xtr, ytr, acts[n]) == labels[n]).mean()) for n in COMPARISONS}

        rng = np.random.default_rng(seed)
        iid = {}
        for n in acts:
            idx = rng.permutation(len(labels[n]))
            cut = int(0.8 * len(idx))
            tr, te = idx[:cut], idx[cut:]
            iid[n] = float((_fit_predict(kind, acts[n][tr], labels[n][tr], acts[n][te]) == labels[n][te]).mean())

        res[kind] = {
            "lofo_mean": float(np.mean(list(lofo.values()))),
            "lofo": lofo,
            "comparisons_mean": float(np.mean(list(comp.values()))),
            "comparisons": comp,
            "iid_mean": float(np.mean(list(iid.values()))),
            "iid": iid,
        }
    return res


def run_probe_eval(model, tokenizer, dev, layers=DEFAULT_LAYERS, eval_dir=EVAL_DIR):
    sets = load_sets(eval_dir)
    names = list(sets)
    statements = [s for n in names for s in sets[n]["statement"].tolist()]
    sizes = [len(sets[n]) for n in names]
    all_acts = extract_acts(model, tokenizer, statements, layers, dev)
    labels = {n: sets[n]["label"].to_numpy().astype(int) for n in names}
    results = {"layers": {}, "n_statements": {n: s for n, s in zip(names, sizes)}}
    for l in layers:
        acts, start = {}, 0
        for n, s in zip(names, sizes):
            acts[n] = all_acts[l][start:start + s]
            start += s
        results["layers"][str(l)] = probe_accuracies(acts, labels)
    return results
