"""Phase 1 diagnostics: why does pruning break cross-topic truth-probe generalization?

Runs on the Bürger sets (data/truth/eval), using the residual stream at the last token of each
statement (as in lib/probe_eval.py). Every test compares a pruned model with the dense one.

Knowledge: is the drop just facts the pruned model no longer knows? (e)
  knowledge            the model's own True/False judgment (chat prompt, logit "True" - logit "False").
                       A fact counts as KNOWN when the model judges both the statement and its negation
                       correctly, which is robust to a yes/no bias.
  lofo_by_knowledge    the standard cross-topic probe's accuracy, split into facts known by both models,
                       facts this model forgot, facts known by neither
  lofo_known_both      cross-topic probe retrained and tested only on facts both models know; compare
                       with dense_lofo_known_both (the dense model on the same statements)
  iid / iid_subject_disjoint   within-topic 80/20 splits, random vs. no subject shared between train
                       and test (subjects appear in both a true and a false statement)

Geometry: if the facts are known, what broke?
  lofo                 cross-topic accuracy and AUROC. AUROC ignores the threshold, so high AUROC with
                       low accuracy points to shifted offsets (b)
  lofo_centered        each set centred on its own mean (label-free) before training and testing (b)
  lofo_aff_only / lofo_neg_only   the same polarity in train and test (d)
  geometry.topic_cosine_mean      mean pairwise cosine of per-topic truth directions (a)
  geometry.aff_neg_cosine_mean    per-topic cosine of affirmative vs. negated truth directions (d)
  geometry.offset_dispersion      std of per-set true/false midpoints along the model's own truth
                       axis, divided by the mean true-false gap (b)
  geometry.noise_ratio            within-class spread along that axis divided by the gap (c)
  geometry_dense_frame            the same, measured along the DENSE model's axis and standardization
  transfer             the dense model's cross-topic probes applied unchanged to this model; and with
                       this model's own standardization (transfer_restandardized)
  sweep                regularization strength and no-standardization, at one layer (f)

Directions are mass-mean differences in a standardized space (pooled over the 12 topic sets).
Cosines between per-topic directions are lowered by sampling noise for the small topics; that noise
is the same for dense and pruned models, so compare across models, not with 1.0.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lib.probe_eval import COMPARISONS, EVAL_DIR, TOPICS, extract_acts

LAYERS = [8, 12, 16, 20, 24]
SWEEP_LAYER = 12
SWEEP_C = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
N_SPLITS = 3
SUBJECT_PATTERNS = {
    "sp_en_trans": r"word '([^']+)'",
    "inventors": r"^(.*?) (?:lived|did not live)",
    "animal_class": r"^The (.*?) (?:is|is not) ",
    "element_symb": r"^(.*?) (?:has|does not have) the symbol",
}
TF_PROMPT = ("Is the following statement true or false? Answer with one word, True or False.\n\n"
             "Statement: {}")
TOPIC_SETS = [s for t in TOPICS for s in (t, f"neg_{t}")]


# ---------------------------------------------------------------- data

def load_statements(eval_dir=EVAL_DIR):
    """One row per statement: set, topic, polarity, row (aligned between t and neg_t), label, subject."""
    frames = []
    for t in TOPICS:
        for name, pol in ((t, "aff"), (f"neg_{t}", "neg")):
            df = pd.read_csv(Path(eval_dir) / f"{name}.csv")
            if t == "cities":
                subj = df["city"].astype(str)
            elif t == "facts":
                subj = df["statement"]
            else:
                subj = df["statement"].str.extract(SUBJECT_PATTERNS[t])[0]
            assert subj.notna().all(), name
            frames.append(pd.DataFrame({
                "set": name, "topic": t, "polarity": pol, "row": np.arange(len(df)),
                "statement": df["statement"], "label": df["label"].astype(int), "subject": subj.values,
            }))
    for c in COMPARISONS:
        df = pd.read_csv(Path(eval_dir) / f"{c}.csv")
        frames.append(pd.DataFrame({
            "set": c, "topic": "comparisons", "polarity": "aff", "row": np.arange(len(df)),
            "statement": df["statement"], "label": df["label"].astype(int), "subject": df["n1"].astype(str),
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- GPU part: activations + judgments

def _tf_prompt(tokenizer, statement):
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template([{"role": "user", "content": TF_PROMPT.format(statement)}],
                                             tokenize=False, add_generation_prompt=True)
        return tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer(TF_PROMPT.format(statement) + "\nAnswer:", add_special_tokens=True)["input_ids"]


@torch.no_grad()
def tf_scores(model, tokenizer, statements, dev, batch_size=32):
    """logit("True") - logit("False") for the model's answer to TF_PROMPT, one per statement."""
    t_ids = tokenizer.encode("True", add_special_tokens=False)
    f_ids = tokenizer.encode("False", add_special_tokens=False)
    assert len(t_ids) == 1 and len(f_ids) == 1, (t_ids, f_ids)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    out = []
    for i in range(0, len(statements), batch_size):
        ids = [_tf_prompt(tokenizer, s) for s in statements[i:i + batch_size]]
        L = max(len(x) for x in ids)
        batch = torch.full((len(ids), L), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), L), dtype=torch.long)
        for j, x in enumerate(ids):
            batch[j, : len(x)] = torch.tensor(x)
            mask[j, : len(x)] = 1
        last = torch.tensor([len(x) - 1 for x in ids])
        h = model.model(input_ids=batch.to(dev), attention_mask=mask.to(dev)).last_hidden_state
        h = h[torch.arange(len(ids)), last.to(h.device)]
        logits = model.lm_head(h).float()  # only the last position, not the full (B, L, vocab) tensor
        out.append((logits[:, t_ids[0]] - logits[:, f_ids[0]]).cpu().numpy())
    return np.concatenate(out)


def extract_diag(model, tokenizer, dev, layers=LAYERS, eval_dir=EVAL_DIR):
    df = load_statements(eval_dir)
    statements = df["statement"].tolist()
    acts = extract_acts(model, tokenizer, statements, layers, dev)
    tf = tf_scores(model, tokenizer, statements, dev)
    return {"acts": {l: a.astype(np.float16) for l, a in acts.items()}, "tf": tf}


def save_diag(d, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, tf=d["tf"], **{f"L{l}": a for l, a in d["acts"].items()})


def load_diag(path):
    z = np.load(path)
    return {"tf": z["tf"], "acts": {int(k[1:]): z[k] for k in z.files if k.startswith("L")}}


# ---------------------------------------------------------------- probes

def _fit(kind, Xtr, ytr, C=1.0, standardize=True):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr) if standardize else None
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr) if sc else Xtr, ytr)
        return ("lr", sc, clf)
    mu_t, mu_f = Xtr[ytr == 1].mean(0), Xtr[ytr == 0].mean(0)
    w = mu_t - mu_f
    return ("mm", w, 0.5 * (mu_t + mu_f) @ w)


def _score(p, X, scaler=None):
    if p[0] == "lr":
        sc = scaler if scaler is not None else p[1]
        return p[2].decision_function(sc.transform(X) if sc is not None else X)
    return X @ p[1] - p[2]


def _center_by_set(X, df, rows):
    X = X.copy()
    sets = df["set"].values
    for s in pd.unique(sets[rows]):
        r = sets == s
        X[r] -= X[r & rows].mean(0)
    return X


def lofo_scores(X, df, keep=None, polarity=None, kind="lr", C=1.0, standardize=True, center=False,
                X_apply=None, probes=None, restandardize=False):
    """Leave-one-topic-out scores, one per statement (NaN where not tested), and the fitted probes.

    X_apply: score these activations instead of X (same statements), e.g. a pruned model's, with
    probes fitted on X. restandardize: rescale X_apply with its own training-set statistics.
    """
    N = len(df)
    keep = np.ones(N, bool) if keep is None else keep
    topic, y = df["topic"].values, df["label"].values
    base = keep & (topic != "comparisons")
    if polarity:
        base &= df["polarity"].values == polarity
    if center:
        X = _center_by_set(X, df, base)
        X_apply = _center_by_set(X_apply, df, base) if X_apply is not None else None
    Xa = X if X_apply is None else X_apply
    scores, fitted = np.full(N, np.nan), {}
    for t in TOPICS:
        tr, te = base & (topic != t), base & (topic == t)
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        p = probes[t] if probes is not None else _fit(kind, X[tr], y[tr], C, standardize)
        fitted[t] = p
        sc = None
        if restandardize and p[0] == "lr" and p[1] is not None:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(Xa[tr])
        scores[te] = _score(p, Xa[te], sc)
    return scores, fitted


def summarize(scores, df, rows=None):
    """Accuracy and AUROC per set and their unweighted mean over sets (as in probe_eval)."""
    from sklearn.metrics import roc_auc_score
    y, sets = df["label"].values, df["set"].values
    m = ~np.isnan(scores) if rows is None else (~np.isnan(scores) & rows)
    per = {}
    for s in pd.unique(sets[m]):
        r = m & (sets == s)
        auc = float(roc_auc_score(y[r], scores[r])) if len(np.unique(y[r])) == 2 else None
        per[s] = {"acc": float(((scores[r] > 0).astype(int) == y[r]).mean()), "auroc": auc, "n": int(r.sum())}
    accs = [v["acc"] for v in per.values()]
    aucs = [v["auroc"] for v in per.values() if v["auroc"] is not None]
    return {"acc": float(np.mean(accs)) if accs else None, "auroc": float(np.mean(aucs)) if aucs else None,
            "n": int(m.sum()), "per_set": per}


def iid(X, df, subject_disjoint, kind="lr", n_splits=N_SPLITS, seed=0):
    from sklearn.model_selection import GroupShuffleSplit
    y, sets, subj = df["label"].values, df["set"].values, df["subject"].values
    per = {}
    for s in TOPIC_SETS:
        idx = np.where(sets == s)[0]
        accs = []
        for k in range(n_splits):
            if subject_disjoint:
                tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed + k).split(idx, groups=subj[idx]))
            else:
                perm = np.random.default_rng(seed + k).permutation(len(idx))
                cut = int(0.8 * len(idx))
                tr, te = perm[:cut], perm[cut:]
            if len(np.unique(y[idx[tr]])) < 2:
                continue
            p = _fit(kind, X[idx[tr]], y[idx[tr]])
            accs.append(float(((_score(p, X[idx[te]]) > 0).astype(int) == y[idx[te]]).mean()))
        per[s] = float(np.mean(accs))
    return {"acc": float(np.mean(list(per.values()))), "per_set": per}


# ---------------------------------------------------------------- geometry

def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def geometry(X, df, frame=None):
    """Truth-direction geometry in a standardized space. frame = (mean, std, axis) of a reference model."""
    topic, pol, y, sets = df["topic"].values, df["polarity"].values, df["label"].values, df["set"].values
    rows = topic != "comparisons"
    if frame is None:
        mu, sd = X[rows].mean(0), X[rows].std(0) + 1e-6
        Z = (X - mu) / sd
        w = Z[rows & (y == 1)].mean(0) - Z[rows & (y == 0)].mean(0)
        frame = (mu, sd, w / np.linalg.norm(w))
    mu, sd, w = frame
    Z = (X - mu) / sd

    def direction(m):
        return Z[m & (y == 1)].mean(0) - Z[m & (y == 0)].mean(0)

    dirs = {t: direction(topic == t) for t in TOPICS}
    pairs = {f"{a}|{b}": _cos(dirs[a], dirs[b]) for i, a in enumerate(TOPICS) for b in TOPICS[i + 1:]}
    aff_neg = {t: _cos(direction((topic == t) & (pol == "aff")), direction((topic == t) & (pol == "neg")))
               for t in TOPICS}
    shared = {t: _cos(dirs[t], w) for t in TOPICS}

    proj = Z @ w
    per = {}
    for s in TOPIC_SETS:
        r = sets == s
        pt, pf = proj[r & (y == 1)], proj[r & (y == 0)]
        per[s] = {"mid": float((pt.mean() + pf.mean()) / 2), "gap": float(pt.mean() - pf.mean()),
                  "within_sd": float(np.sqrt((pt.var() + pf.var()) / 2))}
    mids = np.array([v["mid"] for v in per.values()])
    gaps = np.array([v["gap"] for v in per.values()])
    sds = np.array([v["within_sd"] for v in per.values()])
    res = {
        "topic_cosine_mean": float(np.mean(list(pairs.values()))),
        "topic_cosine": pairs,
        "topic_vs_shared_axis": shared,
        "aff_neg_cosine_mean": float(np.mean(list(aff_neg.values()))),
        "aff_neg_cosine": aff_neg,
        "offset_dispersion": float(mids.std() / gaps.mean()),
        "noise_ratio": float(np.mean(sds / np.abs(gaps).clip(min=1e-6))),
        "per_set": per,
    }
    return res, frame


# ---------------------------------------------------------------- knowledge

def knowledge(df, tf):
    """correct: the model's True/False answer matches the label. known: correct for the statement AND
    its negation (topic sets); for comparisons, just correct."""
    correct = (tf > 0).astype(int) == df["label"].values
    known = correct.copy()
    rows = (df["topic"] != "comparisons").values
    key = (df["topic"] + ":" + df["row"].astype(str)).values
    both = pd.Series(correct).groupby(key).transform("all").values
    known[rows] = both[rows]
    return correct, known


def _knowledge_summary(df, tf, correct, known):
    from sklearn.metrics import roc_auc_score
    y, sets = df["label"].values, df["set"].values
    per = {}
    for s in pd.unique(sets):
        r = sets == s
        per[s] = {"tf_acc": float(correct[r].mean()), "tf_auroc": float(roc_auc_score(y[r], tf[r])),
                  "known_frac": float(known[r].mean())}
    topic_sets = [per[s] for s in TOPIC_SETS]
    return {"tf_acc": float(np.mean([v["tf_acc"] for v in topic_sets])),
            "tf_auroc": float(np.mean([v["tf_auroc"] for v in topic_sets])),
            "known_frac": float(np.mean([v["known_frac"] for v in topic_sets])),
            "per_set": per}


# ---------------------------------------------------------------- one layer, one model

def _analyze_layer(l, X, Xd, df, cats, both_known, sweep):
    X, Xd = X.astype(np.float32), Xd.astype(np.float32)
    out = {}
    s, _ = lofo_scores(X, df)
    out["lofo"] = summarize(s, df)
    out["lofo_by_knowledge"] = {c: summarize(s, df, m) for c, m in cats.items()}
    out["lofo_known_both"] = summarize(lofo_scores(X, df, keep=both_known)[0], df)
    out["dense_lofo_known_both"] = summarize(lofo_scores(Xd, df, keep=both_known)[0], df)
    out["lofo_mm"] = summarize(lofo_scores(X, df, kind="mm")[0], df)
    out["lofo_centered"] = summarize(lofo_scores(X, df, center=True)[0], df)
    out["lofo_aff_only"] = summarize(lofo_scores(X, df, polarity="aff")[0], df)
    out["lofo_neg_only"] = summarize(lofo_scores(X, df, polarity="neg")[0], df)

    _, dense_probes = lofo_scores(Xd, df)
    out["transfer"] = summarize(lofo_scores(Xd, df, X_apply=X, probes=dense_probes)[0], df)
    out["transfer_restandardized"] = summarize(
        lofo_scores(Xd, df, X_apply=X, probes=dense_probes, restandardize=True)[0], df)

    out["iid"] = iid(X, df, subject_disjoint=False)
    out["iid_subject_disjoint"] = iid(X, df, subject_disjoint=True)

    out["geometry"], _ = geometry(X, df)
    _, dense_frame = geometry(Xd, df)
    out["geometry_dense_frame"], _ = geometry(X, df, frame=dense_frame)

    if sweep:
        out["sweep"] = {f"C={C}": summarize(lofo_scores(X, df, C=C)[0], df) for C in SWEEP_C}
        out["sweep"]["no_standardize"] = summarize(lofo_scores(X, df, standardize=False)[0], df)
        for v in out["sweep"].values():
            v.pop("per_set")
    return str(l), out


def analyze(d, dense, df=None, layers=None, sweep_layer=SWEEP_LAYER, n_jobs=None):
    """d, dense: outputs of extract_diag (or load_diag) for this model and the dense model."""
    from joblib import Parallel, delayed
    df = load_statements() if df is None else df
    layers = sorted(d["acts"]) if layers is None else layers
    correct, known = knowledge(df, d["tf"])
    dcorrect, dknown = knowledge(df, dense["tf"])
    cats = {"known_both": known & dknown, "forgotten": dknown & ~known,
            "gained": known & ~dknown, "known_neither": ~known & ~dknown}
    res = {
        "knowledge": _knowledge_summary(df, d["tf"], correct, known),
        "dense_knowledge": _knowledge_summary(df, dense["tf"], dcorrect, dknown),
        "knowledge_categories": {c: float(m[(df["topic"] != "comparisons").values].mean()) for c, m in cats.items()},
    }
    jobs = Parallel(n_jobs=n_jobs or len(layers))(
        delayed(_analyze_layer)(l, d["acts"][l], dense["acts"][l], df, cats, known & dknown, l == sweep_layer)
        for l in layers)
    res["layers"] = dict(jobs)
    return res


def headline(res, layer=SWEEP_LAYER):
    """A one-screen summary of the numbers that decide between the explanations."""
    L = res["layers"][str(layer)]
    g, gd = L["geometry"], L["geometry_dense_frame"]
    k, dk = res["knowledge"], res["dense_knowledge"]
    rows = [
        ("T/F judgment acc (dense)", k["tf_acc"], dk["tf_acc"]),
        ("known fraction (dense)", k["known_frac"], dk["known_frac"]),
        ("forgotten fraction", res["knowledge_categories"]["forgotten"], None),
        ("LOFO acc / auroc", L["lofo"]["acc"], L["lofo"]["auroc"]),
        ("LOFO on forgotten facts", L["lofo_by_knowledge"]["forgotten"]["acc"], None),
        ("LOFO known-both (dense same rows)", L["lofo_known_both"]["acc"], L["dense_lofo_known_both"]["acc"]),
        ("LOFO centered", L["lofo_centered"]["acc"], None),
        ("LOFO aff-only / neg-only", L["lofo_aff_only"]["acc"], L["lofo_neg_only"]["acc"]),
        ("LOFO mass-mean", L["lofo_mm"]["acc"], None),
        ("transfer / restandardized", L["transfer"]["acc"], L["transfer_restandardized"]["acc"]),
        ("iid random / subject-disjoint", L["iid"]["acc"], L["iid_subject_disjoint"]["acc"]),
        ("topic cosine / aff-neg cosine", g["topic_cosine_mean"], g["aff_neg_cosine_mean"]),
        ("offset dispersion / noise ratio", g["offset_dispersion"], g["noise_ratio"]),
        ("dense-frame dispersion / noise", gd["offset_dispersion"], gd["noise_ratio"]),
    ]
    if "sweep" in L:
        rows.append(("sweep acc C=1e-3 / no-std", L["sweep"]["C=0.001"]["acc"], L["sweep"]["no_standardize"]["acc"]))
    fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"
    return "\n".join(f"  {name:36s} {fmt(a)} {fmt(b)}" for name, a, b in rows)
