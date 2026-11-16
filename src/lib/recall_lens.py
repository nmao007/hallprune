"""Recall filter + logit lens: where does a pruned model lose a fact it used to recall?

Facts: CounterFact prompts ("Mount Wade is located in") with their true object ("Antarctica"), from
data/truth/counterfact_quads.jsonl. Nearly every object is a single token, so a model RECALLS the fact
when that token is its top next-token prediction after the prompt.

Logit lens: the hidden state at the prompt's last token after every layer is passed through the final
norm and the unembedding (lm_head), giving the target token's rank (0 = top) at each depth. Pruning only
touches the decoder layers, so norm and lm_head are the same in the dense and pruned models.

For facts the dense model recalls but a pruned model does not (FORGOTTEN), the question is whether the
answer surfaces somewhere in the pruned model's middle layers and is then lost (points to retrieval or
expression), or never surfaces at all (points to storage). The dense model's own failures, and the
facts both models recall, are the reference points.

Caveat: the logit lens is unreliable in early layers (their hidden states are not yet in the output
basis), so only the second half of the network is interpretable.
"""
import json
from pathlib import Path

import numpy as np
import torch

from lib.truth_data import QUADS_FILE

TOP_K = 10      # "surfaced" = the answer is in the top 10 at that layer
FAR = 100       # "never surfaces" = the answer never gets into the top 100


def load_recall_facts(tokenizer, n=4000, path=QUADS_FILE):
    """First n quads whose true and false objects start with different tokens; target = first token."""
    facts = []
    with open(path) as f:
        for line in f:
            q = json.loads(line)
            t = tokenizer(" " + q["true_obj"], add_special_tokens=False)["input_ids"]
            fo = tokenizer(" " + q["false_obj"], add_special_tokens=False)["input_ids"]
            if t[0] == fo[0]:
                continue
            facts.append({"prompt": q["prompt"], "true_obj": q["true_obj"], "relation_id": q["relation_id"],
                          "target": t[0], "single_token": len(t) == 1})
            if len(facts) >= n:
                break
    return facts


@torch.no_grad()
def lens_ranks(model, tokenizer, facts, dev, batch_size=64):
    """(n_facts, n_layers + 1) rank of the target token at the last prompt token after each layer
    (index 0 = embeddings, index n_layers = final output), and the matching probabilities."""
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    norm, head = model.model.norm, model.lm_head
    ranks, probs = [], []
    for i in range(0, len(facts), batch_size):
        chunk = facts[i:i + batch_size]
        ids = [tokenizer(f["prompt"], add_special_tokens=True)["input_ids"] for f in chunk]
        L = max(len(x) for x in ids)
        batch = torch.full((len(ids), L), pad, dtype=torch.long)
        mask = torch.zeros((len(ids), L), dtype=torch.long)
        for j, x in enumerate(ids):
            batch[j, : len(x)] = torch.tensor(x)
            mask[j, : len(x)] = 1
        last = torch.tensor([len(x) - 1 for x in ids])
        hs = model.model(input_ids=batch.to(dev), attention_mask=mask.to(dev), output_hidden_states=True).hidden_states
        tgt = torch.tensor([f["target"] for f in chunk])
        r_b, p_b = [], []
        n_layers = len(hs) - 1
        for l, h in enumerate(hs):
            h = h[torch.arange(len(ids)), last.to(h.device)]
            # HF already applies the final norm to the last hidden state; don't apply it twice
            logits = head(h if l == n_layers else norm(h)).float()
            t = tgt.to(logits.device)
            tl = logits[torch.arange(len(ids)), t]
            r_b.append((logits > tl[:, None]).sum(-1).cpu())
            p_b.append(torch.softmax(logits, -1)[torch.arange(len(ids)), t].cpu())
        ranks.append(torch.stack(r_b, 1))
        probs.append(torch.stack(p_b, 1))
    return torch.cat(ranks).numpy().astype(np.int32), torch.cat(probs).numpy().astype(np.float32)


# ---------------------------------------------------------------- analysis

def _profile(ranks, rows):
    """Per-layer share of facts with the answer in the top 1 / top 10, and the median rank."""
    r = ranks[rows]
    if len(r) == 0:
        return None
    return {"n": int(len(r)),
            "top1": (r == 0).mean(0).round(4).tolist(),
            "top10": (r < TOP_K).mean(0).round(4).tolist(),
            "median_rank": np.median(r, 0).tolist()}


def _failure_shape(ranks, rows, from_layer):
    """For facts a model gets wrong: did the answer surface in a middle/late layer, end as a near miss,
    or never get close?"""
    r = ranks[rows]
    if len(r) == 0:
        return None
    mid = r[:, from_layer:-1]
    final = r[:, -1]
    best = mid.min(1)
    return {
        "n": int(len(r)),
        "surfaced_top10_before_final": float((best < TOP_K).mean()),
        "surfaced_top1_before_final": float((best == 0).mean()),
        "final_near_miss_top10": float((final < TOP_K).mean()),
        "final_rank_median": float(np.median(final)),
        "never_top100": float((r[:, from_layer:].min(1) >= FAR).mean()),
        "best_layer_median": float(np.median(from_layer + mid.argmin(1))),
    }


def first_layer(top_frac, thresh=0.5):
    """First layer at which at least `thresh` of the facts have the answer in the top 10."""
    for l, v in enumerate(top_frac):
        if v >= thresh:
            return l
    return None


def analyze(dense_ranks, pruned_ranks):
    n_layers = dense_ranks.shape[1] - 1
    half = n_layers // 2
    d_ok, p_ok = dense_ranks[:, -1] == 0, pruned_ranks[:, -1] == 0
    forgotten, kept = d_ok & ~p_ok, d_ok & p_ok
    res = {
        "n_facts": int(len(d_ok)),
        "dense_recall": float(d_ok.mean()),
        "pruned_recall": float(p_ok.mean()),
        "gained": int((~d_ok & p_ok).sum()),
        "forgotten": int(forgotten.sum()),
        "forgotten_frac_of_dense_recalled": float(forgotten.sum() / max(d_ok.sum(), 1)),
        # how the pruned model fails on facts it forgot, vs. how the dense model fails on facts it never knew
        "pruned_on_forgotten": _failure_shape(pruned_ranks, forgotten, half),
        "dense_on_its_unknown": _failure_shape(dense_ranks, ~d_ok, half),
        "profiles": {
            "dense_on_forgotten": _profile(dense_ranks, forgotten),
            "pruned_on_forgotten": _profile(pruned_ranks, forgotten),
            "dense_on_kept": _profile(dense_ranks, kept),
            "pruned_on_kept": _profile(pruned_ranks, kept),
        },
    }
    prof = res["profiles"]
    res["answer_emerges_layer"] = {k: (first_layer(v["top10"]) if v else None) for k, v in prof.items()}
    return res


def headline(res):
    f, d = res["pruned_on_forgotten"], res["dense_on_its_unknown"]
    prof = res["profiles"]
    lines = [
        f"  facts {res['n_facts']}: dense recalls {res['dense_recall']:.3f}, pruned {res['pruned_recall']:.3f}; "
        f"forgotten {res['forgotten']} ({res['forgotten_frac_of_dense_recalled']:.1%} of dense-recalled), gained {res['gained']}",
        "  failure shape (layers from the middle on)    pruned-on-forgotten   dense-on-its-unknown",
    ]
    for k, name in [("surfaced_top10_before_final", "answer in top 10 at some layer before final"),
                    ("surfaced_top1_before_final", "answer top 1 at some layer before final"),
                    ("final_near_miss_top10", "final answer rank within top 10"),
                    ("never_top100", "never in top 100"),
                    ("final_rank_median", "median final rank"),
                    ("best_layer_median", "median layer of best rank")]:
        lines.append(f"    {name:44s} {f[k]:>10.3f}          {d[k]:>10.3f}" if f and d else f"    {name}: -")
    em = res["answer_emerges_layer"]
    lines.append("  first layer where >=50% have the answer in top 10: "
                 + ", ".join(f"{k} {v}" for k, v in em.items()))
    lines.append("  top-10 share by layer (every 4th layer, from 16):")
    for k in ("dense_on_forgotten", "pruned_on_forgotten", "dense_on_kept", "pruned_on_kept"):
        if prof[k]:
            t = prof[k]["top10"]
            lines.append(f"    {k:20s} " + " ".join(f"L{l}:{t[l]:.2f}" for l in range(16, len(t), 4)) + f" final:{t[-1]:.2f}")
    return "\n".join(lines)
