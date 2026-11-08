"""Contrastive truth subspace for HallPrune.

The construction set is data/truth/counterfact_quads.jsonl (built by scripts/build_truth_data.py).
Each quad is a 2x2 design over (truth, object):

    aff_true   "X is located in A."                     (true)
    aff_false  "X is located in B."                     (false)
    neg_true   "It is not true that X is located in B." (true)
    neg_false  "It is not true that X is located in A." (false)

At the input of each pruned linear, take the activation at the last token of each statement:
    d_aff = aff_true - aff_false  ~ truth + object(A - B)
    d_neg = neg_true - neg_false  ~ truth - object(A - B)
    t = d_aff + d_neg             ~ 2 * truth   (object identity cancels)
    o = d_aff - d_neg             ~ 2 * object  (control: what the words alone change)

The penalty matrix is the positive part of  E[t t^T] - beta * E[o o^T],  truncated to its top-k
eigenvectors and normalized to unit trace. It is PSD by construction, so H + alpha * s * P stays
positive definite and SparseGPT's Cholesky is unaffected. With k = 1 it reduces to the rank-1 probe
penalty (alpha keeps the same meaning: the penalty's total trace is alpha * s).
"""
import json
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
QUADS_FILE = ROOT_DIR / "data" / "truth" / "counterfact_quads.jsonl"

# Order of the four statements inside every batch of quads
QUAD_FIELDS = ("aff_true", "aff_false", "neg_true", "neg_false")


def load_quads(path=None):
    path = Path(path) if path else QUADS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python scripts/build_truth_data.py` first")
    with open(path) as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def _mean_logprob(model, tokenizer, prompts, completions, dev):
    """Mean per-token log-prob of each completion given its prompt (right-padded batch)."""
    ids, starts = [], []
    for p, c in zip(prompts, completions):
        p_ids = tokenizer(p, add_special_tokens=True)["input_ids"]
        full = tokenizer(p + " " + c, add_special_tokens=True)["input_ids"]
        # BPE can merge across the boundary; fall back to the shared prefix length
        start = len(p_ids)
        while start > 0 and full[:start] != p_ids[:start]:
            start -= 1
        ids.append(full)
        starts.append(start)
    L = max(len(x) for x in ids)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batch = torch.full((len(ids), L), pad, dtype=torch.long)
    for i, x in enumerate(ids):
        batch[i, : len(x)] = torch.tensor(x)
    logits = model(batch.to(dev)).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    out = []
    for i, x in enumerate(ids):
        tgt = torch.tensor(x[starts[i]:], device=logp.device)
        pos = torch.arange(starts[i] - 1, len(x) - 1, device=logp.device)
        out.append(logp[i, pos, tgt].mean().item())
    return out


@torch.no_grad()
def select_known_quads(model, tokenizer, quads, n, dev, batch_size=32, filter_known=True):
    """First n quads (in file order) whose true object the dense model prefers over the false one."""
    if not filter_known:
        return quads[:n], {"scored": 0, "kept": min(n, len(quads))}
    kept, scored = [], 0
    for i in range(0, len(quads), batch_size):
        chunk = quads[i:i + batch_size]
        prompts = [q["prompt"] for q in chunk]
        lp_true = _mean_logprob(model, tokenizer, prompts, [q["true_obj"] for q in chunk], dev)
        lp_false = _mean_logprob(model, tokenizer, prompts, [q["false_obj"] for q in chunk], dev)
        scored += len(chunk)
        for q, a, b in zip(chunk, lp_true, lp_false):
            if a > b:
                kept.append(q)
        if len(kept) >= n:
            break
    return kept[:n], {"scored": scored, "kept": len(kept[:n])}


def tokenize_quads(tokenizer, quads, quads_per_batch):
    """Right-padded input_ids batches of 4 * quads_per_batch statements, plus last-token indices.

    Right padding with a causal mask leaves every real token's activations unchanged, so the
    padded positions are simply ignored. A trailing partial batch is dropped.
    """
    n = (len(quads) // quads_per_batch) * quads_per_batch
    quads = quads[:n]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    batches = []
    for i in range(0, n, quads_per_batch):
        texts = [q[f] for q in quads[i:i + quads_per_batch] for f in QUAD_FIELDS]
        ids = [tokenizer(t, add_special_tokens=True)["input_ids"] for t in texts]
        L = max(len(x) for x in ids)
        batch = torch.full((len(ids), L), pad, dtype=torch.long)
        for j, x in enumerate(ids):
            batch[j, : len(x)] = torch.tensor(x)
        last = torch.tensor([len(x) - 1 for x in ids], dtype=torch.long)
        batches.append((batch, last))
    return batches, n


class ContrastAccumulator:
    """Running E[t t^T] and E[o o^T] for one linear-layer input space."""

    def __init__(self, dim, dev):
        self.Mt = torch.zeros(dim, dim, device=dev, dtype=torch.float32)
        self.Mo = torch.zeros(dim, dim, device=dev, dtype=torch.float32)
        self.n = 0

    def add(self, acts):
        # acts: (4 * q, dim) in QUAD_FIELDS order
        a = acts.float().reshape(-1, 4, acts.shape[-1])
        d_aff = a[:, 0] - a[:, 1]
        d_neg = a[:, 2] - a[:, 3]
        t = d_aff + d_neg
        o = d_aff - d_neg
        self.Mt += t.T @ t
        self.Mo += o.T @ o
        self.n += a.shape[0]

    def penalty(self, rank=64, control_weight=1.0, random_seed=None):
        """Unit-trace PSD penalty matrix, or (None, info) if there is no positive truth signal.

        random_seed: replace the eigenvectors with a random orthonormal basis of the same rank,
        keeping the same eigenvalues (matched random-subspace control).
        """
        if self.n == 0:
            return None, {"reason": "empty"}
        M = (self.Mt - control_weight * self.Mo) / self.n
        M = (M + M.T) / 2
        evals, evecs = torch.linalg.eigh(M)
        pos = evals.clamp(min=0)
        total_abs = evals.abs().sum()
        if pos.sum() <= 1e-8 * max(total_abs.item(), 1e-30) or pos.sum() <= 0:
            return None, {"reason": "no positive eigenvalues"}
        k = int((pos > 0).sum().item()) if rank in (0, None) else min(rank, int((pos > 0).sum().item()))
        lam = pos[-k:]
        U = evecs[:, -k:]
        if random_seed is not None:
            g = torch.Generator(device="cpu").manual_seed(random_seed)
            R = torch.randn(M.shape[0], k, generator=g).to(M.device)
            U, _ = torch.linalg.qr(R)
        P = (U * lam) @ U.T
        P = (P + P.T) / 2
        P = P / torch.trace(P)
        info = {
            "rank": k,
            "pos_mass": (pos.sum() / total_abs).item(),          # share of |spectrum| that is truth > control
            "kept_mass": (lam.sum() / pos.sum()).item(),          # share of positive mass in the top-k
            "top_frac": (lam[-1] / lam.sum()).item(),             # share of the kept mass in the top direction
            "control_ratio": (torch.trace(self.Mo) / torch.trace(self.Mt).clamp(min=1e-30)).item(),
        }
        return P, info

    def free(self):
        self.Mt = self.Mo = None
