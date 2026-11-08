"""Build ORACLE truth quads from the held-out Bürger eval sets (data/truth/eval).

This deliberately leaks the evaluation data into the protected subspace. It is an upper bound,
not a method: if HallPrune cannot preserve probe accuracy even when its subspace is measured on the
very statements the probe is tested on, then the per-layer Hessian penalty is the wrong lever.

Row i of <topic>.csv and neg_<topic>.csv are the same statement and its negation (labels flip).
Within each topic, every true affirmative row i is paired with a false affirmative row j:

    aff_true  = topic[i]       (true)
    aff_false = topic[j]       (false)
    neg_true  = neg_topic[j]   (true)
    neg_false = neg_topic[i]   (false)

so t = (aff_true - aff_false) + (neg_true - neg_false) cancels both the statement content and the
negation, keeping truth, exactly like the CounterFact quads. For cities, the true and false rows of
the same city are paired, so the subject cancels too; other topics are paired randomly (seed 0).

Writes data/truth/oracle_quads.jsonl. Use it with --contrast_file data/truth/oracle_quads.jsonl
--contrast_no_filter (there is no prompt/object split to run the knowledge filter on).
"""
import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "truth" / "eval"
OUT = ROOT / "data" / "truth" / "oracle_quads.jsonl"
TOPICS = ["cities", "sp_en_trans", "inventors", "animal_class", "element_symb", "facts"]


def main():
    rng = random.Random(0)
    quads = []
    for t in TOPICS:
        aff = pd.read_csv(EVAL / f"{t}.csv")
        neg = pd.read_csv(EVAL / f"neg_{t}.csv")
        assert len(aff) == len(neg) and ((aff.label.values + neg.label.values) == 1).all(), t
        if t == "cities":
            pairs = []
            for _, g in aff.groupby("city", sort=False):
                tr, fa = list(g.index[g.label == 1]), list(g.index[g.label == 0])
                pairs += list(zip(tr, fa))
        else:
            tr, fa = list(aff.index[aff.label == 1]), list(aff.index[aff.label == 0])
            rng.shuffle(tr)
            rng.shuffle(fa)
            pairs = list(zip(tr, fa))
        for i, j in pairs:
            quads.append({
                "topic": t,
                "aff_true": aff.statement[i], "aff_false": aff.statement[j],
                "neg_true": neg.statement[j], "neg_false": neg.statement[i],
            })
        print(f"{t}: {len(pairs)} quads")
    rng.shuffle(quads)
    with open(OUT, "w") as f:
        for q in quads:
            f.write(json.dumps(q) + "\n")
    print(f"wrote {len(quads)} quads to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
