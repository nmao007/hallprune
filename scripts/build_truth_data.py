"""Build the true/false data used by HallPrune's contrastive truth subspace.

Downloads (into data/truth/raw/, gitignored):
  - CounterFact (Meng et al., ROME): 21,919 facts, each with the true object and a counterfactual one.
  - Truth_is_Universal (Bürger et al., 2024): 6 topics + negated versions + larger/smaller_than.

Writes (committed):
  - data/truth/counterfact_quads.jsonl : construction set. Each line is a 2x2 quad
        aff_true   "<prompt> <true_obj>."                       label 1
        aff_false  "<prompt> <false_obj>."                      label 0
        neg_true   "It is not true that <prompt> <false_obj>."  label 1
        neg_false  "It is not true that <prompt> <true_obj>."   label 0
    so that (aff_true - aff_false) + (neg_true - neg_false) cancels the object identity and keeps truth,
    and (aff_true - aff_false) - (neg_true - neg_false) keeps the object identity (the control).
  - data/truth/eval/*.csv : the Bürger sets, held out for probe-accuracy evaluation (TPLO's protocol).
    CounterFact records whose subject appears in any of these statements are dropped from the quads.

Usage: python scripts/build_truth_data.py [--skip_download]
"""
import argparse
import json
import random
import re
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / "data" / "truth"
RAW = TRUTH / "raw"

COUNTERFACT_URL = "https://rome.baulab.info/data/dsets/counterfact.json"
TIU_URL = "https://raw.githubusercontent.com/sciai-lab/Truth_is_Universal/main/datasets/{}.csv"
TIU_SETS = [
    "cities", "neg_cities", "sp_en_trans", "neg_sp_en_trans", "inventors", "neg_inventors",
    "animal_class", "neg_animal_class", "element_symb", "neg_element_symb", "facts", "neg_facts",
    "larger_than", "smaller_than",
]
NEG_PREFIX = "It is not true that "


def download(skip):
    (RAW / "counterfact").mkdir(parents=True, exist_ok=True)
    (RAW / "truth_is_universal").mkdir(parents=True, exist_ok=True)
    if skip:
        return
    urllib.request.urlretrieve(COUNTERFACT_URL, RAW / "counterfact" / "counterfact.json")
    for name in TIU_SETS:
        urllib.request.urlretrieve(TIU_URL.format(name), RAW / "truth_is_universal" / f"{name}.csv")


def clausal(template):
    """Keep templates that read as a full declarative clause once the object is appended."""
    t = template.strip()
    if "?" in t or t.startswith("{},") or t.endswith(","):
        return False
    return t.count("{}") == 1


def negate(template, statement):
    # Lowercase the first word unless the statement starts with the (proper-noun) subject
    if template.startswith("{}"):
        return NEG_PREFIX + statement
    return NEG_PREFIX + statement[0].lower() + statement[1:]


def build_quads(eval_text):
    data = json.loads((RAW / "counterfact" / "counterfact.json").read_text())
    quads, seen = [], set()
    dropped = {"template": 0, "eval_overlap": 0, "degenerate": 0, "duplicate": 0}
    for rec in data:
        rw = rec["requested_rewrite"]
        template, subject = rw["prompt"], rw["subject"]
        true_obj, false_obj = rw["target_true"]["str"].strip(), rw["target_new"]["str"].strip()
        if not clausal(template):
            dropped["template"] += 1
            continue
        if not true_obj or not false_obj or true_obj.lower() == false_obj.lower():
            dropped["degenerate"] += 1
            continue
        if re.search(r"\b" + re.escape(subject) + r"\b", eval_text):
            dropped["eval_overlap"] += 1
            continue
        key = (subject, rw["relation_id"])
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)

        prompt = template.format(subject)
        aff_true = f"{prompt} {true_obj}."
        aff_false = f"{prompt} {false_obj}."
        quads.append({
            "case_id": rec["case_id"],
            "relation_id": rw["relation_id"],
            "subject": subject,
            "prompt": prompt,
            "true_obj": true_obj,
            "false_obj": false_obj,
            "aff_true": aff_true,
            "aff_false": aff_false,
            "neg_true": negate(template, aff_false),
            "neg_false": negate(template, aff_true),
        })
    return quads, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    download(args.skip_download)

    eval_dir = TRUTH / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_text = []
    for name in TIU_SETS:
        src = RAW / "truth_is_universal" / f"{name}.csv"
        shutil.copyfile(src, eval_dir / f"{name}.csv")
        eval_text.append(src.read_text())
    eval_text = "\n".join(eval_text)

    quads, dropped = build_quads(eval_text)
    random.Random(args.seed).shuffle(quads)
    out = TRUTH / "counterfact_quads.jsonl"
    with open(out, "w") as f:
        for q in quads:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"wrote {len(quads)} quads to {out.relative_to(ROOT)}")
    print("dropped:", dropped)
    print("relations:", len({q['relation_id'] for q in quads}))
    for q in quads[:3]:
        print({k: q[k] for k in ("aff_true", "aff_false", "neg_true", "neg_false")})


if __name__ == "__main__":
    main()
