import argparse
import json
import re
from collections import defaultdict, Counter


RELS = ["left", "right", "on", "under"]


def norm_gold(x):
    x = str(x).strip().lower()
    for r in RELS:
        if x == r or r in x:
            return r
    return "unknown"


def pred_rel(text):
    text = str(text).strip()
    text = text.replace("</s>", " ")
    text = text.lower()

    # Prefer first clean answer token.
    first = re.split(r"[\s\.,;:!?()\[\]{}\"']+", text.strip())[0]
    if first in RELS:
        return first

    # Fallback: search relation words.
    # Put under before on to avoid weak substring-like errors.
    for r in ["under", "left", "right", "on"]:
        if r == "on":
            if re.search(r"\bon\b", text) and "front" not in text:
                return "on"
        else:
            if re.search(rf"\b{r}\b", text):
                return r

    return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-file", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.answer_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    correct = 0
    total = 0
    by_gold = defaultdict(lambda: [0, 0])
    conf = Counter()

    bad_examples = []

    for r in rows:
        g = norm_gold(r.get("label") or r.get("answer") or r.get("gt-label"))
        p = pred_rel(r.get("response", ""))

        if g not in RELS:
            continue

        ok = p == g
        correct += int(ok)
        total += 1

        by_gold[g][0] += int(ok)
        by_gold[g][1] += 1
        conf[(g, p)] += 1

        if not ok and len(bad_examples) < 30:
            bad_examples.append((r.get("question_id"), g, p, r.get("response", "")))

    print(f"\nFile: {args.answer_file}")
    print(f"overall acc = {correct / total:.4f} ({correct}/{total})")

    print("\nBy relation:")
    for rel in RELS:
        c, t = by_gold[rel]
        print(f"  {rel:6s}: {c}/{t} = {c / t if t else 0.0:.4f}")

    print("\nConfusion matrix:")
    header = "gold   | " + " ".join([f"{r:>7s}" for r in RELS + ["unknown"]])
    print(header)
    print("-" * len(header))

    for g in RELS:
        vals = [conf[(g, p)] for p in RELS + ["unknown"]]
        print(f"{g:6s} | " + " ".join([f"{v:7d}" for v in vals]))

    print("\nBad examples:")
    for qid, g, p, resp in bad_examples:
        print(f"id={qid} gold={g} pred={p} response={resp!r}")


if __name__ == "__main__":
    main()
