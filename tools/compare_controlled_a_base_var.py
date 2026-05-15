import argparse
import json
import re
from collections import defaultdict, Counter


RELS = ["left", "right", "on", "under"]


def norm_gold(x):
    x = str(x).strip().lower()
    for r in RELS:
        if x == r:
            return r
    for r in RELS:
        if re.search(rf"\b{re.escape(r)}\b", x):
            return r
    return "unknown"


def pred_rel(text):
    text = str(text).strip().lower()
    text = text.replace("</s>", " ")

    # Prefer the first generated word.
    toks = re.split(r"[\s\.,;:!?()\[\]{}\"']+", text)
    toks = [t for t in toks if t]

    if toks:
        first = toks[0]
        if first in RELS:
            return first

    # Fallback search.
    # Put under before on.
    if re.search(r"\bunder\b", text):
        return "under"
    if re.search(r"\bleft\b", text):
        return "left"
    if re.search(r"\bright\b", text):
        return "right"
    if re.search(r"\bon\b", text) and "front" not in text:
        return "on"

    return "unknown"


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                qid = int(r.get("question_id", r.get("qid", len(rows))))
                rows.append((qid, r))
    return dict(rows)


def eval_rows(name, rows):
    correct = 0
    total = 0

    by_gold = defaultdict(lambda: [0, 0])
    conf = Counter()

    parsed = {}

    for qid, r in rows.items():
        gold = norm_gold(r.get("label", r.get("answer", "")))
        pred = pred_rel(r.get("response", ""))

        parsed[qid] = {
            "gold": gold,
            "pred": pred,
            "response": r.get("response", ""),
            "prompt": r.get("prompt", ""),
            "image": r.get("image", ""),
        }

        if gold not in RELS:
            continue

        ok = gold == pred
        correct += int(ok)
        total += 1

        by_gold[gold][0] += int(ok)
        by_gold[gold][1] += 1
        conf[(gold, pred)] += 1

    print(f"\n[{name}]")
    print(f"overall acc = {correct / total:.4f} ({correct}/{total})")

    print("\nBy relation:")
    for rel in RELS:
        c, t = by_gold[rel]
        print(f"  {rel:6s}: {c:3d}/{t:3d} = {c / t if t else 0.0:.4f}")

    print("\nConfusion matrix:")
    header = "gold   | " + " ".join([f"{r:>7s}" for r in RELS + ["unknown"]])
    print(header)
    print("-" * len(header))

    for g in RELS:
        vals = [conf[(g, p)] for p in RELS + ["unknown"]]
        print(f"{g:6s} | " + " ".join([f"{v:7d}" for v in vals]))

    return parsed, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="baseline answer jsonl")
    parser.add_argument("--var", required=True, help="VAR answer jsonl")
    parser.add_argument("--show", type=int, default=30)
    args = parser.parse_args()

    base_rows = load_rows(args.base)
    var_rows = load_rows(args.var)

    print("Base file:", args.base)
    print("VAR file: ", args.var)
    print(f"base rows = {len(base_rows)}")
    print(f"var rows  = {len(var_rows)}")

    common_ids = sorted(set(base_rows.keys()) & set(var_rows.keys()))
    print(f"common ids = {len(common_ids)}")

    base_parsed, base_correct, total = eval_rows("baseline logic=0", base_rows)
    var_parsed, var_correct, _ = eval_rows("VAR logic=1", var_rows)

    print("\n" + "=" * 80)
    print("Paired comparison: VAR vs baseline")
    print("=" * 80)
    print(f"baseline correct = {base_correct}/{total} = {base_correct / total:.4f}")
    print(f"VAR correct      = {var_correct}/{total} = {var_correct / total:.4f}")
    print(f"delta            = {var_correct - base_correct:+d}")

    fixed = []
    broken = []
    same_correct = []
    same_wrong = []
    changed = []

    trans = Counter()

    for qid in common_ids:
        b = base_parsed[qid]
        v = var_parsed[qid]

        gold = b["gold"]
        if gold not in RELS:
            continue

        b_ok = b["pred"] == gold
        v_ok = v["pred"] == gold

        trans[(gold, b["pred"], v["pred"])] += 1

        if b["pred"] != v["pred"]:
            changed.append(qid)

        if (not b_ok) and v_ok:
            fixed.append(qid)
        elif b_ok and (not v_ok):
            broken.append(qid)
        elif b_ok and v_ok:
            same_correct.append(qid)
        else:
            same_wrong.append(qid)

    print("\nSummary:")
    print(f"fixed by VAR       = {len(fixed)}")
    print(f"broken by VAR      = {len(broken)}")
    print(f"net improvement    = {len(fixed) - len(broken):+d}")
    print(f"same correct       = {len(same_correct)}")
    print(f"same wrong         = {len(same_wrong)}")
    print(f"prediction changed = {len(changed)}")

    print("\nTransition counts: gold | baseline -> VAR")
    for (gold, bp, vp), n in trans.most_common(40):
        if bp != vp:
            print(f"gold={gold:6s}: base {bp:7s} -> VAR {vp:7s}: {n}")

    def print_examples(title, ids):
        print("\n" + "-" * 80)
        print(title)
        print("-" * 80)

        for qid in ids[: args.show]:
            b = base_parsed[qid]
            v = var_parsed[qid]

            print(
                f"id={qid} image={b['image']} gold={b['gold']} | "
                f"base={b['pred']} resp={b['response']!r} | "
                f"VAR={v['pred']} resp={v['response']!r}"
            )

    print_examples("Fixed examples: baseline wrong -> VAR correct", fixed)
    print_examples("Broken examples: baseline correct -> VAR wrong", broken)
    print_examples("Changed prediction examples", changed)


if __name__ == "__main__":
    main()
