#!/usr/bin/env python3
"""
28_validate_master.py -- hard validation gate for the stimulus master CSV.

Named 28 because 27_clean_stimuli.py already exists.

Exists because an automated coverage metric was mistaken for a data-integrity check. The
clause annotator reported "94.6% regex hit rate, only 4 fallbacks" and that was treated as
a green light -- but it measures how often a pattern MATCHED, not whether the underlying
text was correct. Reading three stories would have shown the next scenario glued on the
end. Every check here fails loudly, and check 9 forces a human to actually read samples.

Exit code is non-zero if any check fails, so it can gate an sbatch chain.

Usage
  python code/experiments/28_validate_master.py --csv dataset/master/moral_2x2_master.csv
  python code/experiments/28_validate_master.py --csv <...> --sample 20 --seed 0
"""
import os, csv, re, sys, argparse, random, statistics
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

EXPECTED_CELLS = {"neutral": 72, "accidental": 77, "attempted": 72, "intentional": 77}
EXPECTED_N = 298

# response-prompt fragment: "<gerund phrase> was:" possibly string-terminal
PROMPT_TAIL = re.compile(r"\b[A-Za-z][^.!?]{0,80}?\b(?:was|were|is|are):")
# ALLCAPS scenario tag such as "LAB", "SAFETY TOWN", "JELLYFISH"
ALLCAPS = re.compile(r"(?<![A-Za-z])[A-Z]{3,}(?![a-z])")
# harm language, for auditing the supposedly harm-free cells
HARM_WORDS = re.compile(
    r"\b(dies?|died|dying|death|dead|kill(?:s|ed|ing)?|poison\w*|drown\w*|"
    r"chokes? to death|fatal\w*|coma|hospital|injur\w*|burn(?:s|ed|ing)?|"
    r"third degree|paralys\w*|suffocat\w*)\b", re.I)

# Acronyms that legitimately appear in the source vignettes. Anything else that trips the
# ALLCAPS check is a scenario tag and therefore contamination.
ACRONYM_ALLOWLIST = {"TV", "CPR", "ID", "US", "UK", "DVD", "SUV", "ATM", "CD"}


class Report:
    def __init__(self):
        self.failures = []
        self.warnings = []

    def check(self, ok, name, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not ok:
            self.failures.append(f"{name}: {detail}")

    def warn(self, name, detail=""):
        print(f"  [WARN] {name}" + (f" -- {detail}" if detail else ""))
        self.warnings.append(f"{name}: {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"))
    ap.add_argument("--offsets", default=os.path.join(ROOT, "dataset", "master", "clause_offsets.csv"))
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    R = Report()
    print(f"=== validating {a.csv} ===\n")

    # 1 -- trailing response-prompt fragments
    bad = [r for r in rows if PROMPT_TAIL.search(r["text"])]
    R.check(not bad, "1. no trailing response-prompt fragment",
            f"{len(bad)} rows: {[r['story_id'] for r in bad][:6]}")

    # 2 -- ALLCAPS scenario tags
    caps = defaultdict(list)
    for r in rows:
        for tok in ALLCAPS.findall(r["text"]):
            if tok not in ACRONYM_ALLOWLIST:
                caps[tok].append(r["story_id"])
    R.check(not caps, "2. no ALLCAPS scenario tags",
            f"{len(caps)} distinct: {dict(list(caps.items())[:5])}")

    # 3 -- row and cell counts
    R.check(len(rows) == EXPECTED_N, "3a. row count", f"{len(rows)} (expected {EXPECTED_N})")
    cells = Counter(r["condition"] for r in rows)
    R.check(dict(cells) == EXPECTED_CELLS, "3b. cell counts", f"{dict(cells)}")

    # 4 -- terminal punctuation
    noterm = [r["story_id"] for r in rows if not r["text"].rstrip().endswith((".", "!", "?", "”", '"'))]
    R.check(not noterm, "4. ends in terminal punctuation", f"{len(noterm)}: {noterm[:6]}")

    # 5 -- word_count matches text
    mism = [(r["story_id"], r["word_count"], len(r["text"].split()))
            for r in rows if int(r["word_count"]) != len(r["text"].split())]
    R.check(not mism, "5. word_count matches len(text.split())",
            f"{len(mism)} rows, e.g. {mism[:3]}")

    # 6 -- word-count distribution
    wc = [len(r["text"].split()) for r in rows]
    mu, sd = statistics.mean(wc), statistics.pstdev(wc)
    out = [(r["story_id"], len(r["text"].split())) for r in rows
           if abs(len(r["text"].split()) - mu) > 2.5 * sd]
    print(f"  [INFO] 6. word_count min={min(wc)} median={statistics.median(wc)} "
          f"max={max(wc)} mean={mu:.1f} sd={sd:.1f}")
    if out:
        R.warn("6. word-count outliers >2.5SD", f"{len(out)}: {out[:8]}")

    # 7 -- one protagonist per story
    STOP = {"The","A","An","He","She","It","They","His","Her","When","After","Before","This",
            "That","There","But","And","If","As","In","On","At","To","For","One","Soon","Later",
            "While","Today","Now","Then","Since","Because","So","Not","No","Yes","During","Both",
            "You","Your","Imagine","Without","Thus","Later","Meanwhile","Everything","Unknown"}
    # Collapse runs of capitalised tokens into one entity so multi-word place names
    # ("Colorado River", "Safety Town", "Logan Airport") count once rather than as extra people.
    multi = []
    for r in rows:
        ents = set()
        for phrase in re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", r["text"]):
            toks = [t for t in phrase.split() if t not in STOP]
            if toks:
                ents.add(" ".join(toks))
        # a story may legitimately name the protagonist plus one other party plus a place
        if len(ents) > 3:
            multi.append((r["story_id"], sorted(ents)))
    R.check(not multi, "7. at most 3 distinct proper-noun entities",
            f"{len(multi)} rows, e.g. {multi[:3]}")

    # 8 -- outcome clause ends at the story end
    if os.path.exists(a.offsets):
        offs = {r["story_id"]: r for r in csv.DictReader(open(a.offsets))}
        byid = {r["story_id"]: r for r in rows}
        far = []
        for sid, o in offs.items():
            if sid in byid:
                d = len(byid[sid]["text"].rstrip()) - int(o["outcome_end"])
                if abs(d) > 5:
                    far.append((sid, d))
        R.check(not far, "8. outcome_end within 5 chars of text end",
                f"{len(far)} rows, e.g. {far[:5]}")
        print(f"  [INFO] 8b. clause method distribution: "
              f"{dict(Counter(o['method'] for o in offs.values()))}")
    else:
        R.warn("8. clause offsets missing", a.offsets)

    # 9 -- harm language in the supposedly harm-free cells (A5 audit)
    print("\n=== A5 audit: harm language in no-harm cells ===")
    flagged = []
    for r in rows:
        if r["condition"] in ("attempted", "neutral"):
            hits = set(m.group(0).lower() for m in HARM_WORDS.finditer(r["text"]))
            if hits:
                flagged.append((r["story_id"], r["condition"], sorted(hits)))
    if flagged:
        R.warn("9. harm language in attempted/neutral", f"{len(flagged)} rows")
        for sid, cond, hits in flagged[:15]:
            print(f"     {sid:34} [{cond:10}] {hits}")
    else:
        print("  [PASS] 9. no harm language in attempted/neutral cells")

    # 10 -- contamination/outcome alignment, the reason this bug mattered
    flag = [bool(PROMPT_TAIL.search(r["text"])) for r in rows]
    harm = [r["outcome_label"] == "harm" for r in rows]
    agree = sum(f == h for f, h in zip(flag, harm)) / len(rows)
    acc = max(agree, 1 - agree)
    base = max(sum(harm) / len(rows), 1 - sum(harm) / len(rows))
    R.check(acc <= base + 0.02, "10. contamination flag no longer predicts outcome",
            f"acc={acc:.3f} vs chance {base:.3f}")

    # ---- manual sample, stratified, seed reported -------------------------
    print(f"\n=== A7 manual verification sample (seed={a.seed}) ===")
    random.seed(a.seed)
    bycell = defaultdict(list)
    for r in rows:
        bycell[(r["condition"], r["source"])].append(r)
    picks, keys = [], sorted(bycell)
    while len(picks) < min(a.sample, len(rows)):
        for k in keys:
            if bycell[k] and len(picks) < a.sample:
                picks.append(bycell[k].pop(random.randrange(len(bycell[k]))))
    for r in picks:
        print(f"\n--- {r['story_id']}  [{r['condition']} | {r['source']} | "
              f"intent={r['intent_label']} outcome={r['outcome_label']} | "
              f"{len(r['text'].split())}w]")
        print(f"    {r['text']}")

    print("\n" + "=" * 70)
    if R.failures:
        print(f"VALIDATION FAILED -- {len(R.failures)} check(s):")
        for f in R.failures:
            print("  -", f)
    else:
        print("ALL CHECKS PASSED")
    if R.warnings:
        print(f"\n{len(R.warnings)} warning(s):")
        for w in R.warnings:
            print("  -", w)
    sys.exit(1 if R.failures else 0)


if __name__ == "__main__":
    main()
