#!/usr/bin/env python3
"""
25_annotate_clauses.py -- Phase 3 / Task P1: locate belief / action / outcome clauses.

WHY
---
01_extract_activations.py stores one vector per story (final token, or the mean over all
tokens). That is a blended summary taken after the model has read everything, so we can say
"intent is decodable at layer 18" but not WHEN the model built that representation.

The belief clause is the payoff. At the belief-clause position **the harm has not yet been
mentioned in the text**, so if intent decodes there, harm words cannot possibly explain it.
That is the cleanest available disconfirmation of the lexical account.

Structure of a typical stimulus:

    ... The container is labeled "toxic", so Grace believes that the white powder is a
    toxic substance left behind by a scientist.        <- BELIEF
    Grace puts the substance in her friend's coffee.   <- ACTION
    Her friend drinks the coffee and dies.             <- OUTCOME

Emits CHARACTER offsets only. Token offsets differ per tokenizer and are resolved at
extraction time, not precomputed globally.

Output
  dataset/master/clause_offsets.csv
    story_id, belief_start, belief_end, action_start, action_end,
    outcome_start, outcome_end, method, n_sentences
"""
import os, csv, re, argparse, random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_CSV = os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv")

BELIEF_VERB = re.compile(r"\b(believes?|thinks?|knows?|realiz\w+|awares?)\b", re.I)
# sentence splitter that does not break on the abbreviations present in these stimuli
SENT_END = re.compile(r'(?<=[.!?])["\u201d\')\]]*\s+')


def split_sentences(text):
    """-> [(start, end, sentence_text)] over the ORIGINAL string, offsets preserved."""
    spans, pos = [], 0
    for m in SENT_END.finditer(text):
        end = m.start() + 1 if text[m.start()] in '.!?' else m.start()
        seg = text[pos:m.start() + 1].strip()
        if seg:
            s = text.index(seg[0], pos)
            spans.append((s, pos + len(text[pos:m.start() + 1].rstrip()), seg))
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        spans.append((text.index(tail[0], pos) if tail[0] in text[pos:] else pos,
                      len(text.rstrip()), tail))
    return spans


def annotate(text):
    """-> dict of clause spans + the method used, or None if the story is unusable."""
    sents = split_sentences(text)
    if len(sents) < 2:
        return None

    belief_i = None
    for i, (_, _, s) in enumerate(sents):
        if BELIEF_VERB.search(s):
            belief_i = i          # last belief sentence: the operative one
    method = "belief_verb"

    if belief_i is None:
        # No explicit mental-state verb (15 stories). Fall back to position: the belief
        # slot in this template is the sentence immediately before the action. Flagged so
        # these can be excluded from the headline analysis.
        belief_i = max(0, len(sents) - 3)
        method = "fallback_position"

    outcome_i = len(sents) - 1
    # action sits between belief and outcome; if they are adjacent there is no separate
    # action sentence and we mark it degenerate rather than inventing one
    if outcome_i - belief_i >= 2:
        action_i = belief_i + 1
    elif outcome_i - belief_i == 1:
        action_i = outcome_i
        method += "+action_eq_outcome"
    else:
        return None

    return {
        "belief_start": sents[belief_i][0], "belief_end": sents[belief_i][1],
        "action_start": sents[action_i][0], "action_end": sents[action_i][1],
        "outcome_start": sents[outcome_i][0], "outcome_end": sents[outcome_i][1],
        "method": method, "n_sentences": len(sents),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=os.path.join(ROOT, "dataset", "master", "clause_offsets.csv"))
    ap.add_argument("--sample", type=int, default=20, help="items to print for manual check")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    out, failed = [], []
    for r in rows:
        ann = annotate(r["text"])
        if ann is None:
            failed.append(r["story_id"])
            continue
        out.append({"story_id": r["story_id"], **ann})

    from collections import Counter
    print(f"stories: {len(rows)}   annotated: {len(out)}   failed: {len(failed)}")
    print(f"regex hit rate: {100*sum(1 for o in out if o['method'].startswith('belief_verb'))/len(rows):.1f}%")
    print("methods:", dict(Counter(o["method"] for o in out)))
    if failed:
        print(f"FAILED (flagged, not silently dropped): {failed}")

    # sanity: does the harm word appear BEFORE the belief clause ends? if so the
    # "no harm mentioned yet" premise breaks for that item
    HARM = re.compile(r"\b(dies?|died|death|kill\w*|poison\w*|injur\w*|hurt|harm\w*|burn\w*|drown\w*)\b", re.I)
    early = 0
    by_id = {r["story_id"]: r for r in rows}
    for o in out:
        pre = by_id[o["story_id"]]["text"][: o["belief_end"]]
        if HARM.search(pre):
            early += 1
    print(f"\nstories where a harm word appears at/before the belief clause: {early} "
          f"({100*early/max(len(out),1):.0f}%)")
    print("  (the belief-clause probe is only confound-free for the remainder; "
          "the analysis must report this subset separately)")

    random.seed(0)
    print(f"\n=== manual verification sample ({a.sample}) ===")
    for o in random.sample(out, min(a.sample, len(out))):
        t = by_id[o["story_id"]]["text"]
        print(f"\n## {o['story_id']}  [{o['method']}]")
        print(f"  BELIEF : {t[o['belief_start']:o['belief_end']][:150]!r}")
        print(f"  ACTION : {t[o['action_start']:o['action_end']][:110]!r}")
        print(f"  OUTCOME: {t[o['outcome_start']:o['outcome_end']][:110]!r}")

    if a.write:
        cols = ["story_id", "belief_start", "belief_end", "action_start", "action_end",
                "outcome_start", "outcome_end", "method", "n_sentences"]
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(out)
        print(f"\n-> {a.out}")
    else:
        print("\n(report only -- rerun with --write to save)")


if __name__ == "__main__":
    main()
