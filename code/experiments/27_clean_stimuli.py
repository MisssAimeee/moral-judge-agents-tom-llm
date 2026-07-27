#!/usr/bin/env python3
"""
27_clean_stimuli.py -- remove rating-prompt / next-scenario bleed from the stimulus texts.

WHY THIS EXISTS
---------------
144 of the 298 stories in dataset/master/moral_2x2_master.csv carry trailing text that is
not part of the story: the study's rating prompt ("Putting the substance in was:") followed
by the opening sentences of the NEXT scenario ("LAB Dan is giving a visitor a tour...").
It is an artefact of parsing the source PDFs by paragraph.

It is not cosmetic, because the contamination is almost perfectly CONFOUNDED WITH OUTCOME:

    no_harm  cells:   0 / 144 contaminated
    harm     cells: 144 / 154 contaminated
    -> a single binary "has trailing junk" flag predicts outcome at 0.966 accuracy

So any probe that decodes outcome may simply be detecting the presence of an appended
fragment. The intent factor is unaffected (48% of innocent and 48% of guilty items are
contaminated), so intent results are not threatened by this particular artefact -- but
intent probes restricted to the harm cells ARE, because there the extra ~180 characters
dilute the story and, under last-token pooling, the final token belongs to a different
scenario entirely.

WHAT IT DOES
------------
Truncates each story at the first rating-prompt marker:
  * "<clause> was:" / "were:" / "is:" / "are:"   (Young & Saxe 2008/2011 style)
  * "How much blame/punishment does X deserve ...?"  (Young & Saxe 2009 style)

Writes a cleaned copy and leaves the original untouched, then verifies the confound is gone.

OBSOLETE AS OF 2026-07-26. The builder itself was fixed (`code/build_dataset.py`) and
`dataset/master/moral_2x2_master.csv` is now the single canonical master. Do not regenerate
`moral_2x2_master_clean.csv` — truncation cannot restore backgrounds that the parser
deleted, and a second CSV is how the chain got forked. Kept only as provenance for the
0.966 confound calculation documented in CONTAMINATION_REPAIR.md §1.1.

Usage
  python code/experiments/27_clean_stimuli.py            # report only
  python code/experiments/27_clean_stimuli.py --write    # write the cleaned sibling CSV
"""
import os, csv, re, argparse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
MASTER = os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv")

# "Putting the substance in was:" / "Letting the child on the chair lift was:"
STUB = re.compile(r"\b[A-Za-z][^.!?]*?\b(?:was|were|is|are):")
# "How much blame does Matt deserve for just watching...?"
BLAME = re.compile(r"How much (?:blame|punishment)[^?]*\?")


def cut_point(text):
    """Index at which the story proper ends, or len(text) if it is already clean."""
    hits = [m.start() for m in (STUB.search(text), BLAME.search(text)) if m]
    return min(hits) if hits else len(text)


def clean(text):
    return text[:cut_point(text)].strip()


def contaminated(text):
    return cut_point(text) < len(text)


def confound_strength(rows, textkey="text"):
    """Accuracy of predicting outcome from the contamination flag alone."""
    flag = [contaminated(r[textkey]) for r in rows]
    harm = [r["outcome_label"] == "harm" for r in rows]
    agree = sum(f == h for f, h in zip(flag, harm)) / len(rows)
    return max(agree, 1 - agree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=MASTER)
    ap.add_argument("--out", default=MASTER.replace(".csv", "_clean.csv"),
                    help="cleaned copy; the input master is never modified")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    fields = list(rows[0].keys())

    n_bad = sum(contaminated(r["text"]) for r in rows)
    print(f"stories: {len(rows)}   contaminated: {n_bad} ({100*n_bad/len(rows):.0f}%)")
    print(f"outcome predictable from contamination flag alone: "
          f"{confound_strength(rows):.3f}\n")

    by_cond = Counter((r["condition"], contaminated(r["text"])) for r in rows)
    for c in ("neutral", "attempted", "accidental", "intentional"):
        print(f"  {c:12} contaminated={by_cond[(c,True)]:3}  clean={by_cond[(c,False)]:3}")

    removed, new_rows = [], []
    for r in rows:
        old = r["text"]
        new = clean(old)
        if len(new) < 40:                      # never let a cut destroy a story
            print(f"  !! {r['story_id']}: cut would leave {len(new)} chars, keeping original")
            new = old
        if new != old:
            removed.append(len(old) - len(new))
        nr = dict(r)
        nr["text"] = new
        nr["word_count"] = len(new.split())
        new_rows.append(nr)

    print(f"\ntruncated {len(removed)} stories, "
          f"mean {sum(removed)/max(len(removed),1):.0f} chars removed, max {max(removed or [0])}")
    wc = [int(r["word_count"]) for r in new_rows]
    print(f"word_count after: min={min(wc)} median={sorted(wc)[len(wc)//2]} max={max(wc)}")

    still = sum(contaminated(r["text"]) for r in new_rows)
    print(f"\nverification: still contaminated = {still}")
    print(f"verification: outcome-from-flag accuracy now = {confound_strength(new_rows):.3f} "
          f"(chance is {max(sum(r['outcome_label']=='harm' for r in new_rows)/len(new_rows), 1-sum(r['outcome_label']=='harm' for r in new_rows)/len(new_rows)):.3f})")

    print("\n--- example ---")
    ex = next(r for r in rows if contaminated(r["text"]))
    print(f"{ex['story_id']}\nBEFORE (tail): ...{ex['text'][-150:]!r}")
    print(f"AFTER  (tail): ...{clean(ex['text'])[-150:]!r}")

    if not a.write:
        print("\n(report only -- rerun with --write to save)")
        return

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(new_rows)
    print(f"\ncleaned copy written -> {a.out}")
    print(f"original left untouched -> {a.csv}")
    print("\nEverything downstream must be regenerated against --csv " + os.path.basename(a.out) +
          ": activations, probes, surface baselines, within-cell, RSA. Behavioural scores "
          "were also collected on the contaminated text.")


if __name__ == "__main__":
    main()
