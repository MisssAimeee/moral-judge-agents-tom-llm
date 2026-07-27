#!/usr/bin/env python3
"""Hand-annotate all 10 YS2011 clause spans (method=manual). Eye-verified substrings."""
import csv, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "dataset/master/moral_2x2_master.csv"
OFF = ROOT / "dataset/master/clause_offsets.csv"

# belief, action, outcome — substrings matched against repaired master text
# (apostrophe / quote characters are resolved against the live text below)
RAW = {
    "YS2011-Allergy-intentional": (
        "You know she is allergic to peanuts.",
        "You grind up the peanuts, add them in, and serve your cousin.",
        "You grind up the peanuts, add them in, and serve your cousin.",
    ),
    "YS2011-Allergy-accidental": (
        "You have no idea she is allergic to peanuts.",
        "You grind up the peanuts, add them in, and serve your cousin.",
        "You grind up the peanuts, add them in, and serve your cousin.",
    ),
    "YS2011-Poison-intentional": (
        "You know full well that you are putting poison in her coffee.",
        "You spoon some powder into your co-worker's coffee.",
        "You know full well that you are putting poison in her coffee.",
    ),
    "YS2011-Poison-accidental": (
        "You do not know that someone replaced the sugar with poison.",
        "You spoon some powder into your co-worker's coffee.",
        "You do not know that someone replaced the sugar with poison.",
    ),
    "YS2011-Sibling-intentional": (
        "That night, you end up talking, and you discover that you are actually long lost siblings.",
        "At the end of the night, you decide to sleep together (using two forms of birth control to be safe).",
        "At the end of the night, you decide to sleep together (using two forms of birth control to be safe).",
    ),
    "YS2011-Sibling-accidental": (
        "The next day, you end up talking, and you discover that you are actually long lost siblings.",
        "At the end of the night, you decide to sleep together (using two forms of birth control to be safe).",
        "The next day, you end up talking, and you discover that you are actually long lost siblings.",
    ),
    "YS2011-Parent-intentional": (
        "You discover that this person is actually your biological parent.",
        "You two get along really well and, that night, end up sleeping together (using two forms of birth control to be safe).",
        "You two get along really well and, that night, end up sleeping together (using two forms of birth control to be safe).",
    ),
    "YS2011-Parent-accidental": (
        "You have no idea that this person is actually your biological parent.",
        "You two get along really well and, that night, end up sleeping together (using two forms of birth control to be safe).",
        "You two get along really well and, that night, end up sleeping together (using two forms of birth control to be safe).",
    ),
    "YS2011-Dog-intentional": (
        'It is correctly labeled "dog".',
        "You end up eating your dog for dinner.",
        "You end up eating your dog for dinner.",
    ),
    "YS2011-Dog-accidental": (
        'It is incorrectly labeled "beef" – but it is actually the meat from your dog.',
        "Without realizing it, you end up eating your dog for dinner.",
        "Without realizing it, you end up eating your dog for dinner.",
    ),
}


def find_span(text, pat):
    if pat in text:
        i = text.index(pat)
        return i, i + len(pat)
    # apostrophe / quote variants
    variants = [pat]
    for a, b in [("'", "\u2019"), ("'", "\u2018"), ('"', "\u201c"), ('"', "\u201d"),
                 ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'), ("–", "-"), ("-", "–")]:
        variants.append(pat.replace(a, b))
    for v in variants:
        if v in text:
            i = text.index(v)
            return i, i + len(v)
    if "labeled" in pat:
        m = re.search(r"It is (?:correctly|incorrectly) labeled .+?\.", text)
        if m:
            return m.start(), m.end()
    raise ValueError(f"not found: {pat!r}\nTEXT: {text!r}")


def main():
    rows = {r["story_id"]: r for r in csv.DictReader(open(MASTER))}
    manual = {}
    print("=== YS2011 manual annotation (eye check) ===")
    for sid, (b, a, o) in RAW.items():
        t = rows[sid]["text"]
        bs, be = find_span(t, b)
        as_, ae = find_span(t, a)
        os_, oe = find_span(t, o)
        manual[sid] = dict(
            story_id=sid, belief_start=bs, belief_end=be,
            action_start=as_, action_end=ae,
            outcome_start=os_, outcome_end=oe,
            method="manual",
            n_sentences=len(re.findall(r"[.!?]+", t)),
        )
        print(f"\n## {sid}")
        print(f"  BELIEF : {t[bs:be]!r}")
        print(f"  ACTION : {t[as_:ae]!r}")
        print(f"  OUTCOME: {t[os_:oe]!r}")
        d = len(t.rstrip()) - oe
        print(f"  delta(len-outcome_end)={d}")
        assert abs(d) <= 5, (sid, d)

    offs = list(csv.DictReader(open(OFF)))
    cols = list(offs[0].keys())
    out = [manual[r["story_id"]] if r["story_id"] in manual else r for r in offs]
    with open(OFF, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print("\nmethods:", dict(Counter(r["method"] for r in out)))
    print("wrote", OFF)


if __name__ == "__main__":
    main()
