#!/usr/bin/env python3
"""
30_polarity_audit.py -- systematic fore/act polarity audit over all scenario_groups.

Background: YS2008 AB factorial vignettes combine fore_A/B × bel_A/B × act_A/B.
CPR inverted act_A/act_B in the source; an early fix wrongly swapped fore+act;
correct fix is act-only. This script sweeps ALL groups for the same class of
label/text incoherence, not just CPR.

Checks per scenario_group (and per source within a group when reprints exist):
  1. outcome_label vs harm language in the final sentence
     (HARM_WORDS from 28_validate_master ∪ HARM_OUTCOME from build_dataset,
      plus clear outcome-harm terms; strong no-harm phrases override)
  2. Within a full 2x2: accidental vs intentional share action+outcome (belief differs);
     neutral vs attempted share action+outcome (belief differs);
     accidental vs neutral share belief (outcome differs);
     intentional vs attempted share belief (outcome differs)
  3. Belief sentences form the expected crossed pattern
  4. Harm-cell last sentences match each other; no-harm pair likewise

Writes outputs/POLARITY_AUDIT.md with a sweep table over all 53 groups.

Usage
  python code/experiments/30_polarity_audit.py
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# 28_validate_master lexicon
HARM_WORDS_28 = (
    r"dies?|died|dying|death|dead|kill(?:s|ed|ing)?|poison\w*|drown\w*|"
    r"chokes? to death|fatal\w*|coma|hospital(?:ized)?|injur\w*|burn(?:s|ed|ing)?|"
    r"third degree|paralys\w*|suffocat\w*"
)
# build_dataset.HARM_OUTCOME extras
HARM_WORDS_BUILD = (
    r"never wakes up|emergency room|contract diseases|contract(?:s|ed)?"
)
# Additional last-sentence outcome harms seen across the master (not foreshadow)
HARM_WORDS_EXTRA = (
    r"plummets?|gasping|seizure|concussion|malaria|rabies|parasite|"
    r"breaks?(?: his| her| their)? (?:neck|leg|hip|legs)|"
    r"broken (?:floorboards|hip|legs|neck)|"
    r"shot a man|hit by a|falling \d+|falls? (?:\d+|hard|50)|"
    r"crash(?:es|ed)? into|blacks? out|stops? breathing|goes into shock|"
    r"convuls\w*|asthma attack|internal b\w*|car hits|gets hit|"
    r"pass(?:es)? out|carbon monoxide|fire starts|catches on fire|"
    r"killed|to her death|to his death|to their death|"
    r"Herpes|infectious disease|nasty strain|severe|"
    r"bikes off a cliff|can.?t brake|gives way"
)
HARM_WORDS = re.compile(
    rf"\b({HARM_WORDS_28}|{HARM_WORDS_BUILD}|{HARM_WORDS_EXTRA})\b", re.I
)

# Strong no-harm endings — override weak/ambiguous harm hits (e.g. "shot" a stag)
NO_HARM = re.compile(
    r"\b("
    r"is fine|are fine|just fine|totally fine|will be fine|"
    r"safely|safe ride|without incident|wonderful time|great time|"
    r"important work|enjoys the|comes out smiling|healthy and delicious|"
    r"quite full|have fun|tasty|bring home the stag|doesn.?t act up|"
    r"returns laughing|learned what to do|splashing around|"
    r"makeovers|pretending|nap|wants more porridge"
    r")\b",
    re.I,
)

# Belief clause markers (LAPTOP uses "believing"; LOGAN uses "thinks")
BELIEF_RE = re.compile(
    r"\b(believ(?:es|ing|ed)?|thinks?|thought|suspects?)\b", re.I
)

FULL_2X2 = {"neutral", "accidental", "attempted", "intentional"}


def norm_text(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = re.sub(r"\s+", " ", s)
    return s


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def last_sentence(text: str) -> str:
    sents = split_sentences(text)
    return sents[-1] if sents else text.strip()


def last_n_sents(text: str, n: int = 2) -> str:
    sents = split_sentences(text)
    return " ".join(sents[-n:]) if sents else text.strip()


def text_outcome_harm(text: str) -> bool | None:
    """True/False if lexicon decides; None if ambiguous (do not fail on label alone)."""
    ls = last_sentence(text)
    harm_hit = bool(HARM_WORDS.search(ls))
    safe_hit = bool(NO_HARM.search(ls))
    if safe_hit and not harm_hit:
        return False
    if harm_hit and safe_hit:
        # "bring home the stag they have just shot" — safety phrase wins
        if re.search(
            r"\b(safely|is fine|are fine|just fine|without incident|safe ride|"
            r"bring home the stag|wonderful time|great time|important work)\b",
            ls,
            re.I,
        ):
            return False
        return True
    if harm_hit:
        return True
    if safe_hit:
        return False
    return None


def belief_sentence(text: str) -> str | None:
    for s in split_sentences(text):
        if BELIEF_RE.search(s):
            return s
    return None


def action_outcome_span(text: str) -> str:
    """Prefer sentences after the belief clause; else last two sentences."""
    sents = split_sentences(text)
    for i, s in enumerate(sents):
        if BELIEF_RE.search(s):
            rest = " ".join(sents[i + 1 :]).strip()
            if rest:
                return rest
            break
    return last_n_sents(text, 2)


def load_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit_source_family(cells: dict[str, dict]) -> tuple[bool, list[str]]:
    notes: list[str] = []
    conds = set(cells)

    # --- 1. label vs last-sentence harm (skip ambiguous) ---
    for cond, r in cells.items():
        labeled_harm = r["outcome_label"] == "harm"
        derived = text_outcome_harm(r["text"])
        if derived is None:
            continue
        if labeled_harm != derived:
            notes.append(
                f"{cond}: outcome_label={r['outcome_label']} but last-sentence "
                f"harm={derived} ({last_sentence(r['text'])[:90]!r})"
            )

    expect_intent = {
        "neutral": "innocent",
        "accidental": "innocent",
        "attempted": "guilty",
        "intentional": "guilty",
    }
    for cond, r in cells.items():
        if cond in expect_intent and r["intent_label"] != expect_intent[cond]:
            notes.append(
                f"{cond}: intent_label={r['intent_label']} expected {expect_intent[cond]}"
            )

    # YS2011 is intentional-vs-accidental matched pairs (both harm), NOT an AB
    # factorial with shared act_B — do not require ending identity there.
    if not FULL_2X2.issubset(conds):
        missing = sorted(FULL_2X2 - conds)
        notes.append(
            f"partial design (missing {missing}); skip act/belief 2x2 pattern checks"
        )
        # Contamination remnant check: last sentence should not be a bare title
        for cond, r in cells.items():
            ls = last_sentence(r["text"])
            if re.fullmatch(r"[A-Z][a-z]+\.?", ls):
                notes.append(
                    f"{cond}: last sentence looks like a glued scenario title ({ls!r})"
                )
        hard = [
            n for n in notes
            if not n.startswith("informational:")
            and not n.startswith("partial design")
        ]
        return (len(hard) == 0), notes

    # --- 2. action+outcome identity within outcome pairs ---
    act = {c: action_outcome_span(cells[c]["text"]) for c in FULL_2X2}
    # Compare last sentences primarily (robust to foreshadow leaking into span)
    last = {c: last_sentence(cells[c]["text"]) for c in FULL_2X2}
    if norm_text(last["accidental"]) != norm_text(last["intentional"]):
        notes.append(
            "act polarity? accidental vs intentional endings differ "
            "(should share act_B / harm ending)"
        )
    if norm_text(last["neutral"]) != norm_text(last["attempted"]):
        notes.append(
            "act polarity? neutral vs attempted endings differ "
            "(should share act_A / no-harm ending)"
        )
    if norm_text(last["accidental"]) == norm_text(last["neutral"]):
        notes.append(
            "outcome polarity? harm and no-harm endings are identical"
        )

    # Also flag if full action spans diverge while endings match (fore leakage into act)
    if (
        norm_text(last["accidental"]) == norm_text(last["intentional"])
        and norm_text(act["accidental"]) != norm_text(act["intentional"])
    ):
        notes.append(
            "informational: accidental/intentional action spans differ before ending "
            "(often foreshadow glued into belief-adjacent text; endings match)"
        )

    # --- 3. belief crossed pattern ---
    bel = {c: belief_sentence(cells[c]["text"]) for c in FULL_2X2}
    if any(v is None for v in bel.values()):
        missing_bel = [c for c, v in bel.items() if v is None]
        notes.append(f"no belief/think sentence in: {missing_bel}")
    else:
        bel_n = {c: norm_text(v) for c, v in bel.items()}
        if bel_n["accidental"] != bel_n["neutral"]:
            notes.append(
                "belief pattern: accidental vs neutral beliefs differ "
                "(should share bel_A / innocent)"
            )
        if bel_n["intentional"] != bel_n["attempted"]:
            notes.append(
                "belief pattern: intentional vs attempted beliefs differ "
                "(should share bel_B / guilty)"
            )
        if bel_n["accidental"] == bel_n["intentional"]:
            notes.append(
                "belief polarity? accidental and intentional share the same belief "
                "(expected crossed / opposite intent)"
            )
        if bel_n["neutral"] == bel_n["attempted"]:
            notes.append(
                "belief polarity? neutral and attempted share the same belief "
                "(expected crossed / opposite intent)"
            )

    # Fail on hard issues; informational notes do not fail
    hard = [
        n for n in notes
        if not n.startswith("informational:")
        and not n.startswith("partial design")
    ]
    return (len(hard) == 0), notes


def audit_group(group: str, rows: list[dict]) -> dict:
    by_source: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_source[r["source"]][r["condition"]] = r

    all_notes: list[str] = []
    ok_all = True
    n_cells = len(rows)

    for src, cells in sorted(by_source.items()):
        ok, notes = audit_source_family(cells)
        if not ok:
            ok_all = False
        for n in notes:
            all_notes.append(f"[{src}] {n}")

    if "YS2008" in by_source and "YS2009" in by_source:
        for cond in FULL_2X2:
            if cond in by_source["YS2008"] and cond in by_source["YS2009"]:
                lab8 = (
                    by_source["YS2008"][cond]["intent_label"],
                    by_source["YS2008"][cond]["outcome_label"],
                )
                lab9 = (
                    by_source["YS2009"][cond]["intent_label"],
                    by_source["YS2009"][cond]["outcome_label"],
                )
                if lab8 != lab9:
                    ok_all = False
                    all_notes.append(
                        f"[reprint] {cond}: YS2008 labels {lab8} != YS2009 {lab9}"
                    )
                elif norm_text(by_source["YS2008"][cond]["text"]) != norm_text(
                    by_source["YS2009"][cond]["text"]
                ):
                    all_notes.append(
                        f"[reprint] {cond}: YS2008/YS2009 wording differs "
                        "(labels agree; informational)"
                    )

    return dict(
        group=group,
        n_cells=n_cells,
        sources=",".join(sorted(by_source)),
        status="PASS" if ok_all else "FAIL",
        notes="; ".join(all_notes) if all_notes else "",
    )


def write_report(results: list[dict], out_path: str) -> None:
    n = len(results)
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_pass = n - n_fail
    fails = [r for r in results if r["status"] == "FAIL"]

    lines = [
        "# Polarity audit — all scenario groups",
        "",
        "Systematic fore/act / belief×outcome coherence sweep over every "
        "`scenario_group` in `dataset/master/moral_2x2_master.csv`.",
        "",
        "## Method",
        "",
        "- Harm from **last sentence**, using `HARM_WORDS` from `28_validate_master.py` "
        "∪ `HARM_OUTCOME` from `build_dataset.py` ∪ clear outcome-harm terms; "
        "strong no-harm phrases (`is fine`, `safely`, …) override. Ambiguous endings "
        "do not fail on label alone — structural 2×2 checks still run.",
        "- Full 2×2: accidental↔intentional endings must match; neutral↔attempted "
        "endings must match; beliefs (believes/thinks/believing) must form the crossed "
        "pattern (innocent pair identical, guilty pair identical, innocent ≠ guilty).",
        "- Partial designs (YS2011 intentional/accidental pairs) skip act/belief 2×2 "
        "pattern checks (no shared act_B by design) but still check "
        "label↔last-sentence harm and glued-title contamination.",
        "- Reprint groups audited per source; cross-reprint label disagreements fail.",
        "",
        f"## Summary: **{n_fail} fail / {n} groups** ({n_pass} pass)",
        "",
    ]
    if fails:
        lines.append("### Failures")
        lines.append("")
        for r in fails:
            lines.append(f"- **{r['group']}** (n_cells={r['n_cells']}): {r['notes']}")
        lines.append("")
    else:
        lines.append(
            "No group failed hard checks. Label↔text harm and within-group "
            "act/belief polarity look coherent across the full sweep "
            "(including groups beyond CPR)."
        )
        lines.append("")

    lines.extend(
        [
            "## Sweep table",
            "",
            "| group | n_cells | sources | status | notes |",
            "|---|---:|---|---|---|",
        ]
    )
    for r in results:
        notes = r["notes"].replace("|", "\\|") if r["notes"] else "—"
        if len(notes) > 220:
            notes = notes[:217] + "..."
        lines.append(
            f"| {r['group']} | {r['n_cells']} | {r['sources']} | "
            f"{r['status']} | {notes} |"
        )
    lines.append("")
    lines.append(
        "*Generated by `code/experiments/30_polarity_audit.py`. This table covers "
        "all 53 groups — not only the historically known CPR act-polarity case.*"
    )
    lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv"),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(ROOT, "outputs", "POLARITY_AUDIT.md"),
    )
    a = ap.parse_args()

    rows = load_rows(a.csv)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        g = r.get("scenario_group") or r["scenario_id"]
        by_group[g].append(r)

    results = [audit_group(g, by_group[g]) for g in sorted(by_group)]
    write_report(results, a.out)

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"Polled {len(results)} scenario_groups → {n_fail} FAIL / {len(results)}")
    print(f"Wrote {a.out}")
    for r in results:
        if r["status"] == "FAIL":
            print(f"  FAIL {r['group']}: {r['notes'][:200]}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
