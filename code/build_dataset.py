#!/usr/bin/env python3
"""
build_dataset.py
Assemble the canonical Young & Saxe intent x outcome (belief x outcome) 2x2 moral
vignettes into a single master CSV for LLM experiments.

Sources (all from saxelab.mit.edu, publicly posted):
  - Young & Saxe 2008  (Neuroimage)  : belief-encoding 2x2, "Grace/coffee" paradigm
  - Young & Saxe 2009  (Neuropsychologia): forgiveness 2x2 (accidental-harm)
  - Young & Saxe 2011  (Cognition)   : Intentional vs Accidental violations (intent contrast)

Design (2008 / 2009): each scenario provides 3 A/B sentence pairs in fixed order
  pair 1 = Foreshadow  (the true state of the world -> tracks OUTCOME)
  pair 2 = Belief      (the protagonist's belief     -> tracks INTENT)
  pair 3 = Action      (what happens                 -> tracks OUTCOME)
A = neutral/innocent/no-harm,  B = negative/guilty/harm.

We cross Belief (intent) x Action (outcome) into the 4 canonical cells:
  neutral     : Fore_A + Belief_A + Action_A   (innocent intent, no harm)
  accidental  : Fore_B + Belief_A + Action_B   (innocent intent, harm)   <- accidental harm
  attempted   : Fore_A + Belief_B + Action_A   (guilty intent,  no harm) <- attempted harm
  intentional : Fore_B + Belief_B + Action_B   (guilty intent,  harm)    <- intentional harm

NO MODEL TRAINING happens here. This only builds text stimuli.
"""
import re, csv, os, sys

RAW = os.path.join(os.path.dirname(__file__), "..", "dataset", "raw_text")
OUT = os.path.join(os.path.dirname(__file__), "..", "dataset", "master")
os.makedirs(OUT, exist_ok=True)

def clean(s):
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_noise(line):
    l = line.strip()
    if not l: return True
    if re.fullmatch(r"\d+", l): return True               # page numbers
    if l.startswith(("Young, L.", "Scenarios are organized",
                     "I. ", "II. ", "III. ", "IV. ", "V. ",
                     "Note:", "*Participant", "When the thought",
                     "Appendix", "Stimuli", "Experiments",
                     "Intentional & Accidental", "“How morally")):
        return True
    return False

def is_judgment(line):
    l = line.strip()
    return l.endswith("?") or l.endswith(":") or l.lower().startswith(
        ("how much", "putting the", "doing", "was:", "how morally"))

def parse_ab_factorial(text, source, named_headers):
    """Parse 2008/2009-style files into 4 cells per scenario."""
    lines = [l for l in text.split("\n")]
    # collect A./B. line indices with their text (may span multiple lines)
    rows = []
    # First, merge wrapped lines so each A./B. item is one logical line.
    logical = []
    buf = None
    for raw in lines:
        if is_noise(raw):
            if buf is not None:
                logical.append(buf); buf = None
            logical.append(("NOISE", ""))
            continue
        if re.match(r"^[AB]\.\s", raw):
            if buf is not None: logical.append(buf)
            buf = ["AB", raw.strip()]
        else:
            if buf is not None:
                buf[1] += " " + raw.strip()
            else:
                logical.append(("TEXT", raw.strip()))
    if buf is not None: logical.append(buf)

    # Walk logical stream, grouping every 6 AB items into a scenario.
    ab = []          # list of (label,text)
    pre_text = []    # text seen since last scenario (background candidates)
    scen = []
    scenarios = []
    pending_pre = []
    for kind, val in logical:
        if kind == "AB":
            ab.append(val)
            if len(ab) == 6:
                scenarios.append((pending_pre[:], ab[:]))
                ab = []
                pending_pre = []
        elif kind in ("TEXT",):
            pending_pre.append(val)
        else:  # NOISE
            pass
    out = []
    for idx, (pre, ab6) in enumerate(scenarios):
        # background = pre text, dropping judgment prompt of previous scenario & header
        bg_lines = [p for p in pre if not is_judgment(p)]
        # scenario name: an ALLCAPS line if present
        name = None
        bg_keep = []
        for p in bg_lines:
            if named_headers and re.fullmatch(r"[A-Z][A-Z0-9 &/\-]{2,}", p.strip()):
                name = p.strip()
            else:
                bg_keep.append(p)
        background = clean(" ".join(bg_keep))
        # strip A./B. prefixes
        items = [re.sub(r"^[AB]\.\s*", "", x) for x in ab6]
        fore_A, fore_B, bel_A, bel_B, act_A, act_B = items
        sid = name if name else f"{source}_{idx+1:02d}"
        cells = {
            "neutral":     (fore_A, bel_A, act_A, "innocent", "no_harm"),
            "accidental":  (fore_B, bel_A, act_B, "innocent", "harm"),
            "attempted":   (fore_A, bel_B, act_A, "guilty",   "no_harm"),
            "intentional": (fore_B, bel_B, act_B, "guilty",   "harm"),
        }
        for cond, (fo, be, ac, intent, outcome) in cells.items():
            vig = clean(f"{background} {fo} {be} {ac}")
            out.append(dict(source=source, scenario_id=sid, condition=cond,
                            intent_label=intent, outcome_label=outcome,
                            text=vig, word_count=len(vig.split())))
    return out

def parse_2011(text):
    """Intentional vs Accidental matched pairs (intent contrast, harmful outcome in both)."""
    out = []
    marker = re.compile(r"([A-Z][a-z]+)\s*-\s*(Intentional|Accidental):", re.M)
    hits = list(marker.finditer(text))
    for i, m in enumerate(hits):
        name, kind = m.group(1), m.group(2)
        start = m.end()
        end = hits[i+1].start() if i+1 < len(hits) else len(text)
        body = text[start:end]
        # strip running footer / page numbers
        body = re.sub(r"\d*\s*When the thought counts less.*?assault", " ", body, flags=re.S)
        body = clean(body)
        out.append(dict(source="YS2011", scenario_id=name,
                        condition=("intentional" if kind=="Intentional" else "accidental"),
                        intent_label=("guilty" if kind=="Intentional" else "innocent"),
                        outcome_label="harm",
                        text=body, word_count=len(body.split())))
    return out

def main():
    all_rows = []
    t08 = open(os.path.join(RAW, "YS2008.txt")).read()
    t09 = open(os.path.join(RAW, "YS2009.txt")).read()
    t11 = open(os.path.join(RAW, "YS2011.txt")).read()
    # Drop the citation/instruction preamble before the first real scenario so it
    # does not contaminate the first scenario's background.
    if "COFFEE" in t08: t08 = t08[t08.index("COFFEE"):]
    if "Matt is babysitting" in t09: t09 = t09[t09.index("Matt is babysitting"):]
    # Defensive: strip any residual running-footer citation fragments.
    t08 = t08.replace("moral judgment. Neuroimage, 40(4), 1912-1920.", " ")
    all_rows += parse_ab_factorial(t08, "YS2008", named_headers=True)
    all_rows += parse_ab_factorial(t09, "YS2009", named_headers=False)
    # 2011 is an intent-only contrast (outcome held constant) and the appendix
    # repeats items across experiments; dedupe and keep only complete int/acc pairs.
    raw11 = parse_2011(t11)
    seen, by_scen = set(), {}
    for r in raw11:
        key = (r["scenario_id"], r["condition"])
        if key in seen: continue
        seen.add(key)
        by_scen.setdefault(r["scenario_id"], []).append(r)
    for scen, items in by_scen.items():
        if len(items) == 2:           # has both intentional & accidental
            all_rows += items

    # add a global id
    for i, r in enumerate(all_rows):
        r["story_id"] = f"{r['source']}-{r['scenario_id']}-{r['condition']}"
    fields = ["story_id","source","scenario_id","condition",
              "intent_label","outcome_label","word_count","text"]
    path = os.path.join(OUT, "moral_2x2_master.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in fields})
    print(f"Wrote {len(all_rows)} rows -> {path}")
    # quick summary
    from collections import Counter
    print("By source:", Counter(r["source"] for r in all_rows))
    print("By condition:", Counter(r["condition"] for r in all_rows))
    print("By intent:", Counter(r["intent_label"] for r in all_rows))
    print("By outcome:", Counter(r["outcome_label"] for r in all_rows))

if __name__ == "__main__":
    main()
