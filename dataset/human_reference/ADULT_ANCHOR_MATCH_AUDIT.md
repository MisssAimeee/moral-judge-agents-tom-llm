# Adult-anchor match audit

**Date:** 2026-08-03  
**Scope:** Text- and condition-match verification for human adult reference lines.  
**Rule:** No figure changes until this audit is filed. Model-vs-model figures are out of scope either way.

---

## Verdict (read this first)

| Question | Answer |
|---|---|
| Are the 10 YS2011 master vignettes the **literal** Young & Saxe (2011) appendix texts? | **Yes** (exact match after unicode/whitespace normalize). |
| Did YS2011 rate a full 2×2 including **attempted / no-harm** on those same harm vignettes? | **No.** Exp. 1 (the texts in the master) is **intentional vs accidental only**. |
| Can YS2011 support the project’s headline contrast `blame(attempted) − blame(accidental)` on matched items? | **No.** |
| Is there a confirmed **item-and-condition** match that can replace or item-match the Young et al. (2007) adult line for the headline contrast? | **No.** |
| Overlap of 2007 SI item list vs YS2008 (192) / YS2009 (96) master texts? | **None confirmed** (SI Text not recovered; main-text cues only). |

**Conditional next step taken:** keep the current Young et al. (2007) adult reference line exactly as is; **strengthen its caveat** everywhere it is described (this audit + human-reference README / methods / paper limitations / presentation README). Do **not** add an item-matched robustness figure for the headline contrast. Do **not** change any model-vs-model figures.

---

## 1. What is `dataset/raw_text/YS2011.txt`?

It is the **published appendix stimuli text** for Young & L. Saxe (2011), *When ignorance is no excuse: Different roles for intent across moral domains*, *Cognition*, 120(2), 202–214 — not a secondary summary.

Cross-check: the local PDF `Young_Saxe_stimuli_2011.pdf` begins with the same appendix block (“Appendix Stimuli / Experiments 1A, 2, and 3 / Intentional & Accidental Violations / Harm / Allergy - Intentional: …”). The `.txt` is a clean extraction of that appendix.

Contents of the appendix file:

1. **Experiments 1A / 2 / 3** — Intentional & Accidental violations; prompt “How morally wrong was the action?”  
   - Harm: Allergy, Poison  
   - Incest: Sibling, Parent  
   - Ingestion: Dog, Urine  
   - Plus Experiment 1B third-person variants (Allergy, Sibling only).
2. **Experiments 4 and 5** — “Failed Attempts”; prompt “How morally wrong was the decision to act?”  
   - Conditions labeled **False Belief / True Belief / Neutral** (not the 2007 2×2 cell names).  
   - Harm and incest examples only in the extracted text.

---

## 2. YS2011 master rows vs appendix (text match)

Master has **10** rows with `source=YS2011` (not 12): Allergy, Poison, Sibling, Parent, Dog × {intentional, accidental}.

**Omitted from master:** `Urine` intentional and accidental (present in the appendix under Ingestion).

### Pairwise comparison method

- Unicode NFKC + curly quotes/dashes → ASCII; collapse whitespace.  
- Compare each master `text` to the corresponding first-person Exp. 1A appendix block.

| story_id | Result |
|---|---|
| YS2011-Allergy-intentional | **Exact match** |
| YS2011-Allergy-accidental | **Exact match** |
| YS2011-Poison-intentional | **Exact match** |
| YS2011-Poison-accidental | **Exact match** |
| YS2011-Sibling-intentional | **Exact match** |
| YS2011-Sibling-accidental | **Exact match** |
| YS2011-Parent-intentional | **Exact match** |
| YS2011-Parent-accidental | **Exact match** |
| YS2011-Dog-intentional | **Exact match** |
| YS2011-Dog-accidental | **Exact match** |

**Conclusion:** these 10 are the **literal same vignette texts** Young & Saxe (2011) used in Exp. 1A (first person), not paraphrases. They are **not** the Exp. 1B third-person (“Sam”) variants, and they are **not** the Exp. 4/5 failed-attempt texts.

Fuzzy overlap of these 10 texts against all YS2008/YS2009 master texts is negligible (best SequenceMatcher ratios ≈ 0.05–0.07). Thematic kinship only: YS2011-Poison ≈ chemical-plant coffee powder; YS2008-COFFEE is a different, third-person 2×2 “Grace” vignette family.

---

## 3. Condition structure Young & Saxe (2011) actually rated

### Exp. 1 (matched texts in the master)

- Design: **2 (intentional vs accidental) × 3 (harm / incest / ingestion)** between-subjects (Exp. 1A).  
- Each participant saw **one** scenario version.  
- Scale: moral wrongness **1–7** (“not at all” → “very”).  
- Harm domain stories in that design: **Allergy** and **Poison** only.

**No attempted / no-harm cell in Exp. 1.**  
**No neutral (all-clear) cell in Exp. 1** corresponding to the project’s `neutral` cell.

### Exp. 4 / 5 (“Failed Attempts”)

- Separate experiments; appendix labels **False Belief / True Belief / Neutral**.  
- Fig. 1 caption in the paper: Exp. 4/5 add attempted violations as “true belief/failed act, false belief/completed act.”  
- Those vignette texts are **not** present as YS2011 rows in `moral_2x2_master.csv`.  
- Even if recovered, they are **not** the same factorial packaging as Young et al. (2007)’s belief×outcome 2×2 on the long fMRI stories.

### Implication for the project contrast

| Contrast | Supported by YS2011 on master-matched texts? |
|---|---|
| `intentional − accidental` (wrongness) | Domain-level human means exist (see §4). Texts match. **Not** the project headline. |
| `attempted − accidental` (headline) | **Not supported** — attempted not rated on these Exp. 1 texts. |
| Full 2×2 (neutral / accidental / attempted / intentional) | **Not supported.** |

---

## 4. Extractable YS2011 human cell means (informational only)

**Not used as a corpus-wide adult line. Not used to replace Young et al. (2007).**

### What is published (Exp. 1A, harm domain, collapsed across Allergy + Poison)

From Young & Saxe (2011), Results 2.2.1 (text means on the 1–7 wrongness scale):

| Condition | Raw mean (1–7 wrongness) | Notes |
|---|---|---|
| Intentional harm | **6.68** | Collapsed across the two harm stories |
| Accidental harm | **2.05** | Collapsed across the two harm stories |

Paper states no story main effect / no story×intent interaction for harm, so they collapse Allergy and Poison. **Per-story means are not reported.**

### Sample size

- Exp. 1A: N = 262 collected → **241** retained; ≈ equal n per cell of the 2×3 between-subjects design → on the order of **~40 participants per intent×domain cell** (harm intentional and harm accidental each pool two stories).  
- Harm-only ANOVA footnote reports F(1, 79) for the intent effect → consistent with ~80 harm judgments total across both intents (both stories).  
- Exp. 1B (third person; Allergy + Sibling only) reports accidental harm **2.09**, intentional harm **5.94** — different perspective; master texts are second-person, so Exp. 1A is the relevant match.

### Normalization identical to the pipeline

`code/03_behavioral.py`: YS2011 scale is `(1, 7, "wrongness")` with  
`norm = (raw − 1) / (7 − 1)` (higher wrongness → higher blame).

| Condition | Raw | `norm_blame` |
|---|---|---|
| intentional harm | 6.68 | **0.9467** |
| accidental harm | 2.05 | **0.1750** |
| intentional − accidental | — | **+0.7717** |

### Why this does **not** trigger the “item-matched robustness figure” branch

1. Headline metric needs **attempted**, which Exp. 1 does not provide on these texts.  
2. Published means are **domain-collapsed**, not per `story_id`.  
3. Design is **between-subjects, one judgment per person** — not the within-corpus 2×2 structure models are scored on.  
4. Audit instruction: do not extrapolate a corpus-wide adult line from this subset.

These numbers are recorded here so a future **intentional−accidental** robustness analysis could cite them without re-digitizing; they are **not** wired into any figure by this audit.

### Matched-item inventory (for the record)

| Quantity | Value |
|---|---|
| n master items with exact text match | **10** vignettes (5 scenarios × 2 conditions) |
| n of those in the **harm** domain | **4** (Allergy, Poison × intentional, accidental) |
| Human conditions with published means on those harm texts | intentional, accidental only |
| Human n (Exp. 1A retained) | 241 (between-subjects; ~40/cell order of magnitude) |
| Attempted / neutral cells on matched texts | **0** |

---

## 5. Young et al. (2007) item list vs YS2008 / YS2009

### What 2007 used (from the main paper)

Young, Cushman, Hauser & Saxe (2007), *PNAS*:

- **12 scenarios × 4 belief×outcome variants = 48 stories** (avg ~86 words).  
- Full text: **“see SI Text for full text of scenarios.”**  
- Behavioral adult cell means currently in `human_reference.csv` come from Exp. 1 Fig. 2 (permissibility 1–4; n=10 fMRI):  
  neutral 3.9, accidental 3.2, attempted 1.2, intentional 1.1 →  
  `norm_blame = 1 − (raw−1)/3` → 0.033, 0.267, 0.933, 0.967.  
- Headline human contrast: attempted − accidental = **+0.666 ≈ +0.67**.

### Was the SI item list recovered?

| Source tried | Result |
|---|---|
| Main PDF (`young_2007_pnas.pdf` / moralitylab copy) | Methods + Fig. 1 schematic; **SI Text not embedded** |
| PNAS / PMC SI PDF URLs | 403 / captcha / 404 from this environment |
| Local stimulus PDFs | 2008 / 2009 / 2011 stimuli present; **no 2007 SI PDF on disk** |

**Recoverable from the main text only (examples / cues), not a full item list:**

- Coffee / white powder / poison vs sugar (Fig. 1 scenario family)  
- “Drowning swimmer” as an example of foreshadowed harm  
- Mentions of bridge / ham appear in the extracted main-text PDF; **not** a complete scenario inventory

### Master corpus sizes

| Source | Rows in `moral_2x2_master.csv` | Scenario groups |
|---|---|---|
| YS2008 | 192 | 48 groups × 4 conditions |
| YS2009 | 96 | 24 groups × 4 conditions |

YS2008-COFFEE is the familiar chemical-plant sugar/poison 2×2 with protagonist **Grace** — same **paradigm family** as the 2007 Fig. 1 example, but **not** verified as the literal 2007 SI wording (SI unavailable). Wording, segmentation (4 cumulative screens in the scanner vs continuous prose in the master), and cast names may differ.

### Overlap finding

**None confirmed.**  
Without the 2007 SI Text, no story_id in YS2008/YS2009 can be asserted as a literal item match to the 2007 behavioral set. Thematic / paradigm continuity (belief×outcome moral luck stories; coffee-poison example) is real but is **not** an item match.

---

## 6. Decision under the audit’s conditional rule

### Branch A — genuine item-and-condition match for the headline contrast  
**Not met.** → Do **not** add an item-matched model-vs-human robustness figure; do **not** replace the adult line.

### Branch B — no confirmed match  
**Taken.**

1. Keep Young et al. (2007) adult cell means and the **+0.67** attempted−accidental reference **unchanged**.  
2. Label it plainly as a **reference-shape estimate from a different, smaller item set (12 scenarios / 48 stories; n=10)**, not a per-item match to the 298-row master corpus.  
3. Treat obtaining per-item (or at least same-item) adult norms on the exact master stimuli as a **top open item** for the paper (Saxe-lab / author norming request; see `outputs/MASTER_SUMMARY.md` “Per-item matched human ratings…” and `dataset/human_reference/README.md`).  
   Note: `NEXT_PHASE_PLAN.md` is referenced in older prompts but is **not present** in this checkout; the live tracker for that request is `MASTER_SUMMARY.md` + this audit.

### What must not change

- No model-vs-model figure edits.  
- No substitution of YS2011 intentional−accidental means into the adult ladder used for attempted−accidental.  
- No silent claim that “our vignettes were behaviorally rated by Young & Saxe (2011)” for the headline contrast — only that 10 short intentional/accidental appendix vignettes are literally included, under a different design.

---

## 7. Caveat language (canonical)

Use this (or a close paraphrase) wherever the adult line appears in talk, deck, or paper:

> The adult reference line is the Young, Cushman, Hauser & Saxe (2007) Exp. 1 cell means (n=10; permissibility 1–4), normalized to 0–1 blame. It is a **reference-shape estimate from a different, smaller item set** (12 scenarios × 4 conditions in SI Text), **not a per-item match** to the 298-item master corpus (YS2008/YS2009/YS2011). The corpus includes 10 literal Young & Saxe (2011) Exp. 1 intentional/accidental vignettes, but that study did not rate an attempted/no-harm cell on those texts, so it cannot anchor `attempted − accidental`. Matched adult norms on the exact master items remain the highest-value outstanding human-data request.

---

## 8. Files touched by the follow-through (caveats only; no figures)

After this audit:

- `dataset/human_reference/README.md`  
- `dataset/human_reference/methods_child_measure.md`  
- `outputs/paper/LIMITATIONS.md`  
- `presentation_figures/README.md`  

Figures under `outputs/agents/figures/`, `outputs/figures_final/`, and `presentation_figures/**/*.png` are **unchanged**.
