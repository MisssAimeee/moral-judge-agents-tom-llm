# Human ground-truth reference data

`05_human_comparison.py` compares each model's blame profile to humans. It reads
`human_reference.csv`, which is **left blank for you to fill** from the published
papers (don't invent numbers). Only the `norm_blame` column is required; the other
columns are helpers so you can show your work.

## Where the numbers come from

### Adults  → Young, Cushman, Hauser & Saxe (2007), *PNAS*
The exact 2×2 (belief × outcome) on the same paradigm as your 2008/2009 stimuli.
Read the four condition means off the behavioral figure/table (neutral, accidental,
attempted, intentional). This is your **adult ground truth**.
- Paper / data: https://www.pnas.org/doi/10.1073/pnas.0701408104
- Open PDF: https://pmc.ncbi.nlm.nih.gov/articles/PMC1895935/  ·  https://moralitylab.bc.edu/wp-content/uploads/sites/192/2011/10/young_2007_pnas.pdf

### Children (developmental curve) → Cushman, Sheketoff, Wharton & Carey (2013), *Cognition*
"The development of intent-based moral judgment" — ages ~4–8 shift from
outcome-based to intent-based judgment. Read condition means per age band.
This is your **developmental axis** (the "judges like a 5-year-old vs an adult" story).
- Open PDF: https://cushmanlab.fas.harvard.edu/files/2022/03/cushman_sheketoff_whartoncarey_2013.pdf
- PubMed: https://pubmed.ncbi.nlm.nih.gov/23318350/

### Saxe-lab developmental ToM (complementary) → Sotomayor-Enriquez, Gweon, Saxe & Richardson (2023/2024)
Open dataset, 321 children (3–12, incl. autistic), with **moral blameworthiness**
items (binary accuracy). Good for a Saxe-lineage developmental anchor and the ASD
contrast; ask Amrita whether you can use the per-item responses.
- OSF dataset: https://osf.io/g5zpv/   ·  Data in Brief: https://www.sciencedirect.com/science/article/pii/S2352340923009484

## How to fill `norm_blame` (normalize to 0–1, higher = more blame)
For a blame/wrongness scale:  `norm_blame = (raw_value - scale_min) / (scale_max - scale_min)`
For a permissibility scale:    `norm_blame = 1 - (raw_value - scale_min) / (scale_max - scale_min)`

Use the SAME normalized 0–1 space the model uses (03_behavioral.py normalizes
identically), so model and human profiles are directly comparable.

## Verification status (checked against the papers)
- **Adults — cell means verified; item match not confirmed.** Young et al. 2007 Exp.1
  (n=10), permissibility scale 1 (forbidden) – 4 (permissible): neutral 3.9,
  accidental (unknowing harm) 3.2, attempted 1.2, intentional 1.1. These are the
  actual reported cell means. They are a **reference-shape estimate from a different,
  smaller item set** (12 scenarios × 4 conditions in SI Text — SI not recovered here),
  **not a per-item match** to the 298-item master corpus. See
  `ADULT_ANCHOR_MATCH_AUDIT.md` (2026-08-03).
- **Children — derived from Cushman 2013, partly interpolated.** Age-4 values are
  directly reported (accidental 59%, attempted 45% judged naughty/punishable). The
  6–7 and 8+ values are interpolated from the paper's reported shift magnitudes
  (~40 pp drop for accidental, ~20 pp rise for attempted across ages 4–8). For a
  publication, digitize Fig. 3 or request the per-age data. Children were only tested
  on **accidental vs attempted** (no neutral/intentional cells).

## Recommended comparison metric
Because children only have two conditions, and adults (1–4 permissibility) and children
(binary proportion condemned) use **different scales and measures**, the cleanest
cross-group number is the **intent-vs-outcome contrast = blame(attempted) − blame(accidental)**:
positive = intent-weighted (adult-like), negative = outcome-weighted (young-child-like).
`05_human_comparison.py` reports this and places each model on the human ladder:
adult +0.67, age 8+ +0.46, age 6–7 +0.15, age 4–5 −0.14.

## Caveats to keep honest
- **Adult line (canonical wording):** The adult reference is Young, Cushman, Hauser
  & Saxe (2007) Exp. 1 cell means (n=10; permissibility 1–4), normalized to 0–1
  blame. It is a **reference-shape estimate from a different, smaller item set**,
  **not a per-item match** to the master corpus. The corpus includes 10 literal
  Young & Saxe (2011) Exp. 1 intentional/accidental vignettes, but that study did
  **not** rate an attempted/no-harm cell on those texts, so it cannot anchor
  `attempted − accidental`. Matched adult norms on the exact master items remain
  the highest-value outstanding human-data request (Saxe-lab / author norming).
- Child bands use different scales/measures than adults; the shared metric is the
  contrast, not absolute cell height.
- Full audit: `ADULT_ANCHOR_MATCH_AUDIT.md`.
