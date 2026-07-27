# Quarantine note — `cushman_calibrated.csv`

**Status:** file is **not present** on the node under that name. Diagnosed 2026-07-26.

## What the user reported

A file `cushman_calibrated.csv` with:
- negative `norm_blame` (−0.029, impossible on a 0–1 scale)
- `child_8plus` apparent attempted/accidental swap

## Diagnosis

1. **No such file in the repo.** `find` / `git log --all -- '*calibrated*'` turns up only
   `code/digitize_cushman_calibrated.py`. That script's trustworthy outputs are:
   - `cushman_naughty_digitized.csv` (hand-calibrated reads, verified by overlay)
   - `cushman_child_bands_PROPOSED.csv` (band averages of those reads)
   It does **not** write `cushman_calibrated.csv`.

2. **The broken auto-detector path.** `digitize_cushman_calibrated.py --auto` runs an
   experimental marker detector that the script itself documents as unreliable:
   > "returned non-monotonic and even negative values… grabbing error-bar tips, legend
   > markers, and axis pixels" (Phase-2 session, 2026-07-10).
   That is the only code path known to have produced negative proportions. Any local
   copy of its stdout-derived CSV is **quarantined — do not use as a human anchor**.

3. **`human_reference.csv` is a different defect, not the same file.** Its child values
   (4–5 = −0.14, 6–7 = +0.15, 8+ = +0.46) come from text-reported
   naughty**+punishable** proportions, not from Fig. 3 Naughty. They are internally
   consistent with that mixed measure but disagree with the figure. Kept as the
   "text-reported" arm of the dual-ladder comparison; not deleted.

4. **No attempted/accidental swap in the digitized files.** Per-age Naughty presented-first:
   age 4 contrast = 0.62 − 0.61 = **+0.01**; age 8 = 0.69 − 0.06 = **+0.63**. Band
   averages in `cushman_child_bands_PROPOSED.csv` match. A swap would flip the sign of
   the age-8 contrast; it does not.

## Rule

- **Use:** `cushman_naughty_digitized.csv` / `cushman_child_bands_PROPOSED.csv` /
  `human_reference_digitized.csv` for the figure-matched Naughty anchor.
- **Use:** `human_reference.csv` only as the text-reported arm of a dual comparison.
- **Do not use:** any file named `cushman_calibrated.csv`, or any output of
  `digitize_cushman_calibrated.py --auto`, as a paper anchor.
