# Pre-repair master backup

Expected file (commit when present on this node):

`moral_2x2_master_CONTAMINATED_20260619.csv`

- **md5:** `5dd904a7609628553319da4acab02f25`
- **Visible trailing-artefact tally (reproducible):** 99  
  accidental 48 / intentional 49 / attempted 2  
- Provenance: local pre-repair snapshot dated 2026-06-19; drives
  `CONTAMINATION_REPAIR.md` §1.2.

This directory was empty on the MIT node as of 2026-07-27 10:48. Place the CSV
here, verify md5, then `git add` it.
