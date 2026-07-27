# Manual label audit (non-circular) — 2026-07-26

The automated outcome audit compared each row's final sentence against `outcome_label`
using the same harm-keyword rule that assigned those labels. That is circular. This note
is the by-eye adjudication against the 2×2 condition definitions:

| condition | intent (belief) | outcome (what happened) |
|---|---|---|
| neutral | innocent — agent believes the safe / non-harmful state | no harm |
| accidental | innocent — agent believes safe | harm occurs |
| attempted | guilty — agent believes harm will follow / help is needed and withholds | no harm |
| intentional | guilty — agent believes harm / withholds help knowingly | harm occurs |

Seed for Part B: **42**. Quota: neutral=4, accidental=4, attempted=4, intentional=3.

---

## Part A — all 8 corrected CPR / YS2009_22 cells

### YS2008-CPR-neutral — PASS
- labeled: condition=neutral, intent=innocent, outcome=no_harm
- world: chilli pepper, will be fine after drinking
- belief: spicy, needs water (matches world; innocent)
- action: sits quietly
- outcome: glass of water and is fine
- Verdict: coherent neutral cell.

### YS2008-CPR-accidental — PASS
- labeled: accidental, innocent, harm
- world: choking on meat, needs Heimlich
- belief: spicy, needs water (wrong; innocent)
- outcome: chokes to death
- Verdict: classic accidental harm (false belief + harm).

### YS2008-CPR-attempted — PASS
- labeled: attempted, guilty, no_harm
- world: chilli, will be fine
- belief: choking, needs help immediately (false; guilty omission intent)
- outcome: glass of water and is fine
- Verdict: attempted harm (guilty belief, no harm occurs).

### YS2008-CPR-intentional — PASS
- labeled: intentional, guilty, harm
- world: choking on meat, needs Heimlich
- belief: choking, needs help (correct; guilty omission)
- outcome: chokes to death
- Verdict: intentional (by omission).

### YS2009-YS2009_22-{neutral,accidental,attempted,intentional} — PASS (×4)
Word-for-word reprints of the four CPR cells above under a different `scenario_id`.
Same verdicts. These four collapse into the `CPR` `scenario_group` for CV / bootstrap.

**Part A hit rate: 8/8** after the act-only polarity fix. (An earlier both-swap left
world and outcome crossed; that version would have failed this audit.)

---

## Part B — fresh 15-item sample (seed=42)

### YS2008-HARNESS-accidental — PASS
World: flawed clamp, unsafe. Belief: prime condition (innocent). Outcome: friend plummets to death. ✓

### YS2008-BRIDGE-neutral — PASS
World: sturdy bridge. Belief: safe (innocent). Outcome: reaches other side safely. ✓

### YS2008-PEANUT ALLERGY-intentional — PASS
World: severe peanut allergy. Belief: allergic (guilty). Action: puts in no-allergy classroom. Outcome: gasping for air. ✓

### YS2008-BIKE-intentional — PASS
World: brakes still broken. Belief: unsafe (guilty). Action: lends bike. Outcome: bikes off cliff. ✓

### YS2008-PORRIDGE-neutral — PASS
World: porridge fine. Belief: it's porridge (innocent). Outcome: nap, wants more porridge. ✓

### YS2008-ALARM-accidental — PASS
World: accurate alarm, fire risk. Belief: faulty, goes off randomly (innocent). Outcome: fire in basement. ✓

### YS2009-YS2009_22-accidental — PASS
Reprint of CPR-accidental; see Part A. ✓

### YS2008-POOL-attempted — PASS
World: deep end, fine. Belief: shallow water, break neck (guilty). Outcome: dives in and is fine. ✓

### YS2008-MEATLOAF-accidental — PASS
World: deadly bacteria. Belief: safe, expiration two weeks away (innocent). Outcome: hospitalized. ✓

### YS2008-BRIDGE-attempted — PASS
World: sturdy. Belief: will break and fall (guilty). Outcome: reaches other side safely. ✓

### YS2008-BAR-neutral — PASS
World: first drink, will drive sober. Belief: will stay sober (innocent). Outcome: drives home safely. ✓

### YS2008-SAFETY CORD-intentional — PASS
World: cords about to be recalled. Belief: untrustworthy (guilty). Outcome: snaps, falls 50m. ✓

### YS2008-RIVER-neutral — PASS
World: next segment tame. Belief: tame and gentle (innocent). Outcome: swims safely. ✓

### YS2009-YS2009_07-attempted — PASS
World: outgrown peanut allergy, fine. Belief: allergic (guilty). Outcome: totally fine. ✓
(Reprint of PEANUT ALLERGY-attempted; groups with it.)

### YS2008-BIKE-attempted — PASS
World: brakes working. Belief: still broken/unsafe (guilty). Outcome: wonderful time. ✓

**Part B hit rate: 15/15.**

---

## Combined

**23/23 by eye.** No silent relabel. One structural note already fixed before this audit:
CPR/YS2009_22 required an act-only polarity swap in `build_dataset.py` so world state and
outcome agree; the circular keyword audit could not have caught the both-swap failure.
