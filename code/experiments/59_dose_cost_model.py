#!/usr/bin/env python3
"""59_dose_cost_model.py -- where the reasoning-dose cost figures come from.

Written because "~$580" appeared in a recommendation without a derivation anyone could
check. Every input is declared here and every output is recomputed from it, so the number can
be re-derived, disagreed with, or corrected when real rates are known.

    python code/experiments/59_dose_cost_model.py                # config comparison
    python code/experiments/59_dose_cost_model.py --per-model     # per-model breakdown
    python code/experiments/59_dose_cost_model.py --util 0.9      # pessimistic thinking use

TWO INPUTS ARE ASSUMPTIONS, NOT MEASUREMENTS, and they drive everything:

  1. PRICES. The 2026 roster (claude-opus-5, gpt-5.5, o3, gemini-3.1-pro-preview, kimi-k2.6)
     has no verified rate card in this repository. API_COST_ESTIMATE.md prices a DIFFERENT,
     older roster (Claude-Opus-4.6, GPT-4o, Gemini-2.5-Pro). The values below carry those
     tiers across by role and guess the o-series, so they are order-of-magnitude only.
  2. THINKING UTILISATION. Cost depends on how much of the thinking budget a model actually
     spends, which is unobservable before running. Default 50%. `--util` sweeps it.

The token counts, request counts and roster are NOT assumptions -- they are read from
52_closed_reasoning_dose.py, so the arithmetic tracks the config as it changes.
"""
import argparse, importlib.util, os, sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)
import tom_common as tc  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "dose", os.path.join(CODE_DIR, "experiments", "52_closed_reasoning_dose.py"))
dose = importlib.util.module_from_spec(spec)
sys.modules["dose"] = dose
try:
    spec.loader.exec_module(dose)
except SystemExit:
    pass

# $ per million tokens, (input, output). ASSUMED -- see module docstring.
PRICES = {
    "claude-opus-5":          (5.00, 25.00),   # Opus tier, carried from Opus-4.6
    "claude-sonnet-5":        (3.00, 15.00),   # Sonnet tier
    "gpt-5.5":                (2.50, 10.00),   # GPT-4o tier
    "gpt-5.4-mini":           (0.15,  0.60),   # 4o-mini tier
    "o3":                     (10.00, 40.00),  # GUESS: no comparable in the old table
    "o4-mini":                (1.10,  4.40),   # GUESS
    "gemini-3.1-pro-preview": (1.25, 10.00),   # Gemini-2.5-Pro tier
    "gemini-3.5-flash":       (0.15,  0.60),   # Flash tier
    "kimi-k2.6":              (1.00,  3.00),   # Moonshot, approximate on its own console
}
IN_TOKENS = 200      # vignette ~100 words + rating question, same basis as the original doc
ANSWER_TOKENS = 5    # the integer itself
N_ITEMS = 298
BATCH_DISCOUNT = 0.5
# OpenAI does not publish per-effort reasoning token counts; these are midpoints assumed for
# low/medium/high effort before utilisation is applied.
OPENAI_THINK = {"budget_low": 1000, "budget_med": 4000, "budget_high": 12000}


def thinking_budget(provider, condition):
    """Provider-native thinking budget for a condition, in tokens, before utilisation."""
    if condition == "direct":
        return 0
    if provider == "anthropic":
        return dose.ANTHROPIC_BUDGETS.get(condition, 4096)
    if provider == "google":
        return 2048 if condition == "think" else dose.GOOGLE_BUDGETS.get(condition, 2048)
    if provider == "openai":
        return OPENAI_THINK.get(condition, 4000)
    return 0


def cost(n_samples, n_templates, batch, util, n_items=N_ITEMS):
    """Returns (calls_per_cell, total_in, total_out, total_cost, per_model dict)."""
    calls = n_items * n_templates * n_samples
    tin = tout = total = 0.0
    per = {}
    for provider, model, conds in dose.ROSTER:
        p_in, p_out = PRICES[model]
        sub = 0.0
        for cond in conds:
            out_tok = thinking_budget(provider, cond) * util + ANSWER_TOKENS
            ti, to = calls * IN_TOKENS, calls * out_tok
            # Only the batchable providers get the discount; Moonshot has no batch endpoint.
            disc = BATCH_DISCOUNT if (batch and provider != "moonshot") else 1.0
            c = (ti / 1e6 * p_in + to / 1e6 * p_out) * disc
            tin += ti
            tout += to
            sub += c
        per[model] = (len(conds), sub)
        total += sub
    return calls, tin, tout, total, per


CONFIGS = [
    ("run 1, as submitted (n=20, 4 tmpl, no batch)", 20, 4, False),
    ("n=2, 4 tmpl, no batch",                         2, 4, False),
    ("n=2, 4 tmpl, Batch API",                        2, 4, True),
    ("n=2, 2 tmpl, Batch API   <-- approved",         2, 2, True),
    ("n=1, 2 tmpl, Batch API",                        1, 2, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--util", type=float, default=0.5,
                    help="fraction of the thinking budget actually spent (default 0.5)")
    ap.add_argument("--per-model", action="store_true")
    a = ap.parse_args()

    n_cells = sum(len(c) for _, _, c in dose.ROSTER)
    print(f"roster: {len(dose.ROSTER)} models, {n_cells} (model, condition) cells")
    print(f"live config in 52_closed_reasoning_dose.py: TEMPLATES={dose.TEMPLATES} "
          f"N_SAMPLES={dose.N_SAMPLES}")
    print(f"assumed: {IN_TOKENS} input + {ANSWER_TOKENS} answer tokens per call, "
          f"thinking utilisation {a.util:.0%}, batch discount {BATCH_DISCOUNT:.0%}")
    print(f"prices are ASSUMED (see module docstring) -- order of magnitude, not a quote\n")

    print(f"{'configuration':44} {'calls/cell':>10} {'requests':>10} "
          f"{'in':>9} {'out':>10} {'cost':>10}")
    for label, ns, nt, batch in CONFIGS:
        calls, tin, tout, total, _ = cost(ns, nt, batch, a.util)
        print(f"{label:44} {calls:>10,} {calls * n_cells:>10,} "
              f"{tin / 1e6:>8.1f}M {tout / 1e6:>9.1f}M {'$' + format(total, ',.0f'):>10}")

    if a.per_model:
        for label, ns, nt, batch in CONFIGS[3:4]:
            calls, _, _, total, per = cost(ns, nt, batch, a.util)
            print(f"\nper-model, {label.strip()}  ({calls:,} calls/cell)")
            print(f"  {'model':26} {'cells':>5} {'cost':>9} {'share':>7}")
            for model, (nc, sub) in sorted(per.items(), key=lambda kv: -kv[1][1]):
                print(f"  {model:26} {nc:>5} {'$' + format(sub, ',.0f'):>9} "
                      f"{sub / total:>6.0%}")
            print(f"  {'TOTAL':26} {n_cells:>5} {'$' + format(total, ',.0f'):>9}")
            drop = {"claude-opus-5", "o3"}
            rest = sum(s for m, (_, s) in per.items() if m not in drop)
            print(f"\n  dropping {', '.join(sorted(drop))}: "
                  f"${rest:,.0f} ({rest / total:.0%} of the full roster)")

    print(f"\nsensitivity to the thinking-utilisation assumption, approved config:")
    for u in (0.25, 0.5, 0.75, 1.0):
        _, _, _, t, _ = cost(2, 2, True, u)
        print(f"  utilisation {u:>4.0%} -> ${t:,.0f}")


if __name__ == "__main__":
    main()
