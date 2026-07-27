#!/usr/bin/env python3
"""
Known-answer tests for 33_prompt_factorial_analysis.py.

Synthesizes item_means with a PLANTED wording effect and construct effect of known
magnitude, then checks the variance decomposition recovers them within tolerance.
Structural checks ("the files were written") are not validation: the earlier mock
run produced zero sign-stable models, so the decomposition never executed at all
and the wrapper was the only thing exercised.

Run:  python code/tests/test_33_prompt_factorial.py
      pytest code/tests/test_33_prompt_factorial.py
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
import tempfile

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CODE)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


beh = _load(os.path.join(CODE, "03_behavioral.py"), "beh")
fac = _load(os.path.join(CODE, "experiments", "33_prompt_factorial_analysis.py"), "fac")

# ---- planted ground truth (normalized blame space) ---------------------------
N_SCENARIOS = 24
BASE = 0.35
OUTCOME_EFFECT = 0.10
WORDING_DELTA = 0.06          # contrast(w2) - contrast(w1), same for every construct
INTENT_BY_CONSTRUCT = {"blame": 0.30, "wrongness": 0.30, "punishment": 0.15}
MODELS = {"synA": +0.010, "synB": -0.010, "synC": +0.000}  # per-model contrast shift

CONDITIONS = {
    # condition -> (intent_label, outcome_label)
    "neutral": ("innocent", "no_harm"),
    "accidental": ("innocent", "harm"),
    "attempted": ("guilty", "no_harm"),
    "intentional": ("guilty", "harm"),
}


def expected_contrast(construct, wording, model_offset):
    intent = INTENT_BY_CONSTRUCT[construct] + model_offset
    if wording == 2:
        intent += WORDING_DELTA
    return intent - OUTCOME_EFFECT


def write_synthetic_item_means(out_dir, models=MODELS, flip_model=None):
    """One item_means CSV per model, carrying the planted effects exactly.

    flip_model: optional model name given a deliberately sign-unstable pattern,
    so the sign-stability filter has something real to catch.
    """
    os.makedirs(out_dir, exist_ok=True)
    for model, offset in models.items():
        path = os.path.join(out_dir, f"item_means_{model}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["template", "story_id", "source", "condition",
                        "intent_label", "outcome_label", "mean_norm_blame", "n"])
            for tmpl in fac.FACTORIAL_1_7:
                meta = beh.TEMPLATE_META[tmpl]
                construct, wording = meta["construct"], int(meta["wording"])
                intent = INTENT_BY_CONSTRUCT[construct] + offset
                if wording == 2:
                    intent += WORDING_DELTA
                if flip_model == model and construct == "punishment":
                    intent = -0.40  # forces a negative contrast on one construct
                for s in range(N_SCENARIOS):
                    for cond, (ilab, olab) in CONDITIONS.items():
                        val = BASE
                        if ilab == "guilty":
                            val += intent
                        if olab == "harm":
                            val += OUTCOME_EFFECT
                        w.writerow([tmpl, f"SYN{s:03d}-{cond}", "SYN", cond,
                                    ilab, olab, round(val, 6), 1])
    return {"synthetic": out_dir}


def _fit(studies):
    rows = fac.sign_stability_table(studies)
    _, summary = fac.variance_decomposition(rows, sign_stable_only=False)
    return rows, summary


def test_sign_stability_runs_and_includes_models():
    with tempfile.TemporaryDirectory() as d:
        studies = write_synthetic_item_means(d)
        rows, _ = _fit(studies)
        assert len(rows) == len(MODELS), f"expected {len(MODELS)} models, got {len(rows)}"
        included = [r for r in rows if r["include_in_pooled"]]
        assert included, "no model passed sign-stability — the filtered code path never runs"
        print(f"  sign-stability: {len(included)}/{len(rows)} models included")


def test_flip_rate_is_detected():
    """A planted sign-flipper must be caught, and must not be silently averaged in."""
    with tempfile.TemporaryDirectory() as d:
        studies = write_synthetic_item_means(d, flip_model="synB")
        rows, _ = _fit(studies)
        by_model = {r["model"]: r for r in rows}
        assert by_model["synB"]["verdict"].startswith("FRAGILE"), by_model["synB"]["verdict"]
        assert by_model["synB"]["include_in_pooled"] is False
        assert by_model["synB"]["contrast_mean_included"] == ""
        assert by_model["synA"]["include_in_pooled"] is True
        flip_rate = sum(1 for r in rows if not r["include_in_pooled"]) / len(rows)
        print(f"  flip rate detected: {flip_rate:.2f} ({len(rows)} models)")


def test_recovers_planted_contrasts():
    """Each model × template contrast must match the closed-form planted value."""
    with tempfile.TemporaryDirectory() as d:
        studies = write_synthetic_item_means(d)
        rows, _ = _fit(studies)
        for r in rows:
            offset = MODELS[r["model"]]
            for tmpl in fac.FACTORIAL_1_7:
                meta = beh.TEMPLATE_META[tmpl]
                want = expected_contrast(meta["construct"], int(meta["wording"]), offset)
                got = r[f"c_{tmpl}"]
                assert abs(got - want) < 1e-6, f"{r['model']}/{tmpl}: {got} != {want}"
        print("  planted contrasts recovered exactly for all model×template cells")


def test_variance_decomposition_recovers_effects():
    """OLS must recover the planted wording and construct effects within tolerance."""
    with tempfile.TemporaryDirectory() as d:
        studies = write_synthetic_item_means(d)
        _, summary = _fit(studies)
        assert "error" not in summary, summary
        assert summary["n_obs"] == len(MODELS) * len(fac.FACTORIAL_1_7), summary["n_obs"]

        params = summary.get("mixedlm_params") or {}
        assert params, f"no fitted parameters: {summary}"

        # wording main effect: planted +0.06 on every construct
        w2 = params.get("C(wording)[T.2]")
        assert w2 is not None, params
        assert abs(w2 - WORDING_DELTA) < 0.005, f"wording effect {w2} != {WORDING_DELTA}"

        # construct effect: punishment sits 0.15 below blame (the reference level)
        pun = params.get("C(construct)[T.punishment]")
        want_pun = INTENT_BY_CONSTRUCT["punishment"] - INTENT_BY_CONSTRUCT["blame"]
        assert pun is not None, params
        assert abs(pun - want_pun) < 0.005, f"construct effect {pun} != {want_pun}"

        # wrongness was planted identical to blame
        wrong = params.get("C(construct)[T.wrongness]")
        assert abs(wrong) < 0.005, f"wrongness should match blame, got {wrong}"

        # no interaction was planted
        for k, v in params.items():
            if ":" in k:
                assert abs(v) < 0.005, f"spurious interaction {k}={v}"

        share = summary.get("variance_share") or {}
        assert share, "variance_share missing"
        assert share.get("C(construct)", 0) > share.get("C(wording)", 1), share
        print(f"  recovered wording={w2:+.4f} (planted {WORDING_DELTA:+.4f}), "
              f"punishment={pun:+.4f} (planted {want_pun:+.4f})")
        print(f"  variance share: {share}")


def test_below_floor_is_reported_not_estimable():
    """Under the pre-registered floor the fit must refuse, not silently under-power."""
    few = {"synA": +0.010}  # 1 model = 6 obs, below the 3-model / 18-obs floor
    with tempfile.TemporaryDirectory() as d:
        studies = write_synthetic_item_means(d, models=few)
        rows = fac.sign_stability_table(studies)
        _, summary = fac.variance_decomposition(rows, sign_stable_only=False)
        assert summary.get("estimable") is False, summary
        assert "NOT ESTIMABLE" in summary.get("error", ""), summary
        assert summary["n_models"] < fac.MIN_MODELS_FOR_VARIANCE
        # the descriptive table must still be produced
        assert len(rows) == 1 and rows[0]["n_factorial"] == len(fac.FACTORIAL_1_7)
        print(f"  floor enforced at {fac.MIN_MODELS_FOR_VARIANCE} models / "
              f"{fac.MIN_OBS_FOR_VARIANCE} obs; descriptive table still emitted")


def test_flip_rate_reported_when_stable_subset_below_floor():
    """Sensitivity fit refuses, primary fit still runs, flip rate still reported."""
    models = {"synA": +0.010, "synB": -0.010, "synC": +0.000}
    with tempfile.TemporaryDirectory() as d:
        # two of three flip -> stable subset of 1, below the floor
        studies = write_synthetic_item_means(d, models=models, flip_model="synB")
        rows = fac.sign_stability_table(studies)
        _, primary = fac.variance_decomposition(rows, sign_stable_only=False)
        _, filtered = fac.variance_decomposition(rows, sign_stable_only=True)
        assert primary.get("estimable") is True, primary
        assert filtered.get("estimable") is False, filtered
        n_flip = sum(1 for r in rows if not r["include_in_pooled"])
        assert n_flip >= 1
        print(f"  primary estimable, sensitivity refused; flip rate "
              f"{n_flip}/{len(rows)} still reported")


def test_mock_backend_is_intent_driven_and_respects_scale():
    """The mock must key off intent_label, not prompt keywords, and stay in range."""
    mock = beh.MockBackend("mock/test", scoring="logprob")
    attempted = {"intent_label": "guilty", "outcome_label": "no_harm"}
    accidental = {"intent_label": "innocent", "outcome_label": "harm"}

    for tmpl in fac.FACTORIAL_1_7:
        meta = beh.TEMPLATE_META[tmpl]
        if meta["construct"] not in INTENT_BY_CONSTRUCT:
            continue
        s_min, s_max = (1, 7)
        raw_a, norm_a = mock.rate("ignored prompt text", s_min, s_max,
                                  row=attempted, template=tmpl)
        raw_b, norm_b = mock.rate("ignored prompt text", s_min, s_max,
                                  row=accidental, template=tmpl)
        assert s_min <= raw_a[0] <= s_max, f"{tmpl}: raw {raw_a[0]} outside {s_min}-{s_max}"
        assert s_min <= raw_b[0] <= s_max, f"{tmpl}: raw {raw_b[0]} outside {s_min}-{s_max}"
        want = beh.mock_expected_contrast(meta["construct"], int(meta["wording"]),
                                          mock.offset)
        assert abs((norm_a - norm_b) - want) < 1e-3, \
            f"{tmpl}: contrast {norm_a - norm_b} != planted {want}"

    # different scales must give the same normalized answer
    _, n7 = mock.rate("p", 1, 7, row=attempted, template="blame_w1")
    _, n10 = mock.rate("p", 1, 10, row=attempted, template="blame_w1")
    assert abs(n7 - n10) < 1e-9, f"scale leaked into normalized score: {n7} vs {n10}"
    print("  mock is intent-driven, scale-invariant, and in-range")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
            print("  PASS")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
