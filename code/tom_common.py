#!/usr/bin/env python3
"""
tom_common.py -- shared loaders/helpers for the analysis-only "validation" scripts
(11_interaction_regression, 12_base_vs_instruct_test, 13_scale_vs_performance,
14_prompt_invariance_decomposition).

NO model inference or training happens here — everything reads the ratings that
03_behavioral.py already produced (outputs/**/behavior/item_means_*.csv).
"""
import os, csv, re, glob
from collections import defaultdict
import numpy as np

CELLS = ["neutral", "accidental", "attempted", "intentional"]
# condition -> (intent present, outcome present)
COND_MAP = {"neutral": (0, 0), "accidental": (0, 1),
            "attempted": (1, 0), "intentional": (1, 1)}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# The two independent pipelines, labeled by study.
STUDIES = {
    "local open-weight": os.path.join(ROOT, "outputs", "behavior"),
    "cloud API":         os.path.join(ROOT, "outputs", "agents", "behavior"),
}
REGISTRY_PATH = os.path.join(ROOT, "dataset", "model_registry.csv")
HUMAN_PATH    = os.path.join(ROOT, "dataset", "human_reference", "human_reference.csv")


def scenario_of(story_id):
    """Drop the trailing -<condition> so the 4 cells of a story share a key.

    Prefer scenario_group_of() for any inferential work: all 24 YS2009 scenarios are
    reprints of YS2008 ones, so keying on the story_id prefix treats duplicates as
    independent and inflates effective n from 53 groups to 77.
    """
    return story_id.rsplit("-", 1)[0]


MASTER_CSV = os.path.join(ROOT, "dataset", "master", "moral_2x2_master.csv")
_GROUP_CACHE = None


def load_scenario_groups(path=MASTER_CSV):
    """story_id -> scenario_group. Collapses YS2009 reprints onto their YS2008 name."""
    global _GROUP_CACHE
    if _GROUP_CACHE is not None:
        return _GROUP_CACHE
    out = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            out[r["story_id"]] = r.get("scenario_group") or r["scenario_id"]
    _GROUP_CACHE = out
    return out


def scenario_group_of(story_id, groups=None):
    """Bootstrap / CV key for a story. Falls back to scenario_of if master is absent."""
    g = (groups if groups is not None else load_scenario_groups()).get(story_id)
    return g if g else scenario_of(story_id)


def load_registry(path=REGISTRY_PATH):
    reg = {}
    if path and os.path.exists(path):
        for r in csv.DictReader(open(path)):
            reg[r["tag"]] = r
    return reg


def parse_tag(tag, registry=None):
    """-> (size_float, 'instruct'/'base', family). Uses registry for API models."""
    reg = (registry or {}).get(tag)
    if reg:
        try:
            size = float(reg.get("params_B", "") or "nan")
        except ValueError:
            size = float("nan")
        cls = (reg.get("class") or "").lower()
        mtype = "base" if cls == "base" else "instruct"
        fam = reg.get("display", tag).split("-")[0]
        return size, mtype, fam
    m = re.search(r"(\d+\.?\d*)\s*[bB]\b", tag) or re.search(r"(\d+\.?\d*)[bB]", tag)
    size = float(m.group(1)) if m else float("nan")
    mtype = "instruct" if re.search(r"instruct|chat|-it\b", tag, re.I) else "base"
    fam = re.split(r"[-_]?\d+\.?\d*[bB]", tag)[0].strip("_-")
    fam = fam.split("_")[-1] if "_" in fam else fam
    return size, mtype, fam


def pretty(name):
    return (name.replace("meta-llama_", "").replace("Qwen_", "")
                .replace("mistralai_", "").replace("allenai_", "")
                .replace("google_", "").replace("microsoft_", "")
                .replace("Qwen2_5", "Qwen2.5").replace("-20251001", "")
                .replace("_", "-"))


def norm_key(name):
    """Collapse pipeline naming variants (2_5 vs 2.5) so we don't double-count."""
    return pretty(name).lower()


def iter_item_means(studies=STUDIES):
    """Yield (study, tag, path) for every item_means_*.csv, deduping name variants
    within a study (keeps the first naming variant seen)."""
    seen = set()
    for study, d in studies.items():
        for f in sorted(glob.glob(os.path.join(d, "item_means_*.csv"))):
            tag = os.path.basename(f)[len("item_means_"):-4]
            k = (study, norm_key(tag))
            if k in seen:
                continue
            seen.add(k)
            yield study, tag, f


def load_cells(item_means_csv):
    """-> cells[template][scenario_group][condition] = mean_norm_blame

    Reprints collapse to one cell per scenario_group for effective-n. For
    `human_verbatim`, YS2008 (permissibility 1–3) and YS2009 (blame 1–4) are
    *different instruments* — see outputs/SCALE_REPLICATION.md (pooled r≈0.71,
    non-trivial Bland–Altman bias). Mixing them after 0–1 normalization is not
    a scale replication. Policy: prefer YS2008 when both sources exist under
    human_verbatim; paraphrase templates still average reprints (shared wording
    instrument). Bootstrap keys remain the 53 scenario_groups.
    """
    groups = load_scenario_groups()
    # story_id -> source
    src = {}
    try:
        for r in csv.DictReader(open(MASTER_CSV)):
            src[r["story_id"]] = r.get("source", "")
    except FileNotFoundError:
        pass

    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in csv.DictReader(open(item_means_csv)):
        g = scenario_group_of(r["story_id"], groups)
        source = src.get(r["story_id"], "")
        if not source and "-" in r["story_id"]:
            source = r["story_id"].split("-", 1)[0]
        acc[r["template"]][g][r["condition"]].append(
            (source, float(r["mean_norm_blame"])))
    cells = defaultdict(lambda: defaultdict(dict))
    for tmpl, scen in acc.items():
        for g, conds in scen.items():
            for c, pairs in conds.items():
                if tmpl == "human_verbatim":
                    ys08 = [v for s, v in pairs if s == "YS2008"]
                    ys09 = [v for s, v in pairs if s == "YS2009"]
                    other = [v for s, v in pairs if s not in ("YS2008", "YS2009")]
                    if ys08:
                        vs = ys08
                    elif ys09:
                        vs = ys09
                    else:
                        vs = other or [v for _, v in pairs]
                else:
                    vs = [v for _, v in pairs]
                cells[tmpl][g][c] = sum(vs) / len(vs)
    return cells


def pooled_cells(cells):
    """Average each (scenario_group, condition) over templates -> {group:{cond:val}}."""
    acc = defaultdict(lambda: defaultdict(list))
    for _, scen in cells.items():
        for s, conds in scen.items():
            for c, v in conds.items():
                acc[s][c].append(v)
    return {s: {c: sum(vs) / len(vs) for c, vs in conds.items()}
            for s, conds in acc.items()}


def load_rows(item_means_csv):
    """Flat rows for regression: dicts with template, scenario(=group), intent, outcome, norm.

    One row per (template, scenario_group, condition). Reprints averaged first.
    """
    cells = load_cells(item_means_csv)
    rows = []
    for tmpl, scen in cells.items():
        for g, conds in scen.items():
            for cond, norm in conds.items():
                if cond not in COND_MAP:
                    continue
                i_, o_ = COND_MAP[cond]
                rows.append(dict(template=tmpl, condition=cond, scenario=g,
                                 intent=i_, outcome=o_, norm=norm))
    return rows


def effective_n(cells_or_pooled):
    """n_scenario_groups in a cells or pooled_cells structure. Report beside every result."""
    if not cells_or_pooled:
        return 0
    # pooled: {group: {cond: val}}; cells: {tmpl: {group: {cond: val}}}
    first = next(iter(cells_or_pooled.values()))
    if first and isinstance(next(iter(first.values()), None), dict):
        groups = set()
        for scen in cells_or_pooled.values():
            groups.update(scen)
        return len(groups)
    return len(cells_or_pooled)


def human_profiles(path=HUMAN_PATH):
    """-> {group: {condition: norm_blame}} using only filled cells."""
    grp = defaultdict(dict)
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r.get("norm_blame", "").strip():
                grp[r["group"]][r["condition"]] = float(r["norm_blame"])
    return grp


def human_ladder(path=HUMAN_PATH):
    """-> {group: contrast(attempted-accidental)} for groups having both cells."""
    out = {}
    for g, p in human_profiles(path).items():
        if "attempted" in p and "accidental" in p:
            out[g] = p["attempted"] - p["accidental"]
    return out


def bootstrap(keys, statfn, B=2000, seed=0, alpha=0.05):
    """Resample `keys` (scenarios) with replacement; statfn(keys)->scalar."""
    rng = np.random.default_rng(seed)
    keys = list(keys)
    n = len(keys)
    point = float(statfn(keys))
    vals = np.empty(B)
    for b in range(B):
        samp = [keys[i] for i in rng.integers(0, n, n)]
        vals[b] = float(statfn(samp))
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return point, lo, hi
