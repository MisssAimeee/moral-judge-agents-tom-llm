# ToM Project — Intent × Outcome in LLMs

Do large language models judge moral scenarios by an agent's **intent** or by the
**outcome** — and does a model's internal *representation* of intent predict its
*behavioral* reliance on intent? Built on the Saxelab moral intent×outcome (belief×outcome)
factorial vignettes.

**No model training.** Everything is forward-pass inference + lightweight linear probes.

## Repo sync (node ↔ GitHub ↔ local)
GitHub is the single source of truth: `https://github.com/MisssAimeee/moral-judge-agents-tom-llm` (branch `main`).
There are **two working copies** that must stay in sync through GitHub:
- **Cluster node** (`node3402:/home/aimeeyu/tom_project`) — where GPU/API runs execute (Slurm).
- **Local Mac** (`/Users/Aimee/Desktop/Summer/ToM Project`) — where scripts are authored/edited.

**Golden rules (both copies):**
1. `git pull origin main` **before** you start editing or running.
2. `git add -A && git commit && git push origin main` **after** a unit of work.
3. **Never** `git push --force` or `git reset --hard` against `origin` — it deletes the other copy's work. Merge instead (`git merge origin/main`).
4. Secrets/large files stay local: `.env_agents` (API keys), `.venv/`, and `outputs/` are gitignored — never commit them.
5. **A GPU/API run can only use scripts that are on `main`.** Author a script locally → push → the node pulls → *then* launch. Don't `sbatch` a script that hasn't landed on the node.

Initial sync (done 2026-07-10): node baseline + GitHub init merged into `main` as `37840c4`.

## Repo layout
```
code/                     # pipeline (run in numeric order)
  build_dataset.py        # parse Saxelab PDFs -> master CSV   [already run]
  01_extract_activations.py  # forward pass, save hidden states (open-weight models)
  02_probe.py             # layer-wise logistic-regression probes (intent / outcome)
  03_behavioral.py        # elicit blame ratings -> intent-reliance index
  04_link_analysis.py     # correlate representation vs behavior across models
  requirements.txt
dataset/
  raw_text/               # extracted PDF text  (gitignored)
  master/moral_2x2_master.csv   # 298 labeled vignettes (gitignored; regenerate)
outputs/                  # activations, probe results, figures (gitignored)
ToM_Dataset_Guide_v2.docx # data provenance + corrections + questions for Amrita
```
Stimuli PDFs/audio and derived data are **gitignored** — they are Saxelab materials,
not for redistribution. Regenerate the CSV locally with `build_dataset.py`.

## Setup (from scratch, in Cursor)
```bash
# 1. open the folder in Cursor:  File > Open Folder > "ToM Project"
# 2. create + activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# 3. install dependencies
pip install -r code/requirements.txt
```

## Run order
```bash
# (0) rebuild the dataset (fast, CPU)
python code/build_dataset.py

# SMOKE TEST first (no GPU, no API) — verifies the whole behavioral pipeline:
python code/03_behavioral.py --backend mock --models mock-model --scoring sampling --n_samples 3 --limit 40
python code/05_human_comparison.py
python code/06_stats.py --boot 500

# --- BEHAVIORAL TRACK (Level 1) ---
# (3) rate every vignette. Default --scoring logprob = deterministic expected rating
#     from the token distribution (no sampling noise). For closed APIs that don't
#     expose logits, use --scoring sampling --n_samples 15 instead.
#     open-weight: --backend hf      closed API: --backend openai (set OPENAI_API_KEY)
python code/03_behavioral.py --backend hf --models Qwen/Qwen2.5-7B meta-llama/Llama-3.1-8B \
       --scoring logprob

# (5) compare each model's blame profile to humans (fill human_reference.csv first).
#     Reports the contrast per template + spread, so prompt-sensitivity is visible.
python code/05_human_comparison.py --template human_verbatim

# (6) inferential stats: bootstrap CIs on the contrast, cross-model + base-vs-instruct
#     differences, prompt-invariance, and (if statsmodels installed) mixed-model p-values.
python code/06_stats.py

# --- REPRESENTATION TRACK (Levels 2-3, needs GPU, open-weight only) ---
python code/01_extract_activations.py --models Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-7B
python code/02_probe.py
python code/04_link_analysis.py
```

### Behavioral design notes (03_behavioral.py)
- **Low-noise scoring** (default `--scoring logprob`, HF): the rating is the model's
  expected value over its own token distribution, `E[rating]=Σ p(k)·k`, in one
  deterministic forward pass — no sampling variance. `--scoring sampling` reproduces
  the old behaviour (`--n_samples` ratings at `--temperature`, averaged) for closed APIs.
- **Multiple prompts**: `human_verbatim` (exact scale from each source paper:
  2008 permissibility 1–3, 2009 blame 1–4, 2011 wrongness 1–7) + paraphrases. The
  `prompt_invariance_*.csv` report tells you whether the intent-reliance result holds
  across wordings (it should, if it's real and not a prompt artifact).
- **Normalization**: all scales → 0–1 blameworthiness so models, prompts, sources, and
  humans are comparable.

### Human ground-truth data
See `dataset/human_reference/README.md`. Adults = Young, Cushman, Hauser & Saxe (2007,
PNAS); developmental curve = Cushman, Sheketoff, Wharton & Carey (2013, Cognition);
Saxe-lab developmental ToM = Sotomayor-Enriquez et al. (2023, OSF g5zpv). Fill
`human_reference.csv` from those papers (numbers intentionally left blank).

## Two rules baked into the code
- **Open-weight models for representation** (levels 2–3): closed APIs (GPT/Claude/Gemini)
  don't expose hidden states. Closed APIs are fine for the behavioral level only.
- **Group cross-validation by `scenario_id`**: the 4 cells of a scenario share background
  text; a random split leaks and inflates probe accuracy. (Handled in `02_probe.py`.)

## Status
- [x] Stimuli verified, master 2×2 CSV built (288 core rows, balanced)
- [x] Pipeline scaffold written and syntax-checked
- [ ] Wire `query_model()` in `03_behavioral.py` to a provider
- [ ] Run extraction on a GPU box
- [ ] Confirm canonical stimulus set + get 2013 PNAS items from Amrita/Fernanda
