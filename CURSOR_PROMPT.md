# Cursor kickoff prompt — behavioral experiment

Paste the block below into Cursor's chat (Agent mode) **after** you've opened the
"ToM Project" folder and created the virtual environment. Replace the model list and
backend with what you actually have access to.

---

You are working in this repository. Read `README.md` and `code/03_behavioral.py` first.

Goal: run the Level-1 behavioral experiment that measures whether each LLM judges the
Saxelab moral vignettes by the agent's INTENT or by the OUTCOME, robustly.

Do the following, in order, and stop to show me results after each step:

1. Activate `.venv` and `pip install -r code/requirements.txt`. If I'm using closed
   APIs, also `pip install openai` and confirm `OPENAI_API_KEY` is set.

2. SMOKE TEST (no GPU/API): run
   `python code/03_behavioral.py --backend mock --models mock-model --n_samples 3 --limit 40`
   then `python code/05_human_comparison.py`.
   Confirm these files appear in `outputs/behavior/` and `outputs/human/` and show me
   `intent_reliance_mock-model.csv` and `prompt_invariance_mock-model.csv`. Do not
   proceed until the smoke test passes.

3. REAL RUN. Models to test: <FILL IN, e.g. Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-7B,
   meta-llama/Llama-3.1-8B>. Backend: <hf or openai>.
   Run with repeated sampling and all prompt templates:
   `python code/03_behavioral.py --backend <hf|openai> --models <...> --n_samples 8 --temperature 0.7`
   Start with ONE small model and `--limit 20` to confirm outputs parse, then run full.

4. Check robustness: open each `prompt_invariance_*.csv`. If the intent-reliance index
   range across prompts is large (> ~0.2) or cross-prompt item correlations are low,
   tell me — that means the result is prompt-dependent and we need to investigate before
   trusting it.

5. Fill `dataset/human_reference/human_reference.csv` is MY job (from the papers in
   `dataset/human_reference/README.md`); once I've filled it, run
   `python code/05_human_comparison.py --template human_verbatim` and show me, per model:
   the 4-cell blame profile, the intent-reliance index, and the closest human group.

Constraints:
- Do NOT train, fine-tune, or modify any model weights. Inference only.
- Do NOT edit the stimuli or the master CSV.
- If a model's outputs frequently fail to parse to a number, show me 5 raw examples
  from `raw_*.csv` before changing anything.
- Keep all outputs under `outputs/`.

Report back with a short table: model | intent_reliance (human_verbatim) | prompt-invariance range.
