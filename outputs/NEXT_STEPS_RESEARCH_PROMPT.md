# Prompt for Claude Research — Next Steps Investigation

Copy everything in the block below into Claude Research (or a deep-research agent).

---

I am running a research project measuring whether large language models judge moral scenarios by an agent's **intent** or by the **outcome**, and how this compares to the human developmental trajectory. I need you to research and propose a rigorous, prioritized plan for the next phase.

## Background / what I have already done

**Design.** I use a 298-item 2×2 factorial moral-vignette set (from Saxe-lab / Young-Cushman-Hauser-Saxe belief×outcome stimuli) crossing the agent's *intent* (innocent vs. guilty belief) with the *outcome* (harm vs. no harm): four conditions = neutral, accidental, attempted, intentional. Each model rates blameworthiness; ratings are normalized to 0–1. My headline metric is the **intent-vs-outcome contrast = blame(attempted) − blame(accidental)**: positive = intent-driven (adult-like), negative = outcome-driven (young-child-like).

**Human ladder (published data):** adult +0.67, age 8+ +0.46, age 6–7 +0.15, age 4–5 −0.14.

**Models tested (21):** Cloud APIs — Claude Haiku-4.5 / Sonnet-4.6 / Opus-4.6, Gemini 2.5 Flash / Pro, GPT-4o / GPT-4o-mini. Local open-weight — Qwen2.5 (0.5B→14B, base + instruct), Llama-3.1-8B, Llama-3.2-3B, Mistral-7B-v0.3, OLMo-2-7B, Gemma-2-9B, Phi-3-mini. Open-weight scored by deterministic logprob expected value; closed APIs by sampling at T=0. Prompts: exact source-paper scale + paraphrases (wrongness-1-7, punishment-1-7); prompt-invariance and bootstrap CIs computed.

**Key empirical findings:**
1. **No model reaches even the 8-year-old level.** Best is Claude Opus 4.6 at +0.09 (≈ age 6–7); most models cluster at the age 4–5 (outcome-biased) floor.
2. **The most capable cloud models are the most outcome-biased** — GPT-4o (−0.38) and GPT-4o-mini (−0.28) are the most child-like judges in the whole set.
3. **Scale does not increase intent-weighting** — size↔contrast Spearman ρ = −0.23 (p=0.43) across disclosed-size models.
4. **Instruction-tuning makes models MORE outcome-biased** — formal paired test across the Qwen2.5 ladder: mean Δ = −0.216 (all 5 pairs negative, paired t p=0.020).
5. **2×2 interaction:** only Claude Opus shows the adult signature (intent effect > outcome effect); for all others the bad-outcome main effect dominates. Models reproduce the adult profile *shape* but compress the attempted-vs-accidental distinction.

## What I want you to research

1. **Situate this in the literature.** Find the most relevant recent (2023–2026) work on: moral judgment in LLMs, intent vs. outcome / moral luck in machine ethics, Theory-of-Mind benchmarks for LLMs, and the developmental-psychology framing (Cushman, Young, Saxe, Gray). Where does my finding (LLMs are outcome-biased / "morally young") agree or conflict with published results? Cite specific papers.

2. **Critique my methodology** and identify the highest-value fixes. Specifically weigh in on: (a) logprob vs. sampling scoring comparability across open/closed models; (b) whether contrast = attempted − accidental is the right summary vs. a full 2×2 regression with an intent×outcome interaction term; (c) prompt sensitivity and how many paraphrases/scales are enough; (d) the fact that my human child bands are approximated from published figures rather than matched per-item.

3. **The instruction-tuning finding.** Propose experiments to test *why* RLHF/instruction-tuning increases outcome bias (e.g., harm-avoidance training, refusal behavior, safety-tuning). Is this documented elsewhere? How would I isolate the mechanism (e.g., base vs. SFT vs. RLHF checkpoints, DPO datasets)?

4. **Mechanistic / representational next steps.** I have a scaffolded but unrun "representation track": extract hidden activations from open-weight models and train linear probes for intent and outcome, then test whether a model's *internal* intent representation predicts its *behavioral* intent-reliance. Advise on best practices: which layers, probe design, cross-validation to avoid scenario leakage, and how to make a rep→behavior causal claim (activation patching / steering?).

5. **Reasoning models & test-time compute.** Should I add explicit chain-of-thought / reasoning models (o-series, DeepSeek-R1, Claude extended-thinking, Gemini thinking budgets) and does reasoning move models up the developmental ladder? Design that comparison.

6. **Prioritized roadmap.** Give me a ranked list of the next 5–8 concrete experiments, each with: hypothesis, method, expected effort, and what result would be publishable / novel. Flag which are "table stakes for a paper" vs. "high-risk high-reward."

7. **Publication framing.** What is the strongest, most defensible narrative here (developmental-analogy? safety-tuning-side-effect? ToM-deficit?), what venue/format fits (workshop, cogsci, ACL/NeurIPS, preprint), and what is the minimum additional evidence needed to support that claim?

Please be specific and cite sources. Where you make empirical claims about what other models/papers found, distinguish established findings from your inference.

---
