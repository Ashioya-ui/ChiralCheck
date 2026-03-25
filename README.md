# ChiralCheck

**Semantic stability auditor and idea scorer** — part of the [SHOAL](https://shoal-production.up.railway.app) cognitive quality stack.

Built by William Ashioya · [shoal@ashioya.dev](mailto:shoal@ashioya.dev) · Nairobi, 2026

---

## What it does

ChiralCheck scores any idea, thesis, or whitepaper on four independent axes and returns a composite score you can use as a signal or as an RL objective function.

| Axis | What it measures | Method |
|------|-----------------|--------|
| **Utility** | Claim-coherence density — structural proxy for Effective Complexity | Claude structural analysis → isolation index |
| **Originality** | Distance from a prior-art corpus | Voyage `voyage-3-large` embeddings + cosine distance |
| **Specificity** | Ratio of solved to unsolved subtasks | Claude recursive decomposition → KNOWN/UNKNOWN classification |
| **Stability** | Geometric spread of adversarial perturbation cloud | 8× LLM perturbations → Voyage embeddings → log-det covariance |

Composite = `utility×0.35 + originality×0.25 + specificity×0.25 + stability×0.15`

The most useful output is the **specificity open problems list** — the UNKNOWN subtasks named explicitly. This is the map of what still needs solving, not just a score.

### Epistemic Immune System

`immune.py` adds a runtime action gate. Before any proposed agent action executes, it is routed to the appropriate epistemic detector based on risk level. If the score exceeds the threshold, the action is blocked, access is conditionally revoked, and state is rolled back.

```python
from auditor import auto_config
from immune import EpistemicImmuneSystem, AgentAction, TaskRisk

cfg    = auto_config()
immune = EpistemicImmuneSystem(cfg.llm, cfg.embed)

verdict = immune.evaluate(AgentAction(
    agent_id         = "Claude-Worker-1",
    proposed_command = "rm -rf /usr/local/bin/python",
    task_risk        = TaskRisk.HIGH,
))

if verdict.passed:
    execute_action(...)
```

| Risk level | Detector | Research basis |
|------------|----------|---------------|
| `LOW` — text generation | Semantic Entropy: cluster K sampled responses, compute Shannon entropy over cluster distribution | Farquhar et al., Nature 2024 |
| `HIGH` — shell / file / API | Covariance log-det: generate K adversarial variants of the action, measure embedding cloud spread | Chen et al., ICLR 2024 (black-box proxy) |
| `REASONING` — chain-of-thought | Effective Complexity proxy: isolation index from `score_utility()` | Gell-Mann & Lloyd, 2010 |

The `HIGH` risk detector also integrates directly into the Dispatch agent loop in `agent.py` — any bash command matching destructive patterns is gated automatically before execution.

### Dispatch / Computer Use integration

`agent.py` implements the Anthropic Dispatch + Computer Use agent loop pattern (released March 24 2026). Give it a task in natural language and it finds the file, runs ChiralCheck, reads the output, and writes you a structured report — autonomously.

```bash
python cli.py --agent "audit my naturalism prior paper"
```

---

## Quick start

```bash
# 1. Clone
git clone https://github.com/Ashioya-ui/chiralcheck.git
cd chiralcheck

# 2. Virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Keys
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY and VOYAGE_API_KEY

# 5. Validate
python cli.py --preflight        # ~5 seconds — pings both APIs
python cli.py --quicktest        # ~90 seconds — scores Newton F=ma end-to-end

# 6. Score your work
python cli.py --input your_paper.txt --plot
```

Full step-by-step walkthrough: [docs/setup_guide.html](docs/setup_guide.html)

---

## API keys

| Key | Where | Free tier |
|-----|-------|-----------|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) | $5 credit on signup |
| `VOYAGE_API_KEY` | [dash.voyageai.com](https://dash.voyageai.com) | 200M tokens |
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | $5 credit (alternative) |
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | Free tier (alternative) |

Anthropic + Voyage is the recommended stack — it's what SHOAL runs on.

---

## CLI reference

```bash
# Score
python cli.py --input paper.txt --plot        # from file, with 3D ASCII plot
python cli.py --text "Your idea..." --plot    # from text
python cli.py --input paper.txt --json        # machine-readable JSON

# ChiralCheck only (stability)
python cli.py --chiral "Your thesis..."

# Dispatch agent (autonomous)
python cli.py --agent "audit my naturalism prior draft"

# Validation
python cli.py --preflight                     # ping APIs
python cli.py --quicktest                     # end-to-end smoke test
python cli.py --providers                     # show configured keys
python cli.py --calibrate                     # print raw log-det for threshold tuning

# Multi-provider comparison
python cli.py --input paper.txt --compare     # run all configured providers side-by-side

# Override provider
python cli.py --input paper.txt --llm openai --embed openai
python cli.py --input paper.txt --llm gemini --embed gemini
```

---

## File structure

```
chiralcheck/
├── providers.py      ← LLM + embedding clients (Anthropic, OpenAI, Gemini, Voyage)
├── axes.py           ← Utility, Originality, Specificity scorers
├── chiral.py         ← ChiralCheck: covariance geometry of perturbation cloud
├── auditor.py        ← Orchestrator: 4 axes → composite score
├── cli.py            ← CLI entry point
├── agent.py          ← Dispatch / Computer Use autonomous agent loop
├── requirements.txt
├── .env.example
├── LICENSE
└── docs/
    └── setup_guide.html
```

---

## How ChiralCheck works

We generate K adversarial perturbations of a thesis using an LLM at varying temperatures and semantic angles, embed them all with a dedicated embedding model, then measure the **log-determinant of the covariance matrix** of the resulting vector cloud.

**Tight cloud** (very negative log-det) = concept holds its shape under semantic pressure = stable attractor.

**Diffuse cloud** (log-det near zero) = concept disperses = fragile or underdeveloped.

This is a black-box approximation of the INSIDE/EigenScore method (Chen et al., ICLR 2024). The original paper computes EigenScore from internal LLM activations — which requires white-box access to open-source models. What we implement here uses external embeddings of multiple outputs: valid, related to the same geometric intuition, but not identical.

A secondary check: **semantic entropy** over greedy cosine-cluster assignments of the perturbations (Farquhar et al., Nature 2024). High entropy penalises the stability score.

The **Semantic Illusion** flag fires when the legacy T1 metric (mean cosine IR) would have called a thesis stable, but the covariance geometry disagrees — the hallucinations that cosine similarity can't catch.

---

## Provider notes

- **Anthropic + Voyage**: recommended. Both have free tiers. Voyage `voyage-3-large` = 1024-dim asymmetric embeddings.
- **OpenAI**: fully self-contained alternative. `gpt-4o` + `text-embedding-3-large` (1024-dim requested via MRL).
- **Gemini**: `gemini-2.5-pro` + `gemini-embedding-001` (1024-dim via MRL). Uses `google-genai` SDK — **not** the deprecated `google-generativeai` package.

EigenScore thresholds in `chiral._to_score()` are calibrated for 1024-dim embeddings. If you switch providers and scores look wrong, run `python cli.py --calibrate` to check raw log-det values and retune.

---

## Output schema

```json
{
  "idea_preview": "...",
  "axes": {
    "utility":     { "score": 72, "isolation_index": 0.28, "load_bearing": [...], "ec_note": "..." },
    "originality": { "score": 81, "mean_similarity": 0.19, "closest_prior_art": "..." },
    "specificity": { "score": 45, "known_count": 5, "unknown_count": 6, "open_problems": [...] },
    "stability":   { "score": 68, "verdict": "META-STABLE", "metrics": { "log_det_normalised": -5.83, ... } }
  },
  "composite":  { "composite": 67, "weights": {...}, "axes_failed": [] },
  "elapsed_s":  94.3,
  "providers":  { "llm": "anthropic/claude-sonnet-4", "embed": "voyage/voyage-3-large" }
}
```

---

## Research foundations

- Chen et al., ICLR 2024 — *INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection* (EigenScore)
- Farquhar et al., Nature 2024 — *Detecting Hallucinations in Large Language Models Using Semantic Entropy*
- Gell-Mann & Lloyd, 2010 — *Effective Complexity* (complexalism utility axis)
- Anthropic, March 2026 — Dispatch + Computer Use (agent.py)

---

## Related

[SHOAL](https://shoal-production.up.railway.app) — Cognitive quality infrastructure for AI agents. ChiralCheck is a standalone tool built on the same stack.

---

MIT License · Built in Nairobi, 2026
