"""
axes.py — Three-axis idea scorer.

RED TEAM FIXES:
  - All ThreadPoolExecutor f.result() calls are wrapped in try/except
  - _prior_art_vecs raises ProviderError explicitly (caller can handle)
  - _llm_json surfaces parse errors as {"_error": ...} not None
  - Input truncation consistent at 4000 chars for LLM, 2000 for embed
"""

from __future__ import annotations
import json, math, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from providers import LLMProvider, EmbedProvider, ProviderError

_PRIOR_ART = [
    "The labour theory of value holds that the value of a commodity is determined by the socially necessary labour time required to produce it.",
    "Marginal utility theory states that the value of a good is determined by the additional satisfaction obtained from consuming one more unit.",
    "Shannon entropy measures the average information content of a message source and quantifies uncertainty in a probability distribution.",
    "Kolmogorov complexity is the length of the shortest computer program that produces a given string as output on a universal Turing machine.",
    "Neural scaling laws describe power-law relationships between model performance and compute, data, or parameters.",
    "Bayesian inference updates the probability of a hypothesis based on prior beliefs and new evidence via Bayes theorem.",
    "The veil of ignorance thought experiment asks what rules of justice people would choose without knowing their place in society.",
    "Effective complexity measures the algorithmic information content of the regularities in a bit string, separating structured from random.",
    "Non-equilibrium thermodynamics describes how systems far from thermal equilibrium dissipate energy and self-organise.",
    "Reinforcement learning trains agents by rewarding desired behaviours, shaping policy through trial and error.",
]

_pa_cache: dict[str, list[list[float]]] = {}
_pa_lock  = threading.Lock()


def _prior_art_vecs(embed: EmbedProvider) -> list[list[float]]:
    with _pa_lock:
        if embed.name not in _pa_cache:
            vecs = embed.embed(_PRIOR_ART, input_type="document")
            _pa_cache[embed.name] = vecs
    return _pa_cache[embed.name]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _llm_json(llm: LLMProvider, system: str, user: str,
              max_tokens: int = 1024, temperature: float = 0.2) -> dict | list | None:
    try:
        raw = llm.complete(system, user, max_tokens=max_tokens, temperature=temperature).strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw   = parts[1][4:].strip() if parts[1].lower().startswith("json") else parts[1].strip()
        return json.loads(raw)
    except ProviderError as e:
        return {"_error": f"LLM failed: {e}"}
    except json.JSONDecodeError as e:
        return {"_error": f"JSON parse: {e}"}


# ── UTILITY ────────────────────────────────────────────────────────────────────

_U_SYS = """
You are an expert in complexity theory (complexalism framework).
Analyse the IDEA. Return ONLY valid JSON — no markdown, no preamble:
{
  "score": <int 0-100>,
  "isolation_index": <float 0-1, fraction of claims with no mutual support>,
  "claim_count": <int>,
  "load_bearing": [<up to 3 strings: claims the idea cannot survive without>],
  "rationale": "<2 sentences>",
  "ec_note": "<1 sentence on relationship to Effective Complexity>"
}
score = round((1-isolation_index)*70 + impact_quality*30)
impact_quality 0-1: net organised complexity added to the world if implemented.
NOTE: this is a STRUCTURAL PROXY for EC, not EC itself.
""".strip()


def score_utility(text: str, llm: LLMProvider) -> dict:
    r = _llm_json(llm, _U_SYS, f"IDEA:\n\n{text[:4000]}")
    if not r or "_error" in r:
        return {"score": None, "error": (r or {}).get("_error", "LLM returned None")}
    try:
        score = max(0, min(100, int(r["score"])))
        iso   = max(0.0, min(1.0, float(r.get("isolation_index", 0.5))))
    except (KeyError, ValueError, TypeError) as e:
        return {"score": None, "error": f"Malformed: {e}"}
    return {
        "score":           score,
        "isolation_index": round(iso, 3),
        "load_bearing":    r.get("load_bearing", [])[:3],
        "rationale":       r.get("rationale", ""),
        "ec_note":         r.get("ec_note", ""),
        "interpretation":  f"Isolation: {round(iso*100)}% unsupported. {r.get('ec_note','')}",
    }


# ── SPECIFICITY ────────────────────────────────────────────────────────────────

_D_SYS = "Break the IDEA into 8-12 concrete subtasks. Each must be specific enough to determine if a solution exists today. Return ONLY a JSON array of strings."
_C_SYS = 'Is this SUBTASK KNOWN (solved today) or UNKNOWN (open problem)? Return ONLY JSON: {"verdict":"KNOWN"|"UNKNOWN","confidence":<0-1>,"evidence":"<1 sentence>"}'


def score_specificity(text: str, llm: LLMProvider) -> dict:
    raw = _llm_json(llm, _D_SYS, f"IDEA:\n\n{text[:4000]}", max_tokens=1024)
    if not isinstance(raw, list):
        return {"score": None, "error": f"Decomposition failed: {(raw or {}).get('_error', type(raw).__name__)}"}

    subtasks = [str(s).strip() for s in raw if str(s).strip()][:12]
    if not subtasks:
        return {"score": None, "error": "No subtasks extracted"}

    known, unknown, detail = [], [], []

    def _classify(s: str) -> dict:
        r = _llm_json(llm, _C_SYS, f"SUBTASK: {s}", max_tokens=200, temperature=0.1)
        if not r or "_error" in r:
            return {"verdict": "UNKNOWN", "confidence": 0.5, "evidence": "Classification failed"}
        return {"verdict": r.get("verdict","UNKNOWN").upper().strip(),
                "confidence": round(float(r.get("confidence", 0.5)), 2),
                "evidence": r.get("evidence", "")}

    with ThreadPoolExecutor(max_workers=min(len(subtasks), 6)) as ex:
        fmap = {ex.submit(_classify, s): s for s in subtasks}
        for f, s in fmap.items():
            try:
                r = f.result()          # FIX: guarded
            except Exception as e:
                r = {"verdict": "UNKNOWN", "confidence": 0.5, "evidence": str(e)}
            detail.append({"subtask": s[:80], **r})
            (known if r["verdict"] == "KNOWN" else unknown).append(s)

    total = len(known) + len(unknown)
    score = round(len(known) / total * 100) if total else 0
    return {
        "score":           score,
        "known_count":     len(known),
        "unknown_count":   len(unknown),
        "total_subtasks":  total,
        "open_problems":   [d for d in detail if d["verdict"] == "UNKNOWN"],
        "solved_subtasks": [d for d in detail if d["verdict"] == "KNOWN"],
        "interpretation":  (
            f"{len(known)}/{total} subtasks have known solutions. "
            + ("Highly implementable." if score > 70
               else "Key open problems remain." if score > 40
               else "Largely at the research frontier.")
        ),
    }


# ── ORIGINALITY ────────────────────────────────────────────────────────────────

def score_originality(text: str, embed: EmbedProvider) -> dict:
    try:
        idea_vecs = embed.embed([text[:2000]], input_type="query")
    except ProviderError as e:
        return {"score": None, "error": f"Idea embedding failed: {e}"}
    if not idea_vecs:
        return {"score": None, "error": "Empty embedding"}

    try:
        prior_vecs = _prior_art_vecs(embed)
    except ProviderError as e:
        return {"score": None, "error": str(e)}

    sims     = [_cosine(idea_vecs[0], p) for p in prior_vecs]
    mean_sim = sum(sims) / len(sims)
    max_sim  = max(sims)
    closest  = _PRIOR_ART[sims.index(max_sim)]
    score    = max(0, min(100, round((1.0 - mean_sim) * 100)))
    return {
        "score":             score,
        "mean_similarity":   round(mean_sim, 4),
        "closest_prior_art": closest[:80] + "...",
        "interpretation": (
            f"Mean cosine sim to prior-art: {round(mean_sim*100)}%. "
            f"Closest: \"{closest[:55]}...\". "
            + ("Highly original." if score > 70
               else "Moderate novelty." if score > 40
               else "Significant overlap with existing frameworks.")
        ),
    }
