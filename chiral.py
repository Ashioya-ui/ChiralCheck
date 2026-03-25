"""
chiral.py — ChiralCheck: Black-box Semantic Stability Auditor
by William Ashioya | SHOAL

Measures geometric spread of adversarial perturbation embeddings.
Tight cloud = stable attractor. Diffuse cloud = fragile/low-coherence.

Black-box approximation of INSIDE/EigenScore (Chen et al. ICLR 2024).
Does not require white-box LLM access — works with any provider.
"""

from __future__ import annotations
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from providers import LLMProvider, EmbedProvider, ProviderError

_ANGLES: list[tuple[str, float]] = [
    ("adversarial counter-argument",           0.9),
    ("simplified Feynman-style explanation",    0.4),
    ("structural rephrasing",                  0.7),
    ("devil's advocate rebuttal",              1.0),
    ("supportive evidence elaboration",         0.5),
    ("cross-domain analogy",                   0.8),
    ("null-hypothesis formulation",             0.7),
    ("Socratic interrogation",                 0.8),
]

_SYS = "Output only the requested rewritten text. No preamble. No labels."


def _perturb(text: str, angle: str, temp: float, llm: LLMProvider) -> str | None:
    try:
        r = llm.complete(
            _SYS,
            f"Rewrite this idea from the angle of a {angle}. "
            f"Radically alter syntax, preserve core claim. ONLY the rewrite.\n\n{text[:2000]}",
            max_tokens=400, temperature=temp,
        )
        return r if r and r.strip() != text.strip() else None
    except ProviderError:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _clip(vecs: list[list[float]], pct: float = 0.90) -> list[list[float]]:
    d   = len(vecs[0])
    ma  = [sum(abs(v[i]) for v in vecs) / len(vecs) for i in range(d)]
    thr = sorted(ma)[min(int(pct * d), d - 1)]
    return [[0.0 if ma[i] > thr else x for i, x in enumerate(v)] for v in vecs]


def _log_det(vecs: list[list[float]]) -> float:
    if len(vecs) < 2: return 0.0
    d  = len(vecs[0])
    mu = [sum(v[i] for v in vecs) / len(vecs) for i in range(d)]
    va = [sum((v[i] - mu[i]) ** 2 for v in vecs) / (len(vecs) - 1) for i in range(d)]
    return sum(math.log(v + 1e-4) for v in va) / d


def _entropy(vecs: list[list[float]], thr: float = 0.82) -> float:
    centroids: list[list[float]] = []; counts: list[int] = []
    for v in vecs:
        j = next((j for j, c in enumerate(centroids) if _cosine(v, c) >= thr), None)
        if j is not None:
            n = counts[j]
            centroids[j] = [(centroids[j][i]*n + v[i])/(n+1) for i in range(len(v))]
            counts[j] += 1
        else:
            centroids.append(list(v)); counts.append(1)
    total = sum(counts)
    if total == 0: return 0.0
    return -sum((c/total)*math.log(c/total) for c in counts if c > 0)


def _count_clusters(vecs: list[list[float]], thr: float = 0.82) -> int:
    cs = []
    for v in vecs:
        if not any(_cosine(v, c) >= thr for c in cs): cs.append(v)
    return len(cs)


def _to_score(nd: float) -> int:
    if nd < -6.5: return min(100, round(85 + abs(nd + 6.5) * 4))
    if nd < -5.5: return round(65 + abs(nd + 5.5) * 20)
    if nd < -4.5: return round(40 + abs(nd + 4.5) * 25)
    return max(0, round(40 + (nd + 4.5) * 8))


def audit(thesis: str, llm: LLMProvider, embed: EmbedProvider, n: int = 8) -> dict:
    configs = _ANGLES[:n]

    perturbs: list[str] = []
    with ThreadPoolExecutor(max_workers=min(n, 8)) as ex:
        futures = [ex.submit(_perturb, thesis, a, t, llm) for a, t in configs]
        for f in as_completed(futures):
            try:
                r = f.result()
                if r: perturbs.append(r)
            except Exception:
                pass

    if len(perturbs) < 3:
        return {"score": None, "error": f"Only {len(perturbs)} perturbations generated (need ≥3)"}

    try:
        all_vecs = embed.embed([thesis] + perturbs, input_type="document")
    except ProviderError as e:
        return {"score": None, "error": f"Embedding failed: {e}"}

    if not all_vecs or len(all_vecs) < 4:
        return {"score": None, "error": "Too few embeddings"}

    clipped   = _clip(all_vecs)
    base_vec  = clipped[0]
    pert_vecs = clipped[1:]

    nd          = _log_det(pert_vecs)
    sem_entropy = _entropy(pert_vecs)
    sims        = [_cosine(base_vec, p) for p in pert_vecs]
    legacy_ir   = round(sum(sims) / len(sims), 4)

    score = _to_score(nd)
    if sem_entropy > 2.0: score = max(0, score - 15)
    elif sem_entropy > 1.5: score = max(0, score - 7)
    score = max(0, min(100, score))

    verdict  = ("STABLE ATTRACTOR"   if score >= 75
                else "META-STABLE"   if score >= 50
                else "FRAGILE / LOW-COHERENCE")
    illusion = legacy_ir > 0.85 and score < 55

    return {
        "score":   score,
        "verdict": verdict,
        "metrics": {
            "log_det_normalised":    round(nd, 4),
            "semantic_entropy_nats": round(sem_entropy, 4),
            "semantic_clusters":     _count_clusters(pert_vecs),
            "legacy_ir_cosine":      legacy_ir,
            "perturbations_run":     len(pert_vecs),
        },
        "semantic_illusion_warning": illusion,
        "interpretation": (
            f"Log-det: {round(nd,3)} | Entropy: {round(sem_entropy,3)} nats | {verdict}."
            + (" ⚠ Semantic Illusion: T1 IR said STABLE — geometry overrules." if illusion else "")
        ),
    }
