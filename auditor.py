"""
auditor.py — Orchestrator: 4 axes → composite.

RED TEAM FIXES:
  - auto_config: anthropic+voyage → openai → gemini (was gemini first)
  - _composite: uses `is not None` check — score=0 is valid, not a failure
  - ThreadPoolExecutor: all f.result() calls guarded
  - load() defaults: anthropic + voyage
"""

from __future__ import annotations
import os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from providers import LLMProvider, EmbedProvider, get_llm, get_embed, available

load_dotenv()

WEIGHTS = {"utility": 0.35, "originality": 0.25, "specificity": 0.25, "stability": 0.15}


class Config:
    def __init__(self, llm: LLMProvider, embed: EmbedProvider,
                 llm_name: str, embed_name: str):
        self.llm        = llm
        self.embed      = embed
        self.llm_name   = llm_name
        self.embed_name = embed_name

    def describe(self) -> str:
        return f"{self.llm.name}  +  {self.embed.name}"


def load(llm_name: str | None = None, embed_name: str | None = None) -> Config:
    # FIX: default is anthropic + voyage, not gemini
    ln = (llm_name   or os.environ.get("LLM_PROVIDER",   "anthropic")).lower()
    en = (embed_name or os.environ.get("EMBED_PROVIDER",  "voyage")).lower()
    _KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
             "gemini": "GEMINI_API_KEY", "voyage": "VOYAGE_API_KEY"}
    missing = [f"  {_KEYS[p]}  (for {p!r})"
               for p in {ln, en} if _KEYS.get(p) and not os.environ.get(_KEYS[p])]
    if missing:
        raise ValueError("Missing keys:\n" + "\n".join(missing) + "\n\nSee .env.example")
    return Config(get_llm(ln), get_embed(en), ln, en)


def auto_config() -> Config:
    # FIX: priority anthropic+voyage → openai+openai → gemini+gemini
    av = available()
    if av.get("anthropic") and av.get("voyage"):  return load("anthropic", "voyage")
    if av.get("openai"):                           return load("openai",    "openai")
    if av.get("gemini"):                           return load("gemini",    "gemini")
    raise ValueError(
        "No API keys found. Set one of:\n"
        "  ANTHROPIC_API_KEY + VOYAGE_API_KEY   (recommended)\n"
        "  OPENAI_API_KEY\n"
        "  GEMINI_API_KEY\n"
        "See .env.example"
    )


def _composite(axes: dict) -> dict:
    # FIX: score is not None (was truthy — treated score=0 as failure)
    valid = {k: v["score"] for k, v in axes.items() if v.get("score") is not None}
    if not valid:
        return {"composite": None, "axes_failed": list(WEIGHTS)}
    tw   = sum(WEIGHTS[k] for k in valid)
    comp = round(sum(valid[k] * WEIGHTS[k] / tw for k in valid))
    return {
        "composite":   comp,
        "weights":     {k: round(WEIGHTS[k], 2) for k in valid},
        "axes_failed": [k for k in WEIGHTS if k not in valid],
    }


def audit(
    text:    str,
    cfg:     Config,
    verbose: bool = True,
    quick:   bool = False,
) -> dict:
    from axes import score_utility, score_specificity, score_originality
    import chiral

    n_perturbs = 4 if quick else 8

    def _done(axis: str, score):
        if verbose:
            s = f"{score}/100" if score is not None else "ERROR"
            print(f"\r  ├─ {axis:<14} {s:<10}", flush=True)

    if verbose:
        print(f"\n  ┌─ AUDIT ──────────────────────────────────────────────────────")
        print(f"  │  {cfg.describe()}")
        print(f"  │  {'[quick — 4 perturbations]' if quick else '[full — 8 perturbations]'}")
        print(f"  │")

    t0      = time.time()
    results: dict[str, dict] = {}

    def _run(key, fn, *args, **kwargs):
        try:
            return key, fn(*args, **kwargs)
        except Exception as e:
            return key, {"score": None, "error": str(e)}

    def r_utility():
        if verbose: print(f"  │  [1/4] utility...   ", end="", flush=True)
        k, v = _run("utility", score_utility, text, cfg.llm)
        _done("UTILITY", v.get("score"))
        return k, v

    def r_originality():
        if verbose: print(f"  │  [2/4] originality..", end="", flush=True)
        k, v = _run("originality", score_originality, text, cfg.embed)
        _done("ORIGINALITY", v.get("score"))
        return k, v

    def r_specificity():
        if verbose: print(f"  │  [3/4] specificity..", end="", flush=True)
        k, v = _run("specificity", score_specificity, text, cfg.llm)
        _done("SPECIFICITY", v.get("score"))
        return k, v

    def r_stability():
        if verbose: print(f"  │  [4/4] stability... ", end="", flush=True)
        k, v = _run("stability", chiral.audit, text, cfg.llm, cfg.embed, n=n_perturbs)
        _done("STABILITY", v.get("score"))
        return k, v

    # Parallel: utility + originality + specificity
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = [ex.submit(r_utility), ex.submit(r_originality), ex.submit(r_specificity)]
        for f in as_completed(futures):
            try:
                k, v = f.result()   # FIX: guarded
                results[k] = v
            except Exception as e:
                pass  # individual axis errors already caught inside _run

    # Stability last
    try:
        k, v = r_stability()
        results[k] = v
    except Exception as e:
        results["stability"] = {"score": None, "error": str(e)}

    # Ensure all keys present
    for key in WEIGHTS:
        if key not in results:
            results[key] = {"score": None, "error": "Task did not complete"}

    elapsed = round(time.time() - t0, 1)
    comp    = _composite(results)

    if verbose:
        c = comp.get("composite")
        print(f"  │")
        print(f"  └─ COMPOSITE ─────── {c}/100   ({elapsed}s)")
        if comp.get("axes_failed"):
            print(f"     failed: {', '.join(comp['axes_failed'])}")
        print()

    return {
        "idea_preview": text.strip()[:120] + ("..." if len(text) > 120 else ""),
        "axes":         results,
        "composite":    comp,
        "elapsed_s":    elapsed,
        "providers":    {"llm": cfg.llm.name, "embed": cfg.embed.name},
        "quick_mode":   quick,
    }
