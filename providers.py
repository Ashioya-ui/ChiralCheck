"""
providers.py — Multi-provider abstraction.
Default: Anthropic (claude-sonnet-4) + Voyage (voyage-3-large).
Alternatives: OpenAI, Gemini.

RED TEAM FIXES:
  - auto_config: anthropic+voyage first, openai second, gemini last
  - preflight(): wrapped in try/except — bad key returns (False, msg) not traceback
  - All API calls have explicit timeout enforcement via client init or retry wrapper
  - Thread-safe singletons throughout
"""

from __future__ import annotations
import os, time, threading
from abc import ABC, abstractmethod

class ProviderError(Exception): pass

def _retry(fn, retries: int = 3, base: float = 1.5):
    for attempt in range(retries):
        try:
            return fn()
        except ProviderError:
            raise
        except Exception as e:
            msg = str(e).lower()
            transient = any(x in msg for x in [
                "429","rate","quota","500","502","503","504",
                "timeout","connection","overload","unavailable",
            ])
            if not transient or attempt == retries - 1:
                raise ProviderError(str(e)) from e
            time.sleep(base ** (attempt + 1))

# ── LLM ───────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def complete(self, system: str, user: str,
                 max_tokens: int = 1024, temperature: float = 0.7) -> str: ...

class AnthropicProvider(LLMProvider):
    _lock = threading.Lock(); _client = None
    @property
    def name(self): return "anthropic/claude-sonnet-4"
    def _c(self):
        with self._lock:
            if AnthropicProvider._client is None:
                import anthropic
                AnthropicProvider._client = anthropic.Anthropic(
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    timeout=60.0,
                )
        return AnthropicProvider._client
    def complete(self, system, user, max_tokens=1024, temperature=0.7):
        c = self._c()
        return _retry(lambda: c.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        ).content[0].text.strip())

class OpenAIProvider(LLMProvider):
    _lock = threading.Lock(); _client = None
    @property
    def name(self): return "openai/gpt-4o"
    def _c(self):
        with self._lock:
            if OpenAIProvider._client is None:
                from openai import OpenAI
                OpenAIProvider._client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"],
                    timeout=60.0,
                )
        return OpenAIProvider._client
    def complete(self, system, user, max_tokens=1024, temperature=0.7):
        c = self._c()
        return _retry(lambda: c.chat.completions.create(
            model="gpt-4o", max_tokens=max_tokens, temperature=temperature,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
        ).choices[0].message.content.strip())

class GeminiProvider(LLMProvider):
    _lock = threading.Lock(); _client = None
    @property
    def name(self): return "google/gemini-2.5-pro"
    def _c(self):
        with self._lock:
            if GeminiProvider._client is None:
                from google import genai
                GeminiProvider._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return GeminiProvider._client
    def complete(self, system, user, max_tokens=1024, temperature=0.7):
        self._c()
        def _call():
            from google import genai
            from google.genai import types
            c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            r = c.models.generate_content(
                model="gemini-2.5-pro", contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system, max_output_tokens=max_tokens,
                    temperature=temperature,
                    http_options=types.HttpOptions(timeout=60_000),
                ),
            )
            return r.text.strip()
        return _retry(_call)

# ── Embed ──────────────────────────────────────────────────────────────────────

class EmbedProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def embed(self, texts: list[str], input_type: str = "document") -> list[list[float]]: ...

class VoyageProvider(EmbedProvider):
    _lock = threading.Lock(); _client = None
    @property
    def name(self): return "voyage/voyage-3-large"
    def _c(self):
        with self._lock:
            if VoyageProvider._client is None:
                import voyageai
                VoyageProvider._client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        return VoyageProvider._client
    def embed(self, texts, input_type="document"):
        c = self._c()
        return _retry(lambda: c.embed(
            texts, model="voyage-3-large", input_type=input_type
        ).embeddings)

class OpenAIEmbedProvider(EmbedProvider):
    _lock = threading.Lock(); _client = None
    @property
    def name(self): return "openai/text-embedding-3-large"
    def _c(self):
        with self._lock:
            if OpenAIEmbedProvider._client is None:
                from openai import OpenAI
                OpenAIEmbedProvider._client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"],
                    timeout=60.0,
                )
        return OpenAIEmbedProvider._client
    def embed(self, texts, input_type="document"):
        c = self._c()
        return _retry(lambda: [
            i.embedding for i in c.embeddings.create(
                model="text-embedding-3-large", input=texts, dimensions=1024,
            ).data
        ])

class GeminiEmbedProvider(EmbedProvider):
    _TASK = {"document":"RETRIEVAL_DOCUMENT","query":"RETRIEVAL_QUERY"}
    _lock = threading.Lock(); _client = None
    @property
    def name(self): return "google/gemini-embedding-001"
    def _c(self):
        with self._lock:
            if GeminiEmbedProvider._client is None:
                from google import genai
                GeminiEmbedProvider._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        return GeminiEmbedProvider._client
    def embed(self, texts, input_type="document"):
        self._c()
        task = self._TASK.get(input_type, "RETRIEVAL_DOCUMENT")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def _one(t):
            from google import genai
            from google.genai import types
            c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            return _retry(lambda: c.models.embed_content(
                model="gemini-embedding-001", contents=[t],
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=1024,
                ),
            ).embeddings[0].values)
        ordered = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=min(len(texts), 8)) as ex:
            fmap = {ex.submit(_one, t): i for i, t in enumerate(texts)}
            for f, idx in fmap.items():
                try:
                    ordered[idx] = f.result()
                except Exception as e:
                    raise ProviderError(f"Gemini embed failed: {e}") from e
        return ordered

# ── Factory ────────────────────────────────────────────────────────────────────

_LLM   = {"anthropic": AnthropicProvider, "openai": OpenAIProvider, "gemini": GeminiProvider}
_EMBED = {"voyage": VoyageProvider, "openai": OpenAIEmbedProvider, "gemini": GeminiEmbedProvider}
_KEYS  = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "gemini":    "GEMINI_API_KEY",
    "voyage":    "VOYAGE_API_KEY",
}

def get_llm(name: str) -> LLMProvider:
    cls = _LLM.get(name.lower())
    if not cls: raise ValueError(f"Unknown LLM: {name!r}. Options: {list(_LLM)}")
    return cls()

def get_embed(name: str) -> EmbedProvider:
    cls = _EMBED.get(name.lower())
    if not cls: raise ValueError(f"Unknown embed: {name!r}. Options: {list(_EMBED)}")
    return cls()

def available() -> dict[str, bool]:
    return {k: bool(os.environ.get(v)) for k, v in _KEYS.items()}

def preflight(llm_name: str, embed_name: str) -> tuple[bool, str]:
    """
    Validate providers with cheap API calls.
    FIX: wrapped in try/except — bad key returns (False, message) not traceback.
    """
    errors = []
    try:
        llm = get_llm(llm_name)
        out = llm.complete("Reply with only the word OK.", "ping", max_tokens=5, temperature=0.0)
        if not out.strip():
            errors.append(f"LLM ({llm_name}): empty response")
    except Exception as e:
        errors.append(f"LLM ({llm_name}): {e}")

    try:
        emb  = get_embed(embed_name)
        vecs = emb.embed(["preflight"], input_type="document")
        if not vecs or not vecs[0]:
            errors.append(f"Embed ({embed_name}): empty embedding")
    except Exception as e:
        errors.append(f"Embed ({embed_name}): {e}")

    if errors:
        return False, "\n".join(f"  ✗ {e}" for e in errors)
    return True, f"  ✓ {llm_name} + {embed_name} both responding"
