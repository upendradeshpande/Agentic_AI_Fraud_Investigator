"""Provider-agnostic LLM layer.

Switch models without code changes, either
  - with environment variables (.env):
        LLM_PROVIDER = gemini | anthropic | openai | openai_compatible | none
        LLM_MODEL    = optional override (else GEMINI_MODEL / ANTHROPIC_MODEL / OPENAI_MODEL / default)
  - or at runtime from the UI ("Model connection"), which calls `configure(...)`.
    A runtime connection is private to one browser session: the UI stores it in that session's server-side
    state and activates it with `use_runtime()` at the start of every page run. Keys are never written to disk,
    logged, or sent back to the browser, and one visitor's key is never used for another visitor.

All providers are called over REST with `requests`, so no vendor SDK is required.
Canonical message format used by the agents:
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Tools: [{"name", "description", "parameters": JSON schema}]

To add a provider: subclass LLMProvider, implement `chat`, register it in PROVIDERS.
"""
from __future__ import annotations

import contextvars
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests
from urllib.parse import quote, urlparse

from ..config import settings


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    raw: Optional[dict] = None


class LLMError(RuntimeError):
    pass


RETRYABLE = {429, 500, 502, 503, 504}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def validate_base_url(url: str) -> None:
    """HTTPS only (plain HTTP allowed for a local dev server); no embedded credentials or query strings."""
    if not url:
        return
    u = urlparse(url)
    if not u.hostname:
        raise LLMError("Invalid API base URL.")
    if u.username or u.password or u.query or u.fragment:
        raise LLMError("API base URL must not contain credentials, query strings or fragments.")
    if u.scheme != "https" and not (u.scheme == "http" and u.hostname in LOCAL_HOSTS):
        raise LLMError("Use an https:// URL (http:// is only allowed for localhost).")


class LLMProvider:
    name = "base"
    key_env: tuple[str, ...] = ()
    default_base_url = ""

    def __init__(self, model: str, timeout: int = 45, api_key: str = "", base_url: str = ""):
        self.model = model
        self.timeout = timeout
        self._api_key = api_key
        self.base_url = (base_url or "").rstrip("/")
        validate_base_url(self.base_url)

    @property
    def api_key(self) -> str:
        if self._api_key:
            return self._api_key
        for name in self.key_env:
            if os.environ.get(name):
                return os.environ[name]
        return ""

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, messages: list[dict], tools: Optional[list[dict]] = None,
             max_tokens: int = 1200, temperature: float = 0.1, json_mode: bool = False) -> LLMResponse:
        raise NotImplementedError

    def _post(self, url, headers, body):
        """POST with one retry on transient errors. Error text never includes keys or request bodies."""
        t0 = time.time()
        for attempt in range(2):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout, allow_redirects=False)
            except requests.Timeout as e:
                if attempt == 0:
                    continue
                raise LLMError(f"{self.name} request timed out after {self.timeout}s") from e
            except requests.RequestException as e:
                if attempt == 0:
                    time.sleep(0.5)
                    continue
                raise LLMError(f"{self.name} connection failed: {e.__class__.__name__}") from e
            if r.status_code in RETRYABLE and attempt == 0:
                time.sleep(1.0)
                continue
            if r.status_code >= 300:
                raise LLMError(f"{self.name} HTTP {r.status_code}: {_safe_error(r)}")
            return r.json(), int((time.time() - t0) * 1000)
        raise LLMError(f"{self.name} request failed after retry")


def _safe_error(r) -> str:
    """Provider error message only (e.g. 'model not found'), truncated. Never echoes headers."""
    try:
        data = r.json()
        err = data.get("error", data)
        msg = err.get("message") if isinstance(err, dict) else str(err)
        return str(msg)[:300]
    except ValueError:
        return r.text[:200]


# --------------------------------------------------------------------------- Anthropic
class AnthropicProvider(LLMProvider):
    name = "anthropic"
    key_env = ("ANTHROPIC_API_KEY", "LLM_API_KEY")
    default_base_url = "https://api.anthropic.com/v1"

    @property
    def url(self):
        return (self.base_url or self.default_base_url) + "/messages"

    def _convert(self, messages):
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                for tc in m.get("tool_calls", []):
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
                out.append({"role": "assistant", "content": blocks})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def chat(self, system, messages, tools=None, max_tokens=1200, temperature=0.1, json_mode=False):
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                "system": system, "messages": self._convert(messages)}
        if tools:
            body["tools"] = [{"name": t["name"], "description": t["description"],
                              "input_schema": t["parameters"]} for t in tools]
        headers = {"x-api-key": self.api_key,
                   "anthropic-version": "2023-06-01", "content-type": "application/json"}
        data, ms = self._post(self.url, headers, body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        calls = [ToolCall(b["id"], b["name"], b.get("input") or {})
                 for b in data.get("content", []) if b.get("type") == "tool_use"]
        u = data.get("usage", {})
        return LLMResponse(text, calls, u.get("input_tokens", 0), u.get("output_tokens", 0), ms, data)


# ------------------------------------------------------------------------------ OpenAI
class OpenAIProvider(LLMProvider):
    name = "openai"
    key_env = ("OPENAI_API_KEY", "LLM_API_KEY")
    default_base_url = "https://api.openai.com/v1"

    @property
    def url(self):
        base = self.base_url or os.environ.get("OPENAI_BASE_URL", "").rstrip("/") or self.default_base_url
        return base + "/chat/completions"


    def _convert(self, system, messages):
        out = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                                          "function": {"name": tc["name"], "arguments": json.dumps(tc["args"])}}
                                         for tc in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            else:
                out.append({"role": "user", "content": m["content"]})
        return out

    def chat(self, system, messages, tools=None, max_tokens=1200, temperature=0.1, json_mode=False):
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": self._convert(system, messages)}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
        if json_mode and not tools:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data, ms = self._post(self.url, headers, body)
        msg = data["choices"][0]["message"]
        calls = [ToolCall(tc["id"], tc["function"]["name"], json.loads(tc["function"].get("arguments") or "{}"))
                 for tc in msg.get("tool_calls") or []]
        u = data.get("usage", {})
        return LLMResponse(msg.get("content") or "", calls, u.get("prompt_tokens", 0),
                           u.get("completion_tokens", 0), ms, data)


class OpenAICompatibleProvider(OpenAIProvider):
    """Any Chat Completions compatible server: vLLM, Ollama, LM Studio, Azure proxy, LiteLLM gateway.
    A key is optional (local servers usually need none) but a base URL is required."""
    name = "openai_compatible"
    key_env = ("LLM_API_KEY", "OPENAI_API_KEY")

    @property
    def available(self):
        return bool(self.base_url or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))

    @property
    def url(self):
        base = self.base_url or os.environ.get("LLM_BASE_URL", "").rstrip("/") or os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
        return base + "/chat/completions"


# ------------------------------------------------------------------------------ Gemini
def _gemini_schema(schema: dict) -> dict:
    """Gemini accepts an OpenAPI subset: upper-case types, no additionalProperties."""
    out = {}
    for k, v in schema.items():
        if k == "additionalProperties":
            continue
        if k == "type" and isinstance(v, str):
            out[k] = v.upper()
        elif k == "properties":
            out[k] = {pk: _gemini_schema(pv) for pk, pv in v.items()}
        elif k == "items":
            out[k] = _gemini_schema(v)
        else:
            out[k] = v
    return out


class GeminiProvider(LLMProvider):
    name = "gemini"
    key_env = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_KEY")
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def _convert(self, messages):
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "parts": [{"text": m["content"]}]})
            elif m["role"] == "assistant":
                parts = [{"text": m["content"]}] if m.get("content") else []
                parts += [{"functionCall": {"name": tc["name"], "args": tc["args"]}} for tc in m.get("tool_calls", [])]
                out.append({"role": "model", "parts": parts})
            elif m["role"] == "tool":
                try:
                    payload = json.loads(m["content"])
                except (TypeError, ValueError):
                    payload = m["content"]
                part = {"functionResponse": {"name": m["name"], "response": {"result": payload}}}
                if out and out[-1]["role"] == "user" and "functionResponse" in out[-1]["parts"][0]:
                    out[-1]["parts"].append(part)
                else:
                    out.append({"role": "user", "parts": [part]})
        return out

    def chat(self, system, messages, tools=None, max_tokens=1200, temperature=0.1, json_mode=False):
        url = f"{self.base_url or self.default_base_url}/models/{quote(self.model, safe='')}:generateContent"
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": self._convert(messages),
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        if tools:
            body["tools"] = [{"functionDeclarations": [
                {"name": t["name"], "description": t["description"], "parameters": _gemini_schema(t["parameters"])}
                for t in tools]}]
        elif json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        data, ms = self._post(url, headers, body)
        cands = data.get("candidates") or []
        if not cands:
            raise LLMError(f"gemini returned no candidates: {json.dumps(data)[:300]}")
        parts = cands[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought"))
        calls = [ToolCall(f"call_{uuid.uuid4().hex[:8]}", p["functionCall"]["name"], p["functionCall"].get("args") or {})
                 for p in parts if "functionCall" in p]
        u = data.get("usageMetadata", {})
        return LLMResponse(text, calls, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0), ms, data)


# ------------------------------------------------------------------------------- None
class NullProvider(LLMProvider):
    """No LLM configured. Agents detect this and use deterministic fallbacks."""
    name = "none"

    @property
    def available(self):
        return False

    def chat(self, *a, **k):
        raise LLMError("No LLM provider configured")


PROVIDERS = {"anthropic": AnthropicProvider, "openai": OpenAIProvider, "gemini": GeminiProvider,
             "openai_compatible": OpenAICompatibleProvider, "none": NullProvider}

# Runtime override for the current session/thread only (a ContextVar, not a process-wide global),
# so concurrent visitors of a hosted app never share a connection or a key.
_RUNTIME_VAR: contextvars.ContextVar = contextvars.ContextVar("llm_runtime", default=None)


def _runtime() -> dict:
    return _RUNTIME_VAR.get() or {}


def runtime_config() -> Optional[dict]:
    """The current session's runtime connection (to store in that session's state)."""
    cfg = _RUNTIME_VAR.get()
    return dict(cfg) if cfg else None


def use_runtime(cfg: Optional[dict]) -> None:
    """Activate a session's stored connection for this page run (None = use .env settings)."""
    _RUNTIME_VAR.set(dict(cfg) if cfg else None)


def configure(provider: str, model: str = "", api_key: str = "", base_url: str = "") -> LLMProvider:
    """Switch provider/model at runtime. Validates, then returns the new provider (not yet called)."""
    provider = provider.lower().strip()
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}'. Options: {', '.join(PROVIDERS)}")
    from ..config import DEFAULT_MODELS
    model = model.strip() or DEFAULT_MODELS.get(provider, "")
    if provider == "openai_compatible" and not base_url:
        raise LLMError("openai_compatible needs a base URL, usually ending in /v1.")
    validate_base_url(base_url.strip())
    _RUNTIME_VAR.set({"provider": provider, "model": model, "api_key": api_key.strip(), "base_url": base_url.strip()})
    return get_provider()


def build(provider: str, model: str = "", api_key: str = "", base_url: str = "") -> LLMProvider:
    """Create a provider from explicit values without making it active (used by the UI's Test button)."""
    provider = provider.lower().strip()
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}'.")
    from ..config import DEFAULT_MODELS
    validate_base_url(base_url.strip())
    return PROVIDERS[provider](model.strip() or DEFAULT_MODELS.get(provider, ""), timeout=settings.llm_timeout_s,
                               api_key=api_key.strip(), base_url=base_url.strip())


def reset_runtime() -> None:
    _RUNTIME_VAR.set(None)


def get_provider(name: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    rt = _runtime() if not name else {}
    name = (name or rt.get("provider") or settings.llm_provider).lower()
    if not settings.enable_llm:
        name = "none"
    cls = PROVIDERS.get(name)
    if cls is None:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Options: {', '.join(PROVIDERS)}")
    from ..config import DEFAULT_MODELS
    model = model or rt.get("model") or (settings.llm_model if name == settings.llm_provider else DEFAULT_MODELS.get(name, ""))
    return cls(model, timeout=settings.llm_timeout_s, api_key=rt.get("api_key", ""),
               base_url=rt.get("base_url") or (os.environ.get("LLM_BASE_URL", "") if name == "openai_compatible" else ""))


def llm_available() -> bool:
    """True when an LLM is enabled and the active provider has what it needs (key or local URL)."""
    return settings.enable_llm and get_provider().available


def describe() -> str:
    p = get_provider()
    if not p.available:
        return "deterministic mode (no LLM connected)"
    return f"{p.name}:{p.model}"


def check_connection(provider: Optional[LLMProvider] = None) -> dict:
    """One tiny synthetic request to confirm key, model and endpoint. Billable (a few tokens)."""
    p = provider or get_provider()
    if not p.available:
        return {"ok": False, "message": "No key or endpoint configured for this provider."}
    try:
        r = p.chat("Reply with the single word OK.", [{"role": "user", "content": "Connectivity test."}], max_tokens=20)
    except LLMError as e:
        return {"ok": False, "message": str(e)}
    return {"ok": True, "message": f"{p.name}:{p.model} replied in {r.latency_ms} ms "
                                   f"({r.input_tokens} in / {r.output_tokens} out tokens).", "reply": r.text[:60]}


def parse_json(text: str) -> dict:
    """Parse model JSON output, tolerating code fences and leading prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[4:] if t.lower().startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    return json.loads(t[start:end + 1])
