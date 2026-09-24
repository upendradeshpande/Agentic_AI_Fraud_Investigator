"""Central configuration. Everything is driven by environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader so the app works without python-dotenv."""
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _load_host_secrets() -> None:
    """Streamlit Community Cloud 'Secrets' (secrets.toml): copy root-level values into the environment so the
    same settings work locally (.env) and hosted. Real environment variables and .env take precedence."""
    try:
        import streamlit as st
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))
    except Exception:  # no Streamlit, no secrets file, or not running under Streamlit
        pass


_load_dotenv()
_load_host_secrets()


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _detect_provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    # Auto-detect: use whichever key is present. Spec default is Gemini.
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
    "none": "deterministic-fallback",
}


@dataclass(frozen=True)
class Settings:
    database_url: str = os.environ.get("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'cockpit.db'}")
    sample_csv: Path = Path(os.environ.get("SAMPLE_CSV", ROOT / "data" / "sample_cases_synthetic.csv"))
    ruleset_path: Path = Path(os.environ.get("RULESET_PATH", ROOT / "cockpit" / "rules" / "ruleset_v1.json"))

    enable_llm: bool = _flag("ENABLE_LLM", True)
    enable_rag: bool = _flag("ENABLE_RAG", False)

    llm_provider: str = _detect_provider()
    llm_model: str = ""
    llm_timeout_s: int = int(os.environ.get("LLM_TIMEOUT_S", "45"))
    agent_max_tool_calls: int = int(os.environ.get("AGENT_MAX_TOOL_CALLS", "6"))
    agent_runtime: str = os.environ.get("AGENT_RUNTIME", "native")  # native | adk

    # Cost estimate only. Verify against your provider's current pricing page.
    price_input_per_mtok: float = float(os.environ.get("LLM_PRICE_INPUT_PER_MTOK", "0.30"))
    price_output_per_mtok: float = float(os.environ.get("LLM_PRICE_OUTPUT_PER_MTOK", "2.50"))

    def __post_init__(self):
        model = os.environ.get("LLM_MODEL") or os.environ.get(
            {"gemini": "GEMINI_MODEL", "anthropic": "ANTHROPIC_MODEL", "openai": "OPENAI_MODEL"}.get(
                self.llm_provider, "LLM_MODEL"), "")
        object.__setattr__(self, "llm_model", model or DEFAULT_MODELS.get(self.llm_provider, ""))

    @property
    def sqlite_path(self) -> str:
        url = self.database_url
        if not url.startswith("sqlite:///"):
            raise ValueError("Prototype supports sqlite:/// URLs only. PostgreSQL is the production path.")
        path = Path(url.replace("sqlite:///", "", 1))
        return str(path if path.is_absolute() else ROOT / path)

    @property
    def llm_active(self) -> bool:
        return self.enable_llm and self.llm_provider != "none"


settings = Settings()
