"""Typed runtime configuration loaded from environment variables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


def _default(name: str) -> object:
    return Settings.__dataclass_fields__[name].default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Application settings with conservative local defaults."""

    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    cpu_model_id: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    force_large_model: bool = False
    data_dir: Path = Path(".agent_data")
    log_level: str = "INFO"
    max_new_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.9
    enable_4bit: bool = True
    enable_tools: bool = True
    shell_timeout: int = 30
    max_reflection_retries: int = 2
    short_term_limit: int = 12
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_mode: str = "build"
    default_effort: str = "medium"

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from `.env` and process environment."""

        load_dotenv()
        data_dir = Path(os.getenv("AGENT_DATA_DIR", ".agent_data")).expanduser()
        return cls(
            model_id=os.getenv("AGENT_MODEL_ID", str(_default("model_id"))),
            cpu_model_id=os.getenv("AGENT_CPU_MODEL_ID", str(_default("cpu_model_id"))),
            force_large_model=_bool_env(
                "AGENT_FORCE_LARGE_MODEL", bool(_default("force_large_model"))
            ),
            data_dir=data_dir,
            log_level=os.getenv("AGENT_LOG_LEVEL", str(_default("log_level"))).upper(),
            max_new_tokens=int(os.getenv("AGENT_MAX_NEW_TOKENS", _default("max_new_tokens"))),
            temperature=float(os.getenv("AGENT_TEMPERATURE", _default("temperature"))),
            top_p=float(os.getenv("AGENT_TOP_P", _default("top_p"))),
            enable_4bit=_bool_env("AGENT_ENABLE_4BIT", bool(_default("enable_4bit"))),
            enable_tools=_bool_env("AGENT_ENABLE_TOOLS", bool(_default("enable_tools"))),
            shell_timeout=int(os.getenv("AGENT_SHELL_TIMEOUT", _default("shell_timeout"))),
            max_reflection_retries=int(
                os.getenv(
                    "AGENT_MAX_REFLECTION_RETRIES",
                    _default("max_reflection_retries"),
                )
            ),
            short_term_limit=int(
                os.getenv("AGENT_SHORT_TERM_LIMIT", _default("short_term_limit"))
            ),
            embedding_model=os.getenv(
                "AGENT_EMBEDDING_MODEL", str(_default("embedding_model"))
            ),
            default_mode=os.getenv("AGENT_DEFAULT_MODE", str(_default("default_mode"))),
            default_effort=os.getenv(
                "AGENT_DEFAULT_EFFORT", str(_default("default_effort"))
            ),
        )

    @property
    def logs_dir(self) -> Path:
        """Directory containing log files."""

        return self.data_dir / "logs"

    @property
    def memory_dir(self) -> Path:
        """Directory containing persistent memory stores."""

        return self.data_dir / "memory"

    @property
    def model_cache_dir(self) -> Path:
        """Directory used by Hugging Face cache downloads."""

        return self.data_dir / "models"
