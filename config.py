"""Small environment-backed settings for the research assistant."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


@dataclass(frozen=True)
class Settings:
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    ncbi_email: str | None = None
    ncbi_api_key: str | None = None
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Load configuration without requiring any model-provider secret."""
    if environ is None:
        load_dotenv(Path(__file__).with_name(".env"))
        environ = os.environ
    return Settings(
        ollama_model=environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL,
        ncbi_email=environ.get("NCBI_EMAIL") or None,
        ncbi_api_key=environ.get("NCBI_API_KEY") or None,
        gemini_api_key=environ.get("GEMINI_API_KEY") or None,
        gemini_model=environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    )


settings = load_settings()
