"""Text-only native Gemini API probe using Google's supported Gen AI SDK."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from config import Settings, settings
from gemini_client import GeminiConfigurationError, format_gemini_error


@dataclass(frozen=True)
class NativeProbeResult:
    model: str
    available: bool
    text: str


class NativeProbeError(RuntimeError):
    """Safe diagnostic containing the stage that failed and provider details."""


def probe_native_gemini(
    config: Settings = settings,
    *,
    client_factory: Callable[..., Any] | None = None,
) -> NativeProbeResult:
    """Check configured model availability and make one text-only generation call.

    The model ID is passed exactly as configured. This function never substitutes
    another model and never logs the key or SDK request headers.
    """
    if not config.gemini_api_key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is not configured. Add it to .env or the process environment."
        )
    if client_factory is None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "The official Google Gen AI SDK is missing. Install requirements.txt."
            ) from exc
        client_factory = genai.Client

    client = client_factory(api_key=config.gemini_api_key)
    stage = "model availability lookup"
    try:
        model_info = client.models.get(model=config.gemini_model)
        stage = "native text generation"
        response = client.models.generate_content(
            model=config.gemini_model,
            contents="Reply with exactly NATIVE_GEMINI_OK.",
        )
        text = getattr(response, "text", None) or ""
        return NativeProbeResult(
            model=getattr(model_info, "name", None) or config.gemini_model,
            available=True,
            text=str(text)[:200],
        )
    except Exception as exc:
        diagnostic = format_gemini_error(exc, api_key=config.gemini_api_key)
        if diagnostic:
            raise NativeProbeError(f"{stage} failed: {diagnostic}") from None
        raise NativeProbeError(
            f"{stage} failed ({type(exc).__name__}); provider details unavailable."
        ) from None
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
