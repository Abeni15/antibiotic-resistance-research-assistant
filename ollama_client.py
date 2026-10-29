"""Local Ollama client using AutoGen's existing OpenAI-compatible adapter."""
from __future__ import annotations

import json
import re

from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import DEFAULT_OLLAMA_BASE_URL, Settings, settings

OLLAMA_OPENAI_COMPAT_BASE_URL = DEFAULT_OLLAMA_BASE_URL


class OllamaConfigurationError(RuntimeError):
    """Local model/client configuration problem."""


class OllamaRequestError(RuntimeError):
    """Sanitized local Ollama request failure safe to display."""


def create_ollama_model_client(config: Settings = settings, *, model: str | None = None) -> OpenAIChatCompletionClient:
    """Build a local-only OpenAI-compatible client; Ollama ignores the placeholder key."""
    model_id = (model or config.ollama_model).strip()
    if not model_id:
        raise OllamaConfigurationError("OLLAMA_MODEL is empty. Set it to an installed Ollama model tag.")
    return OpenAIChatCompletionClient(
        model=model_id,
        api_key="ollama",
        base_url=OLLAMA_OPENAI_COMPAT_BASE_URL,
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
        timeout=300,
        max_retries=0,
        parallel_tool_calls=False,
    )


def format_ollama_error(exc: BaseException) -> str:
    """Extract status/code/message without exposing request headers or credentials."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__

    statuses: list[int] = []
    codes: list[str] = []
    provider_messages: list[str] = []
    names = " ".join(type(item).__name__.casefold() for item in chain)
    for item in chain:
        status = getattr(item, "status_code", None)
        if isinstance(status, int):
            statuses.append(status)
        body = getattr(item, "body", None) or getattr(item, "response_json", None)
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                body = None
        if isinstance(body, dict):
            error = body.get("error", body)
            if isinstance(error, dict):
                code = error.get("code") or error.get("type") or error.get("status")
                if code is not None:
                    codes.append(str(code))
                message = error.get("message")
                if isinstance(message, str):
                    provider_messages.append(message)
        code = getattr(item, "error_code", None) or getattr(item, "code_name", None) or getattr(item, "status", None)
        if code is not None and not isinstance(code, int):
            codes.append(str(code))

    status = statuses[0] if statuses else None
    message = provider_messages[0] if provider_messages else "Local Ollama request failed."
    message = re.sub(r"(?i)(authorization|x-api-key|api[_ -]?key)\s*[:=]\s*[^,;\s]+", r"\1=[REDACTED]", message)
    message = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", message)
    message = re.sub(r"\s+", " ", message).strip()[:400]

    if status == 404 or "not found" in message.casefold() or "pull model" in message.casefold():
        guidance = f"Ollama could not find model '{settings.ollama_model}'. Run `ollama pull {settings.ollama_model}` and retry."
    elif any(token in names for token in ("connection", "connecterror", "connecttimeout")):
        guidance = "Could not connect to local Ollama at http://localhost:11434. Start the Ollama app or run `ollama serve`."
    elif any(token in names for token in ("timeout",)):
        guidance = "The local Ollama request timed out. Check that the model is loaded and the computer has enough memory."
    elif status in (400, 422):
        guidance = "Ollama rejected the local request. Check model support and the tool/function schema."
    else:
        guidance = "Local Ollama request failed. Check the Ollama server and configured model."
    status_text = f"HTTP {status}" if status is not None else "HTTP status unavailable"
    code_text = f"provider code {codes[0]}" if codes else "provider code unavailable"
    return f"{guidance} ({status_text}; {code_text}; {message})"
