"""Gemini API adapter using AutoGen 0.7.5's documented compatibility route.

AutoGen 0.7.5 exposes Gemini through OpenAIChatCompletionClient and Google's
OpenAI-compatible Chat Completions endpoint. The API key and request endpoint
are Google's; the OpenAI Python package is only the transport adapter.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import Settings, settings

GEMINI_OPENAI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiConfigurationError(RuntimeError):
    """Local configuration problem; no API request has been made."""


class GeminiRequestError(RuntimeError):
    """Sanitized Gemini service failure safe to display without secret values."""


def create_gemini_model_client(
    config: Settings = settings,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> OpenAIChatCompletionClient:
    """Build AutoGen's client against Google's Gemini-compatible endpoint."""
    key = api_key or config.gemini_api_key
    if not key:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is not configured. Add your Gemini API key to .env "
            "or set it in the environment, then rerun with --mode gemini."
        )
    model_id = model or config.gemini_model
    return OpenAIChatCompletionClient(
        model=model_id,
        api_key=key,
        base_url=GEMINI_OPENAI_COMPAT_BASE_URL,
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
        timeout=60,
        max_retries=2,
        parallel_tool_calls=False,
    )


def format_gemini_error(exc: BaseException, *, api_key: str | None = None) -> str | None:
    """Report HTTP/provider diagnostics while redacting credentials and headers."""
    seen: set[int] = set()
    current: BaseException | None = exc
    names: list[str] = []
    statuses: list[int] = []
    codes: list[str] = []
    messages: list[str] = []
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.append(type(current).__name__.casefold())
        status = getattr(current, "status_code", None)
        if status is None:
            status = getattr(current, "code", None)
        if isinstance(status, int):
            statuses.append(status)
        elif isinstance(status, str) and status.isdigit():
            statuses.append(int(status))
        provider_code = getattr(current, "error_code", None)
        if provider_code is None:
            provider_code = getattr(current, "code_name", None)
        if provider_code is None:
            provider_code = getattr(current, "status", None)
        response_json = getattr(current, "response_json", None)
        if response_json is None:
            response_json = getattr(current, "body", None)
        if isinstance(response_json, str):
            try:
                response_json = json.loads(response_json)
            except (TypeError, json.JSONDecodeError):
                response_json = None
        if isinstance(response_json, dict):
            error_obj = response_json.get("error", response_json)
            if isinstance(error_obj, dict):
                provider_code = provider_code or error_obj.get("status") or error_obj.get("code")
                provider_message = error_obj.get("message")
                if isinstance(provider_message, str):
                    messages.append(provider_message)
        if provider_code is not None:
            codes.append(str(provider_code))
        message = getattr(current, "message", None)
        if isinstance(message, str):
            messages.append(message)
        response = getattr(current, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            statuses.append(response_status)
        if not messages:
            messages.append(str(current))
        current = current.__cause__ or current.__context__

    status_text = f"HTTP {statuses[0]}" if statuses else "HTTP status unavailable"
    code_text = f"provider code {codes[0]}" if codes else "provider code unavailable"
    # Keep the diagnostic useful, but never echo credentials or header-like data.
    safe_message = next((m for m in messages if m.strip()), "Provider returned no message.")
    safe_message = re.sub(r"(?i)(authorization|x-goog-api-key|api[_ -]?key)\s*[:=]\s*[^,;\s]+", r"\1=[REDACTED]", safe_message)
    safe_message = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", safe_message)
    if api_key:
        safe_message = safe_message.replace(api_key, "[REDACTED]")
    safe_message = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED]", safe_message)
    safe_message = re.sub(r"\s+", " ", safe_message).strip()[:500]

    if any("timeout" in name for name in names):
        category = "Gemini request timed out. Check connectivity and retry."
        return f"{category} ({status_text}; {code_text}; {safe_message})"
    if any("connection" in name for name in names):
        category = "Could not connect to Google's Gemini API. Check connectivity and retry."
        return f"{category} ({status_text}; {code_text}; {safe_message})"
    if 429 in statuses or any("ratelimit" in name for name in names):
        category = "Gemini rate limit or quota reached. Check Google AI Studio usage/quota and try later."
        return f"{category} ({status_text}; {code_text}; {safe_message})"
    if any("authentication" in name for name in names) or any(s in statuses for s in (401, 403)):
        category = "Gemini rejected the credential or its access. Verify key type, API restriction, and model access in Google AI Studio."
        return f"{category} ({status_text}; {code_text}; {safe_message})"
    if 404 in statuses:
        category = "Gemini model or API endpoint was not found. Check GEMINI_MODEL and current model availability."
        return f"{category} ({status_text}; {code_text}; {safe_message})"
    if any("badrequest" in name for name in names) or 400 in statuses:
        category = "Gemini rejected the request. Check the provider message for model-name or request-format details."
        return f"{category} ({status_text}; {code_text}; {safe_message})"
    if any("apierror" in name or "apistatus" in name for name in names):
        category = "Gemini returned an API error. Check credential type/access, model access, and Google AI Studio usage limits."
        return f"{category} ({status_text}; {code_text}; {safe_message})"
    return None
