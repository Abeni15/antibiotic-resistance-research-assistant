"""Diagnose native Gemini first, then AutoGen compatibility and tool calling."""
from __future__ import annotations

import argparse
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import UserMessage

from config import settings
from gemini_client import (
    GeminiConfigurationError,
    GeminiRequestError,
    create_gemini_model_client,
    format_gemini_error,
)
from native_gemini import NativeProbeError, probe_native_gemini


async def compatibility_probe() -> None:
    client = create_gemini_model_client(settings)
    try:
        try:
            response = await client.create(
                [UserMessage(content="Reply with exactly COMPAT_TEXT_OK.", source="user")]
            )
        except Exception as exc:
            diagnostic = format_gemini_error(exc, api_key=settings.gemini_api_key)
            detail = diagnostic or f"{type(exc).__name__}; provider details unavailable"
            raise GeminiRequestError(f"AutoGen-compatible text request failed: {detail}") from None
        print(f"OpenAI-compatible AutoGen text request: PASSED ({settings.gemini_model}).")
        print(f"Compatibility response: {str(response.content)[:200]}")

        called = False

        def diagnostic_tool() -> str:
            """Return a fixed marker to verify one AutoGen function tool round trip."""
            nonlocal called
            called = True
            return "TOOL_SCHEMA_OK"

        agent = AssistantAgent(
            name="gemini_tool_diagnostic",
            model_client=client,
            tools=[diagnostic_tool],
            system_message="Call diagnostic_tool exactly once, then report its returned marker.",
        )
        try:
            await agent.run(task="Call diagnostic_tool now and report its result.")
        except Exception as exc:
            diagnostic = format_gemini_error(exc, api_key=settings.gemini_api_key)
            detail = diagnostic or f"{type(exc).__name__}; provider details unavailable"
            raise GeminiRequestError(f"AutoGen function/tool request failed: {detail}") from None
        if called:
            print("OpenAI-compatible AutoGen function/tool schema round trip: PASSED.")
        else:
            raise GeminiRequestError(
                "AutoGen function/tool request completed without invoking diagnostic_tool; inspect model tool support/schema."
            )
    finally:
        await client.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose the configured Gemini model and AutoGen integration."
    )
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="check native Google model lookup and text generation only",
    )
    parser.add_argument(
        "--compat-only",
        action="store_true",
        help="skip native probe; directly test AutoGen compatibility and function calling",
    )
    args = parser.parse_args()
    if args.native_only and args.compat_only:
        parser.error("--native-only and --compat-only cannot be combined")
    try:
        if not args.compat_only:
            result = probe_native_gemini(settings)
            print(f"Native Google Gemini model availability: PASSED ({result.model}).")
            print("Native Google Gemini text-only generation: PASSED.")
            print(f"Native response: {result.text}")
            if args.native_only:
                return 0
            print("Starting separate AutoGen compatibility and tool-schema checks.")
        elif not settings.gemini_api_key:
            raise GeminiConfigurationError("GEMINI_API_KEY is not configured.")
        asyncio.run(compatibility_probe())
        return 0
    except GeminiConfigurationError as exc:
        print(f"Gemini configuration error: {exc}")
        return 2
    except NativeProbeError as exc:
        print(f"Native Google Gemini diagnostic FAILED: {exc}")
        print("AutoGen compatibility/tool checks were skipped because the native test failed.")
        return 1
    except GeminiRequestError as exc:
        print(f"AutoGen compatibility/tool diagnostic FAILED: {exc}")
        return 1
    except Exception as exc:
        print(f"Diagnostic failed ({type(exc).__name__}).")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
