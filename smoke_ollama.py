"""Opt-in local Ollama request plus a local function-call round trip."""
from __future__ import annotations

import asyncio

from autogen_agentchat.agents import AssistantAgent

from config import settings
from ollama_client import OllamaRequestError, create_ollama_model_client, format_ollama_error


def local_smoke_tool() -> str:
    """Return a fixed local diagnostic value without side effects."""
    return "LOCAL_TOOL_OK"


async def _run() -> None:
    client = create_ollama_model_client()
    try:
        agent = AssistantAgent(
            name="ollama_tool_smoke",
            model_client=client,
            tools=[local_smoke_tool],
            system_message="Call local_smoke_tool once and report its exact result.",
            max_tool_iterations=2,
        )
        result = await agent.run(task="Use the tool and return its result.")
        contents = [getattr(message, "content", "") for message in result.messages]
        if not any("LOCAL_TOOL_OK" in str(content) for content in contents):
            raise RuntimeError("The model response did not include the local tool result.")
        print(f"Local Ollama text and tool-call smoke test passed (model: {settings.ollama_model}).")
    except Exception as exc:
        raise OllamaRequestError(format_ollama_error(exc)) from None
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except OllamaRequestError as exc:
        print(f"Ollama smoke test failed: {exc}")
        raise SystemExit(1)
