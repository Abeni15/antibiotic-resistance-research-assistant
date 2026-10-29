import asyncio, tempfile, unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


from agents import build_agent_team, run_agents
from config import Settings, load_settings
from gemini_client import (
    GEMINI_OPENAI_COMPAT_BASE_URL,
    GeminiConfigurationError,
    create_gemini_model_client,
    format_gemini_error,
)
from native_gemini import NativeProbeError, probe_native_gemini


class GeminiConfigTests(unittest.TestCase):
    def test_loads_gemini_key_and_model(self):
        cfg = load_settings({
            "GEMINI_API_KEY": "unit-test-key",
            "GEMINI_MODEL": "gemini-2.5-pro",
        })
        self.assertEqual(cfg.gemini_api_key, "unit-test-key")
        self.assertEqual(cfg.gemini_model, "gemini-2.5-pro")

    def test_openai_key_does_not_fall_back(self):
        cfg = load_settings({"OPENAI_API_KEY": "old-openai-key"})
        self.assertIsNone(cfg.gemini_api_key)
        self.assertEqual(cfg.gemini_model, "gemini-2.5-flash")
        with self.assertRaisesRegex(GeminiConfigurationError, "GEMINI_API_KEY"):
            create_gemini_model_client(cfg)

    def test_missing_key_message_is_actionable(self):
        cfg = Settings(gemini_model="gemini-2.5-flash")
        with self.assertRaises(GeminiConfigurationError) as caught:
            create_gemini_model_client(cfg)
        self.assertIn(".env", str(caught.exception))
        self.assertIn("--mode gemini", str(caught.exception))

    def test_client_uses_google_compat_endpoint_and_gemini_model(self):
        cfg = Settings(gemini_api_key="unit-test-key", gemini_model="gemini-2.5-flash")
        with patch("gemini_client.OpenAIChatCompletionClient") as factory:
            client = create_gemini_model_client(cfg)
        kwargs = factory.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-2.5-flash")
        self.assertEqual(kwargs["api_key"], "unit-test-key")
        self.assertEqual(kwargs["base_url"], GEMINI_OPENAI_COMPAT_BASE_URL)
        self.assertTrue(kwargs["model_info"]["function_calling"])
        self.assertFalse(kwargs["parallel_tool_calls"])
        self.assertIs(client, factory.return_value)

    def test_provider_errors_are_actionable_and_do_not_echo_exception_text(self):
        class QuotaFailure(RuntimeError):
            status_code = 429
            status = "RESOURCE_EXHAUSTED"
            response_json = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exhausted; retry later."}}
        message = format_gemini_error(QuotaFailure("raw exception body must not appear"))
        self.assertIn("quota", message.lower())
        self.assertNotIn("raw exception body", message)
        self.assertIn("HTTP 429", message)
        self.assertIn("RESOURCE_EXHAUSTED", message)

    def test_error_formatter_redacts_key_and_authorization_header(self):
        class ProviderFailure(RuntimeError):
            status_code = 401
            status = "UNAUTHENTICATED"
        message = format_gemini_error(
            ProviderFailure("authorization: Bearer secret-value"),
            api_key="secret-value",
        )
        self.assertIn("HTTP 401", message)
        self.assertIn("UNAUTHENTICATED", message)
        self.assertNotIn("secret-value", message)
        self.assertIn("REDACTED", message)


class NativeGeminiDiagnosticTests(unittest.TestCase):
    def test_native_probe_uses_configured_model_and_text_only_request(self):
        calls = []

        class Models:
            def get(self, **kwargs):
                calls.append(("get", kwargs))
                return type("Model", (), {"name": "models/gemini 1.5"})()

            def generate_content(self, **kwargs):
                calls.append(("generate", kwargs))
                return type("Response", (), {"text": "NATIVE_GEMINI_OK"})()

        class Client:
            models = Models()
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def close(self):
                calls.append(("close", {}))

        cfg = Settings(gemini_api_key="test-key-not-real", gemini_model="gemini 1.5")
        result = probe_native_gemini(cfg, client_factory=Client)
        self.assertEqual(result.model, "models/gemini 1.5")
        self.assertTrue(result.available)
        self.assertEqual(result.text, "NATIVE_GEMINI_OK")
        self.assertEqual(calls[0], ("get", {"model": "gemini 1.5"}))
        self.assertEqual(calls[1][1]["model"], "gemini 1.5")
        self.assertIn("contents", calls[1][1])

    def test_native_probe_reports_model_lookup_failure_without_generation(self):
        class NotFound(RuntimeError):
            code = 404
            status = "NOT_FOUND"
            response_json = {"error": {"code": 404, "status": "NOT_FOUND", "message": "model unavailable"}}

        class Models:
            def get(self, **kwargs):
                raise NotFound("model unavailable")
            def generate_content(self, **kwargs):
                raise AssertionError("must not generate after failed model lookup")

        class Client:
            models = Models()
            def __init__(self, **kwargs): pass
            def close(self): pass

        with self.assertRaisesRegex(NativeProbeError, "model availability lookup failed") as caught:
            probe_native_gemini(Settings(gemini_api_key="unit-test-key", gemini_model="gemini 1.5"), client_factory=Client)
        self.assertIn("HTTP 404", str(caught.exception))
        self.assertIn("NOT_FOUND", str(caught.exception))

        class TimeoutFailure(TimeoutError):
            pass
        self.assertIn("timed out", format_gemini_error(TimeoutFailure()).lower())


class GeminiAgentTeamTests(unittest.TestCase):
    @staticmethod
    def search(query: str) -> str:
        """Search the PubMed fixture."""
        return query

    @staticmethod
    def details(pmid: str) -> str:
        """Fetch a PubMed fixture by PMID."""
        return pmid

    @staticmethod
    def validate(claims_json: str, sources_json: str) -> str:
        """Validate fixture evidence."""
        return "validated"

    def test_all_four_agents_share_one_client_and_keep_tool_access(self):
        from ollama_client import create_ollama_model_client
        client = create_ollama_model_client()
        try:
            team = build_agent_team(client, self.search, self.details, self.validate)
            self.assertEqual(
                list(team),
                ["planner", "researcher", "analyst", "reviewer"],
            )
            self.assertTrue(all(agent._model_client is client for agent in team.values()))
            self.assertEqual([tool.name for tool in team["researcher"]._tools], ["search", "details"])
            self.assertEqual([tool.name for tool in team["reviewer"]._tools], ["validate"])
        finally:
            asyncio.run(client.close())

    def test_extractively_forced_run_skips_model(self):
        paper = {
            "pmid": "123",
            "title": "Evidence paper",
            "abstract": "A retrieved abstract reports a relevant bacterial resistance finding.",
            "query": "beta-lactamase",
            "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/123/",
        }
        with patch("agents.settings", Settings(ollama_model="llama3.1:8b")):
            with patch("agents.search_pubmed", return_value={"ok": True, "papers": [paper]}):
                result = asyncio.run(run_agents("Question", 1, mode="extractive"))
        self.assertEqual(result["execution_mode"], "extractive_fallback")
        self.assertIsNone(result["provider"])
        self.assertFalse(result["llm_used"])

    def test_explicit_ollama_mode_needs_no_provider_key(self):
        from ollama_client import create_ollama_model_client
        with patch("ollama_client.OpenAIChatCompletionClient") as factory:
            client = create_ollama_model_client(Settings(ollama_model="llama3.1:8b"))
        self.assertIs(client, factory.return_value)


class GeminiReportTests(unittest.TestCase):
    def test_report_records_provider_model_mode_and_traceable_source(self):
        result = {
            "plan": {"subquestions": ["Question"]},
            "queries": ["E. coli resistance"],
            "papers": [{
                "pmid": "123",
                "title": "Evidence paper",
                "year": "2025",
                "journal": "Journal",
                "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                "abstract": "A supported passage from this paper.",
            }],
            "claims": [{
                "text": "A source-linked claim.",
                "topic": "mechanism",
                "source_ids": ["123"],
                "supporting_passages": {"123": "A supported passage from this paper."},
                "evidence_category": "abstract statement",
                "validation_status": "verified_source_and_passage",
                "uncertainty": "Abstract only.",
            }],
            "validation": {"results": [{"status": "verified_source_and_passage"}]},
            "review": "Human review remains necessary.",
            "events": [],
            "provider": "Ollama (local)",
            "model": "llama3.1:8b",
            "execution_mode": "ollama_local",
            "llm_used": True,
            "correction_rounds": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch("research.run_agents", new=AsyncMock(return_value=result)):
                from research import research
                output = asyncio.run(research("Question", 1, Path(tmp), mode="ollama"))
            report = Path(output["report"]).read_text(encoding="utf-8")
            data = Path(output["data"]).read_text(encoding="utf-8")
            import logging
            logging.shutdown()
        self.assertIn("Ollama (local)", report)
        self.assertIn("llama3.1:8b", report)
        self.assertIn("ollama_local", report)
        self.assertIn("PMID:123", report)
        self.assertIn("A supported passage from this paper.", report)
        self.assertIn('"execution_mode": "ollama_local"', data)


if __name__ == "__main__":
    unittest.main()


