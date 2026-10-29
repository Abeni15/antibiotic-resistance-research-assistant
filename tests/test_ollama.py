import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents import (
    _claim_list_from_output,
    _analyst_format_example,
    _parse_analyst_output,
    _plan_from_output,
    _screen_research_papers,
    build_agent_team,
    run_agents,
)
from config import DEFAULT_OLLAMA_MODEL, Settings, load_settings
from ollama_client import (
    OLLAMA_OPENAI_COMPAT_BASE_URL,
    create_ollama_model_client,
    format_ollama_error,
)
from tools import validate_evidence


class OllamaConfigurationTests(unittest.TestCase):
    def test_uses_local_default_and_needs_no_provider_key(self):
        cfg = load_settings({"GROQ_API_KEY": "ignored-old-secret"})
        self.assertEqual(cfg.ollama_model, DEFAULT_OLLAMA_MODEL)
        self.assertEqual(DEFAULT_OLLAMA_MODEL, "llama3.1:8b")

    def test_reads_configured_model(self):
        cfg = load_settings({"OLLAMA_MODEL": "llama3.1:8b-instruct-q4_K_M"})
        self.assertEqual(cfg.ollama_model, "llama3.1:8b-instruct-q4_K_M")

    def test_client_targets_local_openai_compatible_endpoint_and_tool_support(self):
        cfg = Settings(ollama_model="llama3.1:8b")
        with patch("ollama_client.OpenAIChatCompletionClient") as constructor:
            client = create_ollama_model_client(cfg)
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["model"], "llama3.1:8b")
        self.assertEqual(kwargs["base_url"], OLLAMA_OPENAI_COMPAT_BASE_URL)
        self.assertEqual(kwargs["base_url"], "http://localhost:11434/v1")
        self.assertEqual(kwargs["api_key"], "ollama")  # documented placeholder; no secret required
        self.assertTrue(kwargs["model_info"]["function_calling"])
        self.assertFalse(kwargs["parallel_tool_calls"])
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertIs(client, constructor.return_value)

    def test_error_formatter_gives_local_start_instructions_without_secrets(self):
        class ConnectionFailure(RuntimeError):
            pass
        message = format_ollama_error(ConnectionFailure("Authorization: Bearer must-not-leak"))
        self.assertIn("ollama serve", message)
        self.assertIn("HTTP status unavailable", message)
        self.assertIn("provider code unavailable", message)
        self.assertNotIn("must-not-leak", message)

    def test_missing_model_error_suggests_configured_model_pull(self):
        class MissingModel(RuntimeError):
            status_code = 404
            response_json = {"error": {"code": "model_not_found", "message": "model not found"}}
        with patch("ollama_client.settings", Settings(ollama_model="llama3.1:8b")):
            message = format_ollama_error(MissingModel())
        self.assertIn("ollama pull llama3.1:8b", message)
        self.assertIn("HTTP 404", message)
        self.assertIn("model_not_found", message)


class OllamaAgentTeamTests(unittest.TestCase):
    @staticmethod
    def search(query: str) -> str:
        """Search fixture records."""
        return query

    @staticmethod
    def details(pmid: str) -> str:
        """Fetch fixture details."""
        return pmid

    @staticmethod
    def validate(claims_json: str, sources_json: str) -> str:
        """Validate evidence fixture."""
        return "validated"

    def test_four_roles_share_client_and_research_tools_are_registered(self):
        with patch("ollama_client.OpenAIChatCompletionClient"):
            client = create_ollama_model_client()
        team = build_agent_team(client, self.search, self.details, self.validate)
        self.assertEqual(list(team), ["planner", "researcher", "analyst", "reviewer"])
        self.assertTrue(all(agent._model_client is client for agent in team.values()))
        self.assertEqual([tool.name for tool in team["researcher"]._tools], ["search", "details"])
        self.assertEqual([tool.name for tool in team["reviewer"]._tools], ["validate"])

    def test_auto_mode_is_a_local_ollama_alias_not_cloud_fallback(self):
        expected = {"execution_mode": "ollama_local"}
        with patch("agents._run_ollama", new=AsyncMock(return_value=expected)) as run_local:
            result = asyncio.run(run_agents("Question", 1, mode="auto"))
        self.assertEqual(result, expected)
        run_local.assert_awaited_once()


class ModelOutputRegressionTests(unittest.TestCase):
    def test_planner_accepts_targeted_pubmed_queries_alias(self):
        plan = _plan_from_output(
            'Plan follows: ```json\n{"subquestions":["Mechanisms?"],'
            '"targeted PubMed queries":["Escherichia coli beta-lactamase resistance"]}\n```',
            "What are E. coli resistance mechanisms?",
        )
        self.assertEqual(plan["queries"], ["Escherichia coli beta-lactamase resistance"])
        self.assertEqual(plan["subquestions"], ["Mechanisms?"])

    def test_planner_rejects_non_search_urls_and_uses_safe_fallback(self):
        plan = _plan_from_output(
            '{"targeted PubMed queries":["https://pubmed.ncbi.nlm.nih.gov/12345/"]}',
            "What are E. coli resistance mechanisms?",
        )
        self.assertTrue(plan["queries"])
        self.assertTrue(all(not query.startswith("http") for query in plan["queries"]))

    def test_search_title_screen_separates_direct_adjacent_and_background_records(self):
        records = [
            {"pmid": "1", "title": "Antibiotic resistance mechanisms in Escherichia coli"},
            {"pmid": "2", "title": "Gonococcal resistance to beta-lactams"},
            {"pmid": "3", "title": "Detection of beta-lactamase-mediated resistance"},
        ]
        screened, summary = _screen_research_papers(records)
        self.assertEqual(summary["screened_count"], 3)
        self.assertEqual(summary["included_count"], 1)
        self.assertEqual(summary["excluded_count"], 2)
        self.assertEqual(screened[1]["relevance_screen_status"], "excluded_adjacent_organism_or_broad_group")
        self.assertEqual(screened[2]["relevance_screen_status"], "excluded_general_background")
        self.assertIn("not a systematic", summary["note"])

    def test_unstructured_analyst_response_is_diagnosed_without_logging_content(self):
        response = (
            "Here are findings:\n1. A potential claim.\n2. Another potential claim."
        )
        claims, diagnostic = _parse_analyst_output(response)
        self.assertEqual(claims, [])
        self.assertEqual(diagnostic["output_shape"], "unstructured_text")
        self.assertEqual(diagnostic["candidate_count"], 2)
        self.assertEqual(diagnostic["extracted_claim_count"], 0)
        self.assertEqual(diagnostic["rejected_count"], 2)
        self.assertEqual(
            diagnostic["rejection_reasons"],
            {"no_structured_claim_payload_or_source_mapping": 2},
        )

    def test_analyst_accepts_top_level_array_and_normalizes_traceable_fields(self):
        output = '''Claims:
        ```json
        [{"text":"A supported claim.","topic":"Mechanism","source_ids":["123"],
          "supporting_passages":["The efflux pump AcrB contributes to resistance."],
          "validation_status":"verified_source_and_passage"},
         {"text":"   ","source_ids":["123"]},
         {"text":"Untraceable claim."}]
        ```'''
        claims, diagnostic = _parse_analyst_output(output)
        self.assertEqual(len(claims), 1)
        self.assertEqual(diagnostic["candidate_count"], 3)
        self.assertEqual(diagnostic["extracted_claim_count"], 1)
        self.assertEqual(diagnostic["rejected_count"], 2)
        self.assertEqual(claims[0]["source_ids"], ["123"])
        self.assertEqual(
            claims[0]["supporting_passages"],
            {"123": "The efflux pump AcrB contributes to resistance."},
        )
        self.assertNotIn("validation_status", claims[0])
        validation = validate_evidence(
            claims,
            [{"pmid": "123", "title": "Study", "abstract": "The efflux pump AcrB contributes to resistance."}],
        )
        self.assertEqual(validation["results"][0]["status"], "verified_source_and_passage")

    def test_analyst_rejects_top_level_array_of_pmid_strings_and_previews_each(self):
        claims, diagnostic = _parse_analyst_output(
            '["PMID 34490146", "PMID 38638826"]'
        )
        self.assertEqual(claims, [])
        self.assertEqual(diagnostic["output_shape"], "top_level_array")
        self.assertEqual(diagnostic["candidate_count"], 2)
        self.assertEqual(diagnostic["rejected_count"], 2)
        self.assertEqual(diagnostic["rejection_reasons"], {"claim_is_not_an_object": 2})
        self.assertEqual(
            diagnostic["candidate_samples"],
            [
                {"type": "str", "repr": "'PMID 34490146'"},
                {"type": "str", "repr": "'PMID 38638826'"},
            ],
        )

    def test_analyst_accepts_mixed_pmid_string_and_compact_traceable_claim(self):
        output = json.dumps([
            "PMID 34490146",
            {
                "claim": "All ESBL-EC isolates were resistant to third-generation cephalosporin and quinolones.",
                "pmid": "34490146",
                "passage": "All ESBL-EC isolates were resistant to third-generation cephalosporin and quinolones.",
            },
        ])
        claims, diagnostic = _parse_analyst_output(output)
        self.assertEqual(len(claims), 1)
        self.assertEqual(diagnostic["rejected_count"], 1)
        self.assertEqual(len(diagnostic["candidate_samples"]), 1)
        self.assertEqual(diagnostic["candidate_samples"][0]["type"], "str")
        self.assertEqual(claims[0]["source_ids"], ["34490146"])
        self.assertEqual(
            claims[0]["supporting_passages"],
            {"34490146": "All ESBL-EC isolates were resistant to third-generation cephalosporin and quinolones."},
        )
        self.assertNotIn("validation_status", claims[0])
        validation = validate_evidence(claims, [{
            "pmid": "34490146",
            "abstract": "All ESBL-EC isolates were resistant to third-generation cephalosporin and quinolones.",
        }])
        self.assertEqual(validation["results"][0]["status"], "verified_source_and_passage")
        self.assertNotIn("validation_status", claims[0])

    def test_analyst_format_example_uses_only_supplied_passage_and_pmid(self):
        paper = {
            "pmid": "123456",
            "abstract": "A literal source sentence here. Another sentence not used.",
        }
        example = _analyst_format_example([paper])
        self.assertEqual(example, {
            "claims": [{
                "claim": "A literal source sentence here.",
                "pmid": "123456",
                "passage": "A literal source sentence here.",
            }]
        })

    def test_analyst_also_accepts_claims_object_and_keeps_unverified_ids_unverified(self):
        claims = _claim_list_from_output(
            '{"claims":[{"text":"Claim text","source_ids":["999"],'
            '"supporting_passages":{"999":"Passage"}}]}'
        )
        self.assertEqual(len(claims), 1)
        validation = validate_evidence(claims, [{"pmid": "123", "abstract": "Other text"}])
        self.assertEqual(validation["results"][0]["status"], "missing_source")

    def test_analyst_recovers_json_encoded_claim_candidate_from_single_item_array(self):
        # Reproduces the observed top-level array with one non-object candidate.
        # The saved live run retained only counters, so this fixture exercises the
        # plausible JSON-encoded-object shape while preserving strict validation.
        encoded_claim = json.dumps({
            "text": "The supplied abstract reports ESBL-producing E. coli.",
            "source_ids": ["34490146"],
            "supporting_passages": {
                "34490146": "Extended-spectrum β-lactamases producing Escherichia coli (ESBL-EC) lend resistance to most β-lactam antibiotics."
            },
        })
        claims, diagnostic = _parse_analyst_output(json.dumps([encoded_claim]))
        self.assertEqual(diagnostic["output_shape"], "top_level_array")
        self.assertEqual(diagnostic["candidate_count"], 1)
        self.assertEqual(diagnostic["candidate_samples"], [])
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0]["source_ids"], ["34490146"])
        self.assertEqual(
            claims[0]["supporting_passages"]["34490146"],
            "Extended-spectrum β-lactamases producing Escherichia coli (ESBL-EC) lend resistance to most β-lactam antibiotics.",
        )
        self.assertNotIn("validation_status", claims[0])

    def test_analyst_rejects_malformed_encoded_and_prose_candidates_with_bounded_preview(self):
        claims, diagnostic = _parse_analyst_output(
            json.dumps(["not JSON and not a traceable claim", "{broken json"])
        )
        self.assertEqual(claims, [])
        self.assertEqual(diagnostic["candidate_count"], 2)
        self.assertEqual(diagnostic["rejected_count"], 2)
        self.assertEqual(diagnostic["rejection_reasons"], {"claim_is_not_an_object": 2})
        self.assertEqual(diagnostic["candidate_samples"][0]["type"], "str")
        self.assertLessEqual(len(diagnostic["candidate_samples"][0]["repr"]), 180)

    def test_candidate_preview_redacts_credentials(self):
        claims, diagnostic = _parse_analyst_output(
            json.dumps([
                "Bearer secret-token-12345678901234567890",
                "api_key=secret-key-12345678901234567890",
                "AQsecretcredential1234567890",
            ])
        )
        self.assertEqual(claims, [])
        previews = [sample["repr"] for sample in diagnostic["candidate_samples"]]
        self.assertEqual(len(previews), 3)
        self.assertTrue(all("REDACTED" in preview for preview in previews))
        self.assertFalse(any("secret-" in preview or "AQsecret" in preview for preview in previews))

    def test_unsupported_passage_survives_normalization_but_is_not_verified(self):
        claims = _claim_list_from_output(
            '{"claims":[{"text":"Claim text","source_ids":["123"],'
            '"supporting_passages":{"123":"A plausible but absent passage."}}]}'
        )
        validation = validate_evidence(claims, [{"pmid": "123", "abstract": "Actual source text."}])
        self.assertEqual(validation["results"][0]["status"], "passage_not_found")
        self.assertNotIn("validation_status", claims[0])

    def test_missing_passage_claim_is_retained_for_reviewer_but_not_verified(self):
        claims, diagnostic = _parse_analyst_output(
            '{"claims":[{"text":"Claim text","source_ids":["123"]}]}'
        )
        self.assertEqual(len(claims), 1)
        self.assertEqual(diagnostic["claims_without_mapped_passages"], 1)
        validation = validate_evidence(claims, [{"pmid": "123", "abstract": "Source text."}])
        self.assertEqual(validation["results"][0]["status"], "passage_not_found")

    def test_empty_valid_claims_skip_reviewer_and_report_explains_why(self):
        class FakeAgent:
            def __init__(self, content=""):
                self.content = content
                self.run = AsyncMock(return_value=SimpleNamespace(messages=[SimpleNamespace(content=content)]))

        planner = FakeAgent('{"targeted PubMed queries":["E. coli resistance query"]}')
        researcher = FakeAgent("Retrieved records.")
        analyst = FakeAgent("Here are findings:\n1. A possible claim.\n2. Another possible claim.")
        reviewer = FakeAgent("This must not run.")
        client = SimpleNamespace(close=AsyncMock())
        source = {
            "pmid": "123",
            "title": "Antibiotic resistance in Escherichia coli",
            "abstract": "An abstract about E. coli antibiotic resistance.",
            "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/123/",
        }
        with (
            patch("agents.create_ollama_model_client", return_value=client),
            patch("agents.AssistantAgent", return_value=planner),
            patch("agents.build_agent_team", return_value={
                "researcher": researcher,
                "analyst": analyst,
                "reviewer": reviewer,
            }),
            patch("agents.search_pubmed", return_value={"ok": True, "papers": [source]}),
        ):
            result = asyncio.run(run_agents("Question", 1, mode="ollama"))

        self.assertEqual(result["claims"], [])
        self.assertEqual(result["validation"]["results"], [])
        self.assertEqual(analyst.run.await_count, 1)
        analyst_event = next(event for event in result["events"] if event["agent"] == "Evidence Analyst")
        self.assertIn("'output_shape': 'unstructured_text'", analyst_event["output"])
        self.assertIn("'candidate_count': 2", analyst_event["output"])
        self.assertIn("'rejected_count': 2", analyst_event["output"])
        self.assertNotIn("A possible claim", analyst_event["output"])
        self.assertIn("Reviewer model call was skipped", result["review"])
        self.assertIn("No claims were marked as validated", result["review"])
        reviewer.run.assert_not_awaited()
        client.close.assert_awaited_once()

        report_result = {
            "queries": ["E. coli resistance query"],
            "papers": [source],
            "claims": [],
            "validation": {"results": []},
            "review": result["review"],
            "provider": "Ollama (local)",
            "model": "llama3.2:3b",
            "execution_mode": "ollama_local",
            "llm_used": True,
            "screening": result["screening"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch("research.run_agents", new=AsyncMock(return_value=report_result)):
                with patch("research.setup_logging", return_value=None):
                    from research import research
                    report_paths = asyncio.run(research("Question", 1, Path(tmp), mode="ollama"))
            report = Path(report_paths["report"]).read_text(encoding="utf-8")
        self.assertIn("No valid claims were available for review", report)
        self.assertIn("No claims are marked as validated", report)
        self.assertIn("Reviewer model call was skipped", report)
        self.assertNotIn("Four local Ollama-backed AutoGen agents collaborated", report)
        self.assertIn("1 records screened; 1 keyword candidates included; 0 excluded", report)
        self.assertIn("PMID:123 — Antibiotic resistance in Escherichia coli", report)


if __name__ == "__main__":
    unittest.main()
