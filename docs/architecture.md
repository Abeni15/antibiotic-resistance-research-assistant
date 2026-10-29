# Architecture and information flow

## Components

- `config.py` loads `OLLAMA_MODEL` (default `llama3.1:8b`) and optional NCBI settings. Gemini settings are retained only for legacy diagnostics.
- `ollama_client.py` configures AutoGen 0.7.5 `OpenAIChatCompletionClient` for `http://localhost:11434/v1`, with function calling enabled and parallel calls disabled. It uses Ollama's documented placeholder API key and does not require a secret.
- `agents.py` runs Research Planner, Literature Researcher, Evidence Analyst, and Scientific Reviewer.
- `tools.py` uses PubMed E-utilities for retrieval and validates source IDs and exact passage presence.
- `research.py` writes Markdown, JSON trace, and event logs; `evaluation.py` keeps the five fixed questions.
- `smoke_ollama.py` provides an opt-in local text/tool-call check.

## Model selection

The configured tag is used as-is; no fallback model is selected. `llama3.1:8b` is listed by Ollama with tool support. The OpenAI-compatible server runs locally at `http://localhost:11434/v1`. Offline tests verify the configured endpoint and declared tool support, but cannot establish that the server is running, the model is installed, or that this hardware will run it acceptably.

## Workflow

```text
Question -> Planner -> PubMed queries
        -> Researcher (search_pubmed / fetch_paper_details)
        -> Deduplicate and retain source metadata/abstracts
        -> Analyst (claims tied to source passages)
        -> Reviewer (validate_evidence + scientific judgment)
        -> Markdown report + JSON trace + event log
```

## Modes and evidence limits

- `ollama`: local Ollama agents; default mode.
- `auto`: alias for local Ollama. It does not select a remote provider or silently switch to extractive mode.
- `extractive`: real PubMed retrieval plus deterministic abstract excerpts; no model interpretation.

`validate_evidence` checks source membership and normalized literal excerpt presence only. It does not prove entailment or scientific accuracy. The Ollama path applies a preliminary title-keyword triage to separate direct E. coli keyword candidates from adjacent-organism/general-background records; it is not systematic or human relevance screening. Historical reports are preserved; offline tests do not establish live model accuracy.

## Troubleshooting and secret handling

If connection fails, start the Ollama desktop app or run `ollama serve`. If the configured tag is missing, run `ollama pull <OLLAMA_MODEL>`. The application does not read or require `GROQ_API_KEY`; `.env` remains ignored by Git. Optional Gemini diagnostics remain separate.
