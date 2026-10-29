# Antibiotic-resistance research assistant

A PubMed-backed four-agent AutoGen AgentChat 0.7.5 workflow for focused *E. coli* antibiotic-resistance literature investigations. The normal model path uses **Ollama locally** through AutoGen's existing OpenAI-compatible client. It needs no hosted model API key and has no Groq or paid-provider fallback.

## Local model

Default model: `llama3.1:8b`, which Ollama lists as tool-capable. The project connects to `http://localhost:11434/v1`, sets function calling on, and disables parallel tool calls. Ollama's compatibility API accepts the conventional placeholder `api_key="ollama"`; no credential is sent as a real secret. The model must be downloaded on this computer. See [Ollama's Llama 3.1 library page](https://ollama.com/library/llama3.1%3A8b), [tool calling documentation](https://ollama.com/blog/tool-support), and [OpenAI compatibility documentation](https://docs.ollama.com/openai).

Configure `OLLAMA_MODEL` in `.env` if you want another locally installed Ollama model with tool support. The project never silently substitutes a model. `.env` is ignored by Git. NCBI email/API key are optional and do not affect model setup.

## Windows setup

Install Ollama if it is not already installed:

```powershell
irm https://ollama.com/install.ps1 | iex
```

Ollama for Windows normally starts its local server from the desktop app. If it is not running, start a separate PowerShell and leave this running:

```powershell
ollama serve
```

In a second PowerShell, from this project folder:

```powershell
ollama pull llama3.1:8b
ollama list
ollama run llama3.1:8b
```

Copy `.env.example` to `.env` only if `.env` does not exist. Set the model if needed:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Set `OLLAMA_MODEL=llama3.1:8b` (or the exact local tag you pulled). No key is needed. Activate the existing project environment and install pinned dependencies if needed:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The existing `.venv` and dependency list are retained. The `openai` package is used only as the transport adapter for the local Ollama-compatible endpoint; no OpenAI API request is made.

## Tests and smoke check

Run offline tests; they mock PubMed and model behavior and make no model/API requests:

```powershell
python -m unittest discover -s tests -v
```

After Ollama is serving and the configured model is pulled, optionally perform one local text-plus-tool-call check:

```powershell
python smoke_ollama.py
```

It is opt-in and is never invoked automatically by tests or the CLI. Missing-server errors tell you to start Ollama; a missing model error includes the exact `ollama pull <model>` command.

## Research and five-question evaluation

Run one local investigation:

```powershell
python main.py --mode ollama --question "What are the major mechanisms of antibiotic resistance in Escherichia coli?"
```

Run all five preserved questions against the local model:

```powershell
python evaluation.py --mode ollama --output-dir reports/ollama_llama3_1_8b
```

The evaluation writes `evaluation_results.json`, `evaluation_summary.md`, and one Markdown/JSON report plus a run log per question in that directory. These commands perform live local model and PubMed retrieval requests. To run without model interpretation, use `--mode extractive`; this still uses PubMed and is not a model evaluation. `auto` is retained as an alias for local Ollama, never as a provider fallback.

## Workflow, tools, and limits

The roles are Research Planner, Literature Researcher, Evidence Analyst, and Scientific Reviewer. The researcher has PubMed search and PMID detail retrieval functions; the reviewer has the evidence validator. Sources and claims pass through these stages, and the reviewer may request one correction. Offline tests check that the tools are registered and that the PubMed/parser/validation tools work with fixtures. Only a live smoke test verifies that the installed local model actually issues tool calls.

Ollama reports distinguish retrieved records, preliminary direct-E. coli title-keyword candidates, records supplied to the Analyst, and excluded adjacent/general records. This transparent lexical triage is not systematic or human relevance screening. Reports include PubMed titles, PMIDs/links, supporting abstract passages, validation status, uncertainty, and limitations. Validation proves source membership and literal passage presence only; it does not prove scientific entailment. Reports use abstracts, not full-text screening. Human scientific review remains necessary.

The five-question evaluation is a small demonstration, not a scientific benchmark. Offline tests do not answer the questions with a live model. No evaluation result is prefilled or fabricated. Historical reports are preserved.

## Legacy diagnostics and project layout

`gemini_client.py`, `native_gemini.py`, `smoke_gemini.py`, and `requirements-gemini-diagnostic.txt` remain for the earlier opt-in Gemini diagnostic. They are outside the normal workflow and are not installed by `requirements.txt`.

- `agents.py`: four-stage agent orchestration and extractive mode
- `ollama_client.py`: local model client and actionable local error reporting
- `tools.py`: PubMed search, PMID detail retrieval, evidence validation
- `research.py`: source-linked Markdown and JSON reports
- `evaluation.py`: five fixed research questions and evaluation artifacts
- `smoke_ollama.py`: opt-in single-agent local tool-call test
