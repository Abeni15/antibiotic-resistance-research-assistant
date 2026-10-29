# Learning guide

This project uses AutoGen AgentChat 0.7.5 with a four-role workflow and Ollama running locally. AutoGen's existing `OpenAIChatCompletionClient` sends compatible requests to `http://localhost:11434/v1`; the OpenAI package is a protocol adapter, not a dependency on the hosted OpenAI service. No Groq key or paid provider is needed.

## Roles and source tools

The planner prepares PubMed questions and queries. The researcher can call `search_pubmed` and `fetch_paper_details`. The analyst drafts claims only from supplied records. The scientific reviewer checks IDs and passages with `validate_evidence`, reviews limitations, and can request one correction. The report stage retains source IDs, links, passages, uncertainty, and caveats.

The three biological research tools are working PubMed search, PMID record retrieval, and evidence validation. Offline fixtures test them without using PubMed or a model. The opt-in `smoke_ollama.py` checks whether the local model actually invokes a tool.

## Setup and commands

Install Ollama from https://ollama.com/download/windows if needed. Its app starts a local server; when needed run `ollama serve` in a separate terminal. Then:

```powershell
ollama pull llama3.1:8b
ollama list
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python smoke_ollama.py
python main.py --mode ollama --question "What does literature report about plasmid-mediated resistance in E. coli?"
python evaluation.py --mode ollama --output-dir reports/ollama_llama3_1_8b
```

`smoke_ollama.py`, one-question CLI research, and evaluation make live local requests only when explicitly run. Evaluation outputs its JSON, summary Markdown, and individual reports/logs under the selected directory. Offline tests do not answer the questions with a model.

The default `llama3.1:8b` tag is documented by Ollama as tool-capable, but local performance and tool-call success depend on the installed Ollama version, model pull, and available hardware. Set `OLLAMA_MODEL` in `.env` to use an exact alternative local model tag. No silent model substitution occurs.

Evidence checks establish literal source and passage traceability, not entailment. Abstracts may omit important methods or context; scientific review remains necessary.
