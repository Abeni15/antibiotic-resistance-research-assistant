# Evaluation presentation notes

## Scope

The suite retains five antibiotic-resistance questions about *E. coli*. `evaluation.py` records retrieval counts, report paths, literal passage validation, uncertainty, timing, and failures. The suite is a demonstration, not a systematic accuracy benchmark.

## Model setup

Normal execution uses the configured local Ollama model through the existing AutoGen OpenAI-compatible client and `http://localhost:11434/v1`. The default `llama3.1:8b` is documented by Ollama as tool-capable. Offline tests do not test inference. Run the opt-in local smoke test to check text generation plus a tool call on the installed model.

## Interpreting results

The validator checks that source IDs belong to the retrieved set and that quoted passages are present. It does not establish that a passage supports a claim scientifically. Results need human relevance and entailment review; no scores or live model outputs are fabricated.
