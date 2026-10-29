# Antibiotic Resistance Research Assistant

A local, multi-agent research assistant for investigating antibiotic resistance in *Escherichia coli* (*E. coli*) using PubMed, AutoGen AgentChat, and Ollama.

## Overview

The project explores how multiple specialized AI agents can support scientific literature research by retrieving publications, extracting source-linked evidence, validating evidence traceability, and generating structured research reports.

The system uses a locally hosted language model through Ollama, reducing reliance on hosted model APIs for the main research workflow.

## Objectives

- Automate parts of scientific literature discovery using PubMed.
- Organize research tasks across specialized AI agents.
- Extract evidence statements linked to scientific source material.
- Check source identifiers and supporting passages before accepting extracted claims.
- Generate readable Markdown reports and structured JSON outputs.
- Evaluate the reliability and practical limitations of a local multi-agent research workflow.

## Architecture

The workflow consists of four main agent roles:

1. **Research Planner** — defines the research direction and prepares literature-search tasks.
2. **Literature Researcher** — searches PubMed and retrieves publication information.
3. **Evidence Analyst** — extracts candidate evidence statements from retrieved source material.
4. **Scientific Reviewer** — reviews the findings and may request a correction when necessary.

An evidence-validation component checks source traceability and the presence of supporting passages. Validation of source membership and passage presence does not, by itself, establish that a scientific claim is fully supported by the source.

## Technology Stack

- **Python** — application logic and workflow implementation
- **AutoGen AgentChat** — multi-agent orchestration
- **Ollama** — local language-model inference
- **Llama 3.2 3B** — local model
