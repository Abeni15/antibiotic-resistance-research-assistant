"""CLI for a PubMed-backed, local Ollama or extractive evidence report."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from ollama_client import OllamaConfigurationError, OllamaRequestError
from research import research


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Investigate antibiotic-resistance literature using PubMed."
    )
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--mode",
        choices=("ollama", "auto", "extractive"),
        default="ollama",
        help="ollama (default) runs the configured local model; extractive skips model interpretation.",
    )
    args = parser.parse_args()

    if not args.question.strip() or len(args.question) > 1000:
        parser.error("question must contain 1-1000 characters")
    if not 1 <= args.max_results <= 20:
        parser.error("--max-results must be 1-20")

    try:
        result = asyncio.run(
            research(
                args.question.strip(),
                args.max_results,
                args.output_dir,
                mode=args.mode,
            )
        )
    except OllamaConfigurationError as exc:
        print(f"Ollama configuration error: {exc}")
        return 2
    except OllamaRequestError as exc:
        print(f"Ollama request failed: {exc}")
        return 1
    except Exception as exc:
        # Avoid echoing arbitrary exception text that might contain request data.
        print(f"Research run failed ({type(exc).__name__}). Check the run log.")
        return 1

    print(f"PubMed records: {result['paper_count']}; evidence statements: {result['claim_count']}")
    print(f"Provider: {result['provider'] or 'None'}")
    print(f"Model: {result['model'] or 'None'}")
    print(f"Execution mode: {result['execution_mode']}")
    print(f"Elapsed: {result['elapsed_seconds']} seconds")
    print(f"Report: {result['report']}")
    print(f"Data: {result['data']}")
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
