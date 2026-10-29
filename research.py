"""Run the research workflow and write traceable Markdown and JSON reports."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents import run_agents


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    # AutoGen's event logger includes complete model prompts and responses. Keep
    # diagnostics in our bounded workflow events instead of duplicating that content.
    logging.getLogger("autogen_core.events").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.basicConfig(
        filename=path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
        force=True,
    )
    return logging.getLogger("research")


async def research(
    question: str,
    max_results: int = 5,
    output_dir: Path = Path("reports"),
    mode: str = "auto",
) -> dict[str, Any]:
    started = time.monotonic()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = await run_agents(
        question,
        max_results,
        setup_logging(output_dir / f"run_{stamp}.log"),
        mode=mode,
    )
    papers = result["papers"]
    claims = result["claims"]
    validation = result["validation"]
    screening = result.get("screening")
    review_skipped = "reviewer model" in result.get("review", "").lower() and "skipped" in result.get("review", "").lower()
    included_pmids = set(screening.get("included_pmids", [])) if screening else set()
    analyst_pmids = set(screening.get("provided_to_analyst_pmids", [])) if screening else set()
    known_pmids = analyst_pmids if screening else {str(paper["pmid"]) for paper in papers}

    findings = []
    for index, claim in enumerate(claims):
        citations = []
        for source_id in claim.get("source_ids", []):
            if str(source_id) in known_pmids:
                citations.append(
                    f'[PMID:{source_id}](https://pubmed.ncbi.nlm.nih.gov/{source_id}/)'
                )
            else:
                citations.append(f"Unverified source ID: {source_id}")
        citation_text = ", ".join(citations) or "No source"
        passages = "; ".join(
            f"“{passage}”" for passage in claim.get("supporting_passages", {}).values()
        ) or "No supporting passage."
        status = claim.get(
            "validation_status",
            validation["results"][index]["status"]
            if index < len(validation["results"])
            else "needs_scientific_review",
        )
        findings.append(
            f"### {claim.get('topic', 'Evidence')}\n\n"
            f"{claim.get('text', '[Claim text unavailable]')} {citation_text}.\n\n"
            f"Evidence category: {claim.get('evidence_category', 'unspecified')}. "
            f"Validation: {status}. {claim.get('uncertainty', '')}\n\n"
            f"Supporting abstract passage: {passages}\n"
        )
    if not findings:
        if review_skipped:
            findings = [
                "No valid claims were available for review; the Reviewer model call was skipped. "
                "No claims are marked as validated. This does not establish that the retrieved "
                "literature contains no relevant findings."
            ]
        else:
            findings = [
                "No claim could be supported from the retrieved abstracts. This is a gap in "
                "this search sample, not evidence a mechanism is absent."
            ]

    reference_papers = (
        [paper for paper in papers if str(paper.get("pmid", "")) in included_pmids]
        if screening
        else papers
    )
    references = [
        f'- **PMID:{paper["pmid"]}** — {paper.get("title", "Title unavailable")} '
        f'({paper.get("year", "year unavailable")}). {paper.get("journal", "")} '
        f'[PubMed]({paper.get("pubmed_url", "")})'
        + (f' DOI: {paper["doi"]}.' if paper.get("doi") else "")
        for paper in reference_papers
    ]
    queries = "\n".join("- " + query for query in result["queries"])
    ollama_mode = result["execution_mode"] == "ollama_local"
    if ollama_mode and review_skipped:
        if screening and screening.get("provided_to_analyst_count", 0) == 0:
            collaboration_summary = (
                "The Planner and Researcher stages ran. No direct E. coli title-keyword candidates "
                "were available, so the Analyst and Reviewer model calls were skipped."
            )
        else:
            collaboration_summary = (
                "The Planner, Researcher, and Analyst stages ran. No valid claims were available, "
                "so the Reviewer model call was skipped."
            )
    elif ollama_mode:
        collaboration_summary = "Four local Ollama-backed AutoGen agents collaborated on this report."
    else:
        collaboration_summary = (
            "This report used the extractive mode; it contains retrieved abstract excerpts and "
            "deterministic checks, not LLM-based scientific interpretation."
        )
    if screening:
        screening_section = (
            f"- Preliminary title screen: {screening['screened_count']} records screened; "
            f"{screening['included_count']} keyword candidates included; "
            f"{screening['excluded_count']} excluded.\n"
            f"- Records supplied to the Analyst: {screening['provided_to_analyst_count']}; "
            f"{len(screening['not_sent_due_to_source_limit_pmids'])} additional included candidates "
            "were not supplied because of the per-run source cap.\n"
            f"- Exclusion categories: {screening['excluded_by_category'] or 'none'}.\n"
            f"- Method note: {screening['note']}"
        )
        excluded_rows = "\n".join(
            f"- PMID:{item['pmid']} — {item['title']} — {item['category']}: {item['reason']}"
            for item in screening["excluded_records"]
        ) or "No records excluded by the title screen."
        analyst_rows = "\n".join(
            f"- PMID:{paper['pmid']} — {paper.get('title', 'Title unavailable')}"
            for paper in papers
            if str(paper.get("pmid", "")) in set(screening["provided_to_analyst_pmids"])
        ) or "No records were supplied to the Analyst."
        not_sent = "\n".join(
            f"- PMID:{paper['pmid']} — {paper.get('title', 'Title unavailable')}"
            for paper in papers
            if str(paper.get("pmid", "")) in set(screening["not_sent_due_to_source_limit_pmids"])
        ) or "None."
        screening_report = (
            f"\n## Preliminary relevance screening\n\n{screening_section}\n\n"
            f"### Included title candidates supplied to the Analyst\n\n{analyst_rows}\n\n"
            f"### Excluded from claim generation\n\n{excluded_rows}\n\n"
            f"### Included keyword candidates not sent to the Analyst due to source cap\n\n{not_sent}\n"
        )
    else:
        screening_report = (
            "\n## Relevance screening\n\n"
            "No relevance screen was performed in extractive mode; retrieved records are shown as retrieved.\n"
        )
    selection_note = (
        "Records were deduplicated by PMID, then title-keyword triage was applied; this is not systematic relevance screening."
        if screening
        else "Records were deduplicated by PMID; relevance screening was not performed in extractive mode."
    )
    report = f"""# Focused literature investigation

## Research question

{question}

## Executive summary

The PubMed search retrieved {len(papers)} unique records; {len(claims)} evidence statements were produced. {collaboration_summary}

## Execution details

- Provider: {result.get("provider") or "None (extractive fallback)"}
- Model: {result.get("model") or "None"}
- Execution mode: {result["execution_mode"]}

## Background and scope

This is an educational antibiotic-resistance literature investigation, not a systematic review, diagnosis, or treatment advice. Analysis is limited to PubMed metadata and abstracts.

## Literature search methodology

- Database/API: PubMed via NCBI E-utilities.
- Search queries:
{queries}
- Retrieval date (UTC): {datetime.now(timezone.utc).date().isoformat()}
- Unique records retrieved: {len(papers)}.
- Title-screen keyword candidates: {screening['included_count'] if screening else 'not screened'}; records supplied to the Analyst: {screening['provided_to_analyst_count'] if screening else 'not applicable'}.
- Selection note: {selection_note}
- Model calls used: {result["llm_used"]}
{screening_report}

## Findings by mechanism

{'The labels below reflect retrieval query terms only; no independent relevance or mechanism classification was performed.\n' if not ollama_mode else ''}
{chr(10).join(findings)}

## Comparison and synthesis

The abstract sample may represent different strains, drugs, and study designs. Do not assume mechanisms apply to every E. coli strain or antibiotic. {'Interpret model-generated claims alongside their source passages and reviewer comments.' if ollama_mode else 'No model-based cross-source synthesis was performed.'}

## Conflicting findings and uncertainties

No conflicts were adjudicated automatically. Abstract-only retrieval can omit details needed to compare methods, populations, and drug-specific effects. A small sample is insufficient evidence, not proof of absence.

## Limitations of the investigation

- Focused search, not an exhaustive or systematic review.
- Abstracts can omit methods and results; full text was not reviewed.
- Passage matching checks source text, not whether it entails the claim.
- Human scientific review remains necessary for scientific accuracy.

## Conclusion

Conclusions are limited to retrieved PubMed abstracts. If no relevant record was retrieved, this search did not resolve that subquestion.

## References

{'Screen-included title-keyword candidates (not all were necessarily sent to the Analyst):' if screening else 'Retrieved records:'}

{chr(10).join(references) if references else "No PubMed records retrieved."}
"""
    markdown_path = output_dir / f"research_{stamp}.md"
    json_path = output_dir / f"research_{stamp}.json"
    elapsed = round(time.monotonic() - started, 2)
    markdown_path.write_text(report, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "question": question,
                **result,
                "elapsed_seconds": elapsed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "ok": bool(papers),
        "report": str(markdown_path.resolve()),
        "data": str(json_path.resolve()),
        "paper_count": len(papers),
        "included_count": screening["included_count"] if screening else None,
        "claim_count": len(claims),
        "elapsed_seconds": elapsed,
        "llm_used": result["llm_used"],
        "provider": result.get("provider"),
        "model": result.get("model"),
        "execution_mode": result["execution_mode"],
    }
