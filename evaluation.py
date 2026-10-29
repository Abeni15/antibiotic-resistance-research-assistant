"""Five-question evaluation runner; records real results without fabricated scores."""
import argparse,asyncio,json,time
from pathlib import Path
from research import research
QUESTIONS=[
"What are the major mechanisms of antibiotic resistance in Escherichia coli?",
"How do beta-lactamase-mediated resistance and reduced drug permeability differ in Escherichia coli?",
"What does the retrieved literature report about plasmid-mediated antibiotic resistance in E. coli?",
"What are the limitations of methods used to detect antibiotic resistance in Escherichia coli?",
"What can be concluded about a narrowly specified antibiotic-resistance mechanism in Escherichia coli when only a small number of relevant publications are retrieved?",
]
async def evaluate(output_dir=Path("reports"),mode="ollama"):
 if mode not in {"auto","ollama","extractive"}:raise ValueError("mode must be auto (local Ollama), ollama, or extractive")
 out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);rows=[]
 for q in QUESTIONS:
  t=time.monotonic()
  try:
   r=await research(q,3,out,mode=mode)
   payload=json.loads(Path(r["data"]).read_text(encoding="utf-8"))
   validations=payload.get("validation",{}).get("results",[])
   screening=payload.get("screening") or {}
   rows.append({"question":q,"completed":r["ok"],"papers_retrieved":r["paper_count"],"title_keyword_candidates":screening.get("included_count"),"records_provided_to_analyst":screening.get("provided_to_analyst_count"),"selection_note":screening.get("note","No automated relevance screen was run."),"claims_produced":r["claim_count"],"citation_validation":{"verified":sum(x["status"]=="verified_source_and_passage" for x in validations),"failed":sum(x["status"]!="verified_source_and_passage" for x in validations)},"unsupported_or_uncertain_claims":sum(bool(c.get("uncertainty")) or c.get("validation_status") not in ("verified_source_and_passage",None) for c in payload.get("claims",[])),"errors":[] if r["ok"] else ["No papers returned"],"retries":"NCBI request retries are bounded at 3; exact count is not exposed","approx_seconds":round(time.monotonic()-t,2),"api_usage_or_cost":"Not returned by provider; PubMed API calls are unauthenticated unless NCBI_API_KEY is configured.","report":r["report"],"provider":r["provider"],"model":r["model"],"execution_mode":r["execution_mode"],"llm_used":r["llm_used"]})
  except Exception as e:rows.append({"question":q,"completed":False,"papers_retrieved":0,"claims_produced":0,"errors":[type(e).__name__],"approx_seconds":round(time.monotonic()-t,2)})
 summary={"run_date_utc":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),"mode":mode,"provider":"Ollama (local)" if mode in {"auto","ollama"} else "None (extractive)","results":rows}
 (out/"evaluation_results.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
 lines=["# Evaluation summary","",f"Run date (UTC): {summary['run_date_utc']}","",f"Completed workflows: {sum(bool(x.get('completed')) for x in rows)}/{len(rows)}","", "This is a focused demonstration evaluation, not a systematic benchmark. Retrieval counts reflect PubMed responses. Text validation checks only source membership and literal passage presence; claim entailment and scientific accuracy require human review.","","| Question | Completed | Records | Claims | Passage checks verified/failed | Seconds |","|---|---:|---:|---:|---:|---:|"]
 for x in rows:
  cv=x.get("citation_validation",{});lines.append(f"| {x['question']} | {x.get('completed',False)} | {x.get('papers_retrieved',0)} | {x.get('claims_produced',0)} | {cv.get('verified','?')}/{cv.get('failed','?')} | {x.get('approx_seconds','?')} |")
 lines += ["","## Results and observed failures","", "See evaluation_results.json for per-question retrieval, title-screen candidate counts, analyst input counts, errors, uncertainty, run times, and report paths. No results are assumed or prefilled.","","## Limits and next steps","", "- This runner uses a small search limit of three results per query to bound the demonstration.\n- Reported relevance counts are preliminary title-keyword candidates, not human-screened records or systematic review results.","","- The deterministic no-key workflow extracts passages; it does not assess scientific accuracy.","- For the evaluation sample, a human should check whether each abstract passage supports its associated claim and whether study designs were represented accurately.","- Add a reviewer-rated relevance and entailment rubric, then repeat with reproducible query/date settings."]
 (out/"evaluation_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
 return summary
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=Path("reports"));p.add_argument("--mode",choices=("ollama","auto","extractive"),default="ollama");args=p.parse_args()
 result=asyncio.run(evaluate(args.output_dir,args.mode));print(f"Completed {sum(bool(x.get('completed')) for x in result['results'])}/{len(result['results'])}; summary: {Path(args.output_dir,'evaluation_summary.md').resolve()}")
