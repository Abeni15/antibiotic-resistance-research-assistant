"""PubMed E-utilities and deterministic citation/passage validation."""
import json,re,time,threading,unicodedata,urllib.error,urllib.parse,urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime,timezone
from config import settings
from models import Paper
BASE="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ID=re.compile(r"^\d{1,12}$")
_REQUEST_LOCK=threading.Lock()
_LAST_REQUEST=0.0
class PubMedError(RuntimeError): pass
def _get(endpoint,params,attempts=3):
 params=dict(params)
 if settings.ncbi_email: params["email"]=settings.ncbi_email
 if settings.ncbi_api_key: params["api_key"]=settings.ncbi_api_key
 url=BASE+endpoint+"?"+urllib.parse.urlencode(params); delay=.5
 global _LAST_REQUEST
 for attempt in range(attempts):
  try:
   with _REQUEST_LOCK:
    wait=max(0.0,.34-(time.monotonic()-_LAST_REQUEST))
    if wait:time.sleep(wait)
    _LAST_REQUEST=time.monotonic()
    req=urllib.request.Request(url,headers={"User-Agent":"AntibioticResistanceResearchAssistant/1.0"})
    with urllib.request.urlopen(req,timeout=20) as r:return r.read()
  except (urllib.error.URLError,TimeoutError,OSError) as e:
   if attempt+1==attempts: raise PubMedError(f"NCBI request failed ({type(e).__name__})") from e
   time.sleep(delay);delay*=2
 raise PubMedError("NCBI request failed")
def _txt(n): return " ".join("".join(n.itertext()).split()) if n is not None else ""
def _parse(a,query=""):
 pmid=_txt(a.find(".//PMID"))
 if not pmid:return None
 authors=[]
 for x in a.findall(".//AuthorList/Author"):
  n=" ".join(filter(None,[_txt(x.find("ForeName")),_txt(x.find("LastName"))])) or _txt(x.find("CollectiveName"))
  if n:authors.append(n)
 year=_txt(a.find(".//PubDate/Year")) or _txt(a.find(".//PubDate/MedlineDate")).split()[0]
 abstract=" ".join(filter(None,(_txt(x) for x in a.findall(".//Abstract/AbstractText"))))
 doi=next((_txt(x) for x in a.findall(".//ArticleId") if x.get("IdType")=="doi"),"")
 return Paper(pmid,_txt(a.find(".//ArticleTitle")),authors,year,_txt(a.find(".//Journal/Title")),abstract,doi,f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",query,datetime.now(timezone.utc).isoformat())
def fetch_paper_details_batch(pmids,query=""):
 valid=[str(p) for p in pmids if ID.fullmatch(str(p))]
 if not valid:return []
 try: root=ET.fromstring(_get("efetch.fcgi",{"db":"pubmed","id":",".join(valid),"retmode":"xml"}))
 except ET.ParseError as e: raise PubMedError("NCBI returned malformed XML") from e
 return [p for a in root.findall(".//PubmedArticle") if (p:=_parse(a,query))]
def search_pubmed(query:str,max_results:int=10):
 """Search NCBI PubMed. Result limit is 1-50; returns records with abstracts."""
 query=query.strip() if isinstance(query,str) else ""
 if not query or len(query)>500:return {"ok":False,"error":"query must contain 1-500 characters","papers":[]}
 if not isinstance(max_results,int) or isinstance(max_results,bool) or not 1<=max_results<=50:return {"ok":False,"error":"max_results must be 1-50","papers":[]}
 try:
  d=json.loads(_get("esearch.fcgi",{"db":"pubmed","term":query,"retmax":str(max_results),"retmode":"json","sort":"relevance"}))
  ids=d.get("esearchresult",{}).get("idlist",[])
  papers=fetch_paper_details_batch(ids,query) if ids else []
  return {"ok":True,"query":query,"count_reported":int(d.get("esearchresult",{}).get("count",0)),"papers":[p.to_dict() for p in papers]}
 except (PubMedError,ValueError,KeyError,TypeError) as e:return {"ok":False,"query":query,"error":str(e),"papers":[]}
def fetch_paper_details(pmid:str):
 """Fetch one record by numeric PMID, with available abstract."""
 if not isinstance(pmid,str) or not ID.fullmatch(pmid.strip()):return {"ok":False,"pmid":str(pmid),"error":"PMID must contain 1-12 digits"}
 try:
  papers=fetch_paper_details_batch([pmid.strip()])
  if not papers:return {"ok":False,"pmid":pmid,"error":"No PubMed record found"}
  p=papers[0];return {"ok":True,"paper":p.to_dict(),"warning":"Abstract unavailable in PubMed" if not p.abstract else ""}
 except PubMedError as e:return {"ok":False,"pmid":pmid,"error":str(e)}
def _norm(s):
 s=unicodedata.normalize("NFKC",s).casefold()
 return " ".join(re.sub(r"[^\w\s]"," ",s).split())
def validate_evidence(claims:list,sources:list):
 """Check source membership and literal passage presence after text normalization."""
 byid={str(s.get("pmid",s.get("source_id",""))):s for s in sources};out=[]
 for i,c in enumerate(claims):
  ids=c.get("source_ids") or ([c["source_id"]] if c.get("source_id") else [])
  passages=c.get("supporting_passages",{})
  if isinstance(passages,str) and ids:passages={ids[0]:passages}
  if not ids:out.append({"claim_index":i,"status":"missing_source","reason":"Claim has no source identifier","checks":[]});continue
  checks=[]
  for sid in ids:
   src=byid.get(str(sid))
   if src is None:checks.append({"source_id":str(sid),"status":"missing_source","reason":"Identifier absent from retrieved sources"});continue
   passage=passages.get(str(sid),"") if isinstance(passages,dict) else ""
   corpus=" ".join(filter(None,[src.get("abstract",""),src.get("title","")]))
   ok=bool(passage) and _norm(passage) in _norm(corpus)
   checks.append({"source_id":str(sid),"status":"verified_source_and_passage" if ok else "passage_not_found","reason":"Passage occurs in stored title/abstract after normalization" if ok else "No supplied passage, or passage not present"})
  states={x["status"] for x in checks};status="verified_source_and_passage" if states=={"verified_source_and_passage"} else ("missing_source" if "missing_source" in states else "passage_not_found")
  out.append({"claim_index":i,"status":status,"reason":"Source/text check only; scientific entailment requires human review","checks":checks})
 return {"ok":True,"results":out}
