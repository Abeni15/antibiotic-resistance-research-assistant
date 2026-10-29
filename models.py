"""Data models passed between research stages."""
from dataclasses import asdict, dataclass, field
from typing import Any
@dataclass
class Paper:
 pmid: str
 title: str = ""
 authors: list[str] = field(default_factory=list)
 year: str = ""
 journal: str = ""
 abstract: str = ""
 doi: str = ""
 pubmed_url: str = ""
 query: str = ""
 retrieved_at: str = ""
 def to_dict(self) -> dict[str, Any]: return asdict(self)
@dataclass
class Claim:
 text: str
 topic: str = "general"
 source_ids: list[str] = field(default_factory=list)
 supporting_passages: dict[str,str] = field(default_factory=dict)
 evidence_category: str = "not classified"
 uncertainty: str = ""
 limitations: list[str] = field(default_factory=list)
 reviewer_comments: str = ""
 validation_status: str = "needs_scientific_review"
 def to_dict(self) -> dict[str,Any]: return asdict(self)
