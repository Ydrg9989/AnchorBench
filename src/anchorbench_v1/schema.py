"""Data schemas for AnchorBench v1 — ItemSpec and PromptView."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


ANSWER_FORMAT_INSTRUCTION = (
    "Return only a single integer 0\u2013100 on the last line."
)


@dataclass
class ItemSpec:
    """Prompt-agnostic ground-truth record for one benchmark item."""

    item_id: str
    suite: str
    domain: str
    template_family: str

    answer_space: Dict[str, Any] = field(
        default_factory=lambda: {"type": "int", "min": 0, "max": 100}
    )
    theta: int = 0
    y_star: int = 0
    y_star_components: Dict[str, Any] = field(default_factory=dict)

    # v2 dual gold answers (0 = not set / legacy item)
    y_star_theta: int = 0
    y_star_evidence: int = 0

    # v2 difficulty axis ("standard" for legacy, "easy"/"hard" for v2)
    difficulty: str = "standard"

    anchors: Dict[str, Any] = field(default_factory=dict)
    evidence_structured: List[Dict[str, Any]] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)

    # RAG-specific (optional)
    rag: Optional[Dict[str, Any]] = None
    # Tool-specific (optional)
    tool: Optional[Dict[str, Any]] = None
    # History v2 (optional): subset indices, warmup case data for two-stage protocol
    history: Optional[Dict[str, Any]] = None

    # LLM-generated scenario text (None = use template from domains.py)
    scenario_text: Optional[str] = None

    render_version: str = "1.0.0"
    generator_version: str = ""
    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("rag", "tool", "history", "scenario_text"):
            if d.get(k) is None:
                del d[k]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ItemSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PromptView:
    """Rendered prompt for one item × condition, ready for LLM evaluation."""

    item_id: str
    suite: str
    domain: str
    condition: str  # v1: "control"|"low_anchor"|"high_anchor"
                    # v2: "control"|"irrelevant_low"|"irrelevant_high"|"plausible_low"|"plausible_high"

    prompt_text: str
    prompt_components: Dict[str, str] = field(default_factory=dict)

    anchor_string: Optional[str] = None
    anchor_span: Optional[List[int]] = None  # [start, end] char offsets
    anchor_relevance: str = "none"  # "none" | "irrelevant" | "plausible"
    anchor_value: Optional[int] = None

    prompt_hash: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.prompt_hash and self.prompt_text:
            self.prompt_hash = "sha256:" + hashlib.sha256(
                self.prompt_text.encode()
            ).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("anchor_span", "anchor_value"):
            if d.get(k) is None:
                del d[k]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PromptView":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RAGDoc:
    """One document in the frozen RAG corpus."""

    doc_id: str
    domain: str
    doc_type: str  # v1: "evidence"|"distractor"|"anchor_low"|"anchor_high"
                   # v2: "core"|"filler"|"anchor_slot"
    text: str
    role: str = ""          # v2: "core"|"filler"|"anchor_slot"
    relevance: str = "none" # v2: "none"|"irrelevant"|"plausible"
    anchor_value: Optional[int] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("anchor_value",):
            if d.get(k) is None:
                del d[k]
        return d


def compute_prompt_hash(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def write_jsonl(records: list, path, cls=None):
    """Write list of dicts (or dataclass instances) to JSONL."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            obj = r.to_dict() if hasattr(r, "to_dict") else r
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path) -> list:
    """Read JSONL file into list of dicts."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
