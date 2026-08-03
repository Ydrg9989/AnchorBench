"""Data schemas for AnchorBench v1 — ItemSpec, PromptView, RAGDoc."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ANSWER_FORMAT_INSTRUCTION = (
    "Return only a single integer 0\u2013100 on the last line."
)

# Extension scoring functions
SCORING_FUNCTIONS = ("mean", "weighted_mean", "median")


@dataclass
class ItemSpec:
    """Prompt-agnostic ground-truth record for one benchmark item."""

    item_id: str
    suite: str
    domain: str
    template_family: str

    answer_space: dict[str, Any] = field(
        default_factory=lambda: {"type": "int", "min": 0, "max": 100}
    )
    theta: int = 0
    y_star: int = 0
    y_star_components: dict[str, Any] = field(default_factory=dict)

    y_star_theta: int = 0
    y_star_evidence: int = 0

    difficulty: str = "easy"

    anchors: dict[str, Any] = field(default_factory=dict)
    evidence_structured: list[dict[str, Any]] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)

    # Suite-specific metadata (optional)
    rag: dict[str, Any] | None = None
    tool: dict[str, Any] | None = None
    history: dict[str, Any] | None = None

    # LLM-generated scenario text (None = use template from domains.py)
    scenario_text: str | None = None

    # ── Auditable template selection metadata ────────────────────────
    evidence_label_family_idx: int = 0
    scenario_template_idx: int = 0
    question_template_idx: int = 0
    anchor_phrasing_idx: int = 0

    # Extension scoring (core = "mean")
    scoring_function: str = "mean"
    scoring_weights: list[float] | None = None

    render_version: str = "1.0.0"
    generator_version: str = ""
    seed: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("rag", "tool", "history", "scenario_text", "scoring_weights"):
            if d.get(k) is None:
                del d[k]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ItemSpec:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PromptView:
    """Rendered prompt for one item × condition, ready for LLM evaluation."""

    item_id: str
    suite: str
    domain: str
    condition: str

    prompt_text: str
    prompt_components: dict[str, str] = field(default_factory=dict)

    anchor_string: str | None = None
    anchor_span: list[int] | None = None
    anchor_relevance: str = "none"
    anchor_value: int | None = None

    prompt_hash: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

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
        d.pop("provenance", None)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PromptView:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class RAGDoc:
    """One document in the frozen RAG corpus."""

    doc_id: str
    domain: str
    doc_type: str
    text: str
    role: str = ""
    relevance: str = "none"
    anchor_value: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("anchor_value",):
            if d.get(k) is None:
                del d[k]
        return d


def write_jsonl(
    records: list[Any],
    path: str | Path,
    cls: type[json.JSONEncoder] | None = None,
) -> None:
    """Write list of dicts (or dataclass instances) to JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            obj = r.to_dict() if hasattr(r, "to_dict") else r
            f.write(json.dumps(obj, ensure_ascii=False, cls=cls) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL file into list of dicts."""
    items: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items
