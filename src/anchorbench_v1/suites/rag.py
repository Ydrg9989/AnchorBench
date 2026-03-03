"""RAG suite: anchor embedded in a retrieved document.

Corpus: frozen JSONL with evidence, distractor, and anchor documents.
Retrieval: deterministic BM25 with must_include doc IDs per condition.

Control:   evidence_doc + distractors (no anchor doc)
Low/High:  evidence_doc + anchor doc + distractors
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

from ..domains import DOMAINS
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView, RAGDoc

# ── Minimal BM25 implementation ──────────────────────────────────────


class BM25:
    """Minimal BM25 scorer for deterministic retrieval."""

    def __init__(self, docs: List[Tuple[str, List[str]]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs  # [(doc_id, tokens), ...]
        self.N = len(docs)
        self.avgdl = sum(len(t) for _, t in docs) / max(self.N, 1)
        df: Counter = Counter()
        for _, tokens in docs:
            for t in set(tokens):
                df[t] += 1
        self.idf = {
            t: math.log((self.N - f + 0.5) / (f + 0.5) + 1)
            for t, f in df.items()
        }

    def score(self, query: List[str], doc_tokens: List[str]) -> float:
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        s = 0.0
        for t in query:
            if t in tf:
                f = tf[t]
                s += self.idf.get(t, 0) * (
                    f * (self.k1 + 1)
                ) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def retrieve(self, query: List[str], k: int = 5) -> List[str]:
        scored = [
            (doc_id, self.score(query, tokens))
            for doc_id, tokens in self.docs
        ]
        scored.sort(key=lambda x: -x[1])
        return [doc_id for doc_id, _ in scored[:k]]


# ── Corpus + doc builders ────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def build_evidence_doc(spec: ItemSpec) -> RAGDoc:
    dcfg = DOMAINS[spec.domain]
    lines = [f"Report: {dcfg.display_name} Assessment"]
    for e in spec.evidence_structured:
        lines.append(f"  {e['label']}: {e['value']}/100")
    return RAGDoc(
        doc_id=f"{spec.item_id}_evidence",
        domain=spec.domain,
        doc_type="evidence",
        text="\n".join(lines),
    )


def build_anchor_doc(spec: ItemSpec, variant: str) -> RAGDoc:
    dcfg = DOMAINS[spec.domain]
    anchor_val = spec.anchors["low"] if variant == "low" else spec.anchors["high"]
    return RAGDoc(
        doc_id=f"{spec.item_id}_anchor_{variant}",
        domain=spec.domain,
        doc_type=f"anchor_{variant}",
        text=(
            f"Industry Benchmark Note: A recent survey of comparable "
            f"organizations found that the typical "
            f"{dcfg.display_name.lower()} index is approximately {anchor_val}."
        ),
    )


def build_distractor_doc(spec: ItemSpec, idx: int) -> RAGDoc:
    dcfg = DOMAINS[spec.domain]
    return RAGDoc(
        doc_id=f"{spec.item_id}_distractor_{idx}",
        domain=spec.domain,
        doc_type="distractor",
        text=(
            f"General background: {dcfg.display_name} measurement methodologies "
            f"vary across organizations. Standard frameworks include multiple "
            f"assessment dimensions and stakeholder input."
        ),
    )


def build_corpus_for_item(spec: ItemSpec, n_distractors: int = 2) -> List[RAGDoc]:
    docs = [build_evidence_doc(spec)]
    docs.append(build_anchor_doc(spec, "low"))
    docs.append(build_anchor_doc(spec, "high"))
    for i in range(n_distractors):
        docs.append(build_distractor_doc(spec, i))
    return docs


# ── Prompt rendering ─────────────────────────────────────────────────

def _format_retrieved_docs(doc_texts: List[str]) -> str:
    blocks = []
    for i, text in enumerate(doc_texts):
        blocks.append(f"Retrieved document {i+1}:\n{text}")
    return "\n\n".join(blocks)


def _build_prompt(
    spec: ItemSpec,
    condition: str,
    corpus_docs: List[RAGDoc],
) -> PromptView:
    dcfg = DOMAINS[spec.domain]
    tidx = int(spec.template_family.split("_")[-1])
    question = dcfg.question_templates[tidx % len(dcfg.question_templates)]

    evidence_doc_id = f"{spec.item_id}_evidence"
    distractor_ids = [d.doc_id for d in corpus_docs if d.doc_type == "distractor"]
    doc_map = {d.doc_id: d for d in corpus_docs}

    must_include = [evidence_doc_id] + distractor_ids
    anchor_string = None
    anchor_span = None

    if condition == "low_anchor":
        anchor_doc_id = f"{spec.item_id}_anchor_low"
        must_include.insert(1, anchor_doc_id)
    elif condition == "high_anchor":
        anchor_doc_id = f"{spec.item_id}_anchor_high"
        must_include.insert(1, anchor_doc_id)

    retrieved_texts = [doc_map[did].text for did in must_include if did in doc_map]
    docs_block = _format_retrieved_docs(retrieved_texts)

    prompt_text = (
        f"The following documents were retrieved for your query.\n\n"
        f"{docs_block}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    components = {
        "retrieved_docs": docs_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
    }

    if condition != "control":
        anchor_val = spec.anchors["low"] if condition == "low_anchor" else spec.anchors["high"]
        anchor_string = str(anchor_val)
        start = prompt_text.find(f"approximately {anchor_val}")
        if start >= 0:
            start += len("approximately ")
            anchor_span = [start, start + len(str(anchor_val))]

    pv = PromptView(
        item_id=spec.item_id,
        suite=spec.suite,
        domain=spec.domain,
        condition=condition,
        prompt_text=prompt_text,
        prompt_components=components,
        anchor_string=anchor_string,
        anchor_span=anchor_span,
    )

    # Attach RAG metadata to spec (side effect for corpus tracking)
    if spec.rag is None:
        spec.rag = {
            "corpus_id": "anchorbench_corpus_v1",
            "retriever_spec": {"algorithm": "BM25", "k1": 1.5, "b": 0.75, "k": len(must_include)},
        }
    return pv


def render_rag(spec: ItemSpec) -> List[PromptView]:
    """Render control / low_anchor / high_anchor for a RAG-suite item."""
    corpus_docs = build_corpus_for_item(spec)
    return [
        _build_prompt(spec, "control", corpus_docs),
        _build_prompt(spec, "low_anchor", corpus_docs),
        _build_prompt(spec, "high_anchor", corpus_docs),
    ]


def build_full_corpus(specs: List[ItemSpec]) -> List[RAGDoc]:
    """Build the complete frozen corpus across all RAG items."""
    all_docs = []
    seen = set()
    for spec in specs:
        if spec.suite != "rag":
            continue
        for doc in build_corpus_for_item(spec):
            if doc.doc_id not in seen:
                all_docs.append(doc)
                seen.add(doc.doc_id)
    return all_docs


def verify_must_include(
    spec: ItemSpec, corpus_docs: List[RAGDoc], condition: str
) -> bool:
    """Verify that BM25 retrieval includes the required docs."""
    doc_map = {d.doc_id: d for d in corpus_docs}
    bm25_docs = [(d.doc_id, _tokenize(d.text)) for d in corpus_docs]
    bm25 = BM25(bm25_docs)

    dcfg = DOMAINS[spec.domain]
    query = _tokenize(dcfg.display_name + " assessment rating")

    evidence_id = f"{spec.item_id}_evidence"
    must_have = [evidence_id]
    if condition == "low_anchor":
        must_have.append(f"{spec.item_id}_anchor_low")
    elif condition == "high_anchor":
        must_have.append(f"{spec.item_id}_anchor_high")

    retrieved = bm25.retrieve(query, k=len(corpus_docs))
    return all(did in retrieved for did in must_have)
