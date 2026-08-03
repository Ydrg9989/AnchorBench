"""RAG suite: ecological anchoring via retrieved documents.

Each item has a frozen 3-document mini-corpus:
  doc_core        — genuinely relevant context, no anchor
  doc_filler      — neutral methodological background, no anchor
  doc_anchor_slot — condition-dependent (control / irrelevant / plausible)

Retrieval is deterministic: fixed order [doc_core, doc_anchor_slot, doc_filler].
Only doc_anchor_slot varies across matched conditions; all other prompt
components are identical.

Base conditions (5):
  control         — neutral note, no numeric anchor
  irrelevant_low  — arbitrary metadata number (low anchor)
  irrelevant_high — arbitrary metadata number (high anchor)
  plausible_low   — weakly credible prior (low anchor)
  plausible_high  — weakly credible prior (high anchor)

Ablation conditions (8):
  *_order_first     — anchor doc in position 1 (instead of 2)
  *_order_last      — anchor doc in position 3 (instead of 2)
  irrelevant_*_nodiscl — irrelevant without explicit unrelated disclaimer
  plausible_*_authority — plausible with upgraded authority source
"""

from __future__ import annotations

from ..domains import ALL_DOMAINS as DOMAINS
from ..schema import ANSWER_FORMAT_INSTRUCTION, ItemSpec, PromptView, RAGDoc
from ._shared import CONDITIONS, format_evidence, resolve_templates

# ── Document templates ───────────────────────────────────────────────

_DOC_CORE_TEMPLATES: dict[str, list[str]] = {
    "pricing_wtp": [
        "Market Context Report: The product category under assessment has seen variable customer reception across comparable markets. Key factors influencing willingness-to-pay include perceived differentiation, switching costs, and competitive intensity. Assessment methodologies typically combine conjoint analysis with direct survey instruments.",
        "Pricing Research Summary: This assessment examines customer price sensitivity for a product offering. Historical patterns in this category indicate that willingness-to-pay is influenced by brand positioning, feature completeness, and market maturity. Multiple independent research instruments were used to capture price tolerance signals.",
        "Category Assessment Background: The willingness-to-pay evaluation draws on survey-based instruments and historical analogs. Customer segments in this category tend to show heterogeneous price sensitivity, driven by usage frequency, perceived value, and available alternatives.",
        "Valuation Context: This pricing assessment considers a product offering within a competitive landscape. Price tolerance in comparable markets has been shaped by product lifecycle stage, customer loyalty dynamics, and the availability of substitute offerings.",
    ],
    "operations_time": [
        "Operations Assessment Background: The facility under review operates within a sector where efficiency benchmarks are influenced by process maturity, equipment age, and workforce training levels. Independent audits typically assess throughput, cycle time, and quality-adjusted output across multiple dimensions.",
        "Process Efficiency Context: This assessment evaluates operational performance using standardized rating instruments. Comparable operations in this sector show wide variance depending on lean adoption, automation levels, and maintenance scheduling practices.",
        "Operational Review Summary: The efficiency evaluation considers multiple performance dimensions. Industry experience suggests that efficiency scores are driven by capacity utilization, process standardization, and the effectiveness of continuous improvement programs.",
        "Facility Assessment Context: This operational efficiency review draws on independent audit data. Performance in comparable facilities varies based on technology adoption, staffing models, and the maturity of quality management systems.",
    ],
    "transportation_logistics": [
        "Logistics Assessment Background: The transportation operation under review serves markets where reliability depends on route complexity, fleet management practices, and external factors such as weather and infrastructure quality. Multiple performance indicators are collected from independent sources.",
        "Delivery Reliability Context: This assessment evaluates logistics performance across several operational dimensions. Comparable carriers show variability in reliability driven by network density, last-mile execution, and real-time tracking capabilities.",
        "Transportation Review Summary: The reliability evaluation considers on-time performance, asset utilization, and incident rates. Industry patterns suggest that logistics reliability is jointly determined by planning sophistication and operational execution quality.",
        "Carrier Performance Context: This logistics reliability assessment draws on multi-source performance data. Reliability in comparable operations has been influenced by capacity management, route optimization maturity, and customer communication protocols.",
    ],
    "resource_consumption": [
        "Resource Efficiency Context: The entity under assessment operates in a sector where resource consumption patterns depend on technology vintage, process design, and regulatory requirements. Sustainability audits typically evaluate energy use, waste generation, and material efficiency.",
        "Sustainability Assessment Background: This resource efficiency evaluation draws on independent audit scores. Comparable organizations show wide variance in efficiency driven by capital investment in efficiency technologies, operational discipline, and regulatory stringency.",
        "Consumption Review Summary: The assessment considers multiple dimensions of resource use. Historical patterns suggest that resource efficiency scores depend on equipment modernization, staff awareness, and the comprehensiveness of monitoring systems.",
        "Environmental Performance Context: This resource consumption assessment evaluates efficiency across energy, water, materials, and emissions dimensions. Comparable facilities demonstrate that efficiency outcomes are jointly determined by technology, management commitment, and external accountability frameworks.",
    ],
    "market_demographics": [
        "Market Adoption Context: The product under assessment targets a demographic segment where adoption depends on market readiness, competitive alternatives, and distribution channel effectiveness. Research instruments capture signals from early adopters, market penetration data, and brand recognition.",
        "Demographics Assessment Background: This market adoption evaluation draws on survey-based and behavioral research data. Comparable product launches in this segment show that adoption outcomes are influenced by product-market fit, pricing strategy, and marketing channel selection.",
        "Adoption Research Summary: The assessment evaluates adoption potential using multiple independent research signals. Historical patterns suggest that adoption indices in this category are driven by demographic alignment, competitive positioning, and distribution reach.",
        "Market Readiness Context: This adoption potential assessment considers demographic fit, brand awareness, and competitive dynamics. Comparable market entries indicate that adoption trajectories depend on timing, channel strategy, and the strength of the value proposition.",
    ],
    "legal_policy": [
        "Compliance Assessment Background: The organization under review operates in a regulatory environment where compliance posture depends on governance maturity, audit frequency, and the complexity of applicable regulations. Independent assessments typically evaluate policy adherence, risk management, and historical performance.",
        "Regulatory Context Report: This compliance evaluation draws on multi-dimensional audit data. Organizations in comparable regulatory environments show that compliance outcomes are influenced by training programs, internal controls, and the organization of compliance functions.",
        "Policy Compliance Summary: The assessment considers regulatory audit scores, internal compliance ratings, and third-party risk assessments. Industry experience suggests that compliance indices reflect the interplay of governance culture, resource allocation, and regulatory complexity.",
        "Compliance Posture Context: This regulatory compliance assessment evaluates adherence across multiple regulatory frameworks. Comparable organizations demonstrate that compliance maturity depends on leadership commitment, system integration, and the regularity of internal audits.",
    ],
}

_DOC_FILLER_TEMPLATES: dict[str, list[str]] = {
    "pricing_wtp": [
        "Assessment Methodology Note: The evaluation framework employs standardized rating instruments administered independently across multiple research channels. All scores are normalized to a common scale for cross-source comparability. Assessment protocols follow established market research guidelines.",
        "Data Collection Procedures: Research data was collected using parallel independent instruments with standardized administration protocols. Quality controls include respondent screening, consistency checks, and outlier flagging procedures applied uniformly across all data sources.",
    ],
    "operations_time": [
        "Audit Methodology Note: Efficiency ratings were collected using standardized audit protocols administered by independent assessment teams. All scores are calibrated against a common reference framework to ensure cross-facility comparability.",
        "Assessment Procedures: The evaluation employed parallel independent audits with standardized instruments. Data quality assurance includes inter-rater reliability checks, calibration against reference standards, and systematic documentation of assessment conditions.",
    ],
    "transportation_logistics": [
        "Performance Data Methodology: Reliability metrics were collected from independent operational monitoring systems. All indicators are normalized to a common performance scale. Data collection followed standardized logistics performance measurement protocols.",
        "Measurement Procedures: The evaluation draws on parallel independent data sources with standardized collection methods. Quality assurance procedures include sensor calibration, data validation checks, and systematic handling of reporting gaps.",
    ],
    "resource_consumption": [
        "Audit Methodology Note: Resource efficiency scores were derived from standardized sustainability audit protocols. All metrics are normalized to a common efficiency scale. Assessment procedures follow established environmental management evaluation frameworks.",
        "Data Collection Procedures: Efficiency data was collected through independent audit instruments with standardized measurement protocols. Quality controls include equipment calibration records, measurement uncertainty documentation, and systematic verification procedures.",
    ],
    "market_demographics": [
        "Research Methodology Note: Market adoption signals were collected using standardized survey instruments and behavioral analytics. All scores are normalized to a common adoption readiness scale. Research protocols follow established market analysis guidelines.",
        "Data Collection Procedures: Adoption research data was gathered through independent survey channels with standardized administration. Quality assurance includes respondent validation, sample representativeness checks, and systematic documentation of research conditions.",
    ],
    "legal_policy": [
        "Audit Methodology Note: Compliance ratings were derived from standardized regulatory assessment frameworks. All scores are normalized to a common compliance scale. Assessment protocols follow established regulatory audit guidelines.",
        "Assessment Procedures: The evaluation employed independent audit teams using standardized compliance instruments. Data quality controls include auditor certification verification, cross-validation of findings, and systematic documentation of regulatory scope.",
    ],
}


# ── Anchor-slot document phrasing pools ──────────────────────────────

_RAG_IRRELEVANT_PHRASINGS = [
    "Administrative note: document batch reference #{anchor}. "
    "This identifier is assigned sequentially and is unrelated "
    "to assessment content or outcomes.",

    "Processing note: this document was assigned queue number {anchor}. "
    "This number is a sequential processing identifier with no "
    "relation to assessment outcomes.",

    "Filing reference: assessment request #{anchor}. "
    "This is an administrative tracking number unrelated to "
    "the assessment content.",

    "Workflow note: this item is entry {anchor} in the review log. "
    "This sequential identifier does not relate to the assessment.",
]

_RAG_PLAUSIBLE_PHRASINGS = [
    "A recent survey of comparable organizations reported "
    "a {metric} index of approximately {anchor}.",

    "A preliminary benchmarking study estimated the "
    "{metric} index at approximately {anchor}.",

    "Comparable organizations in the sector reported "
    "{metric} scores near {anchor}.",
]

_RAG_AUTHORITY_PHRASINGS = [
    "An expert panel commissioned by the industry regulatory body "
    "estimated the {metric} index at approximately {anchor}.",

    "A panel of domain specialists jointly assessed the "
    "{metric} index at approximately {anchor}.",
]


# ── Anchor-slot document builders ────────────────────────────────────

def _build_anchor_slot_control(item_id: str, domain: str) -> RAGDoc:
    return RAGDoc(
        doc_id=f"{item_id}_anchor_slot_control",
        domain=domain,
        doc_type="anchor_slot",
        role="anchor_slot",
        relevance="none",
        text=(
            "Assessment workflow note: no external benchmark data "
            "was located for this assessment case."
        ),
    )


def _build_anchor_slot_irrelevant(
    item_id: str, domain: str, direction: str, anchor_value: int,
    phrasing_idx: int = 0,
) -> RAGDoc:
    template = _RAG_IRRELEVANT_PHRASINGS[phrasing_idx % len(_RAG_IRRELEVANT_PHRASINGS)]
    return RAGDoc(
        doc_id=f"{item_id}_anchor_slot_irrelevant_{direction}",
        domain=domain,
        doc_type="anchor_slot",
        role="anchor_slot",
        relevance="irrelevant",
        anchor_value=anchor_value,
        text=template.format(anchor=anchor_value),
    )


def _build_anchor_slot_plausible(
    item_id: str, domain: str, direction: str, anchor_value: int,
    phrasing_idx: int = 0,
) -> RAGDoc:
    dcfg = DOMAINS[domain]
    metric = dcfg.metric_name or dcfg.display_name.lower()
    template = _RAG_PLAUSIBLE_PHRASINGS[phrasing_idx % len(_RAG_PLAUSIBLE_PHRASINGS)]
    return RAGDoc(
        doc_id=f"{item_id}_anchor_slot_plausible_{direction}",
        domain=domain,
        doc_type="anchor_slot",
        role="anchor_slot",
        relevance="plausible",
        anchor_value=anchor_value,
        text=template.format(metric=metric, anchor=anchor_value),
    )


def _build_anchor_slot_irrelevant_nodiscl(
    item_id: str, domain: str, direction: str, anchor_value: int,
    phrasing_idx: int = 0,
) -> RAGDoc:
    """Irrelevant anchor without the explicit unrelated disclaimer."""
    return RAGDoc(
        doc_id=f"{item_id}_anchor_slot_irrelevant_nodiscl_{direction}",
        domain=domain,
        doc_type="anchor_slot",
        role="anchor_slot",
        relevance="irrelevant",
        anchor_value=anchor_value,
        text=f"Administrative note: document batch reference #{anchor_value}.",
    )


def _build_anchor_slot_plausible_authority(
    item_id: str, domain: str, direction: str, anchor_value: int,
    phrasing_idx: int = 0,
) -> RAGDoc:
    """Plausible anchor with upgraded authority framing."""
    dcfg = DOMAINS[domain]
    metric = dcfg.metric_name or dcfg.display_name.lower()
    template = _RAG_AUTHORITY_PHRASINGS[phrasing_idx % len(_RAG_AUTHORITY_PHRASINGS)]
    return RAGDoc(
        doc_id=f"{item_id}_anchor_slot_plausible_authority_{direction}",
        domain=domain,
        doc_type="anchor_slot",
        role="anchor_slot",
        relevance="authority",
        anchor_value=anchor_value,
        text=template.format(metric=metric, anchor=anchor_value),
    )


def _build_anchor_slot_intensity(
    item_id: str, domain: str, direction: str, anchor_value: int,
    intensity: str, phrasing_idx: int = 0,
) -> RAGDoc:
    """Plausible anchor with mild/strong source-credibility framing (P1).

    Uses the same preamble pools that External uses (attached in
    ``anchorbench.data.domains`` under ``plausible_mild`` /
    ``plausible_strong``) but wraps them in a RAG-document text body.
    """
    assert intensity in ("plausible_mild", "plausible_strong")
    dcfg = DOMAINS[domain]
    pool = dcfg.anchor_preambles.get(intensity, [])
    if not pool:
        raise KeyError(
            f"Domain {domain!r} missing {intensity!r} preamble pool; "
            "make sure data.domains attached it."
        )
    template = pool[phrasing_idx % len(pool)]
    body = template.format(anchor=anchor_value)
    return RAGDoc(
        doc_id=f"{item_id}_anchor_slot_{intensity}_{direction}",
        domain=domain,
        doc_type="anchor_slot",
        role="anchor_slot",
        relevance=intensity,
        anchor_value=anchor_value,
        text=body,
    )


# ── Per-item corpus construction ─────────────────────────────────────

def build_item_corpus(spec: ItemSpec) -> dict[str, RAGDoc]:
    """Build the frozen 3-document corpus for one RAG item.

    Returns dict keyed by role: {core, filler, anchor_slot_{condition}...}.
    The anchor_slot documents are generated for all 5 conditions.
    """
    domain = spec.domain
    tidx = int(spec.template_family.split("_")[-1])

    core_templates = _DOC_CORE_TEMPLATES[domain]
    filler_templates = _DOC_FILLER_TEMPLATES[domain]

    core_text = core_templates[tidx % len(core_templates)]
    filler_text = filler_templates[tidx % len(filler_templates)]

    doc_core = RAGDoc(
        doc_id=f"{spec.item_id}_core",
        domain=domain,
        doc_type="core",
        role="core",
        relevance="none",
        text=core_text,
    )
    doc_filler = RAGDoc(
        doc_id=f"{spec.item_id}_filler",
        domain=domain,
        doc_type="filler",
        role="filler",
        relevance="none",
        text=filler_text,
    )

    pidx = getattr(spec, "anchor_phrasing_idx", 0)
    anchor_docs = {}
    anchor_docs["control"] = _build_anchor_slot_control(spec.item_id, domain)

    for direction in ("low", "high"):
        anchor_val = spec.anchors[direction]
        anchor_docs[f"irrelevant_{direction}"] = _build_anchor_slot_irrelevant(
            spec.item_id, domain, direction, anchor_val, pidx,
        )
        anchor_docs[f"plausible_{direction}"] = _build_anchor_slot_plausible(
            spec.item_id, domain, direction, anchor_val, pidx,
        )
        anchor_docs[f"irrelevant_{direction}_nodiscl"] = _build_anchor_slot_irrelevant_nodiscl(
            spec.item_id, domain, direction, anchor_val, pidx,
        )
        anchor_docs[f"plausible_{direction}_authority"] = _build_anchor_slot_plausible_authority(
            spec.item_id, domain, direction, anchor_val, pidx,
        )

    result = {"core": doc_core, "filler": doc_filler}
    result.update(anchor_docs)
    return result


def build_full_corpus(specs: list[ItemSpec]) -> list[RAGDoc]:
    """Build complete frozen corpus for all RAG items."""
    all_docs: list[RAGDoc] = []
    seen: set = set()
    for spec in specs:
        if spec.suite != "rag":
            continue
        corpus = build_item_corpus(spec)
        for doc in corpus.values():
            if doc.doc_id not in seen:
                all_docs.append(doc)
                seen.add(doc.doc_id)
    return all_docs


# ── Prompt rendering ─────────────────────────────────────────────────

_RETRIEVAL_HEADER = "The following documents were retrieved for your assessment query.\n"


def _format_retrieved_docs(docs: list[RAGDoc]) -> str:
    blocks = []
    for i, doc in enumerate(docs):
        blocks.append(f"[Document {i+1}]\n{doc.text}")
    return "\n\n".join(blocks)


def _build_prompt(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    corpus: dict[str, RAGDoc],
    *,
    doc_key: str | None = None,
    doc_order: str = "middle",
    ablation_type: str | None = None,
) -> PromptView:
    scenario, question, _, _ = resolve_templates(spec)
    evidence_block = format_evidence(spec.evidence_structured)

    doc_core = corpus["core"]
    doc_filler = corpus["filler"]
    doc_anchor = corpus[doc_key or condition]

    if doc_order == "first":
        retrieved_docs = [doc_anchor, doc_core, doc_filler]
    elif doc_order == "last":
        retrieved_docs = [doc_core, doc_filler, doc_anchor]
    else:
        retrieved_docs = [doc_core, doc_anchor, doc_filler]

    retrieved_ids = [d.doc_id for d in retrieved_docs]
    docs_block = _format_retrieved_docs(retrieved_docs)

    prompt_text = (
        f"{_RETRIEVAL_HEADER}\n"
        f"{docs_block}\n\n"
        f"{scenario}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    components = {
        "retrieval_header": _RETRIEVAL_HEADER,
        "retrieved_docs": docs_block,
        "scenario": scenario,
        "evidence": evidence_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
        "retrieved_doc_ids": ",".join(retrieved_ids),
        "anchor_doc_id": doc_anchor.doc_id,
        "anchor_doc_role": doc_anchor.role,
    }

    anchor_string: str | None = None
    anchor_span: list[int] | None = None
    anchor_value: int | None = None

    base_cond = condition.split("_order_")[0].split("_nodiscl")[0].split("_authority")[0]
    if base_cond != "control":
        anchor_value = doc_anchor.anchor_value
        if anchor_value is not None:
            anchor_string = str(anchor_value)
            start = prompt_text.find(anchor_string, len(_RETRIEVAL_HEADER))
            if start >= 0:
                anchor_span = [start, start + len(anchor_string)]

    provenance = {}
    if ablation_type:
        provenance["ablation_type"] = ablation_type

    return PromptView(
        item_id=spec.item_id,
        suite=spec.suite,
        domain=spec.domain,
        condition=condition,
        prompt_text=prompt_text,
        prompt_components=components,
        anchor_string=anchor_string,
        anchor_span=anchor_span,
        anchor_relevance=relevance,
        anchor_value=anchor_value,
        provenance=provenance,
    )


# ── P2 RAG realism ablation ───────────────────────────────────────────

_DOC_DISTRACTOR_TEMPLATES: dict[str, list[str]] = {
    "pricing_wtp": [
        "Adjacent Market Note: A separate retailer in a tangentially related "
        "category reported quarterly inventory churn of 14%, but this segment "
        "operates on a different value chain and the figure is not comparable.",
        "Tangential Pricing Memo: A different product family within the parent "
        "company maintained a 7% margin variance year-on-year, included here "
        "for retrieval-context completeness only.",
    ],
    "operations_time": [
        "Adjacent Facility Note: A sister plant operating in a different region "
        "logged a 2-hour shift handover overlap; this is a procedural metric "
        "with no bearing on throughput efficiency.",
        "Tangential Operations Memo: Equipment maintenance audits in a "
        "neighbouring facility recorded a 4-day window between major service "
        "events, not relevant to throughput-efficiency assessment.",
    ],
    "transportation_logistics": [
        "Adjacent Network Note: A regional warehouse reported 11 driver "
        "rotations per cycle; this is a labour-scheduling metric not tied "
        "to on-time delivery reliability.",
        "Tangential Logistics Memo: Fleet-level depreciation accounting "
        "showed a 6-year asset replacement window, included for context "
        "only and not relevant to reliability scoring.",
    ],
    "resource_consumption": [
        "Adjacent Utility Note: A separate office facility within the same "
        "campus reported a 3-month water-bill cycle, not comparable to the "
        "industrial consumption metric under review.",
        "Tangential Sustainability Memo: Carbon-offset programme enrolment "
        "stood at 12 employees this quarter, an HR metric unrelated to "
        "resource-consumption efficiency.",
    ],
    "market_demographics": [
        "Adjacent Brand Note: A loyalty-programme adjacent to the assessed "
        "product reported 4% sign-up uplift, but the cohort and channel "
        "differ from the adoption study under review.",
        "Tangential Demographics Memo: Internal HR demographics show 19% "
        "female representation in the regional sales force, not relevant "
        "to product-adoption indexing.",
    ],
    "legal_policy": [
        "Adjacent Compliance Note: A subsidiary entity completed 8 "
        "voluntary disclosures last quarter; these were procedural and "
        "do not bear on the assessed compliance posture.",
        "Tangential Policy Memo: A 6-page revision was filed to the "
        "internal procurement code; this is unrelated to the regulatory "
        "compliance dimension under review.",
    ],
    # Medical pilot domains (defensive fallback to generic text).
    "clinical_readmission_risk": [
        "Adjacent Clinical Note: Ward-level staffing rotations averaged "
        "9 nurses per shift this quarter; not relevant to per-patient "
        "readmission risk.",
        "Tangential Operations Memo: Pharmacy turnaround time held at "
        "21 minutes for non-urgent prescriptions, an operational metric "
        "unrelated to readmission scoring.",
    ],
    "medication_dosage_adjustment": [
        "Adjacent Pharmacy Note: Bulk-order lead time for ancillary "
        "supplies averaged 5 days this period; not relevant to dosage "
        "adjustment decisions.",
        "Tangential Operations Memo: Cold-chain logging compliance reached "
        "98% for non-controlled medications, an operational metric "
        "unrelated to dosage adjustment.",
    ],
    "diagnostic_confidence": [
        "Adjacent Imaging Note: PACS retrieval latency averaged 6 seconds "
        "on the imaging workstation this quarter; not relevant to "
        "diagnostic-confidence scoring.",
        "Tangential Workflow Memo: Imaging room turnover averaged 14 "
        "minutes, an operational metric unrelated to diagnostic "
        "confidence assessment.",
    ],
}


def _build_distractor_docs(domain: str, item_id: str) -> list[RAGDoc]:
    pool = _DOC_DISTRACTOR_TEMPLATES.get(domain) or _DOC_DISTRACTOR_TEMPLATES["pricing_wtp"]
    out: list[RAGDoc] = []
    for i, body in enumerate(pool):
        out.append(RAGDoc(
            doc_id=f"{item_id}_distractor_{i+1}",
            domain=domain,
            doc_type="distractor",
            role="distractor",
            relevance="none",
            text=body,
        ))
    return out


def _format_retrieved_docs_with_scores(
    docs: list[RAGDoc], scores: list[float],
) -> str:
    blocks = []
    for i, (doc, score) in enumerate(zip(docs, scores)):
        blocks.append(f"[Document {i+1} | relevance={score:.2f}]\n{doc.text}")
    return "\n\n".join(blocks)


def _build_realism_prompt(
    spec: ItemSpec,
    condition: str,
    relevance: str,
    anchor_doc: RAGDoc,
    rank: int,
    n_distract: int,
    with_relevance_scores: bool,
) -> PromptView:
    """Render a single P2 realism PromptView.

    Document ordering rules:
      total docs = 1 (core) + 1 (filler) + n_distract (distractors) + 1 (anchor)
      rank: 1-indexed position of the anchor doc in the retrieved list.
    """
    scenario, question, _, _ = resolve_templates(spec)
    evidence_block = format_evidence(spec.evidence_structured)

    domain = spec.domain
    tidx = int(spec.template_family.split("_")[-1])
    core_text = _DOC_CORE_TEMPLATES[domain][tidx % len(_DOC_CORE_TEMPLATES[domain])]
    filler_text = _DOC_FILLER_TEMPLATES[domain][tidx % len(_DOC_FILLER_TEMPLATES[domain])]
    doc_core = RAGDoc(
        doc_id=f"{spec.item_id}_core",
        domain=domain, doc_type="core", role="core",
        relevance="none", text=core_text,
    )
    doc_filler = RAGDoc(
        doc_id=f"{spec.item_id}_filler",
        domain=domain, doc_type="filler", role="filler",
        relevance="none", text=filler_text,
    )
    distractors = _build_distractor_docs(domain, spec.item_id)[:n_distract]

    non_anchor = [doc_core, doc_filler] + list(distractors)
    rank_clamped = max(1, min(rank, len(non_anchor) + 1))
    ordered: list[RAGDoc] = list(non_anchor)
    ordered.insert(rank_clamped - 1, anchor_doc)

    if with_relevance_scores:
        # Decreasing pseudo-relevance scores by rank position (anchor is
        # NOT artificially elevated; this mimics retrievers that surface
        # the anchored doc despite uncertain relevance).
        scores = [round(1.0 - 0.13 * i, 2) for i in range(len(ordered))]
        docs_block = _format_retrieved_docs_with_scores(ordered, scores)
    else:
        docs_block = _format_retrieved_docs(ordered)

    prompt_text = (
        f"{_RETRIEVAL_HEADER}\n"
        f"{docs_block}\n\n"
        f"{scenario}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        f"{question}\n{ANSWER_FORMAT_INSTRUCTION}"
    )

    components = {
        "retrieval_header": _RETRIEVAL_HEADER,
        "retrieved_docs": docs_block,
        "scenario": scenario,
        "evidence": evidence_block,
        "question": question,
        "answer_format": ANSWER_FORMAT_INSTRUCTION,
        "retrieved_doc_ids": ",".join(d.doc_id for d in ordered),
        "anchor_doc_id": anchor_doc.doc_id,
        "anchor_rank": rank_clamped,
        "n_distractors": n_distract,
        "with_relevance_scores": with_relevance_scores,
    }

    anchor_string: str | None = None
    anchor_span: list[int] | None = None
    anchor_value = anchor_doc.anchor_value
    if anchor_value is not None:
        anchor_string = str(anchor_value)
        start = prompt_text.find(anchor_string, len(_RETRIEVAL_HEADER))
        if start >= 0:
            anchor_span = [start, start + len(anchor_string)]

    return PromptView(
        item_id=spec.item_id,
        suite=spec.suite,
        domain=spec.domain,
        condition=condition,
        prompt_text=prompt_text,
        prompt_components=components,
        anchor_string=anchor_string,
        anchor_span=anchor_span,
        anchor_relevance=relevance,
        anchor_value=anchor_value,
        provenance={
            "ablation_type": "realism",
            "anchor_rank": rank_clamped,
            "n_distractors": n_distract,
            "with_relevance_scores": with_relevance_scores,
        },
    )


def build_realism_promptviews(spec: ItemSpec) -> list[PromptView]:
    """P2 RAG realism: 6 new conditions per item using the existing
    plausible/irrelevant anchor docs but varying retrieval realism.

    Conditions (per relevance type in {plausible, irrelevant}):
      *_rank1            anchor doc at rank 1, no distractors
      *_rank5            anchor doc at rank 5 (bottom), 2 distractors
      *_rank5_distract   anchor doc at rank 5, 2 distractors, +relevance scores
    """
    if spec.suite != "rag":
        return []
    corpus = build_item_corpus(spec)
    views: list[PromptView] = []
    for rel in ("plausible", "irrelevant"):
        for direction in ("low", "high"):
            anchor_doc = corpus[f"{rel}_{direction}"]
            # rank1, 0 distractors, no scores
            views.append(_build_realism_prompt(
                spec, f"{rel}_{direction}_rank1", rel,
                anchor_doc, rank=1, n_distract=0,
                with_relevance_scores=False,
            ))
            # rank5, 2 distractors, no scores (pure rank+distractor effect)
            views.append(_build_realism_prompt(
                spec, f"{rel}_{direction}_rank5", rel,
                anchor_doc, rank=5, n_distract=2,
                with_relevance_scores=False,
            ))
            # rank5, 2 distractors, +relevance scores
            views.append(_build_realism_prompt(
                spec, f"{rel}_{direction}_rank5_distract", rel,
                anchor_doc, rank=5, n_distract=2,
                with_relevance_scores=True,
            ))
    return views


def build_intensity_promptviews(spec: ItemSpec) -> list[PromptView]:
    """P1 cross-pathway intensity: render the 4 mild/strong conditions
    on top of an existing RAG itemspec, reusing the standard 3-doc layout
    with the intensity-flavoured anchor-slot document in the middle.

    Conditions emitted: plausible_mild_low/high, plausible_strong_low/high.
    """
    if spec.suite != "rag":
        return []
    corpus = build_item_corpus(spec)
    pidx = getattr(spec, "anchor_phrasing_idx", 0)
    new_corpus = dict(corpus)
    views: list[PromptView] = []
    for intensity in ("plausible_mild", "plausible_strong"):
        for direction in ("low", "high"):
            anchor_val = spec.anchors[direction]
            cond = f"{intensity}_{direction}"
            doc = _build_anchor_slot_intensity(
                spec.item_id, spec.domain, direction, anchor_val,
                intensity, pidx,
            )
            new_corpus[cond] = doc
            views.append(_build_prompt(
                spec, cond, intensity, new_corpus,
                doc_key=cond, doc_order="middle",
                ablation_type=intensity,
            ))
    return views


def render_rag(spec: ItemSpec) -> list[PromptView]:
    """Render base 5 + ablation conditions for a RAG item.

    Base: control, irrelevant_low/high, plausible_low/high (anchor in position 2).
    Ablations: order_first/last, no-disclaimer, authority.
    """
    corpus = build_item_corpus(spec)
    views: list[PromptView] = []

    for cond, rel, _ in CONDITIONS:
        views.append(_build_prompt(spec, cond, rel, corpus))

    for direction in ("low", "high"):
        for rel_type in ("irrelevant", "plausible"):
            base_cond = f"{rel_type}_{direction}"
            for order, order_label in (("first", "order_first"), ("last", "order_last")):
                abl_cond = f"{base_cond}_{order_label}"
                views.append(_build_prompt(
                    spec, abl_cond, rel_type, corpus,
                    doc_key=base_cond, doc_order=order,
                    ablation_type=order_label,
                ))

        views.append(_build_prompt(
            spec, f"irrelevant_{direction}_nodiscl", "irrelevant", corpus,
            doc_key=f"irrelevant_{direction}_nodiscl",
            ablation_type="nodiscl",
        ))
        views.append(_build_prompt(
            spec, f"plausible_{direction}_authority", "authority", corpus,
            doc_key=f"plausible_{direction}_authority",
            ablation_type="authority",
        ))

    return views
