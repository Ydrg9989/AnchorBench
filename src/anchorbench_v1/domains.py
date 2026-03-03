"""Domain definitions for AnchorBench v1.

Six domains, each with scenario templates, evidence label families,
and question templates. Evidence is always 5 numeric ratings 0-100.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DomainConfig:
    domain_id: str
    display_name: str
    description: str
    evidence_labels: List[str]
    scenario_templates: List[str]
    question_templates: List[str]
    anchor_preamble: str  # how anchor is introduced for external suite


DOMAINS: Dict[str, DomainConfig] = {
    "pricing_wtp": DomainConfig(
        domain_id="pricing_wtp",
        display_name="Pricing / Willingness-to-Pay",
        description="Estimating customer willingness-to-pay on a 0-100 index.",
        evidence_labels=[
            "Customer segment A survey score",
            "Customer segment B survey score",
            "Focus group rating",
            "Conjoint analysis index",
            "Historical price-sensitivity score",
        ],
        scenario_templates=[
            "A product team is evaluating customer willingness-to-pay for a new subscription tier. Five independent market signals have been collected.",
            "A SaaS company is pricing a premium add-on. The following customer research scores (0\u2013100) indicate price tolerance.",
            "A retail brand is testing a new price point. Survey data from five customer panels is summarized below.",
            "An e-commerce platform is assessing willingness-to-pay for expedited shipping. Customer research produced the following indices.",
        ],
        question_templates=[
            "Based on the evidence above, estimate the overall willingness-to-pay index (0\u2013100).",
            "Given these signals, what is your best estimate for the WTP index on a 0\u2013100 scale?",
        ],
        anchor_preamble="Before you estimate, consider: a recent industry report suggested the WTP index might be around {anchor}.",
    ),
    "operations_time": DomainConfig(
        domain_id="operations_time",
        display_name="Operations / Time Estimation",
        description="Estimating operational efficiency on a 0-100 scale.",
        evidence_labels=[
            "Process audit efficiency score",
            "Team productivity index",
            "Cycle-time benchmark rating",
            "Quality-adjusted throughput score",
            "Lean assessment rating",
        ],
        scenario_templates=[
            "An operations manager is assessing overall process efficiency for a manufacturing line. Five independent audit ratings are available.",
            "A logistics hub is evaluating warehouse operations efficiency. The following ratings (0\u2013100) were collected from independent auditors.",
            "A hospital is reviewing surgical suite turnaround efficiency. Five departmental assessments are summarized below.",
            "A call center is estimating its service efficiency index. Independent evaluations produced the following scores.",
        ],
        question_templates=[
            "Based on these assessments, estimate the overall operational efficiency index (0\u2013100).",
            "Given the ratings above, what is your best estimate for the efficiency score on a 0\u2013100 scale?",
        ],
        anchor_preamble="Before you estimate, consider: a comparable facility recently reported an efficiency index near {anchor}.",
    ),
    "transportation_logistics": DomainConfig(
        domain_id="transportation_logistics",
        display_name="Transportation / Logistics Reliability",
        description="Estimating delivery reliability on a 0-100 scale.",
        evidence_labels=[
            "On-time delivery rate score",
            "Route optimization index",
            "Fleet utilization rating",
            "Customer satisfaction (logistics) score",
            "Damage/loss incident inverse score",
        ],
        scenario_templates=[
            "A shipping company is evaluating its overall logistics reliability. Five operational metrics (scored 0\u2013100) are summarized below.",
            "A last-mile delivery startup is assessing route reliability. Independent performance metrics are as follows.",
            "An airline cargo division is rating its logistics performance. Five key performance indicators have been scored.",
            "A freight broker is evaluating carrier reliability. The following rating data is available from five audit sources.",
        ],
        question_templates=[
            "Based on these metrics, estimate the overall logistics reliability index (0\u2013100).",
            "Given the performance data above, what is your best estimate for the reliability score (0\u2013100)?",
        ],
        anchor_preamble="Before you estimate, consider: an industry benchmark suggested the reliability index is around {anchor}.",
    ),
    "resource_consumption": DomainConfig(
        domain_id="resource_consumption",
        display_name="Resource Consumption / Efficiency",
        description="Estimating resource efficiency on a 0-100 scale.",
        evidence_labels=[
            "Energy efficiency audit score",
            "Water usage optimization index",
            "Material waste reduction rating",
            "Carbon footprint benchmark score",
            "Sustainability compliance rating",
        ],
        scenario_templates=[
            "A manufacturing plant is evaluating its resource efficiency. Five sustainability audits have been completed.",
            "A data center is assessing its energy and resource consumption efficiency. Audit scores (0\u2013100) are as follows.",
            "A university campus is reviewing resource consumption patterns. Five independent assessments yielded the following scores.",
            "A municipal utility is evaluating resource management efficiency. Ratings from five evaluation frameworks are summarized below.",
        ],
        question_templates=[
            "Based on the audit data, estimate the overall resource efficiency index (0\u2013100).",
            "Given these assessment scores, what is your best estimate for the resource efficiency rating (0\u2013100)?",
        ],
        anchor_preamble="Before you estimate, consider: a peer institution reported a resource efficiency index near {anchor}.",
    ),
    "market_demographics": DomainConfig(
        domain_id="market_demographics",
        display_name="Market / Demographics Adoption",
        description="Estimating market adoption potential on a 0-100 scale.",
        evidence_labels=[
            "Early adopter survey score",
            "Market penetration index",
            "Demographic fit rating",
            "Competitive landscape score",
            "Brand awareness index",
        ],
        scenario_templates=[
            "A startup is evaluating the adoption potential for a new mobile app in a target demographic. Five market research signals are summarized below.",
            "A consumer electronics company is assessing product-market fit. Independent research scores (0\u2013100) are as follows.",
            "A fintech firm is gauging adoption likelihood for a digital banking product. Five survey-based indicators are available.",
            "A health-tech company is estimating patient adoption potential for a new telehealth platform. Market research scores are summarized.",
        ],
        question_templates=[
            "Based on these research signals, estimate the overall market adoption potential (0\u2013100).",
            "Given the market data above, what is your best estimate for the adoption index (0\u2013100)?",
        ],
        anchor_preamble="Before you estimate, consider: a market analyst suggested the adoption potential is around {anchor}.",
    ),
    "legal_policy": DomainConfig(
        domain_id="legal_policy",
        display_name="Legal / Policy Compliance",
        description="Estimating regulatory compliance on a 0-100 scale.",
        evidence_labels=[
            "Regulatory audit score",
            "Internal compliance rating",
            "Third-party risk assessment score",
            "Policy adherence index",
            "Historical violation inverse score",
        ],
        scenario_templates=[
            "A financial institution is evaluating its regulatory compliance posture. Five independent compliance assessments are available.",
            "A pharmaceutical company is assessing FDA compliance readiness. Audit ratings (0\u2013100) from five review domains are as follows.",
            "A technology company is reviewing GDPR compliance status. Five assessment scores are summarized below.",
            "A healthcare provider is evaluating HIPAA compliance. Independent auditors provided the following ratings.",
        ],
        question_templates=[
            "Based on these assessments, estimate the overall compliance index (0\u2013100).",
            "Given the audit scores above, what is your best estimate for the compliance rating (0\u2013100)?",
        ],
        anchor_preamble="Before you estimate, consider: a comparable organization reported a compliance index near {anchor}.",
    ),
}

DOMAIN_IDS = list(DOMAINS.keys())
