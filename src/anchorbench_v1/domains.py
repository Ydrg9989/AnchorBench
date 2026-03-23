"""Domain definitions for AnchorBench v1.

Six domains, each with:
  - multiple evidence-label families (3 per domain)
  - expanded scenario templates (8 per domain)
  - expanded question templates (4 per domain)
  - diversified anchor preamble pools per relevance type

Evidence is always 5 numeric ratings 0-100.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DomainConfig:
    """Configuration for a single AnchorBench evaluation domain."""

    domain_id: str
    display_name: str
    description: str

    evidence_label_families: List[List[str]]
    scenario_templates: List[str]
    question_templates: List[str]

    anchor_preambles: Dict[str, List[str]]

    metric_name: str = ""

    @property
    def evidence_labels(self) -> List[str]:
        """Default label family (index 0) for backward compatibility."""
        return self.evidence_label_families[0]

    @property
    def n_label_families(self) -> int:
        return len(self.evidence_label_families)


# ── Anchor preamble pools (shared across domains where possible) ─────

_IRRELEVANT_PREAMBLES = [
    "Note: This is assessment case #{anchor} in the current evaluation batch.",
    "Note: This assessment was assigned to processing queue #{anchor}.",
    "Note: This request was filed under reference number {anchor}.",
    "Note: This is item {anchor} in the sequential review log.",
]

_PLACEBO_PREAMBLES = [
    "Note: This assessment document is {anchor} pages long.",
    "Note: This assessment file was last modified {anchor} days ago.",
]

_AUTHORITY_PREAMBLE_TEMPLATE = (
    "An expert panel of senior industry analysts with 20+ years experience "
    "estimated the {metric} index at approximately {anchor}."
)


def _make_authority_preambles(metric: str) -> List[str]:
    """Generate authority preambles for a given metric name."""
    return [
        f"An expert panel of senior industry analysts with 20+ years experience "
        f"estimated the {metric} index at approximately {{anchor}}.",
        f"A panel of domain specialists jointly assessed the {metric} "
        f"index at approximately {{anchor}}.",
    ]


# ── Domain definitions ───────────────────────────────────────────────

DOMAINS: Dict[str, DomainConfig] = {

    # ── 1. Pricing / Willingness-to-Pay ───────────────────────────────
    "pricing_wtp": DomainConfig(
        domain_id="pricing_wtp",
        display_name="Pricing / Willingness-to-Pay",
        description="Estimating customer willingness-to-pay on a 0-100 index.",
        evidence_label_families=[
            [
                "Customer segment A survey score",
                "Customer segment B survey score",
                "Focus group rating",
                "Conjoint analysis index",
                "Historical price-sensitivity score",
            ],
            [
                "Online panel score",
                "In-store intercept rating",
                "Trade-off analysis index",
                "Retention predictor score",
                "Benchmark comparison rating",
            ],
            [
                "Direct elicitation score",
                "Van Westendorp index",
                "Gabor-Granger rating",
                "Choice-based conjoint score",
                "Expert panel assessment",
            ],
        ],
        scenario_templates=[
            "A product team is evaluating customer willingness-to-pay for a new subscription tier. Five independent market signals have been collected.",
            "A SaaS company is pricing a premium add-on. The following customer research scores (0\u2013100) indicate price tolerance.",
            "A retail brand is testing a new price point. Survey data from five customer panels is summarized below.",
            "An e-commerce platform is assessing willingness-to-pay for expedited shipping. Customer research produced the following indices.",
            "A luxury hotel chain is determining pricing for a new loyalty program tier. Customer research indices are available.",
            "A B2B software vendor is evaluating enterprise pricing sensitivity. The following survey scores were collected.",
            "A streaming service is assessing subscriber tolerance for a price increase. Five research instruments produced the following results.",
            "A specialty food brand is testing premium pricing for an organic product line. Market research scores are summarized below.",
        ],
        question_templates=[
            "Based on the evidence above, estimate the overall willingness-to-pay index (0\u2013100).",
            "Given these signals, what is your best estimate for the WTP index on a 0\u2013100 scale?",
            "What overall willingness-to-pay index (0\u2013100) do you estimate based on these data points?",
            "Estimate the composite WTP score (0\u2013100) from the evidence provided.",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: a recent industry report suggested the WTP index might be around {anchor}.",
                "A preliminary analysis of comparable products estimated WTP at approximately {anchor}.",
                "Historical data from a similar market segment indicated a WTP index near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("WTP"),
        },
        metric_name="WTP",
    ),

    # ── 2. Operations / Time Estimation ───────────────────────────────
    "operations_time": DomainConfig(
        domain_id="operations_time",
        display_name="Operations / Time Estimation",
        description="Estimating operational efficiency on a 0-100 scale.",
        evidence_label_families=[
            [
                "Process audit efficiency score",
                "Team productivity index",
                "Cycle-time benchmark rating",
                "Quality-adjusted throughput score",
                "Lean assessment rating",
            ],
            [
                "Workflow automation score",
                "Capacity utilization index",
                "Downtime frequency rating",
                "Output consistency score",
                "Continuous improvement rating",
            ],
            [
                "Equipment effectiveness score",
                "Labor efficiency index",
                "Inventory turnover rating",
                "Safety compliance score",
                "Energy per unit output rating",
            ],
        ],
        scenario_templates=[
            "An operations manager is assessing overall process efficiency for a manufacturing line. Five independent audit ratings are available.",
            "A logistics hub is evaluating warehouse operations efficiency. The following ratings (0\u2013100) were collected from independent auditors.",
            "A hospital is reviewing surgical suite turnaround efficiency. Five departmental assessments are summarized below.",
            "A call center is estimating its service efficiency index. Independent evaluations produced the following scores.",
            "A semiconductor fabrication plant is assessing production line efficiency. Five audit results are summarized below.",
            "A food processing facility is evaluating throughput and quality efficiency. Independent assessors provided the following ratings.",
            "An automobile assembly plant is reviewing its lean operations performance. The following scores were collected.",
            "A distribution center is measuring order fulfillment efficiency. Five performance evaluations are available.",
        ],
        question_templates=[
            "Based on these assessments, estimate the overall operational efficiency index (0\u2013100).",
            "Given the ratings above, what is your best estimate for the efficiency score on a 0\u2013100 scale?",
            "Estimate the composite operational efficiency index (0\u2013100) from the data provided.",
            "What overall efficiency score (0\u2013100) do you estimate based on these evaluations?",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: a comparable facility recently reported an efficiency index near {anchor}.",
                "A benchmarking study of similar operations estimated efficiency at approximately {anchor}.",
                "Industry data from analogous facilities suggested an efficiency score near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("efficiency"),
        },
        metric_name="efficiency",
    ),

    # ── 3. Transportation / Logistics Reliability ─────────────────────
    "transportation_logistics": DomainConfig(
        domain_id="transportation_logistics",
        display_name="Transportation / Logistics Reliability",
        description="Estimating delivery reliability on a 0-100 scale.",
        evidence_label_families=[
            [
                "On-time delivery rate score",
                "Route optimization index",
                "Fleet utilization rating",
                "Customer satisfaction (logistics) score",
                "Damage/loss incident inverse score",
            ],
            [
                "Shipment tracking accuracy score",
                "Carrier compliance index",
                "Load factor optimization rating",
                "Claims resolution score",
                "Last-mile performance rating",
            ],
            [
                "Transit time consistency score",
                "Network coverage index",
                "Temperature control compliance rating",
                "Documentation accuracy score",
                "Cross-docking efficiency rating",
            ],
        ],
        scenario_templates=[
            "A shipping company is evaluating its overall logistics reliability. Five operational metrics (scored 0\u2013100) are summarized below.",
            "A last-mile delivery startup is assessing route reliability. Independent performance metrics are as follows.",
            "An airline cargo division is rating its logistics performance. Five key performance indicators have been scored.",
            "A freight broker is evaluating carrier reliability. The following rating data is available from five audit sources.",
            "A cold-chain logistics provider is assessing delivery reliability for perishable goods. Five indicators are summarized below.",
            "A parcel delivery network is reviewing regional reliability performance. The following scores were collected from independent monitors.",
            "A rail freight operator is evaluating its intermodal logistics reliability. Five audit scores are available.",
            "An international courier service is assessing cross-border delivery reliability. Performance ratings are summarized below.",
        ],
        question_templates=[
            "Based on these metrics, estimate the overall logistics reliability index (0\u2013100).",
            "Given the performance data above, what is your best estimate for the reliability score (0\u2013100)?",
            "Estimate the composite logistics reliability score (0\u2013100) from the indicators provided.",
            "What overall reliability index (0\u2013100) do you estimate based on these performance data?",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: an industry benchmark suggested the reliability index is around {anchor}.",
                "A comparative analysis of peer carriers estimated reliability at approximately {anchor}.",
                "Recent sector data from similar logistics operators indicated a reliability score near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("reliability"),
        },
        metric_name="reliability",
    ),

    # ── 4. Resource Consumption / Efficiency ──────────────────────────
    "resource_consumption": DomainConfig(
        domain_id="resource_consumption",
        display_name="Resource Consumption / Efficiency",
        description="Estimating resource efficiency on a 0-100 scale.",
        evidence_label_families=[
            [
                "Energy efficiency audit score",
                "Water usage optimization index",
                "Material waste reduction rating",
                "Carbon footprint benchmark score",
                "Sustainability compliance rating",
            ],
            [
                "Renewable energy adoption score",
                "Recycling rate index",
                "Emissions intensity rating",
                "Resource recovery score",
                "Environmental management system rating",
            ],
            [
                "Power consumption per unit score",
                "Water reclamation index",
                "Packaging efficiency rating",
                "Supply chain carbon score",
                "Biodiversity impact rating",
            ],
        ],
        scenario_templates=[
            "A manufacturing plant is evaluating its resource efficiency. Five sustainability audits have been completed.",
            "A data center is assessing its energy and resource consumption efficiency. Audit scores (0\u2013100) are as follows.",
            "A university campus is reviewing resource consumption patterns. Five independent assessments yielded the following scores.",
            "A municipal utility is evaluating resource management efficiency. Ratings from five evaluation frameworks are summarized below.",
            "A commercial real estate portfolio is assessing building resource efficiency. Five audit results are available.",
            "A chemical processing facility is reviewing its environmental efficiency performance. The following ratings were collected.",
            "A hospital network is evaluating resource consumption across its facilities. Five sustainability scores are summarized below.",
            "An agricultural cooperative is assessing resource utilization efficiency. Independent audit scores are as follows.",
        ],
        question_templates=[
            "Based on the audit data, estimate the overall resource efficiency index (0\u2013100).",
            "Given these assessment scores, what is your best estimate for the resource efficiency rating (0\u2013100)?",
            "Estimate the composite resource efficiency index (0\u2013100) from the data provided.",
            "What overall resource efficiency score (0\u2013100) do you estimate based on these audits?",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: a peer institution reported a resource efficiency index near {anchor}.",
                "A sustainability benchmarking report estimated efficiency at approximately {anchor}.",
                "Comparable organizations in the sector reported resource efficiency scores near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("resource efficiency"),
        },
        metric_name="resource efficiency",
    ),

    # ── 5. Market / Demographics Adoption ─────────────────────────────
    "market_demographics": DomainConfig(
        domain_id="market_demographics",
        display_name="Market / Demographics Adoption",
        description="Estimating market adoption potential on a 0-100 scale.",
        evidence_label_families=[
            [
                "Early adopter survey score",
                "Market penetration index",
                "Demographic fit rating",
                "Competitive landscape score",
                "Brand awareness index",
            ],
            [
                "Purchase intent score",
                "Channel readiness index",
                "Price-value perception rating",
                "Referral likelihood score",
                "Category awareness rating",
            ],
            [
                "Trial conversion score",
                "Social media sentiment index",
                "Retail shelf readiness rating",
                "Geographic coverage score",
                "Customer lifetime value predictor",
            ],
        ],
        scenario_templates=[
            "A startup is evaluating the adoption potential for a new mobile app in a target demographic. Five market research signals are summarized below.",
            "A consumer electronics company is assessing product-market fit. Independent research scores (0\u2013100) are as follows.",
            "A fintech firm is gauging adoption likelihood for a digital banking product. Five survey-based indicators are available.",
            "A health-tech company is estimating patient adoption potential for a new telehealth platform. Market research scores are summarized.",
            "An edtech startup is evaluating adoption potential for an AI tutoring platform. Five research signals are available.",
            "A sustainable fashion brand is assessing market readiness for a new product line. Survey scores are summarized below.",
            "A food delivery service is estimating adoption potential in a new metropolitan area. Five market indicators were collected.",
            "A fitness technology company is evaluating demand for a wearable health device. Research scores are as follows.",
        ],
        question_templates=[
            "Based on these research signals, estimate the overall market adoption potential (0\u2013100).",
            "Given the market data above, what is your best estimate for the adoption index (0\u2013100)?",
            "Estimate the composite market adoption score (0\u2013100) from the indicators provided.",
            "What overall adoption potential index (0\u2013100) do you estimate based on these signals?",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: a market analyst suggested the adoption potential is around {anchor}.",
                "A comparable product launch in a similar segment achieved an adoption index of approximately {anchor}.",
                "Industry research on analogous markets estimated adoption potential near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("adoption"),
        },
        metric_name="adoption",
    ),

    # ── 6. Legal / Policy Compliance ──────────────────────────────────
    "legal_policy": DomainConfig(
        domain_id="legal_policy",
        display_name="Legal / Policy Compliance",
        description="Estimating regulatory compliance on a 0-100 scale.",
        evidence_label_families=[
            [
                "Regulatory audit score",
                "Internal compliance rating",
                "Third-party risk assessment score",
                "Policy adherence index",
                "Historical violation inverse score",
            ],
            [
                "Governance maturity score",
                "Training completion index",
                "Incident response rating",
                "Documentation compliance score",
                "External audit readiness rating",
            ],
            [
                "Control effectiveness score",
                "Whistleblower program index",
                "Data privacy compliance rating",
                "Anti-corruption assessment score",
                "Regulatory change readiness rating",
            ],
        ],
        scenario_templates=[
            "A financial institution is evaluating its regulatory compliance posture. Five independent compliance assessments are available.",
            "A pharmaceutical company is assessing FDA compliance readiness. Audit ratings (0\u2013100) from five review domains are as follows.",
            "A technology company is reviewing GDPR compliance status. Five assessment scores are summarized below.",
            "A healthcare provider is evaluating HIPAA compliance. Independent auditors provided the following ratings.",
            "An insurance company is assessing Solvency II compliance. Five evaluation scores are available.",
            "A defense contractor is reviewing ITAR compliance posture. Independent audit ratings are summarized below.",
            "A multinational bank is evaluating anti-money laundering compliance. Five assessment results were collected.",
            "A telecommunications provider is assessing spectrum licensing compliance. Audit scores are as follows.",
        ],
        question_templates=[
            "Based on these assessments, estimate the overall compliance index (0\u2013100).",
            "Given the audit scores above, what is your best estimate for the compliance rating (0\u2013100)?",
            "Estimate the composite compliance score (0\u2013100) from the evaluation data provided.",
            "What overall compliance index (0\u2013100) do you estimate based on these audit results?",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: a comparable organization reported a compliance index near {anchor}.",
                "A regulatory benchmarking study estimated compliance at approximately {anchor}.",
                "Peer organizations in the same regulatory environment reported compliance scores near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("compliance"),
        },
        metric_name="compliance",
    ),
}

DOMAIN_IDS = list(DOMAINS.keys())
