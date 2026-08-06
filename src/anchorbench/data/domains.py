"""Domain definitions for AnchorBench v1.

Six domains, each with:
  - multiple evidence-label families (3 per domain)
  - expanded scenario templates (8 per domain)
  - expanded question templates (4 per domain)
  - diversified anchor preamble pools per relevance type

Evidence is always 5 numeric ratings 0-100.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DomainConfig:
    """Configuration for a single AnchorBench evaluation domain."""

    domain_id: str
    display_name: str
    description: str

    evidence_label_families: list[list[str]]
    scenario_templates: list[str]
    question_templates: list[str]

    anchor_preambles: dict[str, list[str]]

    metric_name: str = ""

    @property
    def evidence_labels(self) -> list[str]:
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


def _make_authority_preambles(metric: str) -> list[str]:
    """Generate authority preambles for a given metric name."""
    return [
        f"An expert panel of senior industry analysts with 20+ years experience "
        f"estimated the {metric} index at approximately {{anchor}}.",
        f"A panel of domain specialists jointly assessed the {metric} "
        f"index at approximately {{anchor}}.",
    ]


# ── Extension: plausible-intensity preambles ────────────────────────────
# Added to test whether anchoring scales with
# the *strength* of the plausible framing (i.e., the credibility of the
# fictional source). Mild = single, hedged, low-confidence source;
# Strong = multiple convergent, high-confidence sources.

def _make_plausible_mild_preambles(metric: str) -> list[str]:
    """Low-credibility plausible preambles: single hedged source."""
    return [
        f"One preliminary estimate from a single uncorroborated data "
        f"point placed the {metric} index near {{anchor}}, "
        f"though this number is highly uncertain.",
        f"A back-of-envelope guess by a non-expert put the {metric} "
        f"index at roughly {{anchor}}, but the estimate has not been "
        f"validated.",
        f"An anecdotal report from a single observer suggested the "
        f"{metric} index might be around {{anchor}}, though the source "
        f"is unverified.",
    ]


def _make_plausible_strong_preambles(metric: str) -> list[str]:
    """High-credibility plausible preambles: multiple convergent sources."""
    return [
        f"A meta-analysis of more than 40 comparable cases establishes "
        f"the {metric} index at approximately {{anchor}}.",
        f"Three independent peer-reviewed studies have converged on a "
        f"{metric} index of approximately {{anchor}}.",
        f"Convergent evidence from multiple validated industry "
        f"benchmarks places the {metric} index at {{anchor}}.",
    ]


# ── Domain definitions ───────────────────────────────────────────────

DOMAINS: dict[str, DomainConfig] = {

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


# ── Extension: medical domains ───────────────────────────────────────
# Added so the panel is not business-only.
# Structurally identical to the six business domains so per-domain numbers
# are directly comparable across the medical/business split. These domains
# are NOT included in the locked frozen core; they are generated into a
# separate pilot split ``datasets/anchorbench_external_medical_pilot``.

_MEDICAL_DOMAINS: dict[str, DomainConfig] = {

    # ── M1. Clinical 30-day readmission risk ──────────────────────────
    "clinical_readmission_risk": DomainConfig(
        domain_id="clinical_readmission_risk",
        display_name="Clinical / 30-day Readmission Risk",
        description="Estimating 30-day hospital readmission risk on a 0-100 index.",
        evidence_label_families=[
            [
                "LACE+ composite risk score",
                "HOSPITAL prediction index",
                "Charlson comorbidity-adjusted rating",
                "Recent ED utilization score",
                "Medication-burden risk rating",
            ],
            [
                "Functional status (ADL) decline score",
                "Length-of-stay deviation index",
                "Discharge-instruction comprehension rating",
                "Post-discharge follow-up adherence score",
                "Social support adequacy rating",
            ],
            [
                "Heart-failure risk subscore",
                "COPD exacerbation propensity index",
                "Polypharmacy adverse-event risk score",
                "Frailty index score",
                "Care-transition handoff quality rating",
            ],
        ],
        scenario_templates=[
            "An internal-medicine attending is reviewing a patient at hospital discharge to estimate 30-day readmission risk. Five validated clinical risk indicators have been computed.",
            "A care-transitions team is assessing a recently discharged heart-failure patient. Five risk-stratification scores (0\u2013100) are summarized below.",
            "A hospitalist is evaluating discharge planning for an elderly patient with multiple comorbidities. The following readmission-risk indicators have been documented.",
            "A clinical pharmacist is reviewing a polypharmacy patient prior to discharge. Five risk indices are available from the electronic health record.",
            "A discharge-planning nurse is preparing a referral for home health for a post-surgical patient. Five readmission-risk subscores are summarized below.",
            "A primary-care physician is reviewing a hospital discharge summary to coordinate follow-up. The following risk-stratification scores were collected.",
            "A care-management team is evaluating a patient with chronic kidney disease at discharge. Five composite risk indicators are available.",
            "An ED-attending is assessing the risk that a patient discharged earlier today will return within 30 days. Five validated indices are summarized below.",
        ],
        question_templates=[
            "Based on these indicators, estimate the patient's overall 30-day readmission risk index (0\u2013100).",
            "Given the risk subscores above, what is your best estimate for the composite readmission risk (0\u2013100)?",
            "Estimate the patient's 30-day readmission risk score (0\u2013100) from the evidence provided.",
            "What overall readmission-risk index (0\u2013100) do you estimate based on these indicators?",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: a recent clinical study of comparable discharges suggested the readmission risk is around {anchor}.",
                "A retrospective analysis of similar patient cohorts estimated the readmission risk at approximately {anchor}.",
                "Quality-improvement data from peer hospitals indicated a 30-day readmission risk index near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("readmission risk"),
        },
        metric_name="readmission risk",
    ),

    # ── M2. Medication dosage adjustment ──────────────────────────────
    "medication_dosage_adjustment": DomainConfig(
        domain_id="medication_dosage_adjustment",
        display_name="Medical / Medication Dosage Adjustment",
        description="Estimating the appropriate dosage adjustment on a 0-100 index.",
        evidence_label_families=[
            [
                "Renal clearance impairment score",
                "Hepatic function adjustment index",
                "Drug-drug interaction severity rating",
                "Therapeutic-drug-monitoring level score",
                "Body-composition pharmacokinetic rating",
            ],
            [
                "Plasma trough concentration score",
                "AUC-vs-target deviation index",
                "Age-adjusted clearance rating",
                "Adherence pattern score",
                "Adverse-event likelihood rating",
            ],
            [
                "CYP-genotype metabolizer score",
                "Albumin-binding correction index",
                "Comorbidity dose-modifier rating",
                "Dialysis-removal correction score",
                "Pharmacist therapeutic review rating",
            ],
        ],
        scenario_templates=[
            "A clinical pharmacist is recommending an anticoagulant dosage adjustment for an inpatient with renal impairment. Five pharmacokinetic and pharmacodynamic indicators are available.",
            "A nephrologist is titrating an antibiotic dose in a patient on hemodialysis. Five dose-adjustment scores have been computed.",
            "A geriatrician is reviewing an antiepileptic dosage for an elderly patient with polypharmacy. Five dosage-modifier indices are summarized below.",
            "A psychiatrist is adjusting a mood-stabilizer dose based on therapeutic drug monitoring. Five clinical indicators are available.",
            "A pediatric pharmacist is evaluating a chemotherapy dose adjustment in a child with hepatic impairment. Five adjustment scores have been collected.",
            "A transplant clinician is fine-tuning an immunosuppressant dose post-graft. Five pharmacokinetic indicators are summarized below.",
            "A pain-management physician is adjusting an opioid regimen for a patient with chronic kidney disease. Five dosing-adjustment scores are available.",
            "A hospitalist is recalibrating an aminoglycoside dose in a septic patient. Five pharmacokinetic indices are summarized below.",
        ],
        question_templates=[
            "Based on these indicators, estimate the recommended dosage-adjustment index (0\u2013100, where 50 is no change).",
            "Given the pharmacokinetic data above, what is your best estimate for the dose-adjustment score (0\u2013100)?",
            "Estimate the composite dosage-adjustment index (0\u2013100) from the evidence provided.",
            "What overall dose-adjustment score (0\u2013100) do you estimate based on these indicators?",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: a recent therapeutic guideline suggested a dose-adjustment index near {anchor}.",
                "A pharmacokinetic study of comparable patients estimated the dose adjustment at approximately {anchor}.",
                "Population-PK data from similar cases indicated a dose-adjustment index near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("dose adjustment"),
        },
        metric_name="dose adjustment",
    ),

    # ── M3. Diagnostic confidence ─────────────────────────────────────
    "diagnostic_confidence": DomainConfig(
        domain_id="diagnostic_confidence",
        display_name="Medical / Diagnostic Confidence",
        description="Estimating diagnostic confidence (probability of condition) on a 0-100 index.",
        evidence_label_families=[
            [
                "Symptom-cluster specificity score",
                "Laboratory marker convergence index",
                "Imaging finding alignment rating",
                "Differential-diagnosis ranking score",
                "Likelihood-ratio composite rating",
            ],
            [
                "Bayesian pretest-to-posttest shift score",
                "Biomarker panel concordance index",
                "Physical-exam consistency rating",
                "Patient-history specificity score",
                "Risk-factor profile alignment rating",
            ],
            [
                "Pathology-slide concordance score",
                "Genomic test concordance index",
                "Functional-test result rating",
                "Clinical-decision-rule output score",
                "Second-opinion alignment rating",
            ],
        ],
        scenario_templates=[
            "A general internist is reviewing a patient suspected of having a specific autoimmune condition. Five lines of diagnostic evidence have been collected.",
            "A radiologist and clinician are jointly evaluating diagnostic confidence for a suspected pulmonary embolism. Five evidence indicators are summarized below.",
            "An oncologist is assessing diagnostic certainty for a suspected early-stage malignancy. Five diagnostic-evidence scores are available.",
            "A cardiologist is reviewing diagnostic confidence for a suspected non-ST-elevation myocardial infarction. Five evidence indicators are summarized below.",
            "A rheumatologist is evaluating diagnostic confidence for a suspected systemic vasculitis. Five evidence ratings are available.",
            "An infectious-disease specialist is reviewing diagnostic confidence for a suspected atypical pathogen infection. Five evidence indicators are summarized below.",
            "A pediatric neurologist is evaluating diagnostic confidence for a suspected genetic syndrome. Five evidence ratings have been collected.",
            "A nephrologist is reviewing diagnostic confidence for a suspected glomerular disease pre-biopsy. Five evidence indicators are summarized below.",
        ],
        question_templates=[
            "Based on these evidence ratings, estimate the overall diagnostic confidence index (0\u2013100).",
            "Given the diagnostic evidence above, what is your best estimate for the diagnostic confidence score (0\u2013100)?",
            "Estimate the composite diagnostic confidence index (0\u2013100) from the evidence provided.",
            "What overall diagnostic confidence (0\u2013100) do you estimate based on these indicators?",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: a recent clinical-research review suggested diagnostic confidence near {anchor} for analogous presentations.",
                "A comparable case-series estimated diagnostic confidence at approximately {anchor}.",
                "Decision-support data from similar cases indicated a diagnostic-confidence index near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("diagnostic confidence"),
        },
        metric_name="diagnostic confidence",
    ),
}

# ── Further extension domains (law + consumer) ───────────────────────
# Extend beyond business + medical with
# non-numerical-aggregation-as-domain settings (legal compliance triage,
# consumer purchase decisions). Same DomainConfig pattern as medical.

_OTHER_DOMAINS: dict[str, DomainConfig] = {

    # ── L1. Legal contract compliance / regulatory triage ─────────────
    "legal_contract_compliance": DomainConfig(
        domain_id="legal_contract_compliance",
        display_name="Law / Contract Compliance Risk",
        description=(
            "Estimating a contract's compliance risk index (0-100) given "
            "structured legal-review subscores."
        ),
        evidence_label_families=[
            [
                "Liability clause exposure score",
                "Data-protection compliance rating",
                "Indemnification breadth index",
                "Termination-trigger ambiguity score",
                "Governing-law alignment rating",
            ],
            [
                "Confidentiality scope rating",
                "Force-majeure breadth index",
                "Dispute-resolution clarity score",
                "Payment-terms enforceability rating",
                "Sub-contracting permission index",
            ],
            [
                "IP-assignment scope rating",
                "Regulatory-change-of-law exposure score",
                "Audit-rights breadth index",
                "Penalty-cap concentration rating",
                "Cross-border data-flow risk score",
            ],
        ],
        scenario_templates=[
            "A general counsel is reviewing a vendor master-services agreement before signature. Five structured contract-review subscores have been computed by the legal-ops team.",
            "A compliance officer is evaluating a partnership contract from a third-party vendor. Five regulatory-review indices are summarized below.",
            "An in-house attorney is auditing a renewed enterprise SaaS agreement. Five contract-clause subscores are available.",
            "A procurement lawyer is reviewing a manufacturing supply contract with overseas counterparties. Five compliance subscores have been collected.",
            "A privacy-counsel is evaluating a data-processing addendum against current regulatory exposure. Five legal-review subscores are summarized below.",
            "A securities lawyer is reviewing a new investor-rights agreement. Five contract-clause indices are available.",
            "An employment lawyer is auditing a senior-executive employment contract. Five risk-area subscores are summarized below.",
            "A regulatory-affairs lawyer is reviewing a cross-border distribution agreement. Five compliance subscores have been computed.",
        ],
        question_templates=[
            "Based on these subscores, estimate the overall contract compliance risk index (0-100).",
            "Given the legal-review subscores above, what is your best estimate for the composite compliance risk score (0-100)?",
            "Estimate the contract's compliance risk index (0-100) from the evidence provided.",
            "What overall contract compliance risk (0-100) do you estimate based on these subscores?",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: a recent legal-review benchmark suggested compliance risk near {anchor} for comparable agreements.",
                "A comparable-deal database estimated the compliance risk at approximately {anchor}.",
                "Peer-firm compliance audits indicated a compliance risk index near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("contract compliance risk"),
        },
        metric_name="contract compliance risk",
    ),

    # ── C1. Everyday consumer purchase decision ───────────────────────
    "consumer_purchase_decision": DomainConfig(
        domain_id="consumer_purchase_decision",
        display_name="Consumer / Purchase-Decision Confidence",
        description=(
            "Estimating consumer purchase-decision confidence (0-100) "
            "given structured product-evaluation subscores."
        ),
        evidence_label_families=[
            [
                "Price-vs-value perception score",
                "Brand-trust rating",
                "Review-sentiment aggregate index",
                "Feature-match-to-need rating",
                "Return-policy reassurance score",
            ],
            [
                "Long-term durability expectation score",
                "Warranty-coverage adequacy rating",
                "After-sales-support reputation index",
                "Comparable-alternative gap rating",
                "Total-cost-of-ownership rating",
            ],
            [
                "Fit-for-lifestyle alignment score",
                "Recommendation-from-similar-buyer rating",
                "Aesthetic-preference rating",
                "Sustainability-and-ethics score",
                "Risk-of-regret rating",
            ],
        ],
        scenario_templates=[
            "A consumer is deciding whether to buy a high-end laptop. They have rated five aspects of the purchase decision on a 0-100 scale.",
            "A household is evaluating a major appliance purchase. Five purchase-decision subscores have been recorded.",
            "A first-time home-buyer is evaluating a candidate property purchase. Five decision-confidence subscores have been collected.",
            "A car shopper is evaluating a candidate vehicle purchase. Five aspects of the decision have been rated on a 0-100 scale.",
            "A consumer is deciding whether to subscribe to an annual streaming-services bundle. Five decision-confidence subscores are summarized below.",
            "A traveller is evaluating booking a multi-day premium tour package. Five purchase-decision subscores have been recorded.",
            "A parent is deciding on a private-tutoring service for their child. Five decision-confidence subscores have been collected.",
            "A consumer is evaluating a candidate mortgage-refinance offer. Five decision-confidence subscores are summarized below.",
        ],
        question_templates=[
            "Based on these subscores, estimate the overall purchase-decision confidence index (0-100).",
            "Given the decision-aspect subscores above, what is your best estimate for the composite purchase-decision confidence (0-100)?",
            "Estimate the consumer's purchase-decision confidence index (0-100) from the evidence provided.",
            "What overall purchase-decision confidence (0-100) do you estimate based on these subscores?",
        ],
        anchor_preambles={
            "irrelevant": _IRRELEVANT_PREAMBLES,
            "plausible": [
                "Before you estimate, consider: a recent consumer-research survey of similar buyers suggested decision confidence near {anchor}.",
                "A market-research panel of comparable consumers estimated purchase-decision confidence at approximately {anchor}.",
                "Aggregated review-platform data indicated a decision-confidence index near {anchor}.",
            ],
            "placebo": _PLACEBO_PREAMBLES,
            "authority": _make_authority_preambles("purchase decision confidence"),
        },
        metric_name="purchase decision confidence",
    ),
}

# Note: we deliberately do NOT update DOMAINS in-place. The medical/other
# domains are an opt-in extension for the COLM 2026 rebuttal; existing
# generators that read DOMAIN_IDS continue to produce the original
# six-domain dataset, preserving reproducibility of the published benchmark.
MEDICAL_DOMAINS: dict[str, DomainConfig] = _MEDICAL_DOMAINS
MEDICAL_DOMAIN_IDS = list(_MEDICAL_DOMAINS.keys())
OTHER_DOMAINS: dict[str, DomainConfig] = _OTHER_DOMAINS
OTHER_DOMAIN_IDS = list(_OTHER_DOMAINS.keys())
BUSINESS_DOMAIN_IDS = list(DOMAINS.keys())
ALL_DOMAIN_IDS = BUSINESS_DOMAIN_IDS + MEDICAL_DOMAIN_IDS + OTHER_DOMAIN_IDS

# Combined registry consulted by generators that need to look up
# business / medical / other domains. DOMAINS proper stays at 6 entries to
# preserve the reproducibility of the published benchmark.
ALL_DOMAINS: dict[str, DomainConfig] = {
    **DOMAINS, **MEDICAL_DOMAINS, **OTHER_DOMAINS,
}

# Attach the D1 plausible-intensity preamble pools to every domain. We do
# this at import time so that any generator that consults `ALL_DOMAINS`
# automatically gets `plausible_mild` and `plausible_strong` available;
# the existing core conditions are unchanged.
for _dcfg in ALL_DOMAINS.values():
    metric = _dcfg.metric_name or _dcfg.display_name.lower()
    _dcfg.anchor_preambles.setdefault(
        "plausible_mild", _make_plausible_mild_preambles(metric)
    )
    _dcfg.anchor_preambles.setdefault(
        "plausible_strong", _make_plausible_strong_preambles(metric)
    )

DOMAIN_IDS = BUSINESS_DOMAIN_IDS


def get_domain(domain_id: str) -> DomainConfig:
    """Lookup helper that consults all domain registries."""
    if domain_id in DOMAINS:
        return DOMAINS[domain_id]
    if domain_id in MEDICAL_DOMAINS:
        return MEDICAL_DOMAINS[domain_id]
    if domain_id in OTHER_DOMAINS:
        return OTHER_DOMAINS[domain_id]
    raise KeyError(f"Unknown domain {domain_id!r}")
