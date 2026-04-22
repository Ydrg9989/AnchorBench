"""Tests for AnchorBench v1 dataset generation pipeline.

Tests:
  1. Deterministic reproducibility (same seed → same output)
  2. Gold answer correctness for all scoring functions
  3. Pairing invariant (stem identical across conditions)
  4. Template diversity (multiple label families / scenarios / questions)
  5. Anchor bounds and direction
  6. Evidence structure by difficulty
  7. Suite-specific metadata
  8. Validator passes on generated data
"""

import json
import random
import statistics

import pytest

from anchorbench_v1.domains import DOMAINS, DOMAIN_IDS
from anchorbench_v1.itemspec_gen import (
    compute_gold_answer,
    generate_external_itemspecs,
    generate_history_itemspecs,
    generate_icl_dist_itemspecs,
    generate_icl_itemspecs,
    generate_rag_itemspecs,
    generate_tool_itemspecs,
)
from anchorbench_v1.schema import ItemSpec
from anchorbench_v1.suites import SUITE_RENDERERS
from anchorbench_v1.validators import validate_all


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def external_specs():
    return generate_external_itemspecs(n_per_cell=2, seed=42)


@pytest.fixture
def history_specs():
    return generate_history_itemspecs(n_per_cell=2, seed=42)


@pytest.fixture
def icl_specs():
    return generate_icl_itemspecs(n_per_cell=2, seed=42)


@pytest.fixture
def icl_dist_specs():
    return generate_icl_dist_itemspecs(n_per_cell=2, seed=42)


@pytest.fixture
def rag_specs():
    return generate_rag_itemspecs(n_per_cell=2, seed=42)


@pytest.fixture
def tool_specs():
    return generate_tool_itemspecs(n_per_cell=2, seed=42)


# ── 1. Reproducibility ──────────────────────────────────────────────

class TestReproducibility:

    def test_same_seed_same_output(self):
        specs_a = generate_external_itemspecs(n_per_cell=3, seed=99)
        specs_b = generate_external_itemspecs(n_per_cell=3, seed=99)
        assert len(specs_a) == len(specs_b)
        for a, b in zip(specs_a, specs_b):
            assert a.item_id == b.item_id
            assert a.theta == b.theta
            assert a.y_star == b.y_star
            assert a.evidence_structured == b.evidence_structured

    def test_different_seed_different_output(self):
        specs_a = generate_external_itemspecs(n_per_cell=3, seed=1)
        specs_b = generate_external_itemspecs(n_per_cell=3, seed=2)
        thetas_a = [s.theta for s in specs_a]
        thetas_b = [s.theta for s in specs_b]
        assert thetas_a != thetas_b


# ── 2. Gold answer correctness ──────────────────────────────────────

class TestGoldAnswer:

    def test_mean_scoring(self, external_specs):
        for spec in external_specs:
            visible = [e["value"] for e in spec.evidence_structured if not e.get("missing")]
            expected = round(sum(visible) / len(visible))
            assert spec.y_star_evidence == expected, f"{spec.item_id}: {spec.y_star_evidence} != {expected}"
            assert spec.y_star == spec.y_star_evidence

    def test_weighted_mean_scoring(self):
        specs = generate_external_itemspecs(
            n_per_cell=2, seed=42,
            scoring_function="weighted_mean",
            scoring_weights=[1.0, 1.0, 1.5, 1.5, 2.0],
        )
        for spec in specs:
            visible = [e["value"] for e in spec.evidence_structured if not e.get("missing")]
            weights = spec.scoring_weights[:len(visible)]
            expected = round(sum(v * w for v, w in zip(visible, weights)) / sum(weights))
            assert spec.y_star_evidence == expected

    def test_median_scoring(self):
        specs = generate_external_itemspecs(
            n_per_cell=2, seed=42,
            scoring_function="median",
        )
        for spec in specs:
            visible = [e["value"] for e in spec.evidence_structured if not e.get("missing")]
            expected = round(statistics.median(visible))
            assert spec.y_star_evidence == expected

    def test_compute_gold_function(self):
        assert compute_gold_answer([60, 70, 80]) == 70
        assert compute_gold_answer([60, 70, 80], "median") == 70
        assert compute_gold_answer([60, 70, 80], "weighted_mean", [1, 1, 2]) == 72
        assert compute_gold_answer([]) == 0


# ── 3. Pairing invariant ────────────────────────────────────────────

class TestPairingInvariant:

    def _check_suite_pairing(self, specs):
        for spec in specs[:6]:
            renderer = SUITE_RENDERERS[spec.suite]
            views = renderer(spec)
            ctrl = next(v for v in views if v.condition == "control")
            for v in views:
                if v.condition in ("control", "control_twostage"):
                    continue
                assert v.prompt_components.get("evidence") == ctrl.prompt_components.get("evidence"), \
                    f"{spec.item_id}/{v.condition}: evidence differs from control"
                assert v.prompt_components.get("scenario") == ctrl.prompt_components.get("scenario"), \
                    f"{spec.item_id}/{v.condition}: scenario differs from control"
                assert v.prompt_components.get("question") == ctrl.prompt_components.get("question"), \
                    f"{spec.item_id}/{v.condition}: question differs from control"

    def test_external_pairing(self, external_specs):
        self._check_suite_pairing(external_specs)

    def test_icl_pairing(self, icl_specs):
        self._check_suite_pairing(icl_specs)

    def test_icl_dist_pairing(self, icl_dist_specs):
        self._check_suite_pairing(icl_dist_specs)

    def test_rag_pairing(self, rag_specs):
        self._check_suite_pairing(rag_specs)

    def test_tool_pairing(self, tool_specs):
        self._check_suite_pairing(tool_specs)


# ── 4. Template diversity ───────────────────────────────────────────

class TestTemplateDiversity:

    def test_multiple_label_families_used(self, external_specs):
        families = set(s.evidence_label_family_idx for s in external_specs)
        assert len(families) > 1, "All items use the same evidence label family"

    def test_multiple_scenario_templates_used(self, external_specs):
        scenarios = set(s.scenario_template_idx for s in external_specs)
        assert len(scenarios) > 1, "All items use the same scenario template"

    def test_multiple_question_templates_used(self, external_specs):
        questions = set(s.question_template_idx for s in external_specs)
        assert len(questions) > 1, "All items use the same question template"

    def test_multiple_phrasing_indices_used(self, external_specs):
        phrasings = set(s.anchor_phrasing_idx for s in external_specs)
        assert len(phrasings) > 1, "All items use the same anchor phrasing"

    def test_domain_config_label_families(self):
        for domain_id, dcfg in DOMAINS.items():
            assert dcfg.n_label_families >= 2, f"{domain_id}: needs ≥2 label families"
            for family in dcfg.evidence_label_families:
                assert len(family) == 5, f"{domain_id}: each label family must have 5 labels"

    def test_domain_config_template_counts(self):
        for domain_id, dcfg in DOMAINS.items():
            assert len(dcfg.scenario_templates) >= 4, f"{domain_id}: needs ≥4 scenarios"
            assert len(dcfg.question_templates) >= 2, f"{domain_id}: needs ≥2 questions"


# ── 5. Anchor bounds ────────────────────────────────────────────────

class TestAnchorBounds:

    def test_anchors_in_range(self, external_specs):
        for spec in external_specs:
            assert 0 <= spec.anchors["low"] <= 100
            assert 0 <= spec.anchors["high"] <= 100
            assert spec.anchors["low"] < spec.anchors["high"]

    def test_anchor_offsets_stratified(self, external_specs):
        offsets = set(s.anchors["offset"] for s in external_specs)
        assert offsets == {15, 25, 40}


# ── 6. Evidence structure ───────────────────────────────────────────

class TestEvidenceStructure:

    def test_easy_no_missing(self, external_specs):
        for spec in external_specs:
            if spec.difficulty == "easy":
                missing = sum(1 for e in spec.evidence_structured if e.get("missing"))
                assert missing == 0, f"{spec.item_id}: easy item has {missing} missing"

    def test_hard_two_missing(self, external_specs):
        for spec in external_specs:
            if spec.difficulty == "hard":
                missing = sum(1 for e in spec.evidence_structured if e.get("missing"))
                assert missing == 2, f"{spec.item_id}: hard item has {missing} missing"

    def test_medium_one_missing(self):
        specs = generate_external_itemspecs(
            n_per_cell=2, seed=42, difficulties=["medium"],
        )
        for spec in specs:
            assert spec.difficulty == "medium"
            missing = sum(1 for e in spec.evidence_structured if e.get("missing"))
            assert missing == 1, f"{spec.item_id}: medium item has {missing} missing"


# ── 7. Suite-specific metadata ──────────────────────────────────────

class TestSuiteMetadata:

    def test_rag_has_corpus_metadata(self, rag_specs):
        for spec in rag_specs:
            assert spec.rag is not None
            assert spec.rag["corpus_size"] == 3

    def test_tool_has_tool_metadata(self, tool_specs):
        for spec in tool_specs:
            assert spec.tool is not None
            assert "get_evidence_summary" in spec.tool["available_tools"]

    def test_history_has_warmups_and_subsets(self, history_specs):
        for spec in history_specs:
            assert spec.history is not None
            assert len(spec.history["subset_indices_low"]) == 2
            assert "warmup_low" in spec.history

    def test_icl_has_demos(self, icl_specs):
        for spec in icl_specs:
            demos = spec.tags.get("icl_demos")
            assert demos is not None
            assert len(demos) == 3
            for d in demos:
                assert "evidence" in d
                assert 0 <= d["answer"] <= 100

    def test_icl_dist_has_demo_bands(self, icl_dist_specs):
        for spec in icl_dist_specs:
            for key in (
                "icl_dist_demos_control",
                "icl_dist_demos_low",
                "icl_dist_demos_high",
            ):
                demos = spec.tags.get(key)
                assert demos is not None and len(demos) == 3
                for d in demos:
                    assert "evidence" in d
                    assert 0 <= d["answer"] <= 100

    def test_icl_dist_plausible_irrelevant_same_demos(self, icl_dist_specs):
        render = SUITE_RENDERERS["icl_dist"]
        for spec in icl_dist_specs[:4]:
            views = {v.condition: v for v in render(spec)}
            for pl, ir in (("plausible_low", "irrelevant_low"), ("plausible_high", "irrelevant_high")):
                assert views[pl].prompt_components["demos"] == views[ir].prompt_components["demos"]
                assert views[pl].prompt_components["framing_intro"] != views[ir].prompt_components["framing_intro"]


# ── 8. Validator integration ────────────────────────────────────────

class TestValidatorIntegration:

    def _validate_suite(self, specs):
        views = []
        for spec in specs:
            renderer = SUITE_RENDERERS[spec.suite]
            views.extend(renderer(spec))
        errors = validate_all(specs, views)
        assert errors == [], f"Validation errors: {errors}"

    def test_external_validation(self, external_specs):
        self._validate_suite(external_specs)

    def test_history_validation(self, history_specs):
        self._validate_suite(history_specs)

    def test_icl_validation(self, icl_specs):
        self._validate_suite(icl_specs)

    def test_icl_dist_validation(self, icl_dist_specs):
        self._validate_suite(icl_dist_specs)

    def test_rag_validation(self, rag_specs):
        self._validate_suite(rag_specs)

    def test_tool_validation(self, tool_specs):
        self._validate_suite(tool_specs)
