"""Tests for anchorbench.eval.parsing — multi-tier answer parser.

Includes regression tests derived from actual smoke-test failures
(tool-call JSON outputs, verbose CoT, ambiguous intermediate numbers).
"""


from anchorbench.eval.evaluator import parse_response
from anchorbench.eval.parsing import (
    clamp_to_range,
    is_tool_call_output,
    parse_answer_int,
    parse_cot_answer,
    parse_final_answer,
    parse_last_number,
    parse_structured,
    parse_with_fallback,
    parse_xml_answer,
)


class TestParseStructured:
    def test_valid_json(self):
        assert parse_structured('{"answer": 42}') == (42, True)

    def test_valid_with_whitespace(self):
        assert parse_structured('  {"answer": 72}  ') == (72, True)

    def test_zero(self):
        assert parse_structured('{"answer": 0}') == (0, True)

    def test_hundred(self):
        assert parse_structured('{"answer": 100}') == (100, True)

    def test_out_of_range_high(self):
        assert parse_structured('{"answer": 150}') == (None, False)

    def test_out_of_range_negative(self):
        assert parse_structured('{"answer": -5}') == (None, False)

    def test_malformed_json(self):
        assert parse_structured("{answer: 42}") == (None, False)

    def test_missing_key(self):
        assert parse_structured('{"value": 42}') == (None, False)

    def test_empty(self):
        assert parse_structured("") == (None, False)

    def test_float_integer(self):
        assert parse_structured('{"answer": 42.0}') == (42, True)

    def test_float_non_integer(self):
        assert parse_structured('{"answer": 42.5}') == (None, False)

    def test_string_value(self):
        assert parse_structured('{"answer": "42"}') == (None, False)


class TestParseAnswerInt:
    def test_single_number(self):
        assert parse_answer_int("42") == (42, True)

    def test_single_number_with_whitespace(self):
        assert parse_answer_int("  72  ") == (72, True)

    def test_zero(self):
        assert parse_answer_int("0") == (0, True)

    def test_hundred(self):
        assert parse_answer_int("100") == (100, True)

    def test_out_of_range(self):
        assert parse_answer_int("150") == (None, False)

    def test_negative_out_of_range(self):
        assert parse_answer_int("-5") == (None, False)

    def test_empty(self):
        assert parse_answer_int("") == (None, False)

    def test_no_numbers(self):
        assert parse_answer_int("I don't know the answer") == (None, False)

    def test_single_in_sentence(self):
        assert parse_answer_int("The answer is 65.") == (65, True)

    def test_cot_pattern(self):
        assert parse_answer_int("Based on my analysis: 72") == (72, True)

    def test_last_line_extraction(self):
        text = "Let me think...\nConsidering the data...\n55"
        assert parse_answer_int(text) == (55, True)

    def test_echo_filtering(self):
        prompt = "A recent report mentioned the value 80."
        raw = "Based on my analysis with 80 in mind, I estimate 65."
        answer, ok = parse_answer_int(raw, prompt)
        assert ok is True
        assert answer == 65

    def test_multiple_same_number(self):
        assert parse_answer_int("42 and 42 are my estimates") == (42, True)

    def test_ambiguous_multiple(self):
        assert parse_answer_int("30 or 50 or 70") == (None, False)


class TestParseCotAnswer:
    def test_single_last_line(self):
        text = "Step 1: analyze...\nStep 2: estimate...\n55"
        assert parse_cot_answer(text) == (55, True)

    def test_last_valid_in_text(self):
        text = "First 30, then revised to 45, final answer 60."
        assert parse_cot_answer(text) == (60, True)

    def test_empty(self):
        assert parse_cot_answer("") == (None, False)

    def test_no_valid(self):
        assert parse_cot_answer("No numbers here") == (None, False)


class TestParseWithFallback:
    def test_regex_succeeds(self):
        answer, ok, strategy = parse_with_fallback("42", "", False, None)
        assert answer == 42
        assert ok is True
        assert strategy == "regex"

    def test_regex_fails_no_fallback(self):
        answer, ok, strategy = parse_with_fallback("unclear", "", False, None)
        assert answer is None
        assert ok is False
        assert strategy == "failed"


class TestParseResponse:
    def test_structured_first(self):
        answer, ok, strategy = parse_response(
            "some raw text", "prompt",
            structured_raw='{"answer": 42}',
        )
        assert answer == 42
        assert strategy == "structured"

    def test_structured_fails_falls_to_final_answer(self):
        answer, ok, strategy = parse_response(
            "The answer is 65", "prompt",
            structured_raw='{"bad": "json}',
        )
        assert answer == 65
        assert strategy == "final_answer"

    def test_regex_only(self):
        answer, ok, strategy = parse_response(
            "72", "prompt",
        )
        assert answer == 72
        assert strategy == "regex"

    def test_all_fail(self):
        answer, ok, strategy = parse_response(
            "I cannot determine", "prompt",
        )
        assert answer is None
        assert strategy == "failed"

    def test_structured_none_disabled(self):
        answer, ok, strategy = parse_response(
            "42", "prompt",
            structured_raw=None,
        )
        assert answer == 42
        assert strategy == "regex"


class TestClampToRange:
    def test_clamp_high(self):
        assert clamp_to_range(150) == 100

    def test_clamp_low(self):
        assert clamp_to_range(-10) == 0

    def test_in_range(self):
        assert clamp_to_range(42) == 42

    def test_boundaries(self):
        assert clamp_to_range(0) == 0
        assert clamp_to_range(100) == 100


class TestClampedParsing:
    """Verify clamp=True recovers out-of-range values instead of rejecting."""

    def test_structured_clamp_high(self):
        assert parse_structured('{"answer": 150}', clamp=True) == (100, True)

    def test_structured_clamp_negative(self):
        assert parse_structured('{"answer": -5}', clamp=True) == (0, True)

    def test_structured_strict_unchanged(self):
        assert parse_structured('{"answer": 150}', clamp=False) == (None, False)

    def test_parse_answer_int_clamp_bare(self):
        assert parse_answer_int("144", clamp=True) == (100, True)

    def test_parse_answer_int_clamp_negative(self):
        assert parse_answer_int("-20", clamp=True) == (0, True)

    def test_parse_answer_int_strict_bare(self):
        assert parse_answer_int("144") == (None, False)

    def test_xml_answer_clamp(self):
        assert parse_xml_answer("<answer>130</answer>", clamp=True) == (100, True)

    def test_xml_answer_strict(self):
        assert parse_xml_answer("<answer>130</answer>") == (None, False)

    def test_last_number_clamp(self):
        assert parse_last_number("The value is 186", clamp=True) == (100, True)

    def test_last_number_strict(self):
        assert parse_last_number("The value is 186") == (None, False)

    def test_final_answer_clamp(self):
        assert parse_final_answer(
            "The answer is 142", clamp=True,
        ) == (100, True)

    def test_final_answer_strict(self):
        assert parse_final_answer("The answer is 142") == (None, False)

    def test_cot_answer_clamp(self):
        assert parse_cot_answer("Answer: 472", clamp=True) == (100, True)

    def test_cot_answer_strict(self):
        assert parse_cot_answer("Answer: 472") == (None, False)


class TestParseResponseClamped:
    def test_clamp_produces_clamped_strategy(self):
        answer, ok, strategy = parse_response("144", "prompt", clamp=True)
        assert answer == 100
        assert ok is True
        assert strategy == "regex_clamped"

    def test_clamp_in_range_normal_strategy(self):
        answer, ok, strategy = parse_response("72", "prompt", clamp=True)
        assert answer == 72
        assert ok is True
        assert strategy == "regex"

    def test_clamp_false_still_rejects(self):
        answer, ok, strategy = parse_response("144", "prompt", clamp=False)
        assert answer is None
        assert ok is False
        assert strategy == "failed"


# ── Regression tests from smoke-experiment failures ──────────────────


class TestToolCallDetection:
    """Models sometimes output tool-call JSON instead of answers.
    The parser must reject these rather than extracting evidence values."""

    TOOL_CALL_SIMPLE = (
        '{"name": "get_evidence_summary", '
        '"parameters": {"ratings": "[61.0, 73.0, 44.0, 42.0, 57.0]"}}'
    )
    TOOL_CALL_ANCHOR = (
        '{"name": "get_evidence_summary", '
        '"parameters": {"ratings": "[23.0]"}}'
    )
    TOOL_CALL_RANGE = (
        '{"name": "get_evidence_summary", '
        '"parameters": {"ratings": "[34, 35, 36, 37, 38, 39, 40, '
        '41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, '
        '54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, '
        '67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, '
        '80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, '
        '93, 94, 95, 96, 97, 98, 99, 100]", '
        '"type": "array", '
        '"description": "List of numeric ratings to summarize."}}'
    )
    TOOL_CALL_MULTI = (
        '{"name": "get_evidence_summary", '
        '"parameters": {"ratings": "[49.0, 55.0, 53.0, 59.0, 59.0]"}}; '
        '{"name": "check_external_reference", '
        '"parameters": {"domain": "pricing_wtp", "metric": "WTP"}}'
    )

    def test_detect_tool_call(self):
        assert is_tool_call_output(self.TOOL_CALL_SIMPLE) is True
        assert is_tool_call_output(self.TOOL_CALL_ANCHOR) is True
        assert is_tool_call_output(self.TOOL_CALL_RANGE) is True
        assert is_tool_call_output(self.TOOL_CALL_MULTI) is True

    def test_not_tool_call(self):
        assert is_tool_call_output("42") is False
        assert is_tool_call_output("The answer is 55") is False
        assert is_tool_call_output("") is False

    def test_parse_answer_int_rejects_tool_call(self):
        answer, ok = parse_answer_int(self.TOOL_CALL_SIMPLE)
        assert ok is False

    def test_parse_last_number_rejects_tool_call(self):
        answer, ok = parse_last_number(self.TOOL_CALL_SIMPLE)
        assert ok is False

    def test_parse_cot_rejects_tool_call(self):
        answer, ok = parse_cot_answer(self.TOOL_CALL_SIMPLE)
        assert ok is False

    def test_parse_response_rejects_tool_call(self):
        """Full cascade should fail to parse tool-call JSON."""
        answer, ok, strategy = parse_response(self.TOOL_CALL_SIMPLE, "")
        assert ok is False
        assert strategy == "failed"

    def test_tool_call_with_anchor_value_rejected(self):
        """Must not extract the anchor value 23 from tool call."""
        answer, ok, strategy = parse_response(self.TOOL_CALL_ANCHOR, "")
        assert ok is False

    def test_tool_call_range_rejected(self):
        """Must not extract 100 from [34..100] ratings array."""
        answer, ok, strategy = parse_response(self.TOOL_CALL_RANGE, "")
        assert ok is False

    def test_multi_tool_call_rejected(self):
        """Chained tool calls should also be rejected."""
        answer, ok = parse_answer_int(self.TOOL_CALL_MULTI)
        assert ok is False


class TestVerboseCoTRegression:
    """Models produce long chain-of-thought with out-of-range
    intermediate values.  Parser must not grab intermediates."""

    COT_WRONG_MULTIPLY = (
        "Segment A: 20\nSegment B: 51\nSegment C: 20\n"
        "Weighted average:\n(20 + 51 + 20 + 20 + 20) / 5 = 121 / 5 = 24.2\n"
        "Conjoint analysis index: 90\n"
        "24.2 * 90 = 2188\n"
        "The overall willingness-to-pay index is: 2188"
    )

    COT_OUT_OF_RANGE_SUM = (
        "Weighted average: 55\n"
        "55 + 59 = 114\n"
        "114 + 54 = 168\n"
        "168 + 22 = 190\n"
        "So, the estimated WTP index is 190."
    )

    COT_EXPLICIT_ANSWER = (
        "To estimate the overall willingness-to-pay index...\n"
        "Customer segment A survey score: 66\n"
        "Conjoint analysis index: 19\n"
        "Historical price-sensitivity score: 36\n"
        "(66 + 19 + 36) / 3 = 121 / 3 ≈ 40.33\n"
        "Answer: [40]"
    )

    def test_out_of_range_final_rejects(self):
        """When model's declared answer is out of range (2188), no
        intermediate calculation number should be returned as the answer."""
        answer, ok, strategy = parse_response(self.COT_WRONG_MULTIPLY, "")
        if ok:
            assert answer != 24, "Should not extract intermediate 24.2 → 24"
            assert answer != 57, "Should not extract random evidence number"

    def test_out_of_range_sum_rejects(self):
        """190 is out of range; model said 'estimated WTP index is 190',
        so parser should fail rather than grab intermediate 55."""
        answer, ok, strategy = parse_response(self.COT_OUT_OF_RANGE_SUM, "")
        assert ok is False, "Should fail: model's answer (190) is out of range"
        assert strategy == "failed"

    def test_explicit_answer_bracket(self):
        """Answer: [40] should be parsed correctly."""
        answer, ok, strategy = parse_response(self.COT_EXPLICIT_ANSWER, "")
        assert ok is True
        assert answer == 40
        assert strategy == "final_answer"


class TestFinalAnswerPriority:
    """final_answer patterns should fire before regex to avoid
    picking up intermediate CoT numbers."""

    def test_final_answer_beats_intermediate(self):
        text = (
            "Step 1: 30 + 50 = 80\n"
            "Step 2: 80 / 2 = 40\n"
            "My final answer is 40"
        )
        answer, ok, strategy = parse_response(text, "")
        assert ok is True
        assert answer == 40
        assert strategy == "final_answer"

    def test_answer_colon_pattern(self):
        text = "Many calculations... 120, 240, 360\nThe answer is 42"
        answer, ok, strategy = parse_response(text, "")
        assert ok is True
        assert answer == 42
        assert strategy == "final_answer"

    def test_xml_beats_all(self):
        text = "Reasoning: 30, 50, 80\n<answer>42</answer>"
        answer, ok, strategy = parse_response(text, "")
        assert ok is True
        assert answer == 42
        assert strategy == "xml_tag"

    def test_estimate_is_pattern(self):
        text = (
            "Computing (66 + 19 + 36) / 3 = 40.33\n"
            "The estimated WTP index is 40"
        )
        answer, ok, strategy = parse_response(text, "")
        assert ok is True
        assert answer == 40


class TestHistorySuiteRegression:
    """History suite produces two-stage outputs (stage1 + stage2).
    The parser must not pick up stage1's anchor-influenced numbers."""

    def test_qwen_explicit_answer_bracket_decimal(self):
        """Qwen produces 'Answer: [27.5]' — should parse to 28 (rounded)."""
        text = (
            "Weighted Average = (19 + 36) / 2 = 27.5\n"
            "Answer: [27.5]"
        )
        answer, ok, strategy = parse_response(text, "")
        assert ok is True
        assert answer == 28  # rounded from 27.5

    def test_qwen_explicit_answer_bracket_integer(self):
        """Qwen produces 'Answer: [40]' — should parse to 40."""
        text = (
            "Weighted Average = (66 + 19 + 36) / 3 = 40.33\n"
            "Answer: [40]"
        )
        answer, ok, strategy = parse_response(text, "")
        assert ok is True
        assert answer == 40

    def test_llama_last_sentence_answer(self):
        """'the estimated new price point is 65' should parse 65."""
        text = (
            "PED = 684 / -30 = -79.2\n"
            "x = 66 - 5198.8 + 79.2x\n"
            "-78.2x = -5132.8\n"
            "x = 65\n"
            "So, the estimated new price point is 65."
        )
        answer, ok, strategy = parse_response(text, "")
        assert ok is True
        assert answer == 65


class TestLastNumberImproved:
    """Last-number heuristic should prefer the last line."""

    def test_prefers_last_line(self):
        text = "Various numbers: 30, 50, 80\nFinal: 42"
        answer, ok = parse_last_number(text)
        assert ok is True
        assert answer == 42

    def test_rejects_tool_json(self):
        text = '{"name": "tool", "parameters": {"value": 55}}'
        answer, ok = parse_last_number(text)
        assert ok is False

    def test_two_numbers_raw(self):
        """'28.67\\n100' — should return 100 (last line)."""
        text = "28.67\n100"
        answer, ok = parse_last_number(text)
        assert ok is True
        assert answer == 100


class TestTruncatedLlamaICLRegression:
    """Mid-generation cutoff: must not parse evidence (e.g. max index 27) or
    last arithmetic step (100) as the answer."""

    def test_truncated_icl_response_fails(self):
        text = (
            "To estimate the overall willingness-to-pay index...\n\n"
            "Customer segment B survey score: 33\n"
            "Focus group rating: 59\n"
            "Conjoint analysis index: 27\n\n"
            "Since the minimum index is not provided, we assume 0. "
            "The maximum index is 27.\n\n"
            "Willingness-to-pay index = (27 - 0) / (27 - 0) * 100 = 100\n\n"
            "A more reasonable estimate would be the average of"
        )
        answer, ok, strategy = parse_response(text, "")
        assert ok is False
        assert strategy == "failed"
        assert answer is None


class TestParseWithFallbackCascade:
    """parse_with_fallback should use the same ordering as parse_response."""

    def test_xml_beats_regex(self):
        text = "30 or 50\n<answer>42</answer>"
        answer, ok, strategy = parse_with_fallback(text, "", False, None)
        assert ok is True
        assert answer == 42
        assert strategy == "xml_tag"

    def test_final_answer_beats_regex(self):
        text = "Step 1: 30\nStep 2: 50\nMy final answer is 65"
        answer, ok, strategy = parse_with_fallback(text, "", False, None)
        assert ok is True
        assert answer == 65
        assert strategy == "final_answer"

    def test_tool_call_rejected(self):
        text = '{"name": "get_evidence_summary", "parameters": {"ratings": "[55]"}}'
        answer, ok, strategy = parse_with_fallback(text, "", False, None)
        assert ok is False
        assert strategy == "failed"
