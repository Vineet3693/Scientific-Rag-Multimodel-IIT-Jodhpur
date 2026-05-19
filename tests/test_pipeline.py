"""
Tests for the RAG Generation Pipeline and Self-Checker.

Covers:
- VLMOutput dataclass (lightweight wrapper for VLM responses)
- CheckResult dataclass creation and validation
- SelfChecker attribution check (pass/fail)
- SelfChecker faithfulness check
- SelfChecker confidence check
- validate_query function
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from src.generation.self_check import CheckResult, SelfChecker
from src.retrieval.base_retriever import SourceCitation


# ---------------------------------------------------------------------------
# VLMOutput — lightweight wrapper for VLM generation responses
# ---------------------------------------------------------------------------


@dataclass
class VLMOutput:
    """Structured output from a Vision-Language Model.

    Attributes:
        answer: The generated answer text.
        confidence: Model's self-reported confidence in [0, 1].
        citations: List of citation strings found in the answer.
        raw_output: The raw output from the VLM (before parsing).
        generation_time: Wall-clock time for generation in seconds.
    """

    answer: str
    confidence: float = 0.0
    citations: List[str] = None  # type: ignore[assignment]
    raw_output: str = ""
    generation_time: float = 0.0

    def __post_init__(self) -> None:
        """Set defaults and validate after initialisation."""
        if self.citations is None:
            self.citations = []
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(
                f"VLMOutput.confidence must be in [0, 1], got {self.confidence}"
            )
        if self.generation_time < 0:
            raise ValueError(
                f"VLMOutput.generation_time must be >= 0, "
                f"got {self.generation_time}"
            )


# ---------------------------------------------------------------------------
# validate_query — standalone function for pipeline query validation
# ---------------------------------------------------------------------------


def validate_query(
    question: str,
    min_words: int = 3,
    max_length: int = 200,
) -> None:
    """Validate a user query before processing.

    Args:
        question: The user's question string.
        min_words: Minimum number of words required.
        max_length: Maximum character length allowed.

    Raises:
        ValueError: If the query is empty, too short, or too long.
    """
    if not question or not question.strip():
        raise ValueError(
            "Query must not be empty. Please provide a question."
        )

    stripped = question.strip()
    word_count = len(stripped.split())

    if word_count < min_words:
        raise ValueError(
            f"Query must contain at least {min_words} "
            f"words, got {word_count}. Please be more specific."
        )

    if len(stripped) > max_length:
        raise ValueError(
            f"Query must not exceed {max_length} "
            f"characters, got {len(stripped)}. Please shorten your "
            f"question."
        )


# ===========================================================================
# TESTS
# ===========================================================================


class TestVLMOutput:
    """Tests for the VLMOutput dataclass."""

    def test_vlm_output_creation(self) -> None:
        """Test basic creation of a VLMOutput."""
        output = VLMOutput(answer="The attention mechanism computes...")
        assert output.answer == "The attention mechanism computes..."
        assert output.confidence == 0.0
        assert output.citations == []
        assert output.raw_output == ""
        assert output.generation_time == 0.0

    def test_vlm_output_full_creation(self) -> None:
        """Test creation with all fields populated."""
        output = VLMOutput(
            answer="The model uses attention [Source: ViT, Page 3].",
            confidence=0.85,
            citations=["[Source: ViT, Page 3]"],
            raw_output="<|im_start|>The model uses attention...",
            generation_time=3.2,
        )
        assert output.confidence == 0.85
        assert len(output.citations) == 1
        assert output.generation_time == 3.2

    def test_vlm_output_confidence_out_of_range_raises(self) -> None:
        """Test that confidence outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be in"):
            VLMOutput(answer="test", confidence=1.5)

    def test_vlm_output_negative_confidence_raises(self) -> None:
        """Test that negative confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be in"):
            VLMOutput(answer="test", confidence=-0.1)

    def test_vlm_output_negative_generation_time_raises(self) -> None:
        """Test that negative generation_time raises ValueError."""
        with pytest.raises(ValueError, match="generation_time must be >= 0"):
            VLMOutput(answer="test", generation_time=-1.0)

    def test_vlm_output_default_citations_list(self) -> None:
        """Test that citations default to an empty list when None."""
        output = VLMOutput(answer="test", citations=None)
        assert output.citations == []


class TestCheckResult:
    """Tests for the CheckResult dataclass."""

    def test_check_result_creation(self) -> None:
        """Test basic creation of a CheckResult."""
        result = CheckResult(
            passed=True,
            attribution_passed=True,
            faithfulness_passed=True,
            confidence=0.9,
            confidence_passed=True,
            details="All checks passed.",
        )
        assert result.passed is True
        assert result.attribution_passed is True
        assert result.faithfulness_passed is True
        assert result.confidence == 0.9
        assert result.confidence_passed is True

    def test_check_result_failed(self) -> None:
        """Test creation of a failed CheckResult."""
        result = CheckResult(
            passed=False,
            attribution_passed=False,
            faithfulness_passed=True,
            confidence=0.4,
            confidence_passed=False,
            details="Attribution and confidence failed.",
        )
        assert result.passed is False
        assert result.attribution_passed is False
        assert result.confidence_passed is False

    def test_check_result_partial_failure(self) -> None:
        """Test that overall passed is False when any single check fails."""
        # Attribution fails → overall should fail
        result = CheckResult(
            passed=False,
            attribution_passed=False,
            faithfulness_passed=True,
            confidence=0.8,
            confidence_passed=True,
            details="Attribution failed.",
        )
        assert result.passed is False

    def test_check_result_all_must_pass(self) -> None:
        """Test that overall passed requires ALL three checks to pass."""
        # Only when all three pass should overall be True
        result = CheckResult(
            passed=True,
            attribution_passed=True,
            faithfulness_passed=True,
            confidence=0.9,
            confidence_passed=True,
            details="All passed.",
        )
        assert result.passed is True
        # Verify consistency
        assert result.attribution_passed and result.faithfulness_passed and result.confidence_passed


class TestSelfCheckerAttribution:
    """Tests for the SelfChecker attribution check."""

    def test_self_checker_attribution_pass(self) -> None:
        """Test attribution check passes with valid citation markers."""
        checker = SelfChecker(confidence_threshold=0.6)
        answer = (
            "The model uses self-attention [Source: Vaswani, Page 3]. "
            "This allows parallel computation [Source: ViT, Page 5]."
        )
        passed, detail = checker._check_attribution(answer)

        assert passed is True
        assert "2 citation" in detail

    def test_self_checker_attribution_fail(self) -> None:
        """Test attribution check fails without citation markers."""
        checker = SelfChecker(confidence_threshold=0.6)
        answer = "The model uses self-attention. This allows parallel computation."
        passed, detail = checker._check_attribution(answer)

        assert passed is False
        assert "no" in detail.lower() or "citation" in detail.lower()

    def test_self_checker_attribution_empty_answer(self) -> None:
        """Test attribution check fails with empty answer."""
        checker = SelfChecker(confidence_threshold=0.6)
        passed, detail = checker._check_attribution("")

        assert passed is False

    def test_self_checker_attribution_single_citation(self) -> None:
        """Test attribution passes with a single citation marker."""
        checker = SelfChecker(confidence_threshold=0.6)
        answer = "The attention mechanism is key [Source: ViT Paper, Page 2]."
        passed, detail = checker._check_attribution(answer)

        assert passed is True
        assert "1 citation" in detail

    def test_self_checker_attribution_case_insensitive(self) -> None:
        """Test attribution is case-insensitive for [SOURCE: ...]."""
        checker = SelfChecker(confidence_threshold=0.6)
        answer = "Important finding [SOURCE: Paper, Page 1]."
        passed, detail = checker._check_attribution(answer)

        assert passed is True


class TestSelfCheckerFaithfulness:
    """Tests for the SelfChecker faithfulness check."""

    def test_self_checker_faithfulness_pass(self) -> None:
        """Test faithfulness check passes with good keyword overlap."""
        checker = SelfChecker(confidence_threshold=0.6, faithfulness_threshold=0.3)
        answer = "The Vision Transformer uses self-attention mechanisms for image classification."
        context = "The Vision Transformer applies self-attention mechanisms to process images for classification tasks."

        passed, overlap, detail = checker._check_faithfulness(answer, context)

        assert passed is True
        assert overlap >= 0.3

    def test_self_checker_faithfulness_fail(self) -> None:
        """Test faithfulness check fails with poor keyword overlap."""
        checker = SelfChecker(confidence_threshold=0.6, faithfulness_threshold=0.3)
        answer = "Quantum computing leverages superposition for exponential speedups."
        context = "The Vision Transformer applies self-attention mechanisms to process images."

        passed, overlap, detail = checker._check_faithfulness(answer, context)

        assert passed is False
        assert overlap < 0.3

    def test_self_checker_faithfulness_empty_answer(self) -> None:
        """Test faithfulness check fails with empty answer."""
        checker = SelfChecker(confidence_threshold=0.6)
        passed, overlap, detail = checker._check_faithfulness("", "Some context")

        assert passed is False
        assert overlap == 0.0

    def test_self_checker_faithfulness_empty_context(self) -> None:
        """Test faithfulness check fails with empty context."""
        checker = SelfChecker(confidence_threshold=0.6)
        passed, overlap, detail = checker._check_faithfulness("Some answer", "")

        assert passed is False
        assert overlap == 0.0

    def test_self_checker_faithfulness_identical_text(self) -> None:
        """Test faithfulness with identical answer and context."""
        checker = SelfChecker(confidence_threshold=0.6, faithfulness_threshold=0.3)
        text = "The transformer architecture uses multi-head attention mechanisms."
        passed, overlap, detail = checker._check_faithfulness(text, text)

        assert passed is True
        assert overlap == pytest.approx(1.0)


class TestSelfCheckerConfidence:
    """Tests for the SelfChecker confidence check."""

    def test_self_checker_confidence_pass(self) -> None:
        """Test confidence check passes above threshold."""
        checker = SelfChecker(confidence_threshold=0.6)
        passed, detail = checker._check_confidence(0.85)

        assert passed is True
        assert "0.85" in detail

    def test_self_checker_confidence_fail(self) -> None:
        """Test confidence check fails below threshold."""
        checker = SelfChecker(confidence_threshold=0.6)
        passed, detail = checker._check_confidence(0.3)

        assert passed is False
        assert "0.30" in detail or "0.3" in detail

    def test_self_checker_confidence_exact_threshold(self) -> None:
        """Test confidence check passes at exactly the threshold."""
        checker = SelfChecker(confidence_threshold=0.6)
        passed, detail = checker._check_confidence(0.6)

        assert passed is True

    def test_self_checker_confidence_non_numeric_fails(self) -> None:
        """Test confidence check fails with non-numeric input."""
        checker = SelfChecker(confidence_threshold=0.6)
        passed, detail = checker._check_confidence("high")  # type: ignore

        assert passed is False
        assert "not numeric" in detail

    def test_self_checker_confidence_custom_threshold(self) -> None:
        """Test confidence check with a custom threshold."""
        checker = SelfChecker(confidence_threshold=0.8)
        passed_low, _ = checker._check_confidence(0.7)
        passed_high, _ = checker._check_confidence(0.9)

        assert passed_low is False
        assert passed_high is True

    def test_self_checker_full_check_pass(self) -> None:
        """Test the full SelfChecker.check() method with passing inputs."""
        checker = SelfChecker(confidence_threshold=0.6, faithfulness_threshold=0.3)
        result = checker.check(
            answer="The ViT uses patch embeddings [Source: ViT, Page 2].",
            context="The Vision Transformer uses patch embeddings to process images.",
            confidence=0.9,
        )
        assert result.passed is True
        assert result.attribution_passed is True
        assert result.confidence_passed is True

    def test_self_checker_full_check_fail_attribution(self) -> None:
        """Test full check fails when attribution is missing."""
        checker = SelfChecker(confidence_threshold=0.6, faithfulness_threshold=0.3)
        result = checker.check(
            answer="The ViT uses patch embeddings.",
            context="The Vision Transformer uses patch embeddings to process images.",
            confidence=0.9,
        )
        assert result.passed is False
        assert result.attribution_passed is False


class TestValidateQuery:
    """Tests for the validate_query function."""

    def test_validate_query_valid(self) -> None:
        """Test that a valid query passes validation without error."""
        # Should not raise
        validate_query("What is the Vision Transformer?")

    def test_validate_query_empty_raises(self) -> None:
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            validate_query("")

    def test_validate_query_whitespace_only_raises(self) -> None:
        """Test that whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            validate_query("   ")

    def test_validate_query_too_short_raises(self) -> None:
        """Test that a query with fewer than min_words raises ValueError."""
        with pytest.raises(ValueError, match="at least 3 words"):
            validate_query("Hi there")  # Only 2 words

    def test_validate_query_too_long_raises(self) -> None:
        """Test that a query exceeding max_length raises ValueError."""
        long_query = "What " * 50  # ~250 characters
        with pytest.raises(ValueError, match="must not exceed"):
            validate_query(long_query, max_length=100)

    def test_validate_query_custom_min_words(self) -> None:
        """Test validation with custom min_words parameter."""
        # 3 words should pass with min_words=3
        validate_query("One two three", min_words=3)

        # But fail with min_words=5
        with pytest.raises(ValueError, match="at least 5 words"):
            validate_query("One two three", min_words=5)

    def test_validate_query_custom_max_length(self) -> None:
        """Test validation with custom max_length parameter."""
        validate_query("A reasonably long question", max_length=500)

        with pytest.raises(ValueError, match="must not exceed 10"):
            validate_query("This is a very long question", max_length=10)

    def test_validate_query_exactly_min_words(self) -> None:
        """Test that a query with exactly min_words passes."""
        validate_query("One two three", min_words=3)  # Exactly 3 words

    def test_validate_query_single_word_raises(self) -> None:
        """Test that a single-word query raises ValueError with default min."""
        with pytest.raises(ValueError, match="at least 3 words"):
            validate_query("Transformer")
