"""
Self-Check Module — Three-Level Answer Verification.

Implements a three-level verification pipeline that validates generated
answers before returning them to the user:

1. **Attribution** — Checks that the answer contains inline citation
   markers in the format ``[Source: ...]``, ensuring every factual
   claim is traceable to a source document.

2. **Faithfulness** — Measures keyword overlap between the generated
   answer and the retrieved context.  Low overlap suggests the answer
   may contain hallucinated content not grounded in the evidence.

3. **Confidence** — Verifies that the model's self-reported confidence
   meets a minimum threshold, filtering out low-certainty responses.

The overall pass requires all three checks to succeed.  Failed checks
trigger retry logic in the :class:`~src.generation.rag_generator.RAGGenerator`.

Example:
    >>> from src.generation.self_check import SelfChecker, CheckResult
    >>> checker = SelfChecker(confidence_threshold=0.6, faithfulness_threshold=0.3)
    >>> result = checker.check(
    ...     answer="The model uses attention [Source: Vaswani, Page 3].",
    ...     context="The model uses attention mechanisms for sequence processing.",
    ...     confidence=0.85,
    ... )
    >>> print(result.passed)
    True
    >>> print(result.attribution_passed)
    True
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CheckResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of the three-level self-check verification.

    Captures the outcome of each individual check as well as the
    overall pass/fail status.  The *details* field provides a
    human-readable summary suitable for logging or display.

    Attributes:
        passed: Overall pass status — ``True`` only if all three
            checks (attribution, faithfulness, confidence) pass.
        attribution_passed: Whether the answer contains at least one
            ``[Source: ...]`` citation marker.
        faithfulness_passed: Whether the keyword overlap between the
            answer and the context meets the faithfulness threshold.
        confidence: The raw confidence score passed to the check.
        confidence_passed: Whether the confidence meets or exceeds
            the configured threshold.
        details: Human-readable summary of all check outcomes.
    """

    passed: bool
    attribution_passed: bool
    faithfulness_passed: bool
    confidence: float
    confidence_passed: bool
    details: str


# ---------------------------------------------------------------------------
# SelfChecker class
# ---------------------------------------------------------------------------

class SelfChecker:
    """Three-level answer verification pipeline.

    Applies attribution, faithfulness, and confidence checks to a
    generated answer and returns a :class:`CheckResult` indicating
    whether the answer is suitable for returning to the user.

    Args:
        confidence_threshold: Minimum confidence score required to
            pass the confidence check.  Defaults to 0.6 (60%).
        faithfulness_threshold: Minimum keyword overlap ratio required
            to pass the faithfulness check.  Defaults to 0.3 (30%),
            meaning at least 30% of content words in the answer must
            appear in the context.

    Example:
        >>> checker = SelfChecker(confidence_threshold=0.6)
        >>> result = checker.check(
        ...     answer="Transformers use self-attention [Source: Vaswani, Page 2].",
        ...     context="Transformers use self-attention mechanisms.",
        ...     confidence=0.9,
        ... )
        >>> assert result.passed is True
    """

    # Regex pattern for citation markers like [Source: Paper Title, Page 3]
    _ATTRIBUTION_PATTERN: re.Pattern = re.compile(
        r"\[Source:\s*[^]]+\]", re.IGNORECASE
    )

    # Common English stop words to exclude from keyword overlap
    _STOP_WORDS: frozenset = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "shall",
        "can", "it", "its", "this", "that", "these", "those", "i", "me",
        "my", "we", "our", "you", "your", "he", "him", "his", "she",
        "her", "they", "them", "their", "what", "which", "who", "whom",
        "how", "when", "where", "why", "not", "no", "nor", "as", "if",
        "then", "than", "too", "very", "just", "also", "so", "up", "out",
        "about", "into", "over", "after", "before", "between", "under",
        "again", "further", "once", "here", "there", "all", "each",
        "both", "few", "more", "most", "other", "some", "such", "only",
        "own", "same", "any",
    })

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        faithfulness_threshold: float = 0.3,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.faithfulness_threshold = faithfulness_threshold

        logger.info(
            "SelfChecker initialised — confidence_threshold=%.2f, "
            "faithfulness_threshold=%.2f",
            self.confidence_threshold,
            self.faithfulness_threshold,
        )

    # -----------------------------------------------------------------
    # check (main entry point)
    # -----------------------------------------------------------------

    def check(
        self,
        answer: str,
        context: str,
        confidence: float,
    ) -> CheckResult:
        """Run the three-level verification pipeline on a generated answer.

        The checks are applied in order:

        1. **Level 1 — Attribution**: Scans the answer for ``[Source: ...]``
           citation markers using a regular expression.  At least one
           marker must be present to pass.

        2. **Level 2 — Faithfulness**: Computes keyword overlap between
           the answer and the context (excluding stop words).  The overlap
           ratio must be >= *faithfulness_threshold* to pass.

        3. **Level 3 — Confidence**: Checks whether *confidence* is >=
           *confidence_threshold*.

        The overall pass requires **all three** checks to succeed.

        Args:
            answer: The generated answer text to verify.
            context: The retrieved context text that the answer should
                be grounded in.
            confidence: The model's self-reported confidence score,
                typically in the range [0, 1].

        Returns:
            A :class:`CheckResult` with the outcome of each individual
            check and the overall pass/fail status.

        Example:
            >>> checker = SelfChecker()
            >>> result = checker.check(
            ...     answer="The model uses attention [Source: Vaswani, Page 3].",
            ...     context="The model uses attention mechanisms.",
            ...     confidence=0.85,
            ... )
            >>> print(result.passed, result.details)
        """
        logger.info("Running self-check pipeline…")

        # Level 1: Attribution
        attribution_passed, attribution_detail = self._check_attribution(answer)
        logger.info(
            "Level 1 (Attribution): passed=%s — %s",
            attribution_passed,
            attribution_detail,
        )

        # Level 2: Faithfulness
        faithfulness_passed, overlap_ratio, faithfulness_detail = (
            self._check_faithfulness(answer, context)
        )
        logger.info(
            "Level 2 (Faithfulness): passed=%s, overlap=%.2f — %s",
            faithfulness_passed,
            overlap_ratio,
            faithfulness_detail,
        )

        # Level 3: Confidence
        confidence_passed, confidence_detail = self._check_confidence(confidence)
        logger.info(
            "Level 3 (Confidence): passed=%s — %s",
            confidence_passed,
            confidence_detail,
        )

        # Overall pass
        overall_passed = (
            attribution_passed and faithfulness_passed and confidence_passed
        )

        # Build details summary
        lines = [
            f"Attribution: {'PASS' if attribution_passed else 'FAIL'} ({attribution_detail})",
            f"Faithfulness: {'PASS' if faithfulness_passed else 'FAIL'} "
            f"(overlap={overlap_ratio:.2f}, threshold={self.faithfulness_threshold:.2f})",
            f"Confidence: {'PASS' if confidence_passed else 'FAIL'} ({confidence_detail})",
            f"Overall: {'PASS' if overall_passed else 'FAIL'}",
        ]
        details = " | ".join(lines)

        result = CheckResult(
            passed=overall_passed,
            attribution_passed=attribution_passed,
            faithfulness_passed=faithfulness_passed,
            confidence=confidence,
            confidence_passed=confidence_passed,
            details=details,
        )

        logger.info(
            "Self-check complete — overall: %s, attribution: %s, "
            "faithfulness: %s, confidence: %s",
            "PASS" if overall_passed else "FAIL",
            "PASS" if attribution_passed else "FAIL",
            "PASS" if faithfulness_passed else "FAIL",
            "PASS" if confidence_passed else "FAIL",
        )

        return result

    # -----------------------------------------------------------------
    # Level 1: Attribution check
    # -----------------------------------------------------------------

    def _check_attribution(self, answer: str) -> Tuple[bool, str]:
        """Check for citation markers in the answer.

        Scans the answer text for patterns matching ``[Source: ...]``
        using a regular expression.  At least one citation marker must
        be present for the check to pass.

        Args:
            answer: The generated answer text.

        Returns:
            A tuple ``(passed, detail)`` where *passed* is ``True`` if
            at least one citation marker is found, and *detail* is a
            human-readable description of the result.
        """
        if not answer or not answer.strip():
            return False, "answer is empty"

        matches = self._ATTRIBUTION_PATTERN.findall(answer)
        num_citations = len(matches)

        if num_citations > 0:
            passed = True
            detail = f"{num_citations} citation(s) found"
        else:
            passed = False
            detail = "no [Source: ...] citation markers found"

        return passed, detail

    # -----------------------------------------------------------------
    # Level 2: Faithfulness check
    # -----------------------------------------------------------------

    def _check_faithfulness(
        self, answer: str, context: str
    ) -> Tuple[bool, float, str]:
        """Measure keyword overlap between the answer and the context.

        Extracts content words (excluding stop words) from both the
        answer and the context, then computes the ratio of answer
        keywords that also appear in the context.  A high ratio
        indicates the answer is grounded in the evidence; a low ratio
        may indicate hallucination.

        The overlap ratio is defined as::

            overlap = |answer_keywords ∩ context_keywords| / |answer_keywords|

        If the answer contains no keywords after stop-word removal,
        the overlap defaults to 0.0 and the check fails.

        Args:
            answer: The generated answer text.
            context: The retrieved context text.

        Returns:
            A tuple ``(passed, overlap_ratio, detail)`` where *passed*
            is ``True`` if *overlap_ratio* >= *faithfulness_threshold*,
            *overlap_ratio* is the computed keyword overlap in ``[0, 1]``,
            and *detail* is a human-readable description.
        """
        if not answer or not answer.strip():
            return False, 0.0, "answer is empty"
        if not context or not context.strip():
            return False, 0.0, "context is empty"

        # Tokenize: lowercase, split on non-alphanumeric, remove stop words.
        answer_words = self._extract_keywords(answer)
        context_words = self._extract_keywords(context)

        if not answer_words:
            return False, 0.0, "no content keywords in answer"

        # Compute overlap ratio.
        context_set = set(context_words)
        answer_set = set(answer_words)
        overlap_count = len(answer_set & context_set)
        overlap_ratio = overlap_count / len(answer_set)

        passed = overlap_ratio >= self.faithfulness_threshold

        if passed:
            detail = (
                f"overlap {overlap_ratio:.2f} >= "
                f"threshold {self.faithfulness_threshold:.2f}"
            )
        else:
            detail = (
                f"overlap {overlap_ratio:.2f} < "
                f"threshold {self.faithfulness_threshold:.2f}"
            )

        return passed, overlap_ratio, detail

    # -----------------------------------------------------------------
    # Level 3: Confidence check
    # -----------------------------------------------------------------

    def _check_confidence(self, confidence: float) -> Tuple[bool, str]:
        """Check whether the confidence score meets the threshold.

        Args:
            confidence: The model's self-reported confidence score,
                expected to be in the range ``[0, 1]``.

        Returns:
            A tuple ``(passed, detail)`` where *passed* is ``True`` if
            *confidence* >= *confidence_threshold*, and *detail* is a
            human-readable description.
        """
        if not isinstance(confidence, (int, float)):
            return False, f"confidence is not numeric: {type(confidence).__name__}"

        if confidence >= self.confidence_threshold:
            passed = True
            detail = (
                f"confidence {confidence:.2f} >= "
                f"threshold {self.confidence_threshold:.2f}"
            )
        else:
            passed = False
            detail = (
                f"confidence {confidence:.2f} < "
                f"threshold {self.confidence_threshold:.2f}"
            )

        return passed, detail

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract content keywords from text, removing stop words.

        Tokenizes the text by splitting on non-alphanumeric characters,
        lowercasing, and filtering out stop words and very short tokens.

        Args:
            text: The input text.

        Returns:
            A list of keyword strings (lowercase, length >= 3, not
            in the stop word set).
        """
        tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
        keywords = [
            tok for tok in tokens
            if len(tok) >= 3 and tok not in self._STOP_WORDS
        ]
        return keywords
