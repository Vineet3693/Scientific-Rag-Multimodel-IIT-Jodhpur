"""
Query Router — Keyword-Based Query Classification for Hybrid RAG.

Routes incoming queries to either the scientific RAG pipeline or the
medical RAG pipeline based on keyword matching and heuristic rules.

This is a skeleton implementation. Future versions may include:
- ML-based classification (fine-tuned BERT classifier)
- Ensemble of keyword + ML routing
- Confidence-based fallback to both pipelines

Example:
    >>> from future.hybrid.query_router import QueryRouter
    >>> router = QueryRouter()
    >>> route = router.route("What is the Vision Transformer?")
    >>> print(route)
    'scientific'
    >>> route = router.route("What are the side effects of metformin?")
    >>> print(route)
    'medical'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class RouteType(str, Enum):
    """Possible routing destinations for a query."""

    SCIENTIFIC = "scientific"
    MEDICAL = "medical"
    BOTH = "both"
    UNKNOWN = "unknown"


@dataclass
class RoutingResult:
    """Result of query routing classification.

    Attributes:
        route: The selected route type.
        confidence: Routing confidence in [0, 1].
        scientific_score: Keyword match score for scientific domain.
        medical_score: Keyword match score for medical domain.
        matched_keywords: Keywords that triggered the routing decision.
    """

    route: RouteType
    confidence: float = 0.0
    scientific_score: float = 0.0
    medical_score: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate fields after initialisation."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )


class QueryRouter:
    """Keyword-based query router for hybrid scientific/medical RAG.

    Classifies queries by counting keyword matches in two domain-specific
    keyword sets.  The domain with more matches wins.  If scores are
    tied or both are zero, the query is routed to BOTH pipelines.

    Args:
        scientific_keywords: Set of keywords associated with scientific
            domain queries.  Defaults to a built-in set.
        medical_keywords: Set of keywords associated with medical
            domain queries.  Defaults to a built-in set.
        confidence_threshold: Minimum confidence difference required
            to route to a single pipeline.  Below this, BOTH is used.
            Defaults to 0.3.

    Example:
        >>> router = QueryRouter()
        >>> result = router.route("How does self-attention work in ViT?")
        >>> print(result.route, result.confidence)
        RouteType.SCIENTIFIC 0.8
    """

    # Default scientific domain keywords
    DEFAULT_SCIENTIFIC_KEYWORDS: Set[str] = {
        "transformer", "attention", "embedding", "model", "training",
        "benchmark", "architecture", "neural", "network", "layer",
        "encoder", "decoder", "backbone", "feature", "convolution",
        "vision", "image", "classification", "detection", "segmentation",
        "generative", "diffusion", "gan", "autoencoder", "vit",
        "deit", "swin", "clip", "bert", "gpt", "resnet", "vgg",
        "pretraining", "finetuning", "downstream", "sota", "state-of-the-art",
        "ablation", "epoch", "batch", "learning rate", "optimizer",
        "loss function", "gradient", "regularization", "overfitting",
        "arxiv", "paper", "dataset", "evaluation", "metric",
    }

    # Default medical domain keywords
    DEFAULT_MEDICAL_KEYWORDS: Set[str] = {
        "diagnosis", "treatment", "patient", "clinical", "drug",
        "dosage", "symptom", "disease", "therapy", "surgery",
        "medication", "prescription", "side effect", "adverse",
        "contraindication", "prognosis", "pathology", "radiology",
        "biopsy", "blood", "heart", "cancer", "tumor", "diabetes",
        "hypertension", "infection", "vaccine", "immunization",
        "epidemiology", "mortality", "morbidity", "placebo",
        "randomized", "trial", "pubmed", "cochrane", "guideline",
        "protocol", "ward", "icu", "emergency", "triage",
    }

    def __init__(
        self,
        scientific_keywords: Optional[Set[str]] = None,
        medical_keywords: Optional[Set[str]] = None,
        confidence_threshold: float = 0.3,
    ) -> None:
        self.scientific_keywords = scientific_keywords or self.DEFAULT_SCIENTIFIC_KEYWORDS
        self.medical_keywords = medical_keywords or self.DEFAULT_MEDICAL_KEYWORDS
        self.confidence_threshold = confidence_threshold

    def route(self, query: str) -> RoutingResult:
        """Classify a query and determine the routing destination.

        Tokenises the query into lowercase words, then counts how many
        match each keyword set.  The domain with more matches wins,
        provided the confidence difference exceeds the threshold.

        Args:
            query: The user's natural-language question.

        Returns:
            A :class:`RoutingResult` with the selected route, confidence,
            and match details.

        Raises:
            ValueError: If *query* is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        # Tokenise: lowercase, split on non-alphanumeric
        tokens = set(re.split(r"[^a-zA-Z0-9]+", query.lower()))
        tokens.discard("")

        # Count keyword matches
        sci_matches = tokens & self.scientific_keywords
        med_matches = tokens & self.medical_keywords

        scientific_score = len(sci_matches)
        medical_score = len(med_matches)
        total = scientific_score + medical_score

        # Collect all matched keywords
        matched_keywords = list(sci_matches | med_matches)

        # Determine route
        if total == 0:
            # No keyword signals — route to both
            return RoutingResult(
                route=RouteType.BOTH,
                confidence=0.0,
                scientific_score=0.0,
                medical_score=0.0,
                matched_keywords=[],
            )

        # Compute confidence as the proportion of matches in the winning domain
        scientific_confidence = scientific_score / total
        medical_confidence = medical_score / total

        # Determine routing based on score difference
        score_diff = abs(scientific_confidence - medical_confidence)

        if score_diff < self.confidence_threshold:
            # Scores too close — route to both
            route = RouteType.BOTH
            confidence = score_diff
        elif scientific_score > medical_score:
            route = RouteType.SCIENTIFIC
            confidence = scientific_confidence
        else:
            route = RouteType.MEDICAL
            confidence = medical_confidence

        return RoutingResult(
            route=route,
            confidence=confidence,
            scientific_score=scientific_score,
            medical_score=medical_score,
            matched_keywords=sorted(matched_keywords),
        )

    def add_scientific_keywords(self, keywords: Set[str]) -> None:
        """Add keywords to the scientific domain set.

        Args:
            keywords: Set of keywords to add.
        """
        self.scientific_keywords |= keywords

    def add_medical_keywords(self, keywords: Set[str]) -> None:
        """Add keywords to the medical domain set.

        Args:
            keywords: Set of keywords to add.
        """
        self.medical_keywords |= keywords
