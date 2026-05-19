#!/usr/bin/env python3
"""
Evaluate — CLI Script.

Runs evaluation of the online RAG pipeline against a set of questions
with reference answers.  Computes standard metrics (BLEU-4, ROUGE-L,
ANLS, F1), generates comparison charts, and saves the results.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --eval-set outputs/evaluation/eval_set.json
    python scripts/evaluate.py --eval-set eval.json --baseline baseline_results.json

Arguments:
    --eval-set    : Path to evaluation set JSON (default: from evaluation_config.yaml)
    --baseline    : Path to baseline results JSON for comparison (optional)
    --config      : Path to pipeline config YAML (default: configs/pipeline_config.yaml)
    --output-dir  : Output directory for results and charts (default: from config)
    --no-charts   : Skip chart generation

Evaluation Set Format:
    The eval_set.json file should be a list of objects:
    [
        {
            "question": "What is the Vision Transformer?",
            "reference": "The Vision Transformer (ViT) is a model that…",
            "category": "architecture"
        },
        …
    ]

Example:
    $ python scripts/evaluate.py --eval-set my_eval.json
    Running evaluation on 10 questions…
    Mean BLEU-4: 0.45, ROUGE-L: 0.62, ANLS: 0.78, F1: 0.65
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.config_loader import load_config, resolve_paths
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the Scientific Multimodal RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/evaluate.py\n"
            "  python scripts/evaluate.py --eval-set my_eval.json\n"
            "  python scripts/evaluate.py --baseline baseline.json --no-charts\n"
        ),
    )

    parser.add_argument(
        "--eval-set",
        type=str,
        default=None,
        help="Path to evaluation set JSON (default: from evaluation_config.yaml)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to baseline results JSON for comparison (optional)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results and charts",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        default=False,
        help="Skip chart generation",
    )

    return parser.parse_args()


def load_eval_set(path: str) -> List[Dict[str, str]]:
    """Load an evaluation set from a JSON file.

    Args:
        path: Path to the JSON file containing the evaluation set.

    Returns:
        List of dictionaries with at least ``"question"`` and
        ``"reference"`` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or is empty.
    """
    eval_path = Path(path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation set not found: {path}")

    with open(str(eval_path), "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(
            f"Evaluation set must be a non-empty list, got {type(data).__name__}"
        )

    # Validate structure
    for i, item in enumerate(data):
        if "question" not in item:
            raise ValueError(f"Item {i} missing 'question' key.")
        if "reference" not in item:
            raise ValueError(f"Item {i} missing 'reference' key.")

    return data


def create_default_eval_set() -> List[Dict[str, str]]:
    """Create a default evaluation set with sample questions.

    Returns:
        A list of evaluation question/reference pairs about
        Vision Transformer papers.
    """
    default_eval: List[Dict[str, str]] = [
        {
            "question": "What is the Vision Transformer architecture?",
            "reference": (
                "The Vision Transformer (ViT) is a model that applies "
                "a pure Transformer architecture directly to image patches, "
                "without using convolutional layers. It splits an image into "
                "fixed-size patches, linearly embeds them, and processes them "
                "with standard Transformer encoders."
            ),
            "category": "architecture",
        },
        {
            "question": "How does self-attention work in transformers?",
            "reference": (
                "Self-attention computes a weighted sum of all value "
                "vectors, where the weights are derived from the dot "
                "product of query and key vectors. The attention mechanism "
                "allows the model to focus on different parts of the input "
                "sequence when producing each output element."
            ),
            "category": "mechanism",
        },
        {
            "question": "What datasets were used to evaluate ViT?",
            "reference": (
                "Vision Transformer was evaluated on ImageNet, CIFAR-10, "
                "CIFAR-100, and several smaller datasets. When pre-trained "
                "on large datasets like JFT-300M or ImageNet-21k, ViT "
                "achieves state-of-the-art results."
            ),
            "category": "evaluation",
        },
        {
            "question": "What is the difference between ViT and CNNs?",
            "reference": (
                "Unlike CNNs which use local receptive fields and weight "
                "sharing, ViT uses global self-attention over image patches. "
                "ViT lacks inductive biases like translation equivariance "
                "and locality that are built into CNNs, but compensates "
                "with more data during pre-training."
            ),
            "category": "comparison",
        },
        {
            "question": "How does image patching work in Vision Transformers?",
            "reference": (
                "Image patching in ViT works by extracting fixed-size "
                "patches (e.g., 16x16 pixels) from the input image, "
                "flattening each patch into a vector, and linearly "
                "projecting it to the model's embedding dimension. A "
                "learnable position embedding is added to each patch."
            ),
            "category": "mechanism",
        },
        {
            "question": "What is the role of the class token in ViT?",
            "reference": (
                "The class token is a learnable embedding prepended to "
                "the sequence of patch embeddings. Its state at the output "
                "of the Transformer encoder serves as the image "
                "representation, which is used for classification via a "
                "linear layer."
            ),
            "category": "architecture",
        },
        {
            "question": "How is positional encoding used in Vision Transformers?",
            "reference": (
                "Positional encoding in ViT is added to the patch "
                "embeddings to retain positional information. Unlike "
                "original Transformers that use sinusoidal encodings, "
                "ViT typically uses learnable position embeddings that "
                "are trained along with the model."
            ),
            "category": "mechanism",
        },
        {
            "question": "What are the main results reported in ViT papers?",
            "reference": (
                "ViT achieves state-of-the-art performance on image "
                "classification when pre-trained on large datasets. On "
                "ImageNet, ViT-Huge reaches 88.55% top-1 accuracy. The "
                "model performs well even with fewer computational "
                "resources compared to CNNs."
            ),
            "category": "results",
        },
        {
            "question": "What is the patch size used in ViT?",
            "reference": (
                "ViT commonly uses patch sizes of 16x16 or 32x32 "
                "pixels. Smaller patch sizes result in more tokens and "
                "higher computational cost but can capture finer details. "
                "The patch size is a key hyperparameter."
            ),
            "category": "architecture",
        },
        {
            "question": "Why do transformers need large datasets for pre-training?",
            "reference": (
                "Transformers lack the inductive biases of CNNs "
                "(locality, translation equivariance), so they need "
                "larger datasets to learn these patterns from data. "
                "When trained on small datasets, ViT underperforms "
                "compared to CNNs, but matches or exceeds them when "
                "pre-trained on large datasets."
            ),
            "category": "analysis",
        },
    ]

    return default_eval


def run_evaluation(
    pipeline: Any,
    eval_set: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Run the evaluation pipeline on all questions.

    Args:
        pipeline: An initialized :class:`OnlinePipeline` instance.
        eval_set: List of evaluation items with ``"question"`` and
            ``"reference"`` keys.

    Returns:
        List of result dictionaries, each containing the question,
        reference, prediction, confidence, and timing information.
    """
    results: List[Dict[str, Any]] = []

    for i, item in enumerate(eval_set):
        question = item["question"]
        reference = item["reference"]
        category = item.get("category", "general")

        print(
            f"[{i + 1}/{len(eval_set)}] {question[:60]}…",
            end=" ",
            flush=True,
        )

        t_start = time.time()
        try:
            rag_result = pipeline.query(question)
            prediction = rag_result.answer
            confidence = rag_result.confidence
            total_time = time.time() - t_start
            check_passed = rag_result.check_result.passed if rag_result.check_result else False
            retries = rag_result.retries
            print(
                f"✓ conf={confidence:.1%} time={total_time:.1f}s "
                f"retries={retries}"
            )
        except Exception as exc:
            prediction = ""
            confidence = 0.0
            total_time = time.time() - t_start
            check_passed = False
            retries = 0
            print(f"✗ Error: {exc}")

        results.append({
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "category": category,
            "confidence": confidence,
            "time": total_time,
            "check_passed": check_passed,
            "retries": retries,
        })

    return results


def compute_all_metrics(
    results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute aggregate metrics from evaluation results.

    Args:
        results: List of evaluation result dictionaries.

    Returns:
        Dictionary of mean metric scores.
    """
    from src.utils.metrics import compute_bleu4, compute_rouge_l, compute_anls, compute_f1

    bleu_scores = []
    rouge_scores = []
    anls_scores = []
    f1_scores = []

    for r in results:
        pred = r.get("prediction", "")
        ref = r.get("reference", "")
        bleu_scores.append(compute_bleu4(pred, ref))
        rouge_scores.append(compute_rouge_l(pred, ref))
        anls_scores.append(compute_anls(pred, ref))
        f1_scores.append(compute_f1(pred, ref))

    n = len(results) if results else 1
    metrics = {
        "bleu4": sum(bleu_scores) / n,
        "rouge_l": sum(rouge_scores) / n,
        "anls": sum(anls_scores) / n,
        "f1": sum(f1_scores) / n,
    }

    return metrics


def generate_charts(
    metrics: Dict[str, float],
    baseline_metrics: Optional[Dict[str, float]],
    results: List[Dict[str, Any]],
    output_dir: Path,
    confidence_threshold: float = 0.6,
) -> None:
    """Generate evaluation charts and save to disk.

    Creates:
    1. Metrics comparison bar chart (ours vs. baseline if available).
    2. Confidence distribution histogram.
    3. Per-category performance breakdown.

    Args:
        metrics: Our system's aggregate metrics.
        baseline_metrics: Optional baseline metrics for comparison.
        results: Per-question evaluation results.
        output_dir: Directory to save chart images.
        confidence_threshold: Threshold for confidence chart.
    """
    from src.utils.visualization import plot_metrics_comparison, plot_confidence_distribution

    charts_dir = output_dir / "evaluation_charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Metrics comparison
    if baseline_metrics:
        plot_metrics_comparison(
            baseline_metrics=baseline_metrics,
            our_metrics=metrics,
            output_path=charts_dir / "metrics_comparison.png",
            title="Metrics: Baseline vs. Multimodal RAG",
        )
    else:
        # Plot our metrics as standalone bars
        plot_metrics_comparison(
            baseline_metrics={k: 0.0 for k in metrics},
            our_metrics=metrics,
            output_path=charts_dir / "metrics_standalone.png",
            title="Evaluation Metrics — Multimodal RAG",
        )

    # Chart 2: Confidence distribution
    confidences = [r.get("confidence", 0.0) for r in results]
    if confidences:
        plot_confidence_distribution(
            confidences=confidences,
            output_path=charts_dir / "confidence_distribution.png",
            threshold=confidence_threshold,
        )

    logger.info("Charts saved to: %s", charts_dir)


def main() -> None:
    """Main entry point for the evaluate script.

    Loads the evaluation set, runs the pipeline on each question,
    computes metrics, generates charts, and saves the results.
    """
    args = parse_args()

    # Load evaluation config
    try:
        eval_config = load_config("evaluation_config")
    except Exception:
        eval_config = {"evaluation": {}}

    eval_cfg = eval_config.get("evaluation", {})

    # Resolve paths
    eval_set_path = args.eval_set or eval_cfg.get("eval_set_path", "")
    output_dir = args.output_dir or eval_cfg.get("output_dir", "outputs/evaluation/")
    baseline_path = args.baseline
    confidence_threshold = eval_cfg.get("confidence_threshold", 0.6)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load evaluation set
    if eval_set_path:
        print(f"Loading evaluation set: {eval_set_path}")
        try:
            eval_set = load_eval_set(eval_set_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
    else:
        print("No evaluation set specified — using default questions.")
        eval_set = create_default_eval_set()

        # Save default eval set
        default_path = output_dir / "default_eval_set.json"
        with open(str(default_path), "w", encoding="utf-8") as f:
            json.dump(eval_set, f, indent=2, ensure_ascii=False)
        print(f"Default eval set saved to: {default_path}")

    print(f"\nEvaluation set: {len(eval_set)} questions\n")

    # Initialize the online pipeline
    from pipelines.online_pipeline import OnlinePipeline

    try:
        pipeline = OnlinePipeline(config_path=args.config)
    except Exception as exc:
        logger.error("Failed to initialize pipeline: %s", exc)
        print(f"ERROR: Could not initialize pipeline: {exc}")
        sys.exit(1)

    # Run evaluation
    print("=" * 60)
    print("  RUNNING EVALUATION")
    print("=" * 60)

    t_start = time.time()
    results = run_evaluation(pipeline, eval_set)
    total_eval_time = time.time() - t_start

    # Compute metrics
    metrics = compute_all_metrics(results)

    # Load baseline metrics if provided
    baseline_metrics = None
    if baseline_path:
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            if "metrics" in baseline_data:
                baseline_metrics = baseline_data["metrics"]
            elif isinstance(baseline_data, dict):
                baseline_metrics = baseline_data
            print(f"Loaded baseline metrics from: {baseline_path}")
        except Exception as exc:
            logger.warning("Failed to load baseline: %s", exc)

    # Generate charts
    if not args.no_charts:
        try:
            generate_charts(
                metrics=metrics,
                baseline_metrics=baseline_metrics,
                results=results,
                output_dir=output_dir,
                confidence_threshold=confidence_threshold,
            )
        except Exception as exc:
            logger.warning("Chart generation failed: %s", exc)

    # Compute summary statistics
    confidences = [r.get("confidence", 0.0) for r in results]
    pass_rate = sum(1 for c in confidences if c >= confidence_threshold) / max(len(confidences), 1)
    check_pass_rate = sum(1 for r in results if r.get("check_passed", False)) / max(len(results), 1)
    avg_time = sum(r.get("time", 0.0) for r in results) / max(len(results), 1)

    # Save full results
    output_data = {
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "summary": {
            "num_questions": len(eval_set),
            "confidence_pass_rate": pass_rate,
            "check_pass_rate": check_pass_rate,
            "avg_query_time": avg_time,
            "total_eval_time": total_eval_time,
            "confidence_threshold": confidence_threshold,
        },
        "results": results,
    }

    results_path = output_dir / "evaluation_results.json"
    with open(str(results_path), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print()
    print("=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Questions     : {len(eval_set)}")
    print(f"  BLEU-4        : {metrics['bleu4']:.4f}")
    print(f"  ROUGE-L       : {metrics['rouge_l']:.4f}")
    print(f"  ANLS          : {metrics['anls']:.4f}")
    print(f"  F1            : {metrics['f1']:.4f}")
    print()
    print(f"  Confidence pass rate : {pass_rate:.1%}")
    print(f"  Self-check pass rate : {check_pass_rate:.1%}")
    print(f"  Avg query time       : {avg_time:.2f}s")
    print(f"  Total eval time      : {total_eval_time:.1f}s")

    if baseline_metrics:
        print()
        print("  vs. Baseline:")
        for key in sorted(set(metrics.keys()) & set(baseline_metrics.keys())):
            ours = metrics[key]
            theirs = baseline_metrics.get(key, 0.0)
            diff = ours - theirs
            sign = "+" if diff >= 0 else ""
            print(f"    {key}: {ours:.4f} vs {theirs:.4f} ({sign}{diff:.4f})")

    print()
    print(f"  Results saved: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
