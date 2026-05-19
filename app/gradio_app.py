"""
Gradio Demo — Scientific Multimodal RAG
Phase 1: Quick demo for Kaggle testing
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Pipeline wrapper — lazy-loaded to avoid importing heavy deps at launch
# ---------------------------------------------------------------------------

_pipeline = None


def _get_pipeline():
    """Lazily initialise the OnlinePipeline on first query.

    Returns:
        An OnlinePipeline instance ready for querying.
    """
    global _pipeline
    if _pipeline is None:
        from pipelines.online_pipeline import OnlinePipeline

        config_path = str(_project_root / "configs" / "pipeline_config.yaml")
        _pipeline = OnlinePipeline(config_path=config_path)
    return _pipeline


# ---------------------------------------------------------------------------
# Answer function
# ---------------------------------------------------------------------------


def answer_question(question: str) -> Tuple[str, str, str, Any]:
    """Process a user question through the RAG pipeline.

    Executes the full pipeline: validate → encode → retrieve → fuse →
    generate → self-check.  Returns the answer, confidence, sources,
    and the top retrieved page image for display.

    Args:
        question: The user's natural-language question.

    Returns:
        A tuple of (answer, confidence, sources, page_image) where:
        - answer is the generated text,
        - confidence is a percentage string,
        - sources is a formatted string of citations,
        - page_image is the top retrieved page as a PIL image or None.
    """
    if not question or not question.strip():
        return (
            "Please enter a question.",
            "N/A",
            "N/A",
            None,
        )

    try:
        pipeline = _get_pipeline()
        t_start = time.time()
        result = pipeline.query(question)
        elapsed = time.time() - t_start

        # Format answer
        answer = result.answer or "No answer generated."
        answer += f"\n\n---\n⏱ Time: {elapsed:.1f}s | Retries: {result.retries}"

        # Format confidence
        confidence = f"{result.confidence:.0%}"

        # Format sources
        if result.sources:
            source_lines = []
            for i, src in enumerate(result.sources, 1):
                source_lines.append(
                    f"{i}. {src.paper_title} — Pages {src.page_numbers}\n"
                    f"   {src.arxiv_url}\n"
                    f"   Relevance: {src.relevance_score:.3f}"
                )
            sources = "\n".join(source_lines)
        else:
            sources = "No sources found."

        # Get top page image
        page_image = None
        if result.sources:
            for src in result.sources:
                if src.page_images:
                    page_image = src.page_images[0]
                    break

        # Check result summary
        check = result.check_result
        answer += (
            f"\n\n✅ Self-check: {'PASS' if check.passed else 'FAIL'}"
            f" | Attribution: {'✓' if check.attribution_passed else '✗'}"
            f" | Faithfulness: {'✓' if check.faithfulness_passed else '✗'}"
            f" | Confidence: {'✓' if check.confidence_passed else '✗'}"
        )

        return answer, confidence, sources, page_image

    except ValueError as exc:
        return f"❌ Validation error: {exc}", "0%", "", None

    except Exception as exc:
        return f"❌ Pipeline error: {exc}", "0%", "", None


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        placeholder="Ask about Vision Transformers...",
        label="Your Question",
        lines=3,
    ),
    outputs=[
        gr.Textbox(label="Answer", lines=10),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Sources"),
        gr.Image(label="Top Retrieved Page"),
    ],
    title="🔬 Scientific Multimodal RAG",
    description=(
        "Ask questions about Vision Transformer research papers. "
        "Powered by ColPali + SciNCL + Qwen2-VL"
    ),
    examples=[
        "What is Vision Transformer?",
        "How does the patch embedding work in ViT?",
        "What is the difference between ViT and DeiT?",
        "Explain the attention mechanism in Vision Transformers",
        "What does Figure 3 in the ViT paper show?",
    ],
    allow_flagging="never",
    theme=gr.themes.Soft(),
)


if __name__ == "__main__":
    demo.launch(share=True)
