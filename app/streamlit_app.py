"""
Streamlit App — Scientific Multimodal RAG
Full-featured interface with settings sidebar, history, and evaluation.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Ensure project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Scientific Multimodal RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Initialise Streamlit session state with default values."""
    defaults = {
        "history": [],
        "pipeline": None,
        "temperature": 0.3,
        "top_k": 5,
        "colpali_weight": 0.7,
        "scincl_weight": 0.3,
        "confidence_threshold": 0.6,
        "max_retries": 2,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# ---------------------------------------------------------------------------
# Pipeline initialisation
# ---------------------------------------------------------------------------

def _get_pipeline():
    """Lazily initialise the OnlinePipeline.

    Returns:
        An OnlinePipeline instance ready for querying.
    """
    if st.session_state.pipeline is None:
        from pipelines.online_pipeline import OnlinePipeline

        config_path = str(_project_root / "configs" / "pipeline_config.yaml")
        with st.spinner("Loading pipeline... This may take a moment on first run."):
            st.session_state.pipeline = OnlinePipeline(config_path=config_path)
    return st.session_state.pipeline


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    """Render the settings sidebar."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # Generation settings
        st.subheader("Generation")
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            help="Lower = more factual, Higher = more creative",
        )
        st.session_state.max_retries = st.slider(
            "Max retries",
            min_value=0,
            max_value=5,
            value=st.session_state.max_retries,
            help="Number of retry attempts if confidence is below threshold",
        )

        # Retrieval settings
        st.subheader("Retrieval")
        st.session_state.top_k = st.slider(
            "Top-K results",
            min_value=1,
            max_value=20,
            value=st.session_state.top_k,
            help="Number of documents to retrieve",
        )

        # Fusion weights
        st.subheader("Fusion Weights")
        st.session_state.colpali_weight = st.slider(
            "ColPali (vision) weight",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.colpali_weight,
            step=0.05,
            help="Weight for ColPali visual retrieval scores",
        )
        st.session_state.scincl_weight = round(
            1.0 - st.session_state.colpali_weight, 2
        )
        st.info(
            f"SciNCL (text) weight: **{st.session_state.scincl_weight:.2f}**\n\n"
            "Weights must sum to 1.0"
        )

        # Self-check settings
        st.subheader("Self-Check")
        st.session_state.confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_threshold,
            step=0.05,
            help="Minimum confidence for answer acceptance",
        )

        # Info
        st.divider()
        st.caption(
            "🔬 **Scientific Multimodal RAG**\n\n"
            "ColPali (vision) + SciNCL (text) + Qwen2-VL (generation)\n\n"
            "Optimized for Kaggle P100 GPU"
        )


# ---------------------------------------------------------------------------
# Main area — Query tab
# ---------------------------------------------------------------------------

def _render_query_tab() -> None:
    """Render the main query interface."""
    st.header("Ask a Question")

    # Query input
    question = st.text_input(
        "Your question about Vision Transformer research:",
        placeholder="e.g., What is the Vision Transformer architecture?",
    )

    # Example queries
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧠 What is ViT?"):
            question = "What is the Vision Transformer?"
    with col2:
        if st.button("📊 Patch Embedding"):
            question = "How does the patch embedding work in ViT?"
    with col3:
        if st.button("🔄 ViT vs DeiT"):
            question = "What is the difference between ViT and DeiT?"

    # Process query
    if st.button("Submit", type="primary", use_container_width=True):
        if not question or not question.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Processing your question..."):
            try:
                pipeline = _get_pipeline()
                t_start = time.time()
                result = pipeline.query(question)
                elapsed = time.time() - t_start
            except ValueError as exc:
                st.error(f"Validation error: {exc}")
                return
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                return

        # Save to history
        st.session_state.history.append({
            "question": question,
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": [
                {
                    "title": s.paper_title,
                    "page_numbers": s.page_numbers,
                    "url": s.arxiv_url,
                    "score": s.relevance_score,
                }
                for s in result.sources
            ],
            "check_passed": result.check_result.passed,
            "time": elapsed,
            "retries": result.retries,
            "timestamp": datetime.now().isoformat(),
        })

        # Display answer
        st.subheader("Answer")
        if result.answer:
            st.markdown(result.answer)
        else:
            st.warning("No answer was generated.")

        # Confidence and timing
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            conf_color = "green" if result.confidence >= 0.6 else "red"
            st.metric("Confidence", f"{result.confidence:.0%}")
        with col_b:
            st.metric("Time", f"{elapsed:.1f}s")
        with col_c:
            check_emoji = "✅" if result.check_result.passed else "❌"
            st.metric("Self-Check", f"{check_emoji} {'PASS' if result.check_result.passed else 'FAIL'}")

        # Self-check details
        check = result.check_result
        with st.expander("Self-Check Details", expanded=not check.passed):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(
                    f"**Attribution**: {'✅ PASS' if check.attribution_passed else '❌ FAIL'}"
                )
            with c2:
                st.markdown(
                    f"**Faithfulness**: {'✅ PASS' if check.faithfulness_passed else '❌ FAIL'}"
                )
            with c3:
                st.markdown(
                    f"**Confidence**: {'✅ PASS' if check.confidence_passed else '❌ FAIL'}"
                    f" ({check.confidence:.0%})"
                )
            st.caption(check.details)

        # Source cards
        st.subheader("Sources")
        if result.sources:
            for i, src in enumerate(result.sources, 1):
                with st.container():
                    cols = st.columns([3, 1])
                    with cols[0]:
                        st.markdown(f"**{i}. {src.paper_title}**")
                        st.caption(f"Pages: {src.page_numbers}")
                        if src.text_snippet:
                            st.text(src.text_snippet[:200] + "..." if len(src.text_snippet) > 200 else src.text_snippet)
                    with cols[1]:
                        st.metric("Score", f"{src.relevance_score:.3f}")
                        st.markdown(f"[arXiv]({src.arxiv_url})")
                    st.divider()

                    # Show page image if available
                    if src.page_images:
                        for img in src.page_images[:1]:
                            st.image(img, caption=f"Page {src.page_numbers[0]}", use_container_width=True)
        else:
            st.info("No sources found for this query.")


# ---------------------------------------------------------------------------
# History tab
# ---------------------------------------------------------------------------

def _render_history_tab() -> None:
    """Render the query history tab."""
    st.header("Query History")

    if not st.session_state.history:
        st.info("No queries yet. Ask a question to see your history here.")
        return

    for i, entry in enumerate(reversed(st.session_state.history)):
        idx = len(st.session_state.history) - i
        with st.expander(
            f"#{idx} — {entry['question'][:60]}... "
            f"({entry.get('timestamp', 'N/A')[:19]})",
            expanded=False,
        ):
            st.markdown(f"**Question**: {entry['question']}")
            st.markdown(f"**Answer**: {entry['answer'][:300]}..." if len(entry.get("answer", "")) > 300 else f"**Answer**: {entry.get('answer', 'N/A')}")
            st.caption(
                f"Confidence: {entry.get('confidence', 0):.0%} | "
                f"Time: {entry.get('time', 0):.1f}s | "
                f"Check: {'PASS' if entry.get('check_passed', False) else 'FAIL'}"
            )

    # Clear history button
    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------

def _render_evaluation_tab() -> None:
    """Render the evaluation tab."""
    st.header("Evaluation")

    st.markdown("""
    ### Run Evaluation

    Evaluate the RAG pipeline against a set of ground-truth questions and answers.

    **Ground Truth Format** (JSON):
    ```json
    [
      {
        "question": "What is Vision Transformer?",
        "answer": "A model that applies transformer architecture to image patches...",
        "source_pages": ["2010.11929_page_1"]
      }
    ]
    ```
    """)

    # Upload ground truth
    uploaded_file = st.file_uploader(
        "Upload ground truth JSON file",
        type=["json"],
        help="Upload a JSON file with ground truth Q&A pairs",
    )

    if uploaded_file is not None:
        try:
            ground_truth = json.loads(uploaded_file.read().decode("utf-8"))
            st.success(f"Loaded {len(ground_truth)} ground truth entries.")
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON file: {exc}")
            ground_truth = None
    else:
        ground_truth = None

    if st.button("Run Evaluation", disabled=ground_truth is None):
        if ground_truth is None:
            st.error("Please upload a ground truth file first.")
            return

        pipeline = _get_pipeline()
        results = []
        progress = st.progress(0)

        for i, gt in enumerate(ground_truth):
            question = gt.get("question", "")
            if not question:
                continue

            try:
                result = pipeline.query(question)
                results.append({
                    "question": question,
                    "predicted": result.answer,
                    "expected": gt.get("answer", ""),
                    "confidence": result.confidence,
                    "check_passed": result.check_result.passed,
                })
            except Exception as exc:
                results.append({
                    "question": question,
                    "predicted": f"ERROR: {exc}",
                    "expected": gt.get("answer", ""),
                    "confidence": 0.0,
                    "check_passed": False,
                })

            progress.progress((i + 1) / len(ground_truth))

        # Display evaluation results
        st.subheader("Results")
        passed_count = sum(1 for r in results if r["check_passed"])
        avg_confidence = sum(r["confidence"] for r in results) / max(len(results), 1)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Questions", len(results))
        with c2:
            st.metric("Self-Check Pass Rate", f"{passed_count / max(len(results), 1):.0%}")
        with c3:
            st.metric("Avg Confidence", f"{avg_confidence:.0%}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Render the Streamlit application."""
    _render_sidebar()

    st.title("🔬 Scientific Multimodal RAG")
    st.caption(
        "Ask questions about Vision Transformer research papers. "
        "Powered by ColPali + SciNCL + Qwen2-VL"
    )

    tab_query, tab_history, tab_eval = st.tabs(
        ["🔍 Query", "📜 History", "📊 Evaluation"]
    )

    with tab_query:
        _render_query_tab()

    with tab_history:
        _render_history_tab()

    with tab_eval:
        _render_evaluation_tab()


if __name__ == "__main__":
    main()
