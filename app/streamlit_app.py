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

# Activate permanent HuggingFace cache BEFORE any model imports
from src.utils import model_cache  # noqa: F401


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
    if st.button("Submit", type="primary", width='stretch'):
        if not question or not question.strip():
            st.error("Please enter a question.")
            return

        # Beautiful dynamic workflow tracking
        status_box = st.container()
        with status_box:
            st.markdown("""
            <div style='background-color: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 20px;'>
                <h4 style='color: #38bdf8; margin-top: 0;'>🔄 Multimodal RAG Pipeline Execution Flow</h4>
            </div>
            """, unsafe_allow_html=True)
            progress_bar = st.progress(0)
            status_text = st.empty()
            logs_title = st.markdown("**📜 Pipeline Steps & Detailed Time Logs:**")
            logs_area = st.empty()
            
        logs = []
        t_start = time.time()
        
        def update_status_ui(step_name, description, progress_val):
            elapsed_current = time.time() - t_start
            emoji_map = {
                "validate": "🔍",
                "colpali_encode": "🖼️",
                "scincl_encode": "🧬",
                "retrieval": "📚",
                "fusion": "🔗",
                "context": "📦",
                "qwen_generate": "🤖",
                "self_check": "🛡️"
            }
            emoji = emoji_map.get(step_name, "⚙️")
            log_line = f"`[{elapsed_current:4.1f}s]` {emoji} **{step_name.upper()}**: {description}"
            logs.append(log_line)
            progress_bar.progress(progress_val)
            status_text.markdown(f"Current State: **{description}**")
            logs_area.markdown("\n".join(logs))

        try:
            pipeline = _get_pipeline()
            result = pipeline.query(question, status_callback=update_status_ui)
            elapsed = time.time() - t_start
            
            # Show completed state in UI
            progress_bar.progress(100)
            status_text.success(f"🎉 Pipeline Execution Complete in {elapsed:.1f}s!")
        except ValueError as exc:
            st.error(f"Validation error: {exc}")
            return
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            import traceback
            st.code(traceback.format_exc())
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
        if not check:
            st.warning("Self-check could not be performed (generation failed or returned None).")
        else:
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
                            st.image(img, caption=f"Page {src.page_numbers[0]}", width='stretch')
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
# Ingest Document tab
# ---------------------------------------------------------------------------

def _render_ingest_tab() -> None:
    """Render the document ingestion interface."""
    st.header("📤 Ingest New Scientific Document")
    st.caption("Upload a PDF to parse it, chunk the text, generate page multivectors (ColPali) + dense text embeddings (SciNCL), and store them persistently.")

    uploaded_file = st.file_uploader(
        "Upload PDF file",
        type=["pdf"],
        help="Upload a scientific article or paper PDF",
    )

    if uploaded_file is not None:
        st.info(f"File selected: **{uploaded_file.name}**")
        doc_id = Path(uploaded_file.name).stem.replace(" ", "_")
        
        # Ingest button
        if st.button("🚀 Process & Ingest Document", type="primary", width='stretch'):
            # Rich visual logger panel
            status_box = st.container()
            with status_box:
                st.markdown("""
                <div style='background-color: #0f172a; padding: 15px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 20px;'>
                    <h4 style='color: #38bdf8; margin-top: 0;'>🔄 Multimodal Ingestion Pipeline Flow</h4>
                </div>
                """, unsafe_allow_html=True)
                progress_bar = st.progress(0)
                status_text = st.empty()
                logs_title = st.markdown("**📜 Pipeline Steps & Detailed Time Logs:**")
                logs_area = st.empty()
            
            logs = []
            t_start = time.time()
            
            def log_step(step_name, description, progress_val, emoji="⚙️"):
                elapsed = time.time() - t_start
                log_line = f"`[{elapsed:4.1f}s]` {emoji} **{step_name.upper()}**: {description}"
                logs.append(log_line)
                progress_bar.progress(progress_val)
                status_text.markdown(f"Current State: **{description}**")
                logs_area.markdown("\n".join(logs))
            
            try:
                # Step 1: Save uploaded file locally
                log_step("save_pdf", "Saving uploaded PDF locally to data/raw/...", 10, "💾")
                pdf_path = _project_root / "data" / "raw" / uploaded_file.name
                pdf_path.parent.mkdir(parents=True, exist_ok=True)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Step 2: Load merged config
                log_step("load_config", "Loading and resolving application configuration paths...", 20, "⚙️")
                from src.utils.config_loader import load_config, resolve_paths
                data_config = load_config("data_config")
                model_config = load_config("model_config")
                retrieval_config = load_config("retrieval_config")
                merged_config = {
                    "data": data_config.get("data", {}),
                    "paths": data_config.get("paths", {}),
                    "parsing": data_config.get("parsing", {}),
                    "models": model_config.get("models", {}),
                    "retrieval": retrieval_config.get("retrieval", {}),
                }
                merged_config = resolve_paths(merged_config)
                
                # Step 3: Parse PDF (DualPDFParser)
                log_step("parse_pdf", "Parsing PDF pages into high-resolution images & markdown text...", 35, "📄")
                from pipelines.offline_pipeline import DualPDFParser
                parser = DualPDFParser(
                    output_pages_dir=merged_config["paths"].get("parsed_pages", "data/parsed/pages/"),
                    output_markdown_dir=merged_config["paths"].get("parsed_markdown", "data/parsed/markdown/"),
                    dpi=merged_config["parsing"].get("dpi", 200),
                    image_format=merged_config["parsing"].get("image_format", "PNG"),
                )
                parse_result = parser.parse(pdf_path=str(pdf_path), doc_id=doc_id)
                if parse_result["status"] != "success":
                    st.error("Failed to parse PDF document. Please check the logs.")
                    return
                
                num_pages = parse_result["num_pages"]
                log_step("parse_pdf", f"Successfully parsed {num_pages} pages into high-res images.", 45, "✅")
                
                # Step 4: ColPali Embedding
                log_step("colpali_embed", f"Loading ColPali-v1.2 model on CPU/GPU & embedding {num_pages} pages...", 50, "🖼️")
                import torch
                from src.embeddings.colpali_embedder import ColPaliEmbedder
                from PIL import Image
                colpali_cfg = merged_config["models"].get("colpali", {})
                device = "cuda" if torch.cuda.is_available() else "cpu"
                colpali = ColPaliEmbedder(
                    model_name=colpali_cfg.get("model_name", "vidore/colpali-v1.2"),
                    device=device,
                    torch_dtype="float32" if device == "cpu" else "float16",
                    max_pages_per_batch=colpali_cfg.get("max_pages_per_batch", 4),
                )
                colpali.load()
                
                npy_dir = Path(merged_config["paths"].get("multivectors", "data/indices/multivectors/"))
                npy_dir.mkdir(parents=True, exist_ok=True)
                
                log_step("colpali_embed", "Processing page vision embeddings batch-by-batch...", 60, "🖼️")
                page_images = [Image.open(img_path).convert("RGB") for img_path in parse_result["page_images"]]
                outputs = colpali.embed_batch(page_images, item_type="image")
                for i, output in enumerate(outputs):
                    page_num = i + 1
                    npy_path = npy_dir / f"{doc_id}_page_{page_num}.npy"
                    output.doc_id = doc_id
                    output.page_num = page_num
                    colpali.save_vectors(output, str(npy_path))
                
                colpali.unload()
                log_step("colpali_embed", "Saved and persistent vision multivectors completely stored.", 70, "✅")
                
                # Step 5: SciNCL Embedding & ChromaDB Indexing
                log_step("scincl_embed", "Loading SciNCL dense scientific text model and tokenizer...", 75, "🧬")
                from src.embeddings.scincl_embedder import SciNCLEmbedder
                scincl_cfg = merged_config["models"].get("scincl", {})
                scincl = SciNCLEmbedder(
                    model_name=scincl_cfg.get("model_name", "malteos/scincl"),
                    device=device,
                    max_length=scincl_cfg.get("max_length", 512),
                )
                scincl.load()
                
                chroma_dir = Path(merged_config["paths"].get("chroma_index", "data/indices/chroma_index/"))
                collection_name = merged_config["retrieval"].get("chroma_collection", "sci_text")
                
                new_page_metadata = {}
                log_step("scincl_embed", f"Upserting scientific sentence chunks to persistent ChromaDB persistent client...", 85, "🧬")
                for i, img_path in enumerate(parse_result["page_images"]):
                    page_num = i + 1
                    page_key = f"{doc_id}_page_{page_num}"
                    text = parse_result["page_texts"][i] if i < len(parse_result["page_texts"]) else ""
                    new_page_metadata[page_key] = {
                        "doc_id": doc_id,
                        "page_num": page_num,
                        "image_path": str(img_path),
                        "text": text,
                        "paper_title": doc_id,
                    }
                    if text.strip():
                        output = scincl.embed_text(text)
                        output.doc_id = doc_id
                        output.page_num = page_num
                        output.metadata["paper_title"] = doc_id
                        output.metadata["text"] = text[:500]
                        scincl.save_to_chromadb(output, collection_name=collection_name, persist_dir=str(chroma_dir))
                
                scincl.unload()
                log_step("scincl_embed", "Successfully cached and committed dense chunks to ChromaDB.", 90, "✅")
                
                # Step 6: Update Metadata Files
                log_step("update_metadata", "Writing global document metadata and local index mappings...", 95, "🗃️")
                doc_mapping_path = Path(merged_config["paths"].get("doc_mapping", "data/indices/doc_mapping.json"))
                if doc_mapping_path.exists():
                    with open(doc_mapping_path, "r", encoding="utf-8") as f:
                        doc_mapping = json.load(f)
                else:
                    doc_mapping = {}
                    
                doc_mapping[doc_id] = {
                    "arxiv_id": doc_id,
                    "title": doc_id,
                    "url": "",
                    "pdf_path": str(pdf_path),
                    "num_pages": parse_result["num_pages"],
                    "page_images": parse_result["page_images"],
                    "authors": "User Uploaded PDF",
                    "published": datetime.now().isoformat(),
                    "status": "success",
                }
                doc_mapping_path.parent.mkdir(parents=True, exist_ok=True)
                with open(doc_mapping_path, "w", encoding="utf-8") as f:
                    json.dump(doc_mapping, f, indent=2, ensure_ascii=False)
                    
                page_metadata_path = Path(merged_config["paths"].get("page_metadata", "data/indices/page_metadata.json"))
                if page_metadata_path.exists():
                    with open(page_metadata_path, "r", encoding="utf-8") as f:
                        page_metadata = json.load(f)
                else:
                    page_metadata = {}
                    
                page_metadata.update(new_page_metadata)
                with open(page_metadata_path, "w", encoding="utf-8") as f:
                    json.dump(page_metadata, f, indent=2, ensure_ascii=False)
                
                progress_bar.progress(100)
                elapsed_total = time.time() - t_start
                status_text.success(f"🎉 Ingestion complete! {num_pages} pages parsed & persistently indexed in {elapsed_total:.1f}s.")
                
            except Exception as e:
                st.error(f"Ingestion failed with error: {e}")
                import traceback
                st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Browse Chunks tab
# ---------------------------------------------------------------------------

def _render_chunks_tab() -> None:
    """Render the chunk explorer interface."""
    st.header("🗂️ Persistent Chunk Store")
    st.caption("Inspect documents and persistent text chunks currently saved in ChromaDB and page metadata mappings.")
    
    # Load document metadata mapping
    doc_mapping_path = _project_root / "data" / "indices" / "doc_mapping.json"
    if doc_mapping_path.exists():
        with open(doc_mapping_path, "r", encoding="utf-8") as f:
            doc_mapping = json.load(f)
        
        st.subheader(f"📚 Stored Documents ({len(doc_mapping)} papers)")
        
        for doc_id, info in doc_mapping.items():
            with st.expander(f"📄 {info.get('title', doc_id)}", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**Document ID:** `{doc_id}`")
                    st.write(f"**Total Pages:** {info.get('num_pages', 0)}")
                    st.write(f"**Local Path:** `{info.get('pdf_path', 'N/A')}`")
                    st.write(f"**Uploaded/Ingested:** {info.get('published', 'N/A')}")
                with col2:
                    st.metric("Status", info.get("status", "N/A").upper())
                
                # Render mini page explorer
                page_num = st.number_input(f"Preview page image for {doc_id}:", min_value=1, max_value=info.get('num_pages', 1), value=1, key=f"preview_page_{doc_id}")
                page_key = f"{doc_id}_page_{page_num}"
                
                # Check page_metadata for details
                page_metadata_path = _project_root / "data" / "indices" / "page_metadata.json"
                if page_metadata_path.exists():
                    with open(page_metadata_path, "r", encoding="utf-8") as f:
                        page_metadata = json.load(f)
                    
                    page_info = page_metadata.get(page_key)
                    if page_info:
                        col_img, col_txt = st.columns([1, 1])
                        with col_img:
                            img_path = Path(page_info.get("image_path", ""))
                            # Resolve path relative to project root if absolute doesn't work
                            if not img_path.exists():
                                img_path = _project_root / img_path
                            if img_path.exists():
                                st.image(str(img_path), caption=f"Page {page_num}", width='stretch')
                            else:
                                st.warning(f"Page image file not found: `{img_path}`")
                        with col_txt:
                            st.write("**Extracted Page Chunk:**")
                            st.text_area("Chunk text content", page_info.get("text", "(no text)"), height=300, key=f"txt_{page_key}")
                    else:
                        st.info("No detailed page info found.")
    else:
        st.info("No documents have been ingested yet. Use the **Ingest Document** tab to upload and process your first scientific PDF.")

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

    tab_query, tab_ingest, tab_chunks, tab_history, tab_eval = st.tabs(
        ["🔍 Query", "📤 Ingest Document", "🗂️ Browse Chunks", "📜 History", "📊 Evaluation"]
    )

    with tab_query:
        _render_query_tab()

    with tab_ingest:
        _render_ingest_tab()

    with tab_chunks:
        _render_chunks_tab()

    with tab_history:
        _render_history_tab()

    with tab_eval:
        _render_evaluation_tab()


if __name__ == "__main__":
    main()
