#!/usr/bin/env python3
"""
Online RAG Pipeline — HPC Edition
===================================
Converted from: Online_Rag_pipeline_with_modern_gradio.ipynb
Target: IITJ HPC (/scratch/data/divyasaxena_rs/Vineet_internship/)

Key changes vs. Colab notebook:
  1. Full structured logging (file + console) — every step tracked
  2. Paths read from ENV vars so no hardcoding
  3. Gradio launched headless (no share=True tunnel needed — use SSH port-forward)
  4. Data loaded from pre-extracted sci-rag-pages/ & sci-rag-indices/ dirs
  5. HuggingFace cache pinned to /scratch to avoid home-dir quota
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

# ─── ENV SETUP (must be first, before torch/HF imports) ──────────────────────
BASE_DIR   = Path(os.getenv("RAG_BASE_DIR",
                            "/scratch/data/divyasaxena_rs/Vineet_internship"))
DATA_DIR   = BASE_DIR / "data"
PAGES_DIR  = DATA_DIR / "sci-rag-pages"
INDEX_DIR  = DATA_DIR / "sci-rag-indices"
LOG_DIR    = BASE_DIR / "logs"
HF_CACHE   = BASE_DIR / ".cache" / "huggingface"

os.environ["HF_HOME"]             = str(HF_CACHE)
os.environ["TRANSFORMERS_CACHE"]  = str(HF_CACHE / "transformers")
os.environ["HF_DATASETS_CACHE"]   = str(HF_CACHE / "datasets")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# create dirs early
for d in [DATA_DIR, PAGES_DIR, INDEX_DIR, LOG_DIR, HF_CACHE]:
    d.mkdir(parents=True, exist_ok=True)


# ─── LOGGING SETUP ───────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    """
    Dual-sink logger:
      • Console  → INFO level, coloured prefix
      • File     → DEBUG level, full timestamp (logs/online_rag_YYYYMMDD_HHMMSS.log)
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"online_rag_{ts}.log"

    fmt_file    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    fmt_console = "%(asctime)s | %(levelname)-8s | %(message)s"

    logging.basicConfig(
        level=logging.DEBUG,
        format=fmt_file,
        handlers=[logging.FileHandler(log_file, encoding="utf-8")]
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(fmt_console, datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(console)

    log = logging.getLogger("rag.main")
    log.info("=" * 70)
    log.info("  Scientific RAG — Online Pipeline  (HPC Edition)")
    log.info("=" * 70)
    log.info(f"  BASE_DIR  : {BASE_DIR}")
    log.info(f"  PAGES_DIR : {PAGES_DIR}")
    log.info(f"  INDEX_DIR : {INDEX_DIR}")
    log.info(f"  HF_CACHE  : {HF_CACHE}")
    log.info(f"  Log file  : {log_file}")
    log.info("=" * 70)
    return log

log = setup_logging()

# ─── STEP 0 : unzip data if not already extracted ────────────────────────────
def maybe_unzip(zip_path: Path, dest: Path, label: str) -> None:
    """Unzip only when dest dir is empty."""
    if not zip_path.exists():
        log.warning(f"[{label}] zip not found at {zip_path} — skipping unzip")
        return
    if any(dest.iterdir()):
        log.info(f"[{label}] already extracted → {dest}")
        return
    log.info(f"[{label}] extracting {zip_path} → {dest} ...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    log.info(f"[{label}] done in {time.time()-t0:.1f}s  "
             f"({len(list(dest.rglob('*')))} files)")

log.info("[STEP 0] Checking / extracting data zips ...")
maybe_unzip(DATA_DIR / "sci-rag-pages.zip",   PAGES_DIR, "pages")
maybe_unzip(DATA_DIR / "sci-rag-indices.zip",  INDEX_DIR, "indices")


# ─── STEP 1 : imports (after env vars set) ───────────────────────────────────
log.info("[STEP 1] Importing ML libraries ...")
t_import = time.time()

try:
    import torch
    log.info(f"  torch         : {torch.__version__}")
    log.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"  GPU           : {torch.cuda.get_device_name(0)}")
        log.info(f"  VRAM total    : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
except ImportError as e:
    log.error(f"torch import failed: {e}"); sys.exit(1)

try:
    import chromadb
    import numpy as np
    from PIL import Image
    from transformers import AutoTokenizer, AutoModel
    import gradio as gr
    log.info(f"  chromadb      : {chromadb.__version__}")
    log.info(f"  numpy         : {np.__version__}")
    log.info(f"  gradio        : {gr.__version__}")
except ImportError as e:
    log.error(f"Dependency import failed: {e}"); sys.exit(1)

log.info(f"[STEP 1] All imports done in {time.time()-t_import:.1f}s")


# ─── STEP 2 : load SciNCL text retriever ─────────────────────────────────────
log.info("[STEP 2] Loading SciNCL text embedder ...")
t0 = time.time()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"  device = {DEVICE}")

SCINCL_MODEL = os.getenv("SCINCL_MODEL", "malteos/scincl")
try:
    scincl_tokenizer = AutoTokenizer.from_pretrained(
        SCINCL_MODEL, cache_dir=str(HF_CACHE / "transformers")
    )
    scincl_model = AutoModel.from_pretrained(
        SCINCL_MODEL, cache_dir=str(HF_CACHE / "transformers")
    ).to(DEVICE).eval()
    log.info(f"[STEP 2] SciNCL loaded in {time.time()-t0:.1f}s")
except Exception as e:
    log.error(f"[STEP 2] SciNCL load failed: {e}"); raise


def embed_query_scincl(query: str) -> np.ndarray:
    """Embed a text query with SciNCL; returns (768,) float32 array."""
    log.debug(f"embed_query_scincl: '{query[:80]}'")
    tokens = scincl_tokenizer(
        query, return_tensors="pt", truncation=True,
        max_length=512, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        out = scincl_model(**tokens)
    vec = out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    log.debug(f"  → embedding shape {vec.shape}")
    return vec.astype(np.float32)


# ─── STEP 3 : connect ChromaDB ───────────────────────────────────────────────
log.info("[STEP 3] Connecting to ChromaDB ...")
CHROMA_PATH = INDEX_DIR / "chroma_db"
log.info(f"  chroma path: {CHROMA_PATH}")

try:
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collections = chroma_client.list_collections()
    log.info(f"  available collections: {[c.name for c in collections]}")

    TEXT_COLLECTION  = os.getenv("TEXT_COLLECTION",  "scincl_text")
    VISUAL_COLLECTION = os.getenv("VISUAL_COLLECTION", "colpali_visual")

    text_col = chroma_client.get_or_create_collection(TEXT_COLLECTION)
    log.info(f"  text collection  '{TEXT_COLLECTION}': {text_col.count()} docs")
except Exception as e:
    log.error(f"[STEP 3] ChromaDB error: {e}"); raise


# ─── STEP 4 : load page metadata ─────────────────────────────────────────────
log.info("[STEP 4] Loading page metadata ...")
import json, pickle

def load_pages_metadata() -> dict:
    """Try JSON then pickle; return {page_id: metadata_dict}."""
    for fname in ["pages_metadata.json", "pages.json", "metadata.json"]:
        p = PAGES_DIR / fname
        if p.exists():
            log.info(f"  loading metadata from {p}")
            with open(p) as f:
                data = json.load(f)
            log.info(f"  loaded {len(data)} page records")
            return data
    for fname in ["pages_metadata.pkl", "pages.pkl"]:
        p = PAGES_DIR / fname
        if p.exists():
            log.info(f"  loading metadata from {p}")
            with open(p, "rb") as f:
                data = pickle.load(f)
            log.info(f"  loaded {len(data)} page records")
            return data
    log.warning("  no metadata file found — using empty dict")
    return {}

pages_meta = load_pages_metadata()


# ─── STEP 5 : load Qwen2-VL generator ────────────────────────────────────────
log.info("[STEP 5] Loading Qwen2-VL generator ...")
t0 = time.time()
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2-VL-7B-Instruct")

try:
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    qwen_processor = AutoProcessor.from_pretrained(
        QWEN_MODEL, cache_dir=str(HF_CACHE / "transformers")
    )
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto" if DEVICE == "cuda" else None,
        cache_dir=str(HF_CACHE / "transformers"),
    )
    if DEVICE != "cuda":
        qwen_model = qwen_model.to(DEVICE)
    qwen_model.eval()
    log.info(f"[STEP 5] Qwen2-VL loaded in {time.time()-t0:.1f}s")
except Exception as e:
    log.error(f"[STEP 5] Qwen2-VL load failed: {e}")
    log.warning("  Falling back to text-only mode (no VLM generation)")
    qwen_model = None
    qwen_processor = None


# ─── STEP 6 : retrieval helper ────────────────────────────────────────────────
def retrieve_pages(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve top-k pages from ChromaDB using SciNCL text embedding.
    Returns list of dicts with keys: page_id, score, metadata.
    """
    log.info(f"[RETRIEVE] query='{query[:60]}...'  top_k={top_k}")
    t0 = time.time()

    q_vec = embed_query_scincl(query)
    results = text_col.query(
        query_embeddings=[q_vec.tolist()],
        n_results=min(top_k, text_col.count() or 1),
    )

    pages = []
    ids       = results.get("ids",       [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for pid, dist, meta in zip(ids, distances, metadatas):
        score = max(0.0, 1.0 - dist)           # cosine dist → similarity
        pages.append({"page_id": pid, "score": score, "metadata": meta or {}})
        log.debug(f"  page_id={pid}  score={score:.3f}  "
                  f"paper={meta.get('paper_title','?')[:40]}")

    log.info(f"[RETRIEVE] {len(pages)} pages in {time.time()-t0:.2f}s")
    return pages


# ─── STEP 7 : generation helper ───────────────────────────────────────────────
IN_SCOPE_KEYWORDS = [
    "vision transformer", "vit", "swin", "bert", "attention",
    "patch embedding", "self-attention", "image classification",
    "object detection", "transformer", "neural network", "deep learning",
    "dataset", "training", "fine-tun", "arxiv", "paper",
]

def is_in_scope(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in IN_SCOPE_KEYWORDS)


def generate_answer(query: str, pages: list[dict]) -> str:
    """Call Qwen2-VL or fall back to extractive answer from metadata."""
    log.info(f"[GENERATE] building context from {len(pages)} pages")
    t0 = time.time()

    # Build text context from retrieved metadata snippets
    context_parts = []
    for i, p in enumerate(pages[:3], 1):
        meta = p["metadata"]
        snippet = meta.get("text_snippet", meta.get("text", ""))[:400]
        title   = meta.get("paper_title", "Unknown Paper")
        page_no = meta.get("page_number", "?")
        context_parts.append(
            f"[Source {i}] {title} — Page {page_no}\n{snippet}"
        )
    context = "\n\n".join(context_parts)

    if qwen_model is None:
        # Text-only fallback
        log.warning("[GENERATE] No VLM — using extractive fallback")
        answer = (
            f"Based on retrieved documents:\n\n{context}\n\n"
            f"(Note: VLM unavailable on this node — GPU required)"
        )
        log.info(f"[GENERATE] fallback done in {time.time()-t0:.2f}s")
        return answer

    # Full Qwen2-VL generation
    prompt = (
        f"You are a scientific assistant. Answer the question strictly "
        f"based on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    try:
        text_input = qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = qwen_processor(text=[text_input], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out_ids = qwen_model.generate(
                **inputs, max_new_tokens=512, temperature=0.1, do_sample=False
            )
        answer = qwen_processor.decode(
            out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        log.info(f"[GENERATE] VLM answer ({len(answer)} chars) in {time.time()-t0:.2f}s")
        return answer
    except Exception as e:
        log.error(f"[GENERATE] VLM error: {e}")
        return f"Generation failed: {e}\n\nContext:\n{context}"


# ─── STEP 8 : main RAG function ───────────────────────────────────────────────
def rag_query(query: str, top_k: int = 5) -> dict:
    """
    Full RAG pipeline:
      query → [scope check] → [SciNCL embed] → [ChromaDB retrieve]
            → [Qwen2-VL generate] → structured result dict
    """
    log.info("─" * 60)
    log.info(f"[RAG] new query: '{query}'")
    t_total = time.time()

    result = {
        "query":       query,
        "is_from_docs": False,
        "answer":      "",
        "sources":     [],
        "confidence":  0.0,
        "latency_s":   0.0,
    }

    if not query.strip():
        log.warning("[RAG] empty query")
        result["answer"] = "Please enter a question."
        return result

    in_scope = is_in_scope(query)
    log.info(f"[RAG] scope check → {'IN-SCOPE' if in_scope else 'OUT-OF-SCOPE'}")
    result["is_from_docs"] = in_scope

    if not in_scope:
        result["answer"] = (
            "⚠️ This question appears to be outside the scope of the "
            "ingested scientific papers. Please ask about Vision Transformers, "
            "attention mechanisms, or related ML research."
        )
        result["latency_s"] = round(time.time() - t_total, 2)
        log.info(f"[RAG] out-of-scope → done in {result['latency_s']}s")
        return result

    pages   = retrieve_pages(query, top_k=top_k)
    answer  = generate_answer(query, pages)

    # Build source list for UI
    sources = []
    for p in pages:
        meta = p["metadata"]
        sources.append({
            "page_id":     p["page_id"],
            "score":       round(p["score"], 3),
            "paper_title": meta.get("paper_title", "Unknown"),
            "page_number": meta.get("page_number", "?"),
            "arxiv_id":    meta.get("arxiv_id", ""),
            "snippet":     meta.get("text_snippet", meta.get("text",""))[:200],
        })

    avg_score = sum(s["score"] for s in sources) / len(sources) if sources else 0.0
    result["answer"]     = answer
    result["sources"]    = sources
    result["confidence"] = round(avg_score, 3)
    result["latency_s"]  = round(time.time() - t_total, 2)

    log.info(f"[RAG] confidence={avg_score:.3f}  "
             f"sources={len(sources)}  latency={result['latency_s']}s")
    return result


# ─── STEP 9 : Gradio UI ──────────────────────────────────────────────────────
log.info("[STEP 9] Building Gradio UI ...")

def gradio_rag(query: str):
    """Gradio-facing wrapper — returns (answer_html, conf_html, sources_html)."""
    result = rag_query(query)

    # Answer card
    if result["is_from_docs"]:
        ans_html = f"""
        <div style="background:#f0f9ff;border-left:4px solid #2563eb;
                    border-radius:8px;padding:16px;font-size:15px;line-height:1.6">
            {result['answer'].replace(chr(10),'<br>')}
        </div>"""
    else:
        ans_html = f"""
        <div style="background:#fffbeb;border-left:4px solid #f59e0b;
                    border-radius:8px;padding:16px;font-size:15px">
            <strong>⚠️ Out-of-scope:</strong><br>{result['answer']}
        </div>"""

    # Confidence card
    pct  = int(result["confidence"] * 100)
    col  = "#16a34a" if pct > 60 else "#f59e0b" if pct > 30 else "#dc2626"
    conf_html = f"""
    <div style="text-align:center;padding:12px">
        <div style="font-size:36px;font-weight:800;color:{col}">{pct}%</div>
        <div style="color:#6b7280;font-size:13px">retrieval confidence</div>
        <div style="color:#9ca3af;font-size:12px;margin-top:4px">
            ⏱ {result['latency_s']}s
        </div>
    </div>"""

    # Sources panel
    if result["sources"]:
        cards = ""
        for i, s in enumerate(result["sources"], 1):
            arxiv = (f'<a href="https://arxiv.org/abs/{s["arxiv_id"]}" '
                     f'target="_blank">🔗 arXiv</a>' if s["arxiv_id"] else "")
            cards += f"""
            <div style="border:1px solid #e5e7eb;border-radius:8px;
                        padding:12px;margin-bottom:8px;background:white">
                <div style="font-weight:600;color:#1e40af;font-size:13px">
                    [{i}] {s['paper_title'][:60]} — p.{s['page_number']}
                    &nbsp;{arxiv}
                </div>
                <div style="color:#4b5563;font-size:12px;margin-top:4px">
                    Score: {s['score']:.3f}
                </div>
                <div style="color:#6b7280;font-size:12px;margin-top:4px;
                            font-style:italic">
                    {s['snippet'][:150]}...
                </div>
            </div>"""
        src_html = f"""
        <div style="background:#f9fafb;border-radius:10px;padding:12px">
            <div style="font-weight:700;margin-bottom:8px">
                📚 Sources ({len(result['sources'])})
            </div>{cards}
        </div>"""
    else:
        src_html = "<div style='color:#dc2626;padding:12px'>🚫 No sources found.</div>"

    return ans_html, conf_html, src_html


GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Scientific RAG — HPC"
) as demo:

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px">
        <h1 style="font-size:28px;font-weight:800;color:#1e3a5f">
            🔬 Scientific RAG System — IITJ HPC
        </h1>
        <p style="color:#6b7280;font-size:14px">
            ColPali + SciNCL + Qwen2-VL &nbsp;|&nbsp;
            Indexed 10 Vision Transformer papers
        </p>
    </div>""")

    with gr.Row():
        with gr.Column(scale=5):
            q_box = gr.Textbox(
                label="Question",
                placeholder="e.g. How does patch embedding work in ViT?",
                lines=2,
            )
        with gr.Column(scale=1, min_width=120):
            ask_btn   = gr.Button("🔍 Ask",   variant="primary")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary")

    gr.Examples(
        examples=[
            ["What is Vision Transformer (ViT)?"],
            ["How does self-attention work in transformers?"],
            ["What datasets were used to evaluate Swin Transformer?"],
            ["What is the capital of France?"],   # out-of-scope test
        ],
        inputs=q_box,
        label="💡 Examples (last one is out-of-scope)"
    )

    with gr.Row():
        with gr.Column(scale=3):
            ans_out  = gr.HTML("<p style='color:#9ca3af;padding:20px'>Answer appears here...</p>")
        with gr.Column(scale=2):
            conf_out = gr.HTML("<p style='color:#9ca3af;padding:12px'>Confidence...</p>")
            src_out  = gr.HTML("<p style='color:#9ca3af;padding:12px'>Sources...</p>")

    ask_btn.click(fn=gradio_rag, inputs=[q_box], outputs=[ans_out, conf_out, src_out])
    q_box.submit(fn=gradio_rag,  inputs=[q_box], outputs=[ans_out, conf_out, src_out])
    clear_btn.click(
        fn=lambda: (
            "<p style='color:#9ca3af;padding:20px'>Answer appears here...</p>",
            "<p style='color:#9ca3af;padding:12px'>Confidence...</p>",
            "<p style='color:#9ca3af;padding:12px'>Sources...</p>",
            "",
        ),
        outputs=[ans_out, conf_out, src_out, q_box]
    )


# ─── LAUNCH ──────────────────────────────────────────────────────────────────
log.info("[LAUNCH] Starting Gradio ...")
log.info(f"  port          : {GRADIO_PORT}")
log.info(f"  SSH tunnel cmd: ssh -L {GRADIO_PORT}:localhost:{GRADIO_PORT} "
         f"divyasaxena_rs@172.25.0.81")
log.info(f"  Browser URL   : http://localhost:{GRADIO_PORT}")

demo.launch(
    server_name="0.0.0.0",   # listen on all interfaces
    server_port=GRADIO_PORT,
    share=False,              # no Gradio tunnel — use SSH port-forward instead
    debug=False,
    show_error=True,
)
