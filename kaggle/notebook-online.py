"""
Kaggle Session 2: Online Pipeline with Gradio Demo
════════════════════════════════════════════════════
GPU: P100 ON | Internet: OFF (uses saved index)
Runtime: ~15 seconds per query

Steps:
1. Install project + load saved index
2. Start Gradio interface
3. Query and get answers interactively
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install dependencies (no internet needed for inference)
# ═══════════════════════════════════════════════════════════════

import subprocess
import sys
import os

def install(package):
    """Install a pip package and print the result."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", package],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"WARNING: Failed to install {package}")
        print(result.stderr[:500])
    else:
        print(f"Installed: {package}")

# Install the project from Kaggle dataset input
code_dir = "/kaggle/input/sci-rag-code"
if os.path.isdir(code_dir):
    install(f"-e {code_dir}")
else:
    install("-e .")

# Core dependencies (should be pre-installed on Kaggle)
install("chromadb")
install("gradio")

# Verify packages
print("\n--- Package Verification ---")
for pkg in ["torch", "transformers", "chromadb", "PIL"]:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} — NOT AVAILABLE")

print("\nCell 1 complete: Dependencies installed.\n")


# ═══════════════════════════════════════════════════════════════
# Cell 2: Restore saved index data
# ═══════════════════════════════════════════════════════════════

import shutil
from pathlib import Path

# Check for saved index from Session 1
index_input = "/kaggle/input/sci-rag-index"
local_index = "data/indices"

if os.path.isdir(index_input):
    print(f"Restoring index from Kaggle dataset: {index_input}")
    os.makedirs(local_index, exist_ok=True)

    # Copy all index files
    for item in Path(index_input).rglob("*"):
        if item.is_file():
            dest = Path(local_index) / item.relative_to(index_input)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(dest))
            print(f"  Restored: {item.name}")

    print(f"\n✓ Index restored to {local_index}")
else:
    print(f"WARNING: Index dataset not found at {index_input}")
    print("  The online pipeline requires pre-built indices from Session 1.")
    print("  Checking local directory…")
    if os.path.isdir(local_index):
        print(f"  ✓ Local index found at {local_index}")
    else:
        print(f"  ✗ No index available — queries will fail.")

# Check for parsed page images
parsed_input = "/kaggle/input/sci-rag-parsed"
local_parsed = "data/parsed"

if os.path.isdir(parsed_input):
    print(f"\nRestoring parsed data from: {parsed_input}")
    if os.path.isdir(local_parsed):
        shutil.rmtree(local_parsed)
    shutil.copytree(parsed_input, local_parsed)
    print(f"✓ Parsed data restored to {local_parsed}")

print("\nCell 2 complete: Index data restored.\n")


# ═══════════════════════════════════════════════════════════════
# Cell 3: GPU info and load online pipeline
# ═══════════════════════════════════════════════════════════════

import torch

print("=== GPU Information ===")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"VRAM: {total_vram:.2f} GB")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected — inference will be very slow!")

print()

from pipelines.online_pipeline import OnlinePipeline

print("Loading online pipeline…")
pipeline = OnlinePipeline(config_path="configs/pipeline_config.yaml")
print("✓ Online pipeline loaded.\n")

# Quick test query
print("Running test query…")
test_result = pipeline.query("What is the Vision Transformer?")
print(f"Test answer: {test_result.answer[:200]}…")
print(f"Confidence: {test_result.confidence:.1%}")
print(f"Time: {test_result.total_time:.2f}s\n")

print("Cell 3 complete: Online pipeline ready.\n")


# ═══════════════════════════════════════════════════════════════
# Cell 4: Launch Gradio demo
# ═══════════════════════════════════════════════════════════════

import gradio as gr

def answer_question(question):
    """Process a question through the RAG pipeline and return formatted results.

    Args:
        question: User's question string.

    Returns:
        Tuple of (answer, confidence, sources) strings for Gradio output.
    """
    if not question or not question.strip():
        return "Please enter a question.", "0.0%", ""

    try:
        result = pipeline.query(question)

        # Format answer
        answer = result.answer if result.answer else "No answer generated."

        # Format confidence
        confidence = f"{result.confidence:.1%}"

        # Format sources
        sources_lines = []
        for s in result.sources:
            title = s.paper_title if s.paper_title else s.paper_id
            pages = ", ".join(str(p) for p in s.page_numbers)
            score = f"{s.relevance_score:.3f}"
            sources_lines.append(
                f"- {title} (Page {pages}) [score: {score}]"
            )
        sources = "\n".join(sources_lines) if sources_lines else "No sources found."

        # Add self-check info
        check_info = ""
        if result.check_result:
            check_info = (
                f"\n\n--- Self-Check ---\n"
                f"Attribution: {'✓' if result.check_result.attribution_passed else '✗'}\n"
                f"Faithfulness: {'✓' if result.check_result.faithfulness_passed else '✗'}\n"
                f"Confidence: {'✓' if result.check_result.confidence_passed else '✗'}\n"
                f"Overall: {'PASS' if result.check_result.passed else 'FAIL'}"
            )
            answer += check_info

        # Add timing info
        answer += f"\n\n--- Timing ---\nQuery time: {result.total_time:.2f}s\nRetries: {result.retries}"

        return answer, confidence, sources

    except ValueError as e:
        return f"Validation error: {e}", "0.0%", ""
    except Exception as e:
        return f"Error: {e}", "0.0%", ""


# Example questions for the demo
example_questions = [
    "What is the Vision Transformer architecture?",
    "How does self-attention work in transformers?",
    "What datasets were used to evaluate ViT?",
    "What is the difference between ViT and CNNs?",
    "How does image patching work in Vision Transformers?",
    "What are the main results reported in ViT papers?",
    "What is the role of the class token in ViT?",
    "How is positional encoding used in Vision Transformers?",
]

demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        placeholder="Ask about Vision Transformer research papers…",
        label="Question",
        lines=3,
    ),
    outputs=[
        gr.Textbox(label="Answer", lines=15),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Sources", lines=8),
    ],
    title="🔬 Scientific Multimodal RAG",
    description=(
        "Ask questions about Vision Transformer research papers.  "
        "The system uses **ColPali** (vision) + **SciNCL** (text) "
        "hybrid retrieval with **Qwen2-VL** generation.  "
        "Each query takes ~5-15 seconds."
    ),
    examples=example_questions,
    theme="soft",
    allow_flagging="never",
)

print("Launching Gradio demo…")
demo.launch(share=True, debug=False)
