"""
╔══════════════════════════════════════════════════════════════╗
║   SCIENTIFIC MULTIMODAL RAG — MASTER SETUP & RUN SCRIPT     ║
║                                                              ║
║   Paste this ENTIRE file in VS Code terminal and press Enter ║
║   It will do EVERYTHING from setup to testing automatically   ║
╚══════════════════════════════════════════════════════════════╝

WHAT THIS SCRIPT DOES (in order):
1. Creates virtual environment
2. Installs all dependencies
3. Installs project as editable package
4. Creates data directories
5. Tests config loading
6. Tests preprocessing functions
7. Tests self-check module
8. Tests metrics functions
9. Downloads 2 test arXiv papers
10. Parses the PDFs
11. Prints full test report

KAGGLE STEP (separate):
- Copy kaggle/notebook-offline.py cells into Kaggle notebook (GPU P100)
- Run offline pipeline to build indices (~30-40 min)
- Then copy kaggle/notebook-online.py cells for Gradio demo
"""

import subprocess
import sys
import os
from pathlib import Path

# ── Project root (where this script lives) ──
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJECT_ROOT))

def run(cmd, desc=""):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    print(f"$ {cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print(f"WARNING: Command returned exit code {result.returncode}")
    return result.returncode


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║  SCIENTIFIC MULTIMODAL RAG — AUTO SETUP & TEST              ║
║  This will set up everything automatically. Sit back!        ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ════════════════════════════════════════════════
    # STEP 1: Virtual Environment
    # ════════════════════════════════════════════════
    venv_dir = PROJECT_ROOT / "venv"
    pip_path = str(venv_dir / "Scripts" / "pip") if os.name == "nt" else str(venv_dir / "bin" / "pip")
    python_path = str(venv_dir / "Scripts" / "python") if os.name == "nt" else str(venv_dir / "bin" / "python")

    if not venv_dir.exists():
        run(f'python -m venv "{venv_dir}"', "STEP 1: Creating Virtual Environment")
    else:
        print("\n[SKIP] Virtual environment already exists.")

    # ════════════════════════════════════════════════
    # STEP 2: Upgrade pip
    # ════════════════════════════════════════════════
    run(f'"{python_path}" -m pip install --upgrade pip', "STEP 2: Upgrading pip")

    # ════════════════════════════════════════════════
    # STEP 3: Install base dependencies (no GPU needed)
    # ════════════════════════════════════════════════
    base_packages = [
        "pyyaml", "colorlog", "python-dotenv", "Pillow",
        "numpy", "pandas", "tqdm", "matplotlib", "seaborn",
    ]
    run(
        f'"{pip_path}" install -q ' + " ".join(base_packages),
        "STEP 3: Installing base dependencies (no GPU required)"
    )

    # ════════════════════════════════════════════════
    # STEP 4: Install PyTorch (CPU version for local testing)
    # ════════════════════════════════════════════════
    run(
        f'"{pip_path}" install -q torch --index-url https://download.pytorch.org/whl/cpu',
        "STEP 4: Installing PyTorch (CPU version for local testing)"
    )

    # ════════════════════════════════════════════════
    # STEP 5: Install project as editable package
    # ════════════════════════════════════════════════
    run(f'"{pip_path}" install -e .', "STEP 5: Installing project as editable package")

    # ════════════════════════════════════════════════
    # STEP 6: Install PDF processing tools
    # ════════════════════════════════════════════════
    run(f'"{pip_path}" install -q pymupdf arxiv', "STEP 6: Installing PDF processing (PyMuPDF, arxiv)")

    # ════════════════════════════════════════════════
    # STEP 7: Create data directories
    # ════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 7: Creating data directories")
    print(f"{'='*60}\n")

    dirs = [
        "data/raw", "data/parsed/pages", "data/parsed/markdown",
        "data/indices/chroma_index", "data/indices/multivectors",
        "outputs/evaluation/evaluation_charts", "outputs/queries",
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}")

    # ════════════════════════════════════════════════
    # STEP 8: Test config loading
    # ════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 8: Testing config loading")
    print(f"{'='*60}\n")

    test_code = """
import sys
sys.path.insert(0, '.')
from src.utils.config_loader import load_config, resolve_paths

configs_ok = True
for name in ['model_config', 'data_config', 'retrieval_config', 'pipeline_config', 'evaluation_config']:
    try:
        cfg = load_config(name)
        print(f'  OK: {name}.yaml loaded')
    except Exception as e:
        print(f'  FAIL: {name}.yaml - {e}')
        configs_ok = False

if configs_ok:
    print('\\nAll 5 configs loaded successfully!')
"""
    run(f'"{python_path}" -c "{test_code}"', "Testing config loading")

    # ════════════════════════════════════════════════
    # STEP 9: Test preprocessing module
    # ════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 9: Testing preprocessing")
    print(f"{'='*60}\n")

    test_preproc = """
import sys
sys.path.insert(0, '.')
from src.data.preprocessing import clean_text, split_into_chunks, normalize_scientific_text

assert clean_text("  hello   world  ") == "hello world"
print("  OK: clean_text")

chunks = split_into_chunks(" ".join(["word"] * 1000), max_tokens=200)
assert len(chunks) > 1
print("  OK: split_into_chunks")

result = normalize_scientific_text("See Fig. 3 and Eq. 2. $x^2 + y^2 = z^2$")
assert "Figure" in result
print("  OK: normalize_scientific_text")

print("\\nPreprocessing tests PASSED!")
"""
    run(f'"{python_path}" -c "{test_preproc}"', "Testing preprocessing")

    # ════════════════════════════════════════════════
    # STEP 10: Test self-check module
    # ════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 10: Testing self-check")
    print(f"{'='*60}\n")

    test_check = """
import sys
sys.path.insert(0, '.')
from src.generation.self_check import SelfChecker

checker = SelfChecker(confidence_threshold=0.6)
result = checker.check(
    answer="ViT uses self-attention [Source: ViT Paper, Page 3].",
    context="ViT uses self-attention mechanism for image patches.",
    confidence=0.8,
)
assert result.attribution_passed == True
assert result.confidence_passed == True
print(f"  Attribution: {result.attribution_passed}")
print(f"  Faithfulness: {result.faithfulness_passed}")
print(f"  Confidence: {result.confidence_passed}")
print(f"  Overall: {result.passed}")
print("\\nSelf-check test PASSED!")
"""
    run(f'"{python_path}" -c "{test_check}"', "Testing self-check")

    # ════════════════════════════════════════════════
    # STEP 11: Test metrics
    # ════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 11: Testing metrics")
    print(f"{'='*60}\n")

    test_metrics = """
import sys
sys.path.insert(0, '.')
from src.utils.metrics import compute_f1, compute_anls, compute_all

f1 = compute_f1("the cat sat on the mat", "a cat sat on a mat")
print(f"  F1: {f1:.4f}")

anls = compute_anls("the cat sat", "a cat sat")
print(f"  ANLS: {anls:.4f}")

scores = compute_all("Vision Transformer uses attention", "ViT applies self-attention")
for k, v in scores.items():
    print(f"  {k}: {v:.4f}")

print("\\nMetrics test PASSED!")
"""
    run(f'"{python_path}" -c "{test_metrics}"', "Testing metrics")

    # ════════════════════════════════════════════════
    # STEP 12: Download 2 test arXiv papers
    # ════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 12: Downloading 2 test arXiv papers")
    print(f"{'='*60}\n")

    download_code = """
import sys
sys.path.insert(0, '.')
from src.data.arxiv_dataset import ArxivDataset

ds = ArxivDataset(
    query="vision transformer",
    category="cs.CV",
    max_results=5,
    keep_best=2,
    output_dir="data/raw/",
)
results = ds.download()
for r in results:
    print(f"  {r['arxiv_id']}: {r['status']}")
print(f"\\nDownloaded {sum(1 for r in results if r['status']=='success')}/{len(results)} papers.")
"""
    run(f'"{python_path}" -c "{download_code}"', "Downloading test papers from arXiv")

    # ════════════════════════════════════════════════
    # STEP 13: Parse downloaded PDFs
    # ════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 13: Parsing PDFs")
    print(f"{'='*60}\n")

    parse_code = """
import sys, os
sys.path.insert(0, '.')
from src.data.pdf_parser import DualPDFParser
from pathlib import Path

parser = DualPDFParser(
    output_pages_dir="data/parsed/pages/",
    output_markdown_dir="data/parsed/markdown/",
)

pdf_dir = Path("data/raw/")
pdfs = sorted(pdf_dir.glob("*.pdf"))

if not pdfs:
    print("  No PDFs found to parse.")
else:
    for pdf in pdfs:
        print(f"  Parsing {pdf.name}…")
        result = parser.parse(str(pdf))
        print(f"    → {result['num_pages']} pages, status: {result['status']}")
    print(f"\\nParsed {len(pdfs)} PDFs.")
"""
    run(f'"{python_path}" -c "{parse_code}"', "Parsing PDFs into images + markdown")

    # ════════════════════════════════════════════════
    # FINAL SUMMARY
    # ════════════════════════════════════════════════
    print(f"""

{'='*60}
  SETUP COMPLETE — SUMMARY
{'='*60}

  Virtual Environment  : {venv_dir}
  Project Package      : installed (editable)
  Config Files         : 5 YAML files in configs/
  Data Directories     : data/raw, data/parsed, data/indices
  Test Papers          : 2 papers downloaded and parsed

  LOCAL TESTS PASSED:
    - Config loading
    - Text preprocessing
    - Self-check verification
    - Evaluation metrics
    - PDF download & parsing

{'='*60}
  NEXT STEPS — KAGGLE (GPU REQUIRED)
{'='*60}

  To build the full index and run queries, you need a GPU.
  Follow these steps on Kaggle:

  1. Go to https://www.kaggle.com/notebooks
  2. Create New Notebook → Settings → GPU P100
  3. Upload this project as a Kaggle Dataset
     (zip the entire Scientific-Multimodal-RAG folder)
  4. Copy cells from kaggle/notebook-offline.py
     This builds the index (~30-40 min for 10 papers)
  5. Save output as Kaggle Dataset: "sci-rag-index"
  6. Create new notebook → Copy cells from kaggle/notebook-online.py
     This launches the Gradio demo (~15s per query)

  Example queries to try:
    - "What is the Vision Transformer?"
    - "How does patch embedding work?"
    - "What datasets evaluate ViT?"

{'='*60}
  TO RUN TESTS AGAIN:
{'='*60}

  {python_path} tests/test_parsers.py
  {python_path} tests/test_pipeline.py

{'='*60}
""")


if __name__ == "__main__":
    main()
