"""
Kaggle Session 1: Offline Pipeline Runner
═══════════════════════════════════════════
GPU: P100 ON | Internet: ON
Runtime: ~30-40 minutes for 10 PDFs

Steps:
1. Install project as package
2. Run offline pipeline (download → parse → embed → store)
3. Save output as Kaggle Dataset
"""

# ═══════════════════════════════════════════════════════════════
# Cell 1: Install dependencies
# ═══════════════════════════════════════════════════════════════

import subprocess
import sys

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

# Install the project as an editable package (from Kaggle dataset input)
# Adjust the path if your Kaggle dataset has a different structure.
import os

code_dir = "/kaggle/input/sci-rag-code"
if os.path.isdir(code_dir):
    install(f"-e {code_dir}")
else:
    # Fallback: install from the working directory
    install("-e .")

# Install additional dependencies
install("colpali-engine")
install("marker-pdf")
install("arxiv")
install("pdf2image")
install("PyMuPDF")  # Fallback PDF parser
install("chromadb")

# Verify key packages
print("\n--- Package Verification ---")
for pkg in ["torch", "transformers", "colpali_engine", "chromadb", "PIL"]:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} — NOT AVAILABLE")

print("\nCell 1 complete: Dependencies installed.\n")


# ═══════════════════════════════════════════════════════════════
# Cell 2: GPU info and configuration check
# ═══════════════════════════════════════════════════════════════

import torch

print("=== GPU Information ===")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    print(f"VRAM: {total_vram:.2f} GB")
    print(f"CUDA: {torch.version.cuda}")
else:
    print("WARNING: No GPU detected — pipeline will be very slow!")

print("\n=== Configuration ===")
import yaml
from pathlib import Path

config_dir = Path("configs")
for cfg_file in sorted(config_dir.glob("*.yaml")):
    with open(cfg_file) as f:
        cfg = yaml.safe_load(f)
    print(f"\n{cfg_file.name}:")
    for key in list(cfg.keys())[:3]:
        print(f"  {key}: {str(cfg[key])[:80]}...")

print("\nCell 2 complete: GPU and config verified.\n")


# ═══════════════════════════════════════════════════════════════
# Cell 3: Run offline pipeline
# ═══════════════════════════════════════════════════════════════

from pipelines.offline_pipeline import OfflinePipeline

print("=" * 60)
print("  STARTING OFFLINE PIPELINE")
print("=" * 60)

pipeline = OfflinePipeline(config_path="configs/pipeline_config.yaml")
result = pipeline.run()

print("\nCell 3 complete: Offline pipeline finished.\n")


# ═══════════════════════════════════════════════════════════════
# Cell 4: Print summary
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  OFFLINE PIPELINE — RESULTS SUMMARY")
print("=" * 60)
print(f"  Papers processed : {result['papers_processed']}")
print(f"  Pages embedded   : {result['pages_embedded']}")
print(f"  Failed downloads : {result['failed_downloads']}")
print(f"  Failed parses    : {result['failed_parses']}")
print(f"  ColPali time     : {result['colpali_time']:.1f}s")
print(f"  SciNCL time      : {result['scincl_time']:.1f}s")
print(f"  Total time       : {result['total_time']:.1f}s")
print("=" * 60)

print("\nCell 4 complete: Summary printed.\n")


# ═══════════════════════════════════════════════════════════════
# Cell 5: Save as Kaggle Dataset
# ═══════════════════════════════════════════════════════════════

import shutil

output_dir = "/kaggle/working/sci-rag-index"

# Copy the indices directory to Kaggle working directory
src_dir = "data/indices"
if os.path.isdir(src_dir):
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(src_dir, output_dir)
    print(f"Copied index data to {output_dir}")

    # Count files
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            print(f"  {os.path.join(root, f)}")

    # Create zip archive
    shutil.make_archive(
        "/kaggle/working/sci-rag-index", "zip", output_dir
    )
    print("\n✓ Saved sci-rag-index.zip")
    print("  → Upload this as a Kaggle Dataset for Session 2 (online pipeline)")
else:
    print(f"WARNING: Index directory '{src_dir}' not found.")

# Also copy parsed data if available
parsed_dir = "data/parsed"
if os.path.isdir(parsed_dir):
    parsed_output = "/kaggle/working/sci-rag-parsed"
    if os.path.isdir(parsed_output):
        shutil.rmtree(parsed_output)
    shutil.copytree(parsed_dir, parsed_output)
    print(f"\n✓ Copied parsed data to {parsed_output}")

print("\nCell 5 complete: Kaggle dataset ready.\n")
