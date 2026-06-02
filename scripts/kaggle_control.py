# -*- coding: utf-8 -*-
"""kaggle_control.py

Utility for controlling Kaggle notebook execution from the local workstation.

Features
--------
1. **Push** a notebook (offline pipeline) to Kaggle as a kernel.
2. **Poll** Kaggle for kernel status until it finishes or fails.
3. **Download** the output files (embeddings, index, log) to a local directory.
4. Minimal external dependencies – only the `kaggle` Python package (installed
   in the project's virtual environment) and the standard library.

Prerequisites
-------------
* A valid `kaggle.json` file in ``%USERPROFILE%\.kaggle`` (already detected).
* The Kaggle CLI is available via the virtual‑env ``.venv\Scripts\kaggle``.
* A metadata JSON file that describes the kernel.  The repository already
  contains ``kaggle/kaggle-metadata.json`` which points at ``notebook-online.py``.
  This script creates a **copy** ``offline-metadata.json`` that references the
  ``notebook-offline.py`` notebook.

Usage Example
-------------
```bash
# From the project root
.venv\Scripts\python scripts/kaggle_control.py
```
The script will:
* Write ``kaggle/offline-metadata.json`` (if not present).
* Push the kernel.
* Wait for it to finish.
* Download the kernel output into ``kaggle/output/``.
```
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list[str], env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run a command synchronously and raise on error.

    Parameters
    ----------
    cmd: list[str]
        Command and arguments.
    env: Optional[dict]
        Extra environment variables; merged with ``os.environ``.
    """
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    if result.returncode != 0:
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        raise RuntimeError(f"Command {' '.join(cmd)} failed with exit code {result.returncode}")
    return result


def _kaggle_executable() -> str:
    """Return the absolute path to the ``kaggle`` CLI inside the virtual env.
    The virtual environment resides at ``.venv`` relative to the project root.
    """
    root = Path(__file__).resolve().parents[2]  # project root (scripts/..)
    exe = root / ".venv" / "Scripts" / "kaggle"
    if not exe.exists():
        raise FileNotFoundError(f"kaggle executable not found at {exe}. Did you install the package?")
    return str(exe)


def _write_offline_metadata(root: Path) -> Path:
    """Create ``kaggle/offline-metadata.json`` that points at ``notebook-offline.py``.

    The function copies the existing ``kaggle-metadata.json`` (which references the
    online notebook) and swaps the ``code_file`` field.
    """
    src = root / "kaggle" / "kaggle-metadata.json"
    dst = root / "kaggle" / "offline-metadata.json"
    if not src.exists():
        raise FileNotFoundError(f"Source metadata {src} missing")
    data = json.loads(src.read_text(encoding="utf-8"))
    data["code_file"] = "notebook-offline.py"
    data["title"] = "Scientific Multimodal RAG – Offline Indexing"
    dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Written offline metadata to {dst}")
    return dst


def _push_kernel(metadata_path: Path) -> str:
    """Push the notebook to Kaggle and return the kernel slug.

    The ``kaggle kernels push -p <metadata_path>`` command prints the kernel URL
    like ``https://www.kaggle.com/vineet/scientific-multimodal-rag``.  From that we
    extract the *slug* ``vineet/scientific-multimodal-rag`` which is used for
    subsequent status/output commands.
    """
    exe = _kaggle_executable()
    result = _run_cmd([exe, "kernels", "push", "-p", str(metadata_path)])
    # The CLI prints a line containing the kernel URL.
    for line in result.stdout.splitlines():
        if "https://www.kaggle.com/" in line:
            url = line.strip().split()[-1]
            slug = url.replace("https://www.kaggle.com/", "")
            print(f"Kernel pushed – slug: {slug}")
            return slug
    raise RuntimeError("Could not determine kernel slug from push output")


def _kernel_status(slug: str) -> str:
    """Return the current kernel status: ``pending``, ``running``, ``complete`` or ``failed``.
    """
    exe = _kaggle_executable()
    result = _run_cmd([exe, "kernels", "status", slug])
    # Output format: "Status: <status>"
    for line in result.stdout.splitlines():
        if line.lower().startswith("status:"):
            status = line.split(":", 1)[1].strip().lower()
            return status
    raise RuntimeError("Unable to parse kernel status")


def _download_output(slug: str, output_dir: Path) -> None:
    """Download the output folder of a finished kernel.

    ``kaggle kernels output <slug> -p <output_dir>`` creates the directory and
    extracts a zip containing everything the notebook wrote to the ``/kaggle/working``
    folder.
    """
    exe = _kaggle_executable()
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_cmd([exe, "kernels", "output", slug, "-p", str(output_dir)])
    print(f"Kernel output downloaded to {output_dir}")


def push_and_run(
    notebook_dir: Path = Path("kaggle"),
    output_dir: Path = Path("kaggle/output"),
    poll_interval: int = 30,
) -> None:
    """Orchestrates the full Kaggle workflow.

    Parameters
    ----------
    notebook_dir: Path
        Directory containing the notebook and metadata files.
    output_dir: Path
        Where the kernel output zip will be extracted.
    poll_interval: int
        Seconds between status polls.
    """
    root = Path(__file__).resolve().parents[2]
    metadata_path = _write_offline_metadata(root)
    slug = _push_kernel(metadata_path)

    print("Polling kernel status …")
    while True:
        status = _kernel_status(slug)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] status: {status}")
        if status in {"complete", "failed"}:
            break
        time.sleep(poll_interval)

    if status == "failed":
        raise RuntimeError(f"Kernel {slug} failed. Check Kaggle UI for logs.")

    _download_output(slug, output_dir)


if __name__ == "__main__":
    # Simple CLI entry point
    try:
        push_and_run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
