"""
Offline Pipeline — Ingestion Orchestrator
==========================================

Runs the 4-step offline pipeline using src/ modules:
  1. Download arXiv PDFs  (src.data.arxiv_dataset)
  2. Parse PDFs           (src.data.pdf_parser)
  3. ColPali embedding    (src.embeddings.colpali_embedder)
  4. SciNCL → ChromaDB   (src.embeddings.scincl_embedder)
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import torch


class OfflinePipeline:
    """End-to-end offline ingestion pipeline.

    Args:
        cfg: Loaded YAML config dict (from main.py).
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg     = cfg
        self.paths   = cfg.get("paths", {})
        self.papers  = cfg.get("papers", {})
        self.models  = cfg.get("models", {})
        self.parsing = cfg.get("parsing", {})

        self.raw_dir      = self.paths.get("raw",          "data/raw")
        self.pages_dir    = self.paths.get("pages",        "data/parsed/pages")
        self.markdown_dir = self.paths.get("markdown",     "data/parsed/markdown")
        self.npy_dir      = self.paths.get("multivectors", "data/indices/multivectors")
        self.chroma_dir   = self.paths.get("chroma_index", "data/indices/chroma_index")
        self.indices_dir  = self.paths.get("indices",      "data/indices")

        # Ensure all dirs exist
        for d in [self.raw_dir, self.pages_dir, self.markdown_dir,
                  self.npy_dir, self.chroma_dir]:
            os.makedirs(d, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # run — full pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute all 4 offline steps sequentially."""
        t0 = time.time()

        print("  Step 1/4: Downloading PDFs ...")
        doc_results = self.step1_download()

        print("\n  Step 2/4: Parsing PDFs ...")
        page_metadata, doc_mapping = self.step2_parse(doc_results)

        print("\n  Step 3/4: ColPali visual embedding ...")
        self.step3_colpali(page_metadata)

        print("\n  Step 4/4: SciNCL text embedding → ChromaDB ...")
        self.step4_scincl(page_metadata)

        elapsed = time.time() - t0
        self._save_summary(doc_mapping, page_metadata)

        print(f"\n  ✅ ALL STEPS DONE in {elapsed/60:.1f} min")
        print(f"  Papers  : {len(doc_mapping)}")
        print(f"  Pages   : {len(page_metadata)}")
        print(f"  ColPali : {len(list(Path(self.npy_dir).glob('*.npy')))} .npy files")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: Download
    # ─────────────────────────────────────────────────────────────────────────

    def step1_download(self) -> list:
        """Download arXiv PDFs using src.data.arxiv_dataset."""
        from src.data.arxiv_dataset import ArxivDataset

        dataset   = ArxivDataset(output_dir=self.raw_dir)
        results   = []
        arxiv_ids = list(self.papers.keys())

        print(f"  Downloading {len(arxiv_ids)} papers ...")

        for i, arxiv_id in enumerate(arxiv_ids):
            title   = self.papers[arxiv_id]
            pdf_path = os.path.join(self.raw_dir, f"{arxiv_id}.pdf")

            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10_000:
                size_mb = os.path.getsize(pdf_path) / 1e6
                print(f"  [{i+1:2d}/{len(arxiv_ids)}] SKIP (exists): {arxiv_id} ({size_mb:.1f} MB)")
                results.append({"arxiv_id": arxiv_id, "title": title,
                                 "status": "exists", "pdf_path": pdf_path})
                continue

            try:
                result = dataset.download(arxiv_id, title)
                results.append(result)
                status = "✅" if result.get("status") == "success" else "❌"
                print(f"  [{i+1:2d}/{len(arxiv_ids)}] {status} {arxiv_id}")
            except Exception as e:
                print(f"  [{i+1:2d}/{len(arxiv_ids)}] ❌ {arxiv_id}: {e}")
                results.append({"arxiv_id": arxiv_id, "title": title,
                                 "status": "failed", "error": str(e)})

        success = sum(1 for r in results if r.get("status") in ("success", "exists"))
        print(f"\n  Downloads: {success}/{len(arxiv_ids)} successful")

        # Save manifest
        with open(os.path.join(self.indices_dir, "download_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Parse
    # ─────────────────────────────────────────────────────────────────────────

    def step2_parse(self, download_results: list):
        """Parse PDFs using src.data.pdf_parser."""
        from src.data.pdf_parser import PDFParser

        dpi    = self.parsing.get("dpi", 200)
        parser = PDFParser(
            pages_dir=self.pages_dir,
            markdown_dir=self.markdown_dir,
            dpi=dpi,
        )

        successful = [d for d in download_results if d.get("status") in ("success", "exists")]
        print(f"  Parsing {len(successful)} PDFs (DPI={dpi}) ...")

        doc_mapping   = {}
        page_metadata = {}
        total_pages   = 0

        for idx, dl in enumerate(successful):
            arxiv_id = dl["arxiv_id"]
            title    = self.papers.get(arxiv_id, arxiv_id)
            pdf_path = dl.get("pdf_path", os.path.join(self.raw_dir, f"{arxiv_id}.pdf"))

            if not os.path.exists(pdf_path):
                print(f"  [{idx+1:2d}] SKIP (no PDF): {arxiv_id}")
                continue

            try:
                result = parser.parse(arxiv_id=arxiv_id, title=title, pdf_path=pdf_path)
                doc_mapping[arxiv_id] = result
                for page_key, meta in result.get("pages", {}).items():
                    page_metadata[page_key] = meta
                total_pages += result.get("num_pages", 0)
                print(f"  [{idx+1:2d}/{len(successful)}] ✅ {arxiv_id} — {result.get('num_pages',0)} pages")
            except Exception as e:
                print(f"  [{idx+1:2d}/{len(successful)}] ❌ {arxiv_id}: {e}")

        # Save metadata files
        with open(os.path.join(self.indices_dir, "doc_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(doc_mapping, f, indent=2, ensure_ascii=False)

        with open(os.path.join(self.indices_dir, "page_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(page_metadata, f, indent=2, ensure_ascii=False)

        print(f"\n  Parsed: {len(doc_mapping)} papers, {total_pages} pages total")
        return page_metadata, doc_mapping

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: ColPali embedding
    # ─────────────────────────────────────────────────────────────────────────

    def step3_colpali(self, page_metadata: dict) -> None:
        """Embed page images with ColPali using src.embeddings.colpali_embedder."""
        from src.embeddings.colpali_embedder import ColPaliEmbedder
        import numpy as np
        from PIL import Image

        colpali_cfg = self.models.get("colpali", {})
        embedder    = ColPaliEmbedder(
            model_name=colpali_cfg.get("model_name", "vidore/colpali-v1.2"),
            device=colpali_cfg.get("device", "cuda"),
            torch_dtype=colpali_cfg.get("torch_dtype", "float16"),
        )

        # Find already embedded pages
        existing = {f.replace(".npy", "") for f in os.listdir(self.npy_dir) if f.endswith(".npy")}
        to_embed = [
            (pk, meta) for pk, meta in page_metadata.items()
            if pk not in existing and meta.get("image_path") and os.path.exists(meta["image_path"])
        ]

        print(f"  Already embedded : {len(existing)} pages")
        print(f"  To embed         : {len(to_embed)} pages")

        if not to_embed:
            print("  All pages already embedded — skipping.")
            return

        print("  Loading ColPali model ...")
        embedder.load()

        embedded = 0
        errors   = 0
        t_start  = time.time()

        for i, (page_key, meta) in enumerate(to_embed):
            try:
                img = Image.open(meta["image_path"]).convert("RGB")
                result = embedder.embed_image(img)
                vectors = result.vectors  # numpy array (num_patches, 128)
                np.save(os.path.join(self.npy_dir, f"{page_key}.npy"), vectors)
                embedded += 1

                if (i + 1) % 20 == 0 or (i + 1) == len(to_embed):
                    elapsed = time.time() - t_start
                    rate    = embedded / max(elapsed, 1)
                    eta     = (len(to_embed) - (i + 1)) / max(rate, 0.01) / 60
                    print(f"  [{i+1:3d}/{len(to_embed)}] embedded={embedded} | "
                          f"rate={rate:.2f}p/s | ETA={eta:.1f}min")
            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️  OOM on {page_key} — skipping")
                torch.cuda.empty_cache()
                errors += 1
            except Exception as e:
                print(f"  ❌ {page_key}: {e}")
                errors += 1

        print("  Unloading ColPali ...")
        embedder.unload()
        gc.collect()

        print(f"\n  ColPali: embedded={embedded}, errors={errors}, "
              f"time={( time.time()-t_start)/60:.1f}min")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: SciNCL → ChromaDB
    # ─────────────────────────────────────────────────────────────────────────

    def step4_scincl(self, page_metadata: dict) -> None:
        """Build SciNCL text index using src.embeddings.scincl_embedder."""
        from src.embeddings.scincl_embedder import SciNCLEmbedder
        import chromadb

        scincl_cfg  = self.models.get("scincl", {})
        retrieval_cfg = self.cfg.get("retrieval", {})
        collection_name = retrieval_cfg.get("chroma_collection", "sci_rag_pages")
        batch_size  = scincl_cfg.get("batch_size", 32)

        embedder = SciNCLEmbedder(
            model_name=scincl_cfg.get("model_name", "malteos/scincl"),
            device=scincl_cfg.get("device", "cuda"),
            max_length=scincl_cfg.get("max_length", 512),
        )

        # Init ChromaDB
        chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        try:
            chroma_client.delete_collection(collection_name)
            print(f"  Deleted existing collection '{collection_name}'")
        except Exception:
            pass

        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  Created ChromaDB collection: {collection_name}")

        # Prepare texts
        texts, metas, ids = [], [], []
        for page_key, meta in page_metadata.items():
            text = meta.get("text", "").strip()
            if not text or len(text) < 20:
                continue
            texts.append(text[:512])
            metas.append({
                "doc_id":      meta.get("doc_id", ""),
                "page_num":    str(meta.get("page_num", 0)),
                "paper_title": meta.get("paper_title", ""),
            })
            ids.append(page_key)

        print(f"  Pages with valid text: {len(texts)}")
        print("  Loading SciNCL model ...")
        embedder.load()

        count   = 0
        t_start = time.time()

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids   = ids[i:i + batch_size]
            batch_metas = metas[i:i + batch_size]

            result = embedder.embed_texts(batch_texts)
            embeddings = result.vectors  # numpy (batch, 768)

            collection.upsert(
                ids=batch_ids,
                embeddings=embeddings.tolist(),
                documents=[t[:500] for t in batch_texts],
                metadatas=batch_metas,
            )
            count += len(batch_ids)

            elapsed = time.time() - t_start
            rate    = count / max(elapsed, 1)
            eta     = (len(texts) - count) / max(rate, 0.1) / 60
            print(f"  [{count:3d}/{len(texts)}] rate={rate:.1f}p/s | ETA={eta:.1f}min")

        print("  Unloading SciNCL ...")
        embedder.unload()
        gc.collect()

        print(f"\n  SciNCL: indexed={collection.count()} entries, "
              f"time={( time.time()-t_start)/60:.1f}min")

    # ─────────────────────────────────────────────────────────────────────────
    # Save summary
    # ─────────────────────────────────────────────────────────────────────────

    def _save_summary(self, doc_mapping: dict, page_metadata: dict) -> None:
        npy_files = list(Path(self.npy_dir).glob("*.npy"))
        summary = {
            "created_at":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_papers":     len(doc_mapping),
            "total_pages":    len(page_metadata),
            "colpali_files":  len(npy_files),
            "chroma_entries": len(page_metadata),
            "models_used": {
                "colpali": self.models.get("colpali", {}).get("model_name", "vidore/colpali-v1.2"),
                "scincl":  self.models.get("scincl", {}).get("model_name",  "malteos/scincl"),
            },
            "papers": {aid: {"title": title} for aid, title in self.papers.items()},
        }

        out_path = os.path.join(self.indices_dir, "summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved → {out_path}")
