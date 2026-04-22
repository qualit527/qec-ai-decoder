"""
Convert every PDF in knowledge/papers/ to markdown in knowledge/papers_md/ via
pymupdf4llm (fast, structure-preserving). Parallelized across CPU cores.
Skips files already converted.

Output is suitable for: (a) Ideator agent reading as reference text,
(b) building a vector index / BM25 index later for RAG.
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import time
import traceback
from pathlib import Path

try:
    import pymupdf4llm
except ImportError:
    print("pymupdf4llm not installed. Run: pip install pymupdf4llm", file=sys.stderr)
    sys.exit(1)

HERE = Path(__file__).parent.resolve()
PDF_DIR = HERE / "papers"
MD_DIR = HERE / "papers_md"
MD_DIR.mkdir(exist_ok=True)
LOG_FILE = HERE / "convert_markdown.log"
STATUS_FILE = HERE / "markdown_status.json"


def convert_one(pdf_path: Path) -> dict:
    """Convert a single PDF. Returns {'file', 'status', 'md_chars', 'n_images', 'time_s', 'error'}."""
    stem = pdf_path.stem
    md_path = MD_DIR / (stem + ".md")
    img_dir = MD_DIR / (stem + "_images")
    # Reconsider skip: require both .md and image dir to exist (previous runs had no images).
    if md_path.exists() and md_path.stat().st_size > 1000 and img_dir.exists():
        n_imgs = len(list(img_dir.glob("*")))
        return {
            "file": pdf_path.name,
            "status": "skipped",
            "md_chars": md_path.stat().st_size,
            "n_images": n_imgs,
            "time_s": 0.0,
            "error": "",
        }
    t0 = time.time()
    try:
        img_dir.mkdir(exist_ok=True)
        md = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=False,
            write_images=True,
            image_path=str(img_dir),
            image_format="png",
            dpi=150,
            image_size_limit=0.02,     # skip tiny images (<2% of page area) to cut noise
            show_progress=False,
        )
        # Rewrite image paths to be relative to the .md file location
        import re as _re
        def _rewrite(m):
            full = m.group(1)
            # Keep only basename and prefix with the image-dir folder name
            base = Path(full).name
            return f"]({stem}_images/{base})"
        md = _re.sub(r"\]\(([^)]+\.png)\)", _rewrite, md)

        header = (
            f"<!-- source: papers/{pdf_path.name} -->\n"
            f"<!-- images: {stem}_images/ -->\n\n"
        )
        md_path.write_text(header + md, encoding="utf-8")

        n_imgs = len(list(img_dir.glob("*")))
        # Clean up empty image dir
        if n_imgs == 0:
            try:
                img_dir.rmdir()
            except OSError:
                pass

        return {
            "file": pdf_path.name,
            "status": "converted",
            "md_chars": len(md),
            "n_images": n_imgs,
            "time_s": time.time() - t0,
            "error": "",
        }
    except Exception as e:
        return {
            "file": pdf_path.name,
            "status": "failed",
            "md_chars": 0,
            "n_images": 0,
            "time_s": time.time() - t0,
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=2)}",
        }


def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs. Output → {MD_DIR}", file=sys.stderr)

    results = []
    # pymupdf is thread-safe with GIL released internally; use threads for I/O overlap
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(convert_one, pdf): pdf for pdf in pdfs}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            mark = {"converted": "✓", "skipped": "·", "failed": "✗"}.get(r["status"], "?")
            print(
                f"[{i}/{len(pdfs)}] {mark} {r['file']:50s} "
                f"{r['md_chars']:>7} chars  {r.get('n_images',0):>3} imgs  {r['time_s']:5.2f}s"
                + (f"  ERR: {r['error'][:80]}" if r["status"] == "failed" else ""),
                file=sys.stderr,
            )

    # Summary
    conv = sum(1 for r in results if r["status"] == "converted")
    skip = sum(1 for r in results if r["status"] == "skipped")
    fail = sum(1 for r in results if r["status"] == "failed")
    total_chars = sum(r["md_chars"] for r in results)
    total_time = sum(r["time_s"] for r in results)

    print(file=sys.stderr)
    print(
        f"Summary: converted={conv}  skipped={skip}  failed={fail}  "
        f"total_md={total_chars/1e6:.1f}MB  cpu_time={total_time:.1f}s",
        file=sys.stderr,
    )

    # Persist status
    STATUS_FILE.write_text(
        json.dumps({"results": results, "summary": {
            "converted": conv, "skipped": skip, "failed": fail,
            "total_md_mb": total_chars / 1e6, "cpu_seconds": total_time,
        }}, indent=2, default=str),
        encoding="utf-8",
    )

    if fail:
        # Dump failures to log
        LOG_FILE.write_text(
            "\n".join(f"FAIL {r['file']}:\n{r['error']}\n" for r in results if r["status"] == "failed"),
            encoding="utf-8",
        )
        print(f"Failures logged to {LOG_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
