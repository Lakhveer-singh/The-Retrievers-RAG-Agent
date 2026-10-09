"""
Assignment 0 — Corpus Preparation: Text Extraction
Extracts clean text from PDF/TXT/HTML documents in data/sources/
Outputs data/corpus.json, data/sample.json, and prints corpus stats.
"""

import json
import os
import re
from pathlib import Path


# ──────────────────────────── Configuration ────────────────────────────
SOURCES_DIR = Path("data/sources")
CORPUS_FILE = Path("data/corpus.json")
SAMPLE_FILE = Path("data/sample.json")
NEAR_EMPTY_THRESHOLD = 50  # characters


# ──────────────────────────── Text Cleanup ─────────────────────────────
def clean_text(text: str) -> str:
    """Clean raw extracted text: fix whitespace, line breaks, and noise."""
    # Remove form feed characters (page breaks embedded in text)
    text = text.replace("\f", "")

    # Fix hyphenated line breaks (e.g., "con-\ntinue" → "continue")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Replace single line breaks within paragraphs with spaces
    # (PDFs often break lines mid-sentence)
    # Keep double newlines as paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Collapse 3+ newlines into 2 (paragraph separator)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Strip overall leading/trailing whitespace
    text = text.strip()

    return text


def remove_repeated_headers_footers(entries: list[dict]) -> list[dict]:
    """
    Detect and remove repeated headers/footers and page numbers from PDF pages.
    Works on raw text blocks (before line-joining), so short header lines are
    still separate and can be matched. A block counts as a header/footer if the
    same text appears in the first/last 2 blocks of 30%+ of pages.
    Then each page's remaining blocks are cleaned and joined as paragraphs.
    """
    page_num_pattern = re.compile(r"^\s*(page\s+)?\d{1,4}\s*$", re.IGNORECASE)
    pdf_entries = [e for e in entries if "_blocks" in e]

    header_candidates, footer_candidates = set(), set()
    if len(pdf_entries) >= 5:
        # normalise digits so "Page 12" and "Page 13" count as the same line
        norm = lambda t: re.sub(r"\d+", "#", t)
        first_counts, last_counts = {}, {}
        # only short single-line blocks can be headers/footers (never tables/paragraphs)
        short = lambda t: "\n" not in t and len(t) < 80
        for e in pdf_entries:
            for b in set(norm(x) for x in e["_blocks"][:2] if short(x)):
                first_counts[b] = first_counts.get(b, 0) + 1
            for b in set(norm(x) for x in e["_blocks"][-2:] if short(x)):
                last_counts[b] = last_counts.get(b, 0) + 1
        threshold = len(pdf_entries) * 0.3
        header_candidates = {b for b, c in first_counts.items() if c >= threshold}
        footer_candidates = {b for b, c in last_counts.items() if c >= threshold}
        print(f"  Detected repeated headers: {header_candidates or 'none'}")
        print(f"  Detected repeated footers: {footer_candidates or 'none'}")

    for e in pdf_entries:
        blocks = e.pop("_blocks")
        kept = []
        for i, b in enumerate(blocks):
            nb = re.sub(r"\d+", "#", b)
            if i < 2 and "\n" not in b and nb in header_candidates:
                continue
            if i >= len(blocks) - 2 and "\n" not in b and nb in footer_candidates:
                continue
            if page_num_pattern.match(b):
                continue
            kept.append(clean_text(b))
        e["text"] = "\n\n".join(k for k in kept if k)
        e["char_count"] = len(e["text"])

    return entries


# ──────────────────────────── PDF Extraction ───────────────────────────
def extract_pdf(filepath: Path) -> list[dict]:
    """Extract text from a PDF file, one entry per page.
    Keeps PyMuPDF text blocks separate so header/footer/page-number
    removal can run BEFORE lines are joined into paragraphs."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("ERROR: PyMuPDF not installed. Run: uv pip install pymupdf")
        raise

    entries = []
    doc = fitz.open(str(filepath))

    for page_num in range(len(doc)):
        page = doc[page_num]
        # blocks = (x0, y0, x1, y1, text, block_no, block_type); type 0 = text
        blocks = page.get_text("blocks", sort=True)
        block_texts = [b[4].strip() for b in blocks if b[6] == 0 and b[4].strip()]

        entries.append({
            "source": filepath.name,
            "page": page_num + 1,  # 1-indexed
            "char_count": 0,       # set after cleanup
            "text": "",
            "_blocks": block_texts,  # temporary, removed before saving
        })

    doc.close()
    return entries


# ──────────────────────────── TXT Extraction ───────────────────────────
def extract_txt(filepath: Path) -> list[dict]:
    """Extract text from a plain text or markdown file."""
    text = filepath.read_text(encoding="utf-8", errors="replace")
    text = clean_text(text)

    return [{
        "source": filepath.name,
        "page": 1,
        "char_count": len(text),
        "text": text,
    }]


# ──────────────────────────── HTML Extraction ──────────────────────────
def extract_html(filepath: Path) -> list[dict]:
    """Extract text from an HTML file using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("ERROR: BeautifulSoup not installed. Run: uv pip install beautifulsoup4")
        raise

    html = filepath.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = clean_text(text)

    return [{
        "source": filepath.name,
        "page": 1,
        "char_count": len(text),
        "text": text,
    }]


# ──────────────────────────── Main Pipeline ────────────────────────────
def extract_all() -> list[dict]:
    """Extract text from all documents in data/sources/."""
    if not SOURCES_DIR.exists():
        print(f"ERROR: {SOURCES_DIR} directory not found.")
        print("Create it and place your source documents inside.")
        return []

    # Map file extensions to extractors
    extractors = {
        ".pdf": extract_pdf,
        ".txt": extract_txt,
        ".md": extract_txt,
        ".html": extract_html,
        ".htm": extract_html,
    }

    all_entries = []
    source_files = sorted(SOURCES_DIR.iterdir())
    supported_files = [f for f in source_files if f.suffix.lower() in extractors]

    if not supported_files:
        print(f"ERROR: No supported files found in {SOURCES_DIR}")
        print(f"Supported formats: {', '.join(extractors.keys())}")
        return []

    for filepath in supported_files:
        ext = filepath.suffix.lower()
        extractor = extractors[ext]
        print(f"Extracting: {filepath.name} ({ext})")

        entries = extractor(filepath)
        all_entries.extend(entries)
        print(f"  → {len(entries)} entries extracted")

    # Remove repeated headers/footers across pages
    print("\nDetecting repeated headers/footers...")
    all_entries = remove_repeated_headers_footers(all_entries)

    return all_entries


def print_stats(entries: list[dict]) -> None:
    """Print corpus statistics."""
    sources = set(e["source"] for e in entries)
    total_chars = sum(e["char_count"] for e in entries)
    avg_chars = total_chars / len(entries) if entries else 0
    empty_entries = [e for e in entries if e["char_count"] < NEAR_EMPTY_THRESHOLD]

    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    print(f"  Number of source files : {len(sources)}")
    print(f"  Total entries          : {len(entries)}")
    print(f"  Total characters       : {total_chars:,}")
    print(f"  Avg characters/entry   : {avg_chars:,.0f}")
    print(f"  Empty/near-empty (<{NEAR_EMPTY_THRESHOLD} chars) : {len(empty_entries)}")

    if empty_entries:
        print(f"\n  ⚠ Near-empty entries:")
        for e in empty_entries:
            preview = e["text"][:80].replace("\n", " ") if e["text"] else "(empty)"
            print(f"    - {e['source']} p.{e['page']} ({e['char_count']} chars): {preview}")

    # ready-to-paste line for EXTRACTION_REPORT.md
    pages = ", ".join(f"{e['page']}" for e in empty_entries) or "none"
    print(f"\nFor EXTRACTION_REPORT.md -> {len(empty_entries)} near-empty pages: {pages}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("CORPUS EXTRACTION — GenAI Hackathon")
    print("=" * 60)

    # Extract
    entries = extract_all()
    if not entries:
        return

    # Save corpus.json
    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {CORPUS_FILE} ({len(entries)} entries)")

    # Save sample.json (first 10 entries)
    sample = entries[:10]
    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {SAMPLE_FILE} ({len(sample)} entries)")

    # Print stats
    print_stats(entries)


if __name__ == "__main__":
    main()
