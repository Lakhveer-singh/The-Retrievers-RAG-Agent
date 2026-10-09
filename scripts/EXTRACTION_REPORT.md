# Extraction Report — NVIDIA Annual Report 2026

## Corpus Source

- **Document:** NVIDIA Corporation Annual Report 2026
- **File:** `2026-Annual-Report-Web.pdf` (175 pages, ~15.8 MB)
- **Source URL:** NVIDIA Investor Relations (publicly available; includes the Form 10-K)
- **Language:** English

## Extraction Method

- **Library:** PyMuPDF (`fitz`), text blocks via `page.get_text("blocks", sort=True)`
- **Granularity:** One entry per page (175 entries total)
- **Output:** `data/corpus.json` (all pages) and `data/sample.json` (first 10 pages), UTF-8

## Corpus Statistics

```
Number of source files : 1
Total entries          : 175
Total characters       : 629,713
Avg characters/entry   : 3,598
Empty/near-empty (<50 chars) : 0
```

## Text Cleanup Applied

Cleanup runs in this order:

1. **Block extraction:** each page is read as PyMuPDF text blocks (one block ≈ one paragraph, heading or table area), sorted top-to-bottom.
2. **Repeated headers/footers:** a short, single-line block that appears among the first 2 or last 2 blocks on 30%+ of pages is treated as a header/footer and removed. Digits are normalised during matching so lines like "Page 12" / "Page 13" count as the same line. Multi-line blocks (paragraphs, tables) are never removed.
3. **Page numbers:** standalone numbers or "Page X" blocks are removed.
4. **Hyphenated line breaks:** words split across lines are rejoined (e.g. "con-\ntinue" → "continue").
5. **Mid-sentence line breaks:** single newlines inside a block are replaced with spaces.
6. **Whitespace:** multiple spaces collapsed; blocks are joined with a blank line so paragraph structure is kept for chunking.

## Near-empty pages

0 pages have fewer than 50 characters. Every page has some extractable text: even
image-heavy pages (cover, section dividers, charts) contain titles or captions.
No entries were dropped.

## Known Issues

### 1. Tables
Financial tables (income statement, balance sheet, cash flow) extract cell by cell, so each table is flattened onto one line (e.g. `Revenue $ 130,497 $ 60,922 ...`). Labels and numbers stay in order, but row/column structure is lost. If table questions fail during the hackathon, we will re-extract the financial statement pages with `pdfplumber`.

### 2. Hyphen rejoining
The hyphen rule can also merge real hyphenated words split at a line end (e.g. "AI-\npowered" → "AIpowered"). Rare; left as is.

### 3. Footnotes and legal disclaimers
SEC-mandated disclaimers and footnotes repeat across sections. They are kept because they contain factual information that may be asked about (risk factors, forward-looking statements).

## Fix history

- **v1:** joined all lines *before* header/footer detection. PyMuPDF puts every PDF line on its own line with no blank lines between paragraphs, so each page became one long line. As a result, headers and page numbers were never removed (the page number stayed glued to the end of the text) and paragraph breaks were lost.
- **v2 (current):** extracts text blocks, removes repeated headers/footers and page numbers first, then cleans each block and joins blocks with blank lines. Page numbering is unchanged (still 1–175), so `questions.json` did not need updating.

## Conclusion

Extraction worked well for this corpus. Most pages are text-heavy (financial data, management discussion, notes to the financial statements) and extract cleanly. The only losses are garbled chart labels and the flattened table layout.
