# Failure Log — The Retrievers

> Updated after each eval run. Documents failures and fixes applied.

## Log

| # | Question | Category | Retrieval OK? | Problem | Fix applied | Fixed? |
|---|----------|----------|:---:|---------|-------------|:---:|
| 9 | How much did NVIDIA pay in cash dividends... | factual | ✅ | Generation error | TBD | 🔄 |
| 14 | By how many dollars did Data Center revenue increase... | cross-reference | ✅ | Generation error — failed to compute dollar diff across pages | TBD | 🔄 |
| 15 | How did April 2025 licensing affect risk disclosures and compensation... | cross-reference | ✅ | Generation error — missed connecting both sections | TBD | 🔄 |
| 17 | Annual Review highlights $6B Physical AI vs Automotive segment... | cross-reference | ❌ | Retrieval miss — needed pages 24+167 | TBD | 🔄 |
| 18 | How much was H20 charge and in how many parts disclosed... | cross-reference | ✅ | Generation error — did not enumerate all disclosure locations | TBD | 🔄 |

## Ambiguous questions (checked by hand)

| # | Question | Response quality | Notes |
|---|----------|:---:|-------|
| 40 | Tell me about NVIDIA's compensation | ✅ | Answered broadly with key compensation components |
| 41 | What are the risks? | ✅ | Listed risk categories from the report |
| 42 | How is NVIDIA doing this year? | ✅ | Gave revenue summary with key metrics |

## Summary

- **Eval run date:** September 28, 2026
- **Total questions:** 43
- **Retrieval hit rate:** 28/35 (80%)
- **Answer accuracy:** 35/40 (88%)
- **Priority accuracy:** 5/5 (100%)
- **Failures fixed:** 0 / 5
- **Regressions introduced:** 0
