"""
Checks data/questions.json against the Assignment 1 checklist and against data/corpus.json.
Usage (from repo root): python scripts/check_questions.py
No API calls, free to run.
"""
import json
import re
from collections import Counter

with open("data/questions.json", encoding="utf-8") as f:
    questions = json.load(f)
with open("data/corpus.json", encoding="utf-8") as f:
    corpus = json.load(f)

# page text lookup, whitespace removed so line breaks don't break the search
pages = {(e["source"], e["page"]): re.sub(r"\s+", "", e["text"]) for e in corpus}

# ---------------- checklist ----------------
print("=" * 60)
print("CHECKLIST")
print("=" * 60)
cats = Counter(q["category"] for q in questions)
print(f"total questions: {len(questions)}  (need 40-50)")
mins = {"factual": 10, "cross-reference": 8, "stress-test": 10, "out-of-scope": 3, "ambiguous": 2}
for c, m in mins.items():
    print(f"  {c:16} {cats[c]:3}  (min {m}) {'OK' if cats[c] >= m else 'TOO FEW'}")

cross = {q["sub_type"] for q in questions if q["category"] == "cross-reference"}
stress = {q["sub_type"] for q in questions if q["category"] == "stress-test"}
print(f"cross-reference sub-types: {len(cross)}/6 (need 5) {sorted(cross)}")
print(f"stress-test sub-types:     {len(stress)}/11 (need 8)")
print(f"priority questions: {sum(q['priority'] for q in questions)} (need exactly 5)")

problems = 0
for i, q in enumerate(questions):
    if q["difficulty"] not in ("easy", "medium", "hard"):
        print(f"  [{i}] bad difficulty"); problems += 1
    if q["category"] in ("cross-reference", "stress-test") and "sub_type" not in q:
        print(f"  [{i}] missing sub_type"); problems += 1
    for c in q["expected_chunks"] or []:
        if (c["source"], c["page"]) not in pages:
            print(f"  [{i}] expected chunk {c['source']} p.{c['page']} NOT in corpus.json"); problems += 1
print(f"format problems: {problems}")

# ---------------- page check ----------------
# take the numbers from the expected answer (e.g. 215.9, 71.1%, 36,343,830)
# and check they appear on the expected pages
print("\n" + "=" * 60)
print("PAGE CHECK (numbers from expected_answer found on expected pages?)")
print("=" * 60)
for i, q in enumerate(questions):
    if not q["expected_chunks"] or not q["expected_answer"]:
        continue
    terms = set(re.findall(r"\d[\d,]*\.?\d*", q["expected_answer"]))
    terms = {t.rstrip(".,") for t in terms if len(t.rstrip(".,")) >= 3}  # skip short numbers like 65, too common
    if not terms:
        print(f"  [{i:2}] ?   no numbers to check, verify by hand: {q['question'][:55]}")
        continue
    text = "".join(pages.get((c["source"], c["page"]), "") for c in q["expected_chunks"])
    found = {t for t in terms if t in text}
    exp_pages = [c["page"] for c in q["expected_chunks"]]
    if len(found) == len(terms):
        print(f"  [{i:2}] OK  p.{exp_pages}")
    else:
        missing = terms - found
        # where are the missing numbers actually?
        where = {t: [p for (s, p), txt in pages.items() if t in txt][:5] for t in missing}
        flag = "!!" if not found else "~ "
        print(f"  [{i:2}] {flag}  p.{exp_pages} missing {sorted(missing)} -> found on pages {where}")

print("\n'!!' = none of the numbers are on the expected pages -> page number is probably wrong")
print("'~ ' = some numbers missing -> often fine (rounded/computed values), check by hand")
