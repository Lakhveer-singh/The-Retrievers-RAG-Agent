"""
Usage:
  python scripts/retrieval_checker.py "What was total revenue in fiscal 2026?"
  python scripts/retrieval_checker.py 7
  python scripts/retrieval_checker.py 7 10
"""
import json
import sys

sys.path.append("app")
from rag import retrieve, TOP_K_FINAL, CONFIDENCE_THRESHOLD

with open("data/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

arg = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_K_FINAL

if arg.isdigit():
    q = questions[int(arg)]
else:
    q = next((x for x in questions if x["question"].strip().lower() == arg.strip().lower()),
             {"question": arg, "expected_chunks": None})

print(f"\nQUESTION: {q['question']}")
if q.get("category"):
    print(f"category: {q['category']} / {q.get('sub_type', '-')} / {q.get('difficulty')}")
if q.get("expected_answer"):
    print(f"expected answer: {q['expected_answer']}")

hits = retrieve(q["question"], k)
found = [(h["source"], h["page"]) for h in hits]

print(f"\nTOP-{k} RETRIEVED:")
for rank, h in enumerate(hits, 1):
    print(f"\n#{rank}  rerank_score={h['score']:.4f}  {h['source']} p.{h['page']}")
    print("   " + h["text"][:300].replace("\n", " ") + "...")

if hits and hits[0]["score"] < CONFIDENCE_THRESHOLD:
    print(f"\n⚠️ Top score {hits[0]['score']:.4f} < CONFIDENCE_THRESHOLD {CONFIDENCE_THRESHOLD} -> system will refuse to answer.")

if q.get("expected_chunks"):
    print("\nEXPECTED CHUNKS:")
    for e in q["expected_chunks"]:
        key = (e["source"], e["page"])
        if key in found:
            print(f"  ✅ {e['source']} p.{e['page']}  (rank {found.index(key) + 1})")
        else:
            print(f"  ❌ {e['source']} p.{e['page']}  (missed)")
else:
    print("\n(no expected chunks for this question)")
