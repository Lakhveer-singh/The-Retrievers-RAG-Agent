"""
Usage:
  python scripts/eval_runner.py --retrieval-only
  python scripts/eval_runner.py --full
"""
import argparse
import json
import sys
import time
from collections import defaultdict

sys.path.append("app")
import rag
from rag import answer, retrieve, NO_INFO

parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
parser.add_argument("--retrieval-only", action="store_true", help="Run retrieval only (calculates recall@k and MRR without generating answers)")
parser.add_argument("--full", action="store_true", help="Run full pipeline including answer generation and LLM judging")
args = parser.parse_args()

mode = "retrieval-only" if args.retrieval_only or not args.full else "full"
print(f"Running evaluation in [{mode.upper()}] mode...\n")

JUDGE_PROMPT = """You grade a question-answering system.
Question: {question}
Expected answer: {expected}
System answer: {answer}

Is the system answer correct, i.e. does it contain the key facts of the expected answer without contradicting it?
Reply with one word: CORRECT or INCORRECT."""

def judge(question, expected, got):
    resp = rag.client.chat.completions.create(
        model=rag.CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, expected=expected, answer=got)}],
    )
    rag.track(rag.CHAT_MODEL, resp.usage, "judge")
    return resp.choices[0].message.content.strip().upper().startswith("CORRECT")

with open("data/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

results = []
recalls_1, recalls_3, recalls_5, mrrs, latencies = [], [], [], [], []

for i, q in enumerate(questions):
    t0 = time.time()
    
    # Handle conversation history if available
    history = [{"q": h, "a": ""} for h in q.get("history", [])]

    if mode == "retrieval-only":
        search_query = q["question"]
        if history:
            search_query = history[-1]["q"] + " " + q["question"]
        hits = retrieve(search_query, k=5)
        text = ""
    else:
        text, hits = answer(q["question"], history=history)

    latency = time.time() - t0
    latencies.append(latency)

    found_pages = [(h["source"], h["page"]) for h in hits]
    
    # Calculate retrieval metrics on questions with expected chunks
    r1 = r3 = r5 = mrr = None
    if q.get("expected_chunks"):
        expected_tuples = {(e["source"], e["page"]) for e in q["expected_chunks"]}
        
        # Rank of first matching expected chunk
        first_rank = None
        for rank, p in enumerate(found_pages, 1):
            if p in expected_tuples:
                first_rank = rank
                break
        
        r1 = 1 if (first_rank is not None and first_rank <= 1) else 0
        r3 = 1 if (first_rank is not None and first_rank <= 3) else 0
        r5 = 1 if (first_rank is not None and first_rank <= 5) else 0
        mrr = (1.0 / first_rank) if first_rank is not None else 0.0
        
        recalls_1.append(r1)
        recalls_3.append(r3)
        recalls_5.append(r5)
        mrrs.append(mrr)

    correct = None
    if mode == "full":
        refused = (text.strip() == NO_INFO or "not confident" in text.lower())
        if q["category"] == "out-of-scope":
            correct = refused
        elif q["category"] == "ambiguous":
            correct = None
        elif q.get("sub_type") == "on_topic_unanswerable":
            correct = refused
        else:
            correct = False if refused else judge(q["question"], q["expected_answer"], text)

    results.append({
        "id": i,
        "question": q["question"],
        "category": q["category"],
        "sub_type": q.get("sub_type"),
        "latency": round(latency, 3),
        "recall@1": r1, "recall@3": r3, "recall@5": r5, "mrr": round(mrr, 4) if mrr is not None else None,
        "answer": text,
        "correct": correct
    })

    status = f"MRR={mrr:.2f}" if mrr is not None else "Out-of-scope"
    print(f"[{i:2}] {q['category']:15} {status:<12} | {q['question'][:50]}")

with open("data/eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print(f"EVALUATION SUMMARY ({mode.upper()})")
print("=" * 60)
if recalls_5:
    print(f"Recall@1: {sum(recalls_1) / len(recalls_1):.2%}")
    print(f"Recall@3: {sum(recalls_3) / len(recalls_3):.2%}")
    print(f"Recall@5: {sum(recalls_5) / len(recalls_5):.2%}")
    print(f"MRR:      {sum(mrrs) / len(mrrs):.4f}")
print(f"Avg Latency: {sum(latencies) / len(latencies):.2f}s")
if mode == "full":
    valid = [r for r in results if r["correct"] is not None]
    acc = sum(r["correct"] for r in valid) / len(valid) if valid else 0
    print(f"Accuracy: {acc:.2%}")
print("=" * 60)
