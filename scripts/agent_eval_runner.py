"""
Usage (from repository root):
  python scripts/agent_eval_runner.py
"""
import json
import sys
import time

sys.path.append("app")
import rag
from agent import run_agent
from rag import answer

def run_agent_eval():
    with open("data/questions.json", "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    # Filter to agent eval questions + priority questions as required
    eval_questions = [q for q in all_questions if q.get("category") == "agent" or q.get("priority")]

    print(f"Running Agent Evaluation on {len(eval_questions)} questions...\n")
    print(f"{'Q#':<4} | {'Category/Sub-Type':<25} | {'Tools Used':<30} | {'Expected Tools':<25} | {'Steps':<6} | {'Agent Latency':<13} | {'RAG Latency':<11}")
    print("-" * 125)

    agent_results = []

    for idx, q in enumerate(eval_questions, 1):
        q_text = q["question"]
        expected_tools = set(q.get("expected_tools", []))

        # 1. Run Agent
        t0_agent = time.time()
        agent_run = run_agent(q_text)
        agent_latency = time.time() - t0_agent

        actual_tools = [s["tool"] for s in agent_run["steps"] if s["tool"] != "(step cap reached)"]
        tools_match = expected_tools.issubset(set(actual_tools)) if expected_tools else (len(actual_tools) == 0)

        # 2. Run RAG Pipeline (for Agent vs RAG comparison table)
        t0_rag = time.time()
        rag_answer_text, _ = answer(q_text)
        rag_latency = time.time() - t0_rag

        tool_str = ", ".join(actual_tools) if actual_tools else "None"
        exp_str = ", ".join(expected_tools) if expected_tools else "None"
        sub_type = q.get("sub_type", q.get("category", "-"))

        print(f"{idx:<4} | {sub_type:<25} | {tool_str[:28]:<30} | {exp_str[:23]:<25} | {agent_run['model_calls']:<6} | {agent_latency:<13.2f}s | {rag_latency:<11.2f}s")

        agent_results.append({
            "question": q_text,
            "category": q.get("category"),
            "sub_type": sub_type,
            "expected_tools": list(expected_tools),
            "actual_tools": actual_tools,
            "tools_match": tools_match,
            "agent_steps": agent_run["model_calls"],
            "agent_latency": round(agent_latency, 3),
            "rag_latency": round(rag_latency, 3),
            "agent_answer": agent_run["answer"],
            "rag_answer": rag_answer_text,
            "hit_step_cap": agent_run.get("hit_step_cap", False),
        })

    with open("data/agent_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(agent_results, f, ensure_ascii=False, indent=2)

    # Summary Output
    print("\n" + "=" * 60)
    print("AGENT EVALUATION SUMMARY")
    print("=" * 60)
    correct_tools = sum(1 for r in agent_results if r["tools_match"])
    print(f"Tool Trajectory Match: {correct_tools}/{len(agent_results)} ({correct_tools / len(agent_results):.1%})")
    avg_steps = sum(r["agent_steps"] for r in agent_results) / len(agent_results)
    print(f"Average Agent Steps:   {avg_steps:.2f}")
    step_cap_hits = sum(1 for r in agent_results if r["hit_step_cap"])
    print(f"Hit Step Cap:          {step_cap_hits}/{len(agent_results)}")
    print(f"Avg Agent Latency:     {sum(r['agent_latency'] for r in agent_results) / len(agent_results):.2f}s")
    print(f"Avg RAG Latency:       {sum(r['rag_latency'] for r in agent_results) / len(agent_results):.2f}s")
    print("=" * 60)
    print("Saved results to data/agent_eval_results.json")

if __name__ == "__main__":
    run_agent_eval()
