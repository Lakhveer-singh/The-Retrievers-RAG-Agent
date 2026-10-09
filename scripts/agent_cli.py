"""
Run the agent on one question in the terminal and print its full trajectory.
Usage (from repo root):
  python scripts/agent_cli.py "What share of fiscal 2026 revenue came from the Data Center?"
If the agent proposes a message, you are asked to approve it (y/n) before anything is sent.
"""
import sys

sys.path.append("app")
from agent import run_agent, log_send
import messaging

question = " ".join(sys.argv[1:]) or input("Question: ")
run = run_agent(question)

print(f"\nQUESTION: {question}\n")
print("TRAJECTORY:")
for s in run["steps"]:
    print(f"  step {s['step']}: {s['tool']}({s['args']})")
    print("     -> " + s["result"][:300].replace("\n", " ") + ("..." if len(s["result"]) > 300 else ""))
if not run["steps"]:
    print("  (no tools used)")

print(f"\nANSWER:\n{run['answer']}")
print(f"\nsources: {run['pages']} | model calls: {run['model_calls']} | cost: ${run['cost']:.4f}")

if run["pending_message"]:
    print("\n" + "=" * 60)
    print("THE AGENT WANTS TO SEND THIS MESSAGE:\n")
    print(run["pending_message"])
    print("=" * 60)
    if input("Send it? (y/n): ").strip().lower() == "y":
        result = messaging.send(run["pending_message"])
        log_send(run["pending_message"], "approved: " + result)
        print(result)
    else:
        log_send(run["pending_message"], "rejected")
        print("Rejected. Nothing was sent.")
