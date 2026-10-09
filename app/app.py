import re

import gradio as gr

from rag import answer, NO_INFO, total_cost
from agent import run_agent, log_send
import messaging

# Final Demo Questions for RAG tab (covering single chunk, multi-chunk, memory, out-of-scope, injection, BM25)
EXAMPLE_QUESTIONS = [
    "What was NVIDIA's gross margin for fiscal year 2026?",
    "Who is NVIDIA's Chief Executive Officer and who is the Chief Financial Officer?",
    "What was NVIDIA's total revenue in fiscal 2026?",
    "And what was it for fiscal 2025?",
    "What was Apple's total iPhone revenue in 2026?",
    "Summarize the section on cybersecurity risks.",
    "What does Item 1A discuss in the Form 10-K?",
    "Under the H200 export licensing program, what tariff applies to shipments into the U.S., and what problem does NVIDIA flag with recovering that cost?",
    "How much dough did the company hand back to its owners as dividends last year?",
    "How much did NVIDIA's total revenue grow in dollars from fiscal 2025 to fiscal 2026, and what is that as a percentage?",
    "Who is the CEO of NVIDIA?",
    "Divide his total compensation by the median employee's. Does it match the reported ratio?",
    "What is NVIDIA's official revenue guidance for fiscal year 2027?",
    "What was the Data Center revenue for fiscal year 2026?"
]

# Final Demo Questions for Agent tab (covering all 9 validated scenarios)
AGENT_EXAMPLES = [
    "What was NVIDIA's gross margin for fiscal year 2026?",
    "Who is NVIDIA's President and CEO, and who is the Chief Financial Officer?",
    "What was the total capital distributed as financial remuneration to equity proprietors last year?",
    "What is the exact numerical difference in Data Center revenue between fiscal 2025 and fiscal 2026?",
    "Which documents do you have access to?",
    "Hi! What can you help me with?",
    "What was NVIDIA's total revenue for fiscal year 2026? Post the answer to our team channel.",
    "How many smartphones did NVIDIA manufacture and sell in fiscal year 2026?",
    "Summarize the section on cybersecurity risks.",
    "Check this claim: Did NVIDIA's Data Center revenue decrease in fiscal year 2026?",
    "What does NVIDIA mean by the term 'Omniverse'?",
    "If a product launched in September 2025, which NVIDIA fiscal quarter does that fall into?",
    "What were the revenues for the Data Center, Gaming, and Professional Visualization segments in fiscal 2026? Use the chart generator to visualize the comparison.",
    "Assess the severity of the risk regarding US export controls to China."
]

MAX_HISTORY_TURNS = 3  # how many past turns the model sees for follow-ups


def format_history(history):
    """Render the whole conversation as Markdown, newest first."""
    if not history:
        return "_No questions asked yet._"
    parts = []
    for n, turn in reversed(list(enumerate(history, start=1))):
        parts.append(f"**Q{n}: {turn['q']}**\n\n{turn['a']}")
    return "\n\n---\n\n".join(parts)


# ---------------- RAG tab ----------------
def ask(question, history):
    history = history or []
    if not question.strip():
        return "Please type a question.", "", "", format_history(history), history

    text, hits = answer(question, history[-MAX_HISTORY_TURNS:])

    if text == NO_INFO:
        sources = "_No relevant source found._"
    else:
        pages = []
        for h in hits:
            if (h["source"], h["page"]) not in pages:
                pages.append((h["source"], h["page"]))
        sources = "\n".join(f"- {src}, page {page}" for src, page in pages)

    passages = "\n\n---\n\n".join(
        f"**{h['source']} p. {h['page']}** (score {h['score']:.2f})\n\n{h['text'][:600]}..." for h in hits
    )
    cost = f"Total spent so far: ${total_cost():.4f} / $5.00"

    history = history + [{"q": question, "a": text}]
    return text, sources, passages + "\n\n" + cost, format_history(history), history


def clear_history():
    return "", "", "", "_No questions asked yet._", []


# ---------------- Agent tab ----------------
def ask_agent(question, history):
    history = history or []
    if not question.strip():
        return "Please type a question.", "", "", "", "", "", format_history(history), history

    run = run_agent(question, history[-MAX_HISTORY_TURNS:])
    text = run["answer"] or ""

    # 1. Step trace: tool and arguments in order
    lines = []
    for s in run["steps"]:
        args = ", ".join(f"{k}={v!r}" for k, v in s["args"].items())
        lines.append(f"{len(lines) + 1}. `{s['tool']}({args})`")
    tool_calls = "\n".join(lines) if lines else "_No tools used (answered directly)._"
    
    tool_calls += f"\n\n**Run Stats:** {run['model_calls']} model calls | Cost: ${run['cost']:.4f} | Total:${total_cost():.4f} / $5.00"

    # 2. Full trace: every tool call with arguments and output result
    full_trace_blocks = []
    for s in run["steps"]:
        if s["tool"] == "(step cap reached)":
            full_trace_blocks.append("### Step: Step cap reached\nForced final response generation.")
            continue
        args_str = ", ".join(f"{k}={v!r}" for k, v in s["args"].items())
        res_text = str(s.get("result", "")).strip()
        if len(res_text) > 800:
            res_text = res_text[:800] + " ... [truncated]"
        full_trace_blocks.append(
            f"**Step {s['step']}:** `{s['tool']}`\n\n"
            f"- **Arguments:** `{args_str}`\n"
            f"- **Result:**\n```text\n{res_text}\n```"
        )
    full_trace = "\n\n---\n\n".join(full_trace_blocks) if full_trace_blocks else "_No tool executions recorded._"

    cited = sorted({int(p) for p in re.findall(r"p\.\s*(\d+)", text)})
    shown = cited or run["pages"]
    sources = "\n".join(f"- page {p}" for p in shown) if shown else "_No source used._"

    if run["pending_message"]:
        pending = run["pending_message"]
        status = "⚠️ **Action pending approval:** The agent proposes sending this message. Review and confirm below."
    else:
        pending = ""
        status = "No message waiting for approval."

    history = history + [{"q": question, "a": text}]
    return text, tool_calls, sources, full_trace, pending, status, format_history(history), history


def clear_agent():
    return "", "_No tools used yet._", "_No source used._", "_Detailed tool results will appear here._", "", "", "_No questions asked yet._", []


def approve(pending):
    if not pending.strip():
        return pending, "Nothing to send."
    try:
        result = messaging.send(pending)
        log_send(pending, "approved: " + result)
        return "", "✅ " + result
    except Exception as e:
        log_send(pending, f"approved but failed: {e}")
        return pending, f"❌ Sending failed: {e}"


def reject(pending):
    if pending.strip():
        log_send(pending, "rejected")
    return "", "🚫 Message rejected. Nothing was sent."


with gr.Blocks(title="The Retrievers - NVIDIA Annual Report Q&A") as demo:
    gr.Markdown("# NVIDIA Annual Report 2026 - Q&A\nAsk anything about NVIDIA's fiscal year 2026 annual report.")

    # ---------- RAG tab ----------
    with gr.Tab("RAG"):
        history_state = gr.State([])

        example_dd = gr.Dropdown(
            choices=EXAMPLE_QUESTIONS,
            value=None,
            label="Example questions (pick one, or type your own below)",
        )
        question = gr.Textbox(label="Your question", placeholder="e.g. What was NVIDIA's total revenue in fiscal 2026?")
        with gr.Row():
            ask_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear conversation")

        answer_box = gr.Markdown(label="Answer")
        gr.Markdown("### Sources")
        sources_box = gr.Markdown()
        with gr.Accordion("Retrieved passages", open=False):
            passages_box = gr.Markdown()
        with gr.Accordion("Conversation history", open=False):
            history_box = gr.Markdown("_No questions asked yet._")

        outputs = [answer_box, sources_box, passages_box, history_box, history_state]

        example_dd.change(lambda q: q or "", example_dd, question)
        ask_btn.click(ask, [question, history_state], outputs)
        question.submit(ask, [question, history_state], outputs)
        clear_btn.click(clear_history, None, [question, answer_box, sources_box, history_box, history_state])

    # ---------- Agent tab ----------
    with gr.Tab("Agent"):
        agent_history = gr.State([])

        a_example = gr.Dropdown(
            choices=AGENT_EXAMPLES,
            value=None,
            label="Example multi-step questions (pick one, or type your own below)",
        )
        a_question = gr.Textbox(
            label="Your question",
            placeholder="e.g. What is the exact numerical difference in Data Center revenue between fiscal 2025 and fiscal 2026?",
        )
        with gr.Row():
            a_btn = gr.Button("Ask the agent", variant="primary", scale=2)
            a_clear = gr.Button("Clear conversation", scale=1)

        # Side-by-side layout: Answer + Action Gate (Left), Step Trace + Sources + Full Trace (Right)
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Answer")
                a_answer = gr.Markdown(value="_The agent's answer will appear here._")

                with gr.Group():
                    gr.Markdown("### Send to Teams channel (Human Approval)")
                    a_pending = gr.Textbox(label="Proposed message (editable before dispatch)", lines=4)
                    with gr.Row():
                        approve_btn = gr.Button("✅ Approve & send", variant="primary")
                        reject_btn = gr.Button("🚫 Reject")
                    a_status = gr.Markdown("No message waiting for approval.")

            with gr.Column(scale=2):
                gr.Markdown("### Step Trace")
                a_tools = gr.Markdown(value="_No tools used yet._")

                gr.Markdown("### Sources")
                a_sources = gr.Markdown(value="_No source used._")

                with gr.Accordion("Full Trace (Tool Arguments & Raw Results)", open=False):
                    a_full_trace = gr.Markdown(value="_Detailed tool results will appear here._")

        with gr.Accordion("Conversation history", open=False):
            a_history_box = gr.Markdown("_No questions asked yet._")

        a_outputs = [
            a_answer,
            a_tools,
            a_sources,
            a_full_trace,
            a_pending,
            a_status,
            a_history_box,
            agent_history,
        ]

        a_example.change(lambda q: q or "", a_example, a_question)
        a_btn.click(ask_agent, [a_question, agent_history], a_outputs)
        a_question.submit(ask_agent, [a_question, agent_history], a_outputs)
        a_clear.click(
            clear_agent,
            None,
            [
                a_answer,
                a_tools,
                a_sources,
                a_full_trace,
                a_pending,
                a_status,
                a_history_box,
                agent_history,
            ],
        )
        approve_btn.click(approve, a_pending, [a_pending, a_status])
        reject_btn.click(reject, a_pending, [a_pending, a_status])

if __name__ == "__main__":
    demo.launch()
