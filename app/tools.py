import json
import re
from datetime import datetime

from simpleeval import simple_eval
import rag

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Search the NVIDIA Annual Report 2026 and return the most relevant passages "
                           "with their page numbers. Use this for ANY fact about NVIDIA. If the results "
                           "are not useful, call it again with different keywords.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'Data Center revenue fiscal 2026'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression, e.g. '193737 - 115186' or '9156 / 139297 * 100'. "
                           "Use it for every calculation instead of doing math in your head.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression using numbers and + - * / ( )"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List the source documents available in the corpus and how many pages each has.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Retrieve the text of a document or specific section by its source name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "The exact source name of the document to retrieve."},
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "today",
            "description": "Get the current date and time to answer date-relative questions.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fact_checker",
            "description": "Verify if a specific claim about NVIDIA is true, false, or unsupported based on the 10-K report. Returns a definitive verdict.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "description": "The specific statement to verify, e.g. 'NVIDIA's Data Center revenue dropped in 2026.'"}
                },
                "required": ["claim"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "financial_glossary",
            "description": "Lookup precise definitions for NVIDIA-specific technologies (e.g., Hopper, Blackwell, CUDA) and SEC financial jargon.",
            "parameters": {
                "type": "object",
                "properties": {
                    "term": {"type": "string", "description": "The term to define."}
                },
                "required": ["term"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fiscal_date_mapper",
            "description": "Map standard calendar dates to NVIDIA's specific fiscal quarters for FY26.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ascii_chart_generator",
            "description": "Generate a visual text-based bar chart for financial comparisons. Pass a JSON string mapping category names to numerical values.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_json": {"type": "string", "description": "JSON string of category:value pairs, e.g. '{\"Data Center\": 193.7, \"Gaming\": 10.4}'"}
                },
                "required": ["data_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "risk_severity_assessor",
            "description": "Assess the severity (High, Medium, Low) of a specific risk factor (e.g., 'export controls') based on the SEC 10-K language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_topic": {"type": "string", "description": "The risk topic to assess."}
                },
                "required": ["risk_topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Post a message to the team's Slack/Teams channel. ONLY use this when the user explicitly "
                           "asks to send, post or share something. A human must approve before it is sent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The full message to post, including page citations"},
                },
                "required": ["text"],
            },
        },
    },
]

def search_corpus(query):
    hits = rag.retrieve(query)
    if not hits or hits[0]["score"] < rag.CONFIDENCE_THRESHOLD:
        best = hits[0]["score"] if hits else 0
        return {
            "result": f"No matching passages found (best score {best:.4f}). "
                      f"Try again with different keywords, or say you don't have the information.",
            "pages": []
        }
    passages = [f"[p. {h['page']}] (score {h['score']:.4f})\n{h['text']}" for h in hits]
    return {"result": "\n\n".join(passages), "pages": [h["page"] for h in hits]}

def calculator(expression):
    expr = expression.replace("$", "").replace("%", "")
    expr = re.sub(r"(?<=\d),(?=\d{3})", "", expr)
    try:
        value = simple_eval(expr)
    except Exception as e:
        return {"result": f"Error: could not evaluate '{expression}' ({e}). Use only numbers and + - * / ( )."}
    if isinstance(value, float):
        value = round(value, 4)
    return {"result": f"{expression} = {value}"}

def list_documents():
    rag.load()
    pages = {}
    for c in rag.chunks:
        pages.setdefault(c["source"], set()).add(c["page"])
    docs = [f"{src} ({len(p)} pages with text, pages {min(p)}-{max(p)})" for src, p in pages.items()]
    return {"result": "Available documents:\n" + "\n".join(docs)}

def get_document(source):
    rag.load()
    doc_chunks = sorted([c for c in rag.chunks if c["source"] == source], key=lambda x: x["page"])
    if not doc_chunks:
        return {"result": f"Error: Document '{source}' not found. Use list_documents to see available sources."}
    full_text = "\n\n".join(f"[p. {c['page']}]\n{c['text']}" for c in doc_chunks[:10])
    return {"result": full_text}

def today():
    return {"result": f"Today's date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."}

def fact_checker(claim):
    hits = rag.retrieve(claim, k=3)
    if not hits or hits[0]["score"] < rag.CONFIDENCE_THRESHOLD:
        return {"result": "VERDICT: UNSUPPORTED. No relevant information found in the corpus to verify this claim."}
    
    context = "\n".join(f"[p. {h['page']}] {h['text'][:400]}" for h in hits)
    prompt = f"Context from NVIDIA 10-K:\n{context}\n\nClaim to verify: '{claim}'\n\nIs this claim TRUE, FALSE, or PARTIALLY TRUE based ONLY on the context? Provide a 1-sentence explanation."
    
    resp = rag.client.chat.completions.create(
        model=rag.CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"result": f"VERDICT: {resp.choices[0].message.content}"}

def financial_glossary(term):
    glossary = {
        "omniverse": "NVIDIA's computing platform that enables teams to develop Universal Scene Description (OpenUSD)-based 3D workflows.",
        "cuda": "Compute Unified Device Architecture. A parallel computing platform and programming model developed by NVIDIA.",
        "hopper": "NVIDIA's GPU architecture designed to accelerate AI, dynamic programming, and data analytics (e.g., H100 GPUs).",
        "blackwell": "NVIDIA's next-generation GPU architecture succeeding Hopper.",
        "gaap": "Generally Accepted Accounting Principles. The standard framework of guidelines for financial accounting.",
        "non-gaap": "Financial measures that exclude certain items (like stock-based compensation) to provide a different view of operating performance."
    }
    term_lower = term.lower()
    for key, definition in glossary.items():
        if key in term_lower:
            return {"result": f"Definition of {key}: {definition}"}
    return {"result": f"Term '{term}' not found in the glossary. Use search_corpus instead."}

def fiscal_date_mapper():
    mapping = (
        "NVIDIA's Fiscal Year 2026 ends on January 25, 2026.\n"
        "- Q1 FY26: February, March, April 2025\n"
        "- Q2 FY26: May, June, July 2025\n"
        "- Q3 FY26: August, September, October 2025\n"
        "- Q4 FY26: November, December 2025, January 2026"
    )
    return {"result": mapping}

def ascii_chart_generator(data_json):
    try:
        data = json.loads(data_json)
        if not data:
            return {"result": "No data provided to chart."}
        
        max_val = max(data.values())
        max_bar_length = 30
        
        chart_lines = ["\n📊 **Financial Visualization:**\n```text"]
        for category, value in data.items():
            bar_length = int((value / max_val) * max_bar_length) if max_val > 0 else 0
            bar = "█" * bar_length
            chart_lines.append(f"{category.ljust(20)} | {bar} {value}")
        chart_lines.append("```\n")
        
        return {"result": "\n".join(chart_lines)}
    except Exception as e:
        return {"result": f"Error generating chart: {e}"}

def risk_severity_assessor(risk_topic):
    hits = rag.retrieve(risk_topic, k=3)
    if not hits or hits[0]["score"] < rag.CONFIDENCE_THRESHOLD:
        return {"result": "ASSESSMENT: Unknown. No relevant risk disclosures found."}
    
    context = "\n".join(f"[p. {h['page']}] {h['text'][:400]}" for h in hits)
    prompt = (
        f"Context from NVIDIA 10-K:\n{context}\n\n"
        f"Evaluate the severity of the risk regarding '{risk_topic}'. "
        "Reply strictly with 'SEVERITY: HIGH', 'SEVERITY: MEDIUM', or 'SEVERITY: LOW' followed by a 1-sentence justification."
    )
    
    resp = rag.client.chat.completions.create(
        model=rag.CHAT_MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"result": resp.choices[0].message.content}

def run_tool(name, args):
    if name == "search_corpus":
        return search_corpus(args.get("query", ""))
    if name == "calculator":
        return calculator(args.get("expression", ""))
    if name == "list_documents":
        return list_documents()
    if name == "get_document":
        return get_document(args.get("source", ""))
    if name == "today":
        return today()
    if name == "fact_checker":
        return fact_checker(args.get("claim", ""))
    if name == "financial_glossary":
        return financial_glossary(args.get("term", ""))
    if name == "fiscal_date_mapper":
        return fiscal_date_mapper()
    if name == "ascii_chart_generator":
        return ascii_chart_generator(args.get("data_json", "{}"))
    if name == "risk_severity_assessor":
        return risk_severity_assessor(args.get("risk_topic", ""))
    return {"result": f"Error: unknown tool '{name}'"}
