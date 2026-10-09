"""
Sends a message to the team channel. Two ways, chosen with SEND_MODE in .env:

  SEND_MODE=mcp      -> primary path: MCP client connected to Slack's official MCP server
                        needs SLACK_USER_TOKEN (xoxp-..., scope chat:write) and SLACK_CHANNEL_ID
  SEND_MODE=webhook  -> fallback: plain HTTP POST to an incoming webhook
                        needs WEBHOOK_URL and WEBHOOK_KIND=slack or teams

Secrets stay in .env (gitignored), never in the code.
"""
import asyncio
import os

import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_MCP_URL = "https://mcp.slack.com/mcp"


def send(text):
    mode = os.getenv("SEND_MODE", "webhook").lower()
    if mode == "mcp":
        return asyncio.run(send_via_mcp(text))
    return send_via_webhook(text)


# ---------------- primary path: MCP client ----------------
async def send_via_mcp(text):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    token = os.environ["SLACK_USER_TOKEN"]
    channel = os.environ["SLACK_CHANNEL_ID"]
    headers = {"Authorization": f"Bearer {token}"}

    async with streamablehttp_client(SLACK_MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools

            # find the server's "send message" tool and its argument names from its schema
            tool = next((t for t in tools if "send" in t.name.lower() and "message" in t.name.lower()), None)
            if tool is None:
                raise RuntimeError(f"No send-message tool on the MCP server. Tools: {[t.name for t in tools]}")
            props = tool.inputSchema.get("properties", {})
            channel_key = next(k for k in props if "channel" in k.lower())
            text_key = next((k for k in props if k.lower() in ("text", "message", "markdown_text", "content")),
                            next(k for k in props if k != channel_key))

            result = await session.call_tool(tool.name, {channel_key: channel, text_key: text})
            if result.isError:
                raise RuntimeError(f"MCP tool {tool.name} failed: {result.content}")
            return f"Sent to Slack via MCP (tool '{tool.name}')"


# ---------------- fallback: webhook ----------------
def send_via_webhook(text):
    url = os.environ["WEBHOOK_URL"]
    kind = os.getenv("WEBHOOK_KIND", "slack").lower()
    if kind == "teams":
        # format expected by a Teams Workflows "post to a channel when a webhook request is received" flow
        payload = {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [{"type": "TextBlock", "text": text, "wrap": True}],
                },
            }],
        }
    else:
        payload = {"text": text}  # Slack incoming webhook
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()
    return f"Sent to {kind} via webhook (HTTP {r.status_code})"
