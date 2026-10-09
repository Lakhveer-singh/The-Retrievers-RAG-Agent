"""
Connects to Slack's MCP server as an MCP client and prints the tools it offers.
Use it to check your token works and to see the exact name/arguments of the send-message tool.
Usage (from repo root): python scripts/mcp_list_tools.py
Needs SLACK_USER_TOKEN in .env
"""
import asyncio
import json
import os

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

load_dotenv()
URL = "https://mcp.slack.com/mcp"


async def main():
    headers = {"Authorization": f"Bearer {os.environ['SLACK_USER_TOKEN']}"}
    async with streamablehttp_client(URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print(f"{len(tools)} tools on {URL}:\n")
            for t in tools:
                print(f"- {t.name}: {(t.description or '')[:100]}")
                if "send" in t.name.lower() and "message" in t.name.lower():
                    print("  arguments:", json.dumps(t.inputSchema.get("properties", {}), indent=2)[:800])


asyncio.run(main())
