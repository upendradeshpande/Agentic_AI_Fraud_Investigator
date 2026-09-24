"""Optional MCP server exposing the read and analysis tools (spec 11).

    pip install mcp
    python -m cockpit.mcp_server          # stdio transport

Write tools are intentionally not exposed: in this design only the UI, after investigator
confirmation, can change case state. The browser never talks to MCP directly.
"""
from __future__ import annotations


def main():
    from mcp.server.fastmcp import FastMCP

    from .agents.tools import build_tools
    from .service import CockpitService

    svc = CockpitService()
    svc.ensure_loaded()
    server = FastMCP("ai-investigation-cockpit")
    for tool in build_tools(svc).values():
        if tool.kind in ("read", "analysis"):
            server.add_tool(tool.fn, name=f"{tool.namespace}_{tool.name}", description=tool.description)
    server.run()


if __name__ == "__main__":
    main()
