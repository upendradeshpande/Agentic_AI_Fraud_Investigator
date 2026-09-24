"""Live connectivity check for the configured LLM (makes one tiny, billable request).

    python -m cockpit.check_provider
    python -m cockpit.check_provider --provider anthropic --model claude-sonnet-5
"""
from __future__ import annotations

import argparse
import sys

from .llm import providers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", help="gemini | anthropic | openai | openai_compatible")
    ap.add_argument("--model")
    ap.add_argument("--base-url", default="")
    args = ap.parse_args()
    if args.provider:
        providers.configure(args.provider, args.model or "", "", args.base_url)
    p = providers.get_provider()
    print(f"Provider: {p.name}  model: {p.model}  key present: {bool(p.api_key)}")
    res = providers.check_connection(p)
    print(("PASS: " if res["ok"] else "FAIL: ") + res["message"])
    if res["ok"]:
        tools = [{"name": "ping", "description": "Return pong.", "parameters": {"type": "object", "properties": {}}}]
        try:
            r = p.chat("Call the ping tool.", [{"role": "user", "content": "Please call ping."}], tools=tools, max_tokens=60)
            print("PASS: tool calling works" if r.tool_calls else "WARN: model answered without calling the tool")
        except providers.LLMError as e:
            print(f"FAIL: tool calling: {e}")
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
