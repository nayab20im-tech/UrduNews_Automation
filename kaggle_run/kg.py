#!/usr/bin/env python3
"""Minimal Kaggle MCP JSON-RPC client (Bearer KGAT token from ~/.qoder/mcp.json).

Usage:
    python3 kg.py <tool_name> '<request-json>'
Prints the tool's text content (pretty JSON when possible).
"""
import json, sys, urllib.request

def token() -> str:
    cfg = json.load(open("/home/guest/.qoder/mcp.json"))
    args = cfg["mcpServers"]["kaggle"]["args"]
    return args[args.index("Authorization: Bearer") + 1].split()[-1] \
        if "Authorization: Bearer" in args else args[-1].split()[-1]

def call(name: str, request: dict, rid: int = 1):
    body = json.dumps({
        "jsonrpc": "2.0", "id": rid, "method": "tools/call",
        "params": {"name": name, "arguments": {"request": request}},
    }).encode()
    req = urllib.request.Request(
        "https://www.kaggle.com/mcp", data=body, headers={
            "Authorization": f"Bearer {token()}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }, method="POST")
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read().decode()
    # SSE: lines like "event: message" / "data: {...}"
    for line in raw.splitlines():
        if line.startswith("data: "):
            msg = json.loads(line[6:])
            if "error" in msg:
                raise SystemExit(f"MCP error: {msg['error']}")
            content = msg["result"].get("content", [])
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            out = "\n".join(texts)
            if msg["result"].get("isError"):
                raise SystemExit(f"tool error: {out}")
            try:
                return json.loads(out)
            except (json.JSONDecodeError, ValueError):
                return out
    raise SystemExit(f"no data frame in response:\n{raw}")

if __name__ == "__main__":
    name = sys.argv[1]
    request = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    result = call(name, request)
    print(result if isinstance(result, str) else json.dumps(result, indent=2))
