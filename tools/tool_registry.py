"""
tools/tool_registry.py
─────────────────────────────────────────────────────────────────────────────
Tool definitions and execution engine for Stage-3 tool-use training and
model inference.

Tools:
  web_search       – Execute a web search and return top snippets
  fetch_url        – Retrieve and summarise a web page
  calculator       – Evaluate safe mathematical expressions
  python_exec      – Execute sandboxed Python code snippets
  summarise        – Summarise a long block of retrieved text

The ToolCall / ToolResult dataclasses mirror the XML token format the model
generates during Stage 3:

    <tool_call>{"name": "web_search", "query": "Avogadro's number"}</tool_call>
    <tool_result>6.02214076 × 10²³ mol⁻¹ …</tool_result>
─────────────────────────────────────────────────────────────────────────────
"""

import ast
import io
import json
import re
import textwrap
import traceback
import urllib.parse
import urllib.request
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


# ──────────────────────────────────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class ToolCall:
    name: str
    params: Dict[str, Any]

    @classmethod
    def from_json(cls, json_str: str) -> "ToolCall":
        data = json.loads(json_str)
        name = data.pop("name")
        return cls(name=name, params=data)

    def to_json(self) -> str:
        return json.dumps({"name": self.name, **self.params}, ensure_ascii=False)


@dataclass
class ToolResult:
    tool_name: str
    content: str
    success: bool
    error: Optional[str] = None

    def format_for_model(self) -> str:
        """Returns the string the model sees inside <tool_result>…</tool_result>."""
        if self.success:
            return self.content
        return f"[Error in {self.tool_name}] {self.error}"


# ──────────────────────────────────────────────────────────────────────────
# Individual Tool Implementations
# ──────────────────────────────────────────────────────────────────────────

def tool_web_search(query: str, max_results: int = 5, **_) -> ToolResult:
    """
    Perform a DuckDuckGo Instant-Answer API search (no API key required).
    For production, swap in a proper search API (Bing, Google, SerpAPI, etc.)
    """
    try:
        encoded = urllib.parse.quote_plus(query)
        url     = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_redirect=1&no_html=1"
        req     = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        snippets = []
        # Abstract text (instant answer)
        if data.get("AbstractText"):
            snippets.append(data["AbstractText"])
        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                snippets.append(topic["Text"])

        if not snippets:
            return ToolResult(
                tool_name="web_search",
                content=f"No instant-answer results found for: {query}",
                success=True,
            )
        content = "\n\n".join(snippets[:max_results])
        return ToolResult(tool_name="web_search", content=content, success=True)

    except Exception as e:
        return ToolResult(
            tool_name="web_search", content="", success=False,
            error=str(e),
        )


def tool_fetch_url(url: str, max_chars: int = 2000, **_) -> ToolResult:
    """
    Fetch a URL and return its plain-text content (stripped HTML).
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        # Very basic HTML stripping
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        text = text[:max_chars]

        return ToolResult(tool_name="fetch_url", content=text, success=True)
    except Exception as e:
        return ToolResult(tool_name="fetch_url", content="", success=False, error=str(e))


def tool_calculator(expression: str, **_) -> ToolResult:
    """
    Safely evaluate a mathematical expression using Python's `ast` module.
    Only allows: numbers, +, -, *, /, **, (, ), math functions.
    """
    import math as _math

    # Whitelist: literals and safe operators only
    SAFE_NAMES = {
        k: v for k, v in vars(_math).items() if not k.startswith("_")
    }
    SAFE_NAMES.update({"abs": abs, "round": round, "int": int, "float": float})

    try:
        # Parse and check AST for forbidden nodes
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
                # Allow calls only to whitelisted names
                if isinstance(node, ast.Call):
                    if not (isinstance(node.func, ast.Name) and node.func.id in SAFE_NAMES):
                        raise ValueError(f"Function '{getattr(node.func, 'id', '?')}' not allowed")
            elif isinstance(node, ast.Name) and node.id not in SAFE_NAMES:
                raise ValueError(f"Name '{node.id}' not allowed")

        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, SAFE_NAMES)
        return ToolResult(tool_name="calculator", content=str(result), success=True)
    except Exception as e:
        return ToolResult(tool_name="calculator", content="", success=False, error=str(e))


def tool_python_exec(code: str, timeout: int = 5, **_) -> ToolResult:
    """
    Execute a sandboxed Python snippet and capture stdout.

    Safety measures:
      • No import of os, sys, subprocess, socket, or open() calls
      • stdout captured; no file I/O
      • Execution timeout via threading

    WARNING: This is a best-effort sandbox. For production use, execute
    inside a container or use a proper sandbox like RestrictedPython.
    """
    import threading

    FORBIDDEN_PATTERNS = [
        r"\bimport\s+os\b", r"\bimport\s+sys\b", r"\bimport\s+subprocess\b",
        r"\bimport\s+socket\b", r"\bopen\s*\(", r"__import__",
        r"exec\s*\(", r"eval\s*\(",
    ]
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            return ToolResult(
                tool_name="python_exec", content="", success=False,
                error=f"Forbidden pattern detected: {pat}",
            )

    output_buf = io.StringIO()
    error_msg  = []

    def run():
        try:
            with redirect_stdout(output_buf):
                exec(code, {"__builtins__": {"print": print, "range": range, "len": len,
                                              "sum": sum, "map": map, "zip": zip,
                                              "list": list, "dict": dict, "set": set,
                                              "tuple": tuple, "str": str, "int": int,
                                              "float": float, "bool": bool,
                                              "abs": abs, "round": round, "min": min,
                                              "max": max, "sorted": sorted, "enumerate": enumerate}})
        except Exception:
            error_msg.append(traceback.format_exc())

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        return ToolResult(tool_name="python_exec", content="", success=False,
                          error="Execution timed out")

    if error_msg:
        return ToolResult(tool_name="python_exec", content="", success=False,
                          error=error_msg[0][:500])

    return ToolResult(tool_name="python_exec",
                      content=output_buf.getvalue()[:2000], success=True)


def tool_summarise(text: str, max_sentences: int = 5, **_) -> ToolResult:
    """
    Simple extractive summarisation: pick the first `max_sentences` sentences
    from the text. (Replace with a neural summariser in production.)
    """
    try:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        summary   = " ".join(sentences[:max_sentences])
        return ToolResult(tool_name="summarise", content=summary, success=True)
    except Exception as e:
        return ToolResult(tool_name="summarise", content="", success=False, error=str(e))


# ──────────────────────────────────────────────────────────────────────────
# Tool Registry
# ──────────────────────────────────────────────────────────────────────────
TOOL_REGISTRY: Dict[str, Callable] = {
    "web_search":   tool_web_search,
    "fetch_url":    tool_fetch_url,
    "calculator":   tool_calculator,
    "python_exec":  tool_python_exec,
    "summarise":    tool_summarise,
}

TOOL_DESCRIPTIONS = {
    "web_search": {
        "description": "Search the web for current information.",
        "params": {"query": "string – the search query"},
    },
    "fetch_url": {
        "description": "Retrieve and read the content of a web page.",
        "params": {"url": "string – the full URL to fetch"},
    },
    "calculator": {
        "description": "Evaluate a mathematical expression (supports standard math functions).",
        "params": {"expression": "string – e.g. 'sqrt(2) * pi'"},
    },
    "python_exec": {
        "description": "Execute a short Python code snippet and return stdout.",
        "params": {"code": "string – Python source code"},
    },
    "summarise": {
        "description": "Summarise a long piece of retrieved text.",
        "params": {"text": "string – the text to summarise"},
    },
}


def execute_tool(call: ToolCall) -> ToolResult:
    """Look up and execute a tool by name."""
    fn = TOOL_REGISTRY.get(call.name)
    if fn is None:
        return ToolResult(
            tool_name=call.name, content="", success=False,
            error=f"Unknown tool '{call.name}'. Available: {list(TOOL_REGISTRY.keys())}",
        )
    return fn(**call.params)


# ──────────────────────────────────────────────────────────────────────────
# Tool-Call Parser (extracts calls from model output)
# ──────────────────────────────────────────────────────────────────────────
TOOL_CALL_RE  = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_RESULT_RE = re.compile(r"<tool_result>(.*?)</tool_result>", re.DOTALL)

def parse_tool_calls(model_output: str) -> List[ToolCall]:
    """Extract all tool calls from a model response string."""
    calls = []
    for match in TOOL_CALL_RE.finditer(model_output):
        try:
            calls.append(ToolCall.from_json(match.group(1).strip()))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ToolRegistry] Failed to parse tool call: {e}")
    return calls


def inject_tool_results(model_output: str, results: List[ToolResult]) -> str:
    """
    Insert <tool_result>…</tool_result> tags into the model output
    after each matching <tool_call> block.
    """
    output = model_output
    for result in results:
        # Find the first <tool_call>…</tool_call> block that isn't already
        # followed by a <tool_result> block
        m = TOOL_CALL_RE.search(output)
        if m and not TOOL_RESULT_RE.search(output[m.end(): m.end() + 50]):
            insert_pos = m.end()
            result_tag = f"<tool_result>{result.format_for_model()}</tool_result>"
            output = output[:insert_pos] + result_tag + output[insert_pos:]
    return output


# ──────────────────────────────────────────────────────────────────────────
# System Prompt Builder
# ──────────────────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    """
    Builds a system prompt that describes available tools to the model.
    Used during Stage-3 synthetic data generation and inference.
    """
    tool_docs = []
    for name, info in TOOL_DESCRIPTIONS.items():
        params_str = ", ".join(f"{k}: {v}" for k, v in info["params"].items())
        tool_docs.append(
            f'• {name}({params_str})\n'
            f'  {info["description"]}'
        )

    tools_block = "\n".join(tool_docs)

    return textwrap.dedent(f"""\
        You are a precise, helpful assistant with access to external tools.

        ## Available Tools
        {tools_block}

        ## Tool Usage Format
        To call a tool, output a JSON block wrapped in <tool_call> tags:
            <tool_call>{{"name": "web_search", "query": "your query here"}}</tool_call>

        The system will reply with a <tool_result> block:
            <tool_result>...retrieved content...</tool_result>

        ## Guidelines
        - Use <think>…</think> to reason before deciding to call a tool.
        - Call tools ONLY when the answer cannot be derived from your training knowledge.
        - After receiving a result, synthesise it into a clear, complete answer.
        - For multi-step problems, chain multiple tool calls if needed.
        - Always provide a final answer in plain language after tool results.
    """)


if __name__ == "__main__":
    # Quick smoke-test
    print(build_system_prompt())
    print("\n--- Calculator test ---")
    result = execute_tool(ToolCall(name="calculator", params={"expression": "sqrt(144) + pi"}))
    print(result)
    print("\n--- Web search test ---")
    result = execute_tool(ToolCall(name="web_search", params={"query": "speed of light"}))
    print(result.content[:300])
