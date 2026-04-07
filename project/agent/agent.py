"""
agent.py
--------
LangChain + Ollama agent that answers natural-language questions about the
NYC Airbnb Hospitality Lakehouse by orchestrating the four MCP tools in
mcp_server/server.py.

Framework: LangGraph ReAct agent (langgraph.prebuilt.create_react_agent)
LLM:       Ollama (local, no API key required) — default model: llama3.2
           Override: set OLLAMA_MODEL env variable (e.g. llama3.1, qwen2.5)
           Requires: ollama pull llama3.2  (or your chosen model)

MCP tools (loaded automatically from server.py at startup):
  query_warehouse          — SELECT queries against PostgreSQL star schema
  query_geodata            — MongoDB geospatial / document queries
  search_reviews           — semantic search over 56k guest reviews (ChromaDB)
  get_neighbourhood_context — semantic search over Wikipedia paragraphs (ChromaDB)

Transparency:
  Every reasoning step, tool call, tool input, and tool output is printed to
  stdout as it occurs. This makes the agent's decision trace fully auditable
  without reading framework internals.

  Output format per step:
    [STEP N] <message type>
    Tool call:   <tool_name>(<args>)
    Tool result: <truncated result>
    Final:       <answer text>

Usage:
  # Interactive REPL
  python3 agent/agent.py

  # Single question
  python3 agent/agent.py -q "Which borough has the most Airbnb listings?"

  # Use a different Ollama model
  OLLAMA_MODEL=qwen2.5 python3 agent/agent.py -q "..."

Environment variables (from .env):
  OLLAMA_MODEL   — Ollama model name (default: llama3.1)
  OLLAMA_HOST    — Ollama server URL (default: http://localhost:11434)
  All store credentials for server.py (POSTGRES_*, MONGO_*, CHROMA_PATH)

Version justification:
  See project/versions.txt — documents dry-run conflict testing, import
  compatibility tests, and the AgentExecutor→create_react_agent migration
  finding that determined the final pinned versions.
"""

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

import logging

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

console = Console(highlight=False)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

AGENT_DIR   = Path(__file__).resolve().parent
PROJECT_DIR = AGENT_DIR.parent
REPO_ROOT   = PROJECT_DIR.parent
SERVER_PATH = PROJECT_DIR / "mcp_server" / "server.py"

load_dotenv(REPO_ROOT / ".env")

# ── Schema introspection ───────────────────────────────────────────────────────

_WAREHOUSE_TABLES = [
    "dim_neighbourhood", "dim_listing", "dim_host", "dim_date",
    "fact_review", "fact_review_weather", "noise_borough_month",
]


def _fetch_pg_schema() -> str:
    """
    Query information_schema at startup to get the live column list for every
    warehouse table. Returns a formatted string ready for the system prompt.

    Falls back to an empty string if the database is unreachable — the agent
    can still run; it just won't have schema context.
    """
    try:
        from sqlalchemy import create_engine, text
        import pandas as pd

        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        db   = os.getenv("POSTGRES_DB",   "airbnb")
        user = os.getenv("POSTGRES_USER", "postgres")
        pw   = os.getenv("POSTGRES_PASSWORD", "postgres")
        url  = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"

        engine = create_engine(url, pool_pre_ping=True)
        table_list = ", ".join(f"'{t}'" for t in _WAREHOUSE_TABLES)
        sql = text(f"""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name IN ({table_list})
            ORDER BY table_name, ordinal_position
        """)

        with engine.connect() as conn:
            df = pd.read_sql(sql, conn)

        lines = []
        for table, group in df.groupby("table_name"):
            cols = ", ".join(group["column_name"].tolist())
            lines.append(f"   {table}: {cols}")
        return "\n".join(lines)

    except Exception as exc:
        log.warning(f"Could not fetch live schema from PostgreSQL: {exc}")
        return "   (schema unavailable — database unreachable at startup)"


# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """You are a data analyst with access to the NYC Airbnb Hospitality Lakehouse.
You have four tools to answer questions:

1. query_warehouse — SQL SELECT against PostgreSQL. Live schema (columns fetched from the database):

{schema}

   IMPORTANT rules:
   - Use ONLY the column names listed above. Never invent columns.
   - dim_listing has NO borough or neighbourhood name columns — only neighbourhood_id.
     To get borough or neighbourhood name: JOIN dim_listing ON neighbourhood_id to dim_neighbourhood.
   - Example: SELECT n.borough, COUNT(*) FROM dim_listing l JOIN dim_neighbourhood n ON l.neighbourhood_id = n.neighbourhood_id GROUP BY n.borough
   - If a query returns an error, call the tool again with corrected SQL. NEVER write tool calls as JSON in your text response — always use the actual tool call mechanism.

2. query_geodata — MongoDB. Collections: neighbourhoods, places (POIs).

3. search_reviews — Semantic search over 56,132 guest reviews.

4. get_neighbourhood_context — Semantic search over 598 Wikipedia paragraphs.

Tool selection rules (follow strictly):
- "What restaurants/bars/hotels/places are near X?" → query_geodata on collection "places" with filter {{"neighbourhood": "X", "types": "restaurant"}}
- "What do guests say / reviews mention" → search_reviews
- Counts, averages, rankings, occupancy, scores → query_warehouse
- Neighbourhood history, character, demographics → get_neighbourhood_context
- Noise levels → query_warehouse on noise_borough_month
- "Closest to subway/transit" → query_warehouse: SELECT n.neighbourhood, n.poi_transit_station FROM dim_neighbourhood n ORDER BY n.poi_transit_station DESC LIMIT 10
- Multi-part questions: use multiple tools in sequence

CRITICAL: You MUST invoke tools using the function-call interface — never write {{"name": "tool_name", "parameters": {{...}}}} as plain text. Plain-text JSON will NOT execute and will produce no answer. Call the tool directly.

Always explain findings in plain language with specific numbers. Be concise."""


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY — spinner, tool call formatting, result tables
# ══════════════════════════════════════════════════════════════════════════════

# Human-readable tool labels
_TOOL_LABEL = {
    "query_warehouse":          "Query Warehouse",
    "query_geodata":            "Query Geodata",
    "search_reviews":           "Search Reviews",
    "get_neighbourhood_context": "Get Neighbourhood Context",
}

# Spinner state
_spinner_stop  = threading.Event()
_spinner_thread: threading.Thread | None = None


def _spinner_loop():
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not _spinner_stop.is_set():
        sys.stdout.write(f"\r\033[32m{frames[i % len(frames)]} Thinking...\033[0m")
        sys.stdout.flush()
        i += 1
        time.sleep(0.09)
    sys.stdout.write("\r\033[K")   # clear line
    sys.stdout.flush()


def _start_spinner():
    global _spinner_thread
    _spinner_stop.clear()
    _spinner_thread = threading.Thread(target=_spinner_loop, daemon=True)
    _spinner_thread.start()


def _stop_spinner():
    _spinner_stop.set()
    if _spinner_thread:
        _spinner_thread.join(timeout=0.5)


def _print_tool_call(tool_name: str, args: dict):
    label = _TOOL_LABEL.get(tool_name, tool_name)
    console.print(f"\n[bold]→ {label}[/bold]")
    if tool_name == "query_warehouse" and "sql" in args:
        console.print(Text(args["sql"], style="color(93)"))   # violet
    elif tool_name == "query_geodata":
        filt = args.get("filter", {})
        console.print(Text(json.dumps(filt, ensure_ascii=False), style="color(93)"))
    elif tool_name in ("search_reviews", "get_neighbourhood_context"):
        console.print(Text(args.get("query", ""), style="color(93)"))


def _print_tool_result(tool_name: str, raw: str):
    """Render tool results as clean tables where possible."""
    # raw is a list of content blocks from MCP: [{"type":"text","text":"..."}]
    try:
        blocks = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(blocks, list) and blocks and "text" in blocks[0]:
            data_str = blocks[0]["text"]
        else:
            data_str = raw
        data = json.loads(data_str)
    except Exception:
        console.print(f"  [dim]{str(raw)[:300]}[/dim]")
        return

    if "error" in data:
        console.print(f"  [red]Error:[/red] {data['error']}")
        return

    if tool_name == "query_warehouse" and "rows" in data:
        rows = data["rows"]
        if not rows:
            console.print("  [dim](no rows returned)[/dim]")
            return
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        for col in rows[0].keys():
            table.add_column(str(col))
        for row in rows[:20]:
            table.add_row(*[str(v) if v is not None else "" for v in row.values()])
        console.print(table)

    elif tool_name == "query_geodata" and "documents" in data:
        docs = data["documents"]
        if not docs:
            console.print("  [dim](no documents found)[/dim]")
            return
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        for col in ["name", "rating", "vicinity", "neighbourhood"]:
            table.add_column(col)
        for doc in docs[:10]:
            types = doc.get("types", [])
            table.add_row(
                str(doc.get("name", "")),
                str(doc.get("rating", "")),
                str(doc.get("vicinity", ""))[:50],
                str(doc.get("neighbourhood", "")),
            )
        console.print(table)

    elif tool_name == "search_reviews" and "results" in data:
        results = data["results"]
        if not results:
            console.print("  [dim](no reviews found)[/dim]")
            return
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        table.add_column("Review", max_width=70)
        table.add_column("Neighbourhood", max_width=20)
        table.add_column("Date", max_width=12)
        for r in results[:5]:
            table.add_row(
                r.get("document", "")[:120],
                r.get("metadata", {}).get("neighbourhood", ""),
                r.get("metadata", {}).get("review_date", ""),
            )
        console.print(table)

    elif tool_name == "get_neighbourhood_context" and "results" in data:
        results = data["results"]
        if not results:
            console.print("  [dim](no context found)[/dim]")
            return
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
        table.add_column("Paragraph", max_width=80)
        table.add_column("Neighbourhood", max_width=20)
        for r in results[:3]:
            table.add_row(
                r.get("document", "")[:200],
                r.get("metadata", {}).get("neighbourhood", ""),
            )
        console.print(table)

    else:
        console.print(f"  [dim]{data_str[:400]}[/dim]")


def print_step(step_num: int, chunk: dict):
    """
    Print one LangGraph stream chunk — stops spinner first, prints, restarts spinner.
    """
    from langchain_core.messages import AIMessage, ToolMessage

    for node, node_update in chunk.items():
        if node == "__end__":
            continue
        if isinstance(node_update, dict) and "messages" in node_update:
            msgs = node_update["messages"]
        elif isinstance(node_update, list):
            msgs = node_update
        else:
            msgs = [node_update]

        for msg in msgs:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                _stop_spinner()
                for tc in msg.tool_calls:
                    _print_tool_call(tc["name"], tc["args"])
                _start_spinner()   # tool is running — show progress again
            elif isinstance(msg, ToolMessage):
                _stop_spinner()
                _print_tool_result(msg.name, msg.content)
                _start_spinner()   # model is processing result


# ══════════════════════════════════════════════════════════════════════════════
# AGENT
# ══════════════════════════════════════════════════════════════════════════════

async def run_agent(question: str) -> str:
    """
    Run the LangGraph ReAct agent against the question.

    Flow:
      1. Fetch live schema from PostgreSQL (information_schema)
      2. Build system prompt with real column names
      3. Launch server.py as a stdio subprocess via MultiServerMCPClient
      4. Load MCP tools as LangChain tools (load_mcp_tools)
      5. Build a LangGraph ReAct agent (create_react_agent)
      6. Stream execution — print each step as it occurs
      7. Return the final answer text

    Returns the final answer string.
    """
    from langchain_ollama import ChatOllama
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain.agents import create_agent

    # Build system prompt from the live database schema so the model always
    # has accurate column names regardless of schema evolution.
    schema_text = _fetch_pg_schema()
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(schema=schema_text)

    model_name = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=ollama_host,
        temperature=0,          # deterministic — important for reproducibility
        num_predict=2048,        # max tokens per response
    )

    # MultiServerMCPClient spawns server.py as a subprocess and translates
    # MCP tool schemas into LangChain BaseTool objects automatically.
    # Note: async context manager removed in langchain-mcp-adapters>=0.1.0
    # (see project/versions.txt, Test 4). Use await client.get_tools() instead.
    client = MultiServerMCPClient(
        {
            "nyc-airbnb-lakehouse": {
                "command": sys.executable,
                "args": [str(SERVER_PATH)],
                "transport": "stdio",
                "env": dict(os.environ),
            }
        }
    )
    tools = await client.get_tools()

    # Green tick list — one tool per line
    console.print()
    for t in tools:
        label = _TOOL_LABEL.get(t.name, t.name)
        console.print(f"[green]✓[/green] {label}")
    console.print()

    _start_spinner()

    # create_react_agent moved to langchain.agents.create_agent in langchain 1.x
    # (see project/versions.txt, Test 5). Parameter renamed prompt → system_prompt.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    # astream with "updates" yields one dict per node — each dict shows what
    # changed in that step (tool calls, tool results, model responses).
    # We print each step for auditability and capture the last AI text as answer.
    from langchain_core.messages import AIMessage, ToolMessage
    final_answer = ""
    step_num = 0

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="updates",
    ):
        step_num += 1
        print_step(step_num, chunk)

        # Capture last AIMessage with text content (no tool calls) as final answer
        for node_update in chunk.values():
            if isinstance(node_update, dict) and "messages" in node_update:
                msgs = node_update["messages"]
            elif isinstance(node_update, list):
                msgs = node_update
            else:
                msgs = [node_update]
            for msg in msgs:
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    text = msg.content if isinstance(msg.content, str) else \
                           " ".join(b.get("text", "") for b in msg.content if isinstance(b, dict))
                    if text.strip():
                        final_answer = text

    # Safety net: llama3.2 sometimes writes a tool call as JSON text instead of
    # using the actual tool mechanism. Detect any embedded JSON tool call and
    # execute it by importing the server functions directly.
    _TOOL_NAMES = (
        "query_warehouse", "query_geodata",
        "search_reviews", "get_neighbourhood_context",
    )
    if final_answer:
        import re
        import importlib
        detected_tool = next(
            (t for t in _TOOL_NAMES if f'"name": "{t}"' in final_answer), None
        )
        if detected_tool:
            match = re.search(r'\{.*?"name".*?\}', final_answer, re.DOTALL)
            if match:
                try:
                    embedded  = json.loads(match.group())
                    params    = embedded.get("parameters", {})

                    # Sanitise common llama3.2 parameter bugs
                    if "projection" in params and isinstance(params["projection"], str):
                        params["projection"] = {}
                    if "sort" in params and isinstance(params["sort"], list):
                        clean = []
                        for item in params["sort"]:
                            if isinstance(item, list) and len(item) == 2:
                                field, direction = item
                                direction = direction if isinstance(direction, int) else -1
                                clean.append([field, direction])
                        params["sort"] = clean or None
                    if "limit" in params and not isinstance(params["limit"], int):
                        try:
                            params["limit"] = int(params["limit"])
                        except (ValueError, TypeError):
                            params["limit"] = 20

                    console.print(f"\n[dim]→ recovering {detected_tool} call[/dim]")
                    sys.path.insert(0, str(PROJECT_DIR / "mcp_server"))
                    server_mod = importlib.import_module("server")
                    fn         = getattr(server_mod, detected_tool)
                    result     = fn(**params)
                    _print_tool_result(detected_tool, result)
                    final_answer = "(see result above)"
                except Exception as e:
                    log.warning(f"Recovery failed: {e}")

    _stop_spinner()
    return final_answer or "(Agent completed — check step output above)"


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="NYC Airbnb Lakehouse Agent — LangChain + Ollama + MCP"
    )
    parser.add_argument("-q", "--question", help="Single question (non-interactive)")
    args = parser.parse_args()

    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    print(f"Agent initialising — model: {model} | server: {SERVER_PATH.name}")

    if args.question:
        console.print(f"\n[dim]Question:[/dim] {args.question}")
        console.print("─" * 70)
        answer = asyncio.run(run_agent(args.question))
        console.print(f"\n{'─' * 70}")
        console.print(answer)
        return

    # Interactive REPL
    console.print("What would you like to know about Airbnb in New York City?\n")
    first = True
    while True:
        try:
            prompt = "Ask anything about NYC Airbnb → " if first else "Next question → "
            question = input(prompt).strip()
            first = False
        except (KeyboardInterrupt, EOFError):
            console.print("\nBye.")
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        answer = asyncio.run(run_agent(question))
        console.print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
