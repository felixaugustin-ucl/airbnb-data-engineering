"""
agent.py
--------
Multi-agent LangGraph pipeline for the NYC Airbnb Hospitality Lakehouse.

Architecture — Planner → Executor(s) → Synthesizer:

  1. Planner Node
       ChatOllama(format="json") classifies the question into a routing plan.
       Structured output eliminates the free-text JSON tool-call problem that
       affects single-agent ReAct with small models (llama3.2).

  2. Executor Nodes  (each handles exactly one tool)
       Warehouse  — generates SQL and calls query_warehouse (PostgreSQL)
       Geo        — generates MongoDB filter and calls query_geodata
       Semantic   — calls search_reviews + get_neighbourhood_context (ChromaDB)

  3. Synthesizer Node
       Composes a plain-language answer from all accumulated tool results.

Routes:
  warehouse → Warehouse  → Synthesizer
  geodata   → Geo        → Synthesizer
  semantic  → Semantic   → Synthesizer
  multi     → Warehouse  → Semantic → Synthesizer

Rationale:
  The single-agent ReAct loop (feature/mcp) required llama3.2 to simultaneously
  choose between 4 tools, generate complex parameters, and write a final answer.
  Separating these into dedicated nodes with narrow responsibilities improves
  reliability significantly — each node solves an easier sub-problem.
  See project/versions.txt for benchmark notes.

  Pattern reference: LangGraph supervisor / planner-executor architecture.
  ReAct paper (Yao et al., 2022) informs the thought–act–observe loop at the
  node level; the supervisor pattern extends this to multi-agent coordination.

LLM: Groq (llama-3.3-70b-versatile) when GROQ_API_KEY is set in .env;
     local Ollama (default: llama3.2) otherwise — no API key required.
     Override model: GROQ_MODEL / OLLAMA_MODEL env variables.

Usage:
  python3 agent/agent.py            # interactive REPL
  python3 agent/agent.py -q "..."   # single question
"""

import argparse
import asyncio
import json
import logging
import warnings
warnings.filterwarnings("ignore", message=".*create_react_agent has been moved.*")
import operator
import os
import re
import sys
import threading
import time
from functools import partial
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

console = Console(highlight=False)
log     = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────────

AGENT_DIR   = Path(__file__).resolve().parent
PROJECT_DIR = AGENT_DIR.parent
REPO_ROOT   = PROJECT_DIR.parent
SERVER_PATH = PROJECT_DIR / "mcp_server" / "server.py"

load_dotenv(REPO_ROOT / ".env")

# ── Live schema introspection ───────────────────────────────────────────────────

_WAREHOUSE_TABLES = [
    "dim_neighbourhood", "dim_listing", "dim_host", "dim_date",
    "fact_review", "fact_review_weather", "noise_borough_month",
]


def _fetch_pg_schema() -> str:
    """
    Query information_schema at startup to get live column names for every
    warehouse table. Used to build the SQL-generation prompt in warehouse_node.
    Falls back gracefully if the database is unreachable.
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

        engine     = create_engine(url, pool_pre_ping=True)
        table_list = ", ".join(f"'{t}'" for t in _WAREHOUSE_TABLES)
        sql        = text(f"""
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
        log.warning(f"Could not fetch live schema: {exc}")
        return "   (schema unavailable — database unreachable at startup)"


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY — spinner, tool headers, result tables
# ══════════════════════════════════════════════════════════════════════════════

_TOOL_LABEL = {
    "query_warehouse":            "Query Warehouse",
    "query_geodata":              "Query Geodata",
    "search_reviews":             "Search Reviews",
    "get_neighbourhood_context":  "Get Neighbourhood Context",
}

# Place types actually stored in MongoDB places collection.
# Derived from Google Places API types returned for NYC POIs — used for
# routing correction so the signal is data-driven, not hand-picked keywords.
_GEO_PLACE_TYPES = frozenset({
    "restaurant", "bar", "lodging", "tourist_attraction", "transit_station",
    "cafe", "food", "night_club", "park", "museum", "bakery",
    "subway_station", "meal_takeaway", "meal_delivery", "bowling_alley",
})

_spinner_stop    = threading.Event()
_spinner_thread: threading.Thread | None = None
_spinner_message = "Thinking..."


def _spinner_loop():
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not _spinner_stop.is_set():
        sys.stdout.write(f"\r\033[32m{frames[i % len(frames)]} {_spinner_message}\033[0m")
        sys.stdout.flush()
        i += 1
        time.sleep(0.09)
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


def _start_spinner(message: str = "Thinking..."):
    global _spinner_thread, _spinner_message
    _spinner_message = message
    _spinner_stop.clear()
    _spinner_thread = threading.Thread(target=_spinner_loop, daemon=True)
    _spinner_thread.start()


def _stop_spinner():
    _spinner_stop.set()
    if _spinner_thread:
        _spinner_thread.join(timeout=0.5)


def _print_tool_call(tool_name: str, args: dict):
    """Print a tool invocation header with the query in violet."""
    label = _TOOL_LABEL.get(tool_name, tool_name)
    console.print(f"\n[bold]→ {label}[/bold]")
    if tool_name == "query_warehouse" and "sql" in args:
        console.print(Text(args["sql"], style="color(93)"))
    elif tool_name == "query_geodata":
        if args.get("pipeline"):
            # Show pipeline stages compactly (one stage per line)
            stages = ", ".join(
                list(stage.keys())[0] for stage in args["pipeline"] if isinstance(stage, dict)
            )
            console.print(Text(f"[pipeline: {stages}]", style="color(93)"))
        else:
            console.print(Text(json.dumps(args.get("filter", {}), ensure_ascii=False), style="color(93)"))
    elif tool_name in ("search_reviews", "get_neighbourhood_context"):
        console.print(Text(args.get("query", ""), style="color(93)"))


def _extract_result_data(raw: str) -> tuple[dict | None, str]:
    """
    Parse the raw tool result string into a Python dict.
    MCP tools return either a JSON string or a list of content blocks.
    Returns (parsed_dict, raw_str) — parsed_dict is None on failure.
    """
    try:
        blocks = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(blocks, list) and blocks and "text" in blocks[0]:
            data_str = blocks[0]["text"]
        else:
            data_str = raw if isinstance(raw, str) else json.dumps(raw)
        return json.loads(data_str), data_str
    except Exception:
        return None, str(raw)


def _print_tool_result(tool_name: str, raw: str):
    """Render a tool result as a rich table where possible."""
    data, data_str = _extract_result_data(raw)

    if data is None:
        console.print(f"  [dim]{data_str[:300]}[/dim]")
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
        # Aggregation results have dynamic computed keys (e.g. avg_rating, _id=borough).
        # Find mode always has a "name" field. Detect by its absence.
        is_aggregation = "name" not in docs[0]
        if is_aggregation:
            # Rename _id → group_by for readability, show all computed fields
            cols = list(docs[0].keys())
            display_cols = ["group_by" if c == "_id" else c for c in cols]
            for c in display_cols:
                table.add_column(str(c))
            for doc in docs[:20]:
                row = []
                for c in cols:
                    v = doc.get(c, "")
                    row.append(f"{v:.2f}" if isinstance(v, float) else str(v))
                table.add_row(*row)
        else:
            for col in ["name", "rating", "vicinity", "neighbourhood"]:
                table.add_column(col)
            for doc in docs[:10]:
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
        table.add_column("Review",        max_width=70)
        table.add_column("Neighbourhood", max_width=20)
        table.add_column("Date",          max_width=12)
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
        table.add_column("Paragraph",     max_width=80)
        table.add_column("Neighbourhood", max_width=20)
        for r in results[:3]:
            table.add_row(
                r.get("document", "")[:200],
                r.get("metadata", {}).get("neighbourhood", ""),
            )
        console.print(table)

    else:
        console.print(f"  [dim]{data_str[:400]}[/dim]")


def _summarise_for_synthesizer(tool_name: str, raw: str) -> str:
    """
    Extract a concise, text-readable summary of a tool result for the
    synthesizer prompt. Full rows/docs for warehouse/geo; document text for
    semantic results.
    """
    data, data_str = _extract_result_data(raw)
    if data is None:
        return data_str[:600]
    if "error" in data:
        return f"Error: {data['error']}"
    def _round_floats(obj):
        """Recursively round floats to 2dp so the synthesizer doesn't get 16-digit numbers."""
        if isinstance(obj, float):
            return round(obj, 2)
        if isinstance(obj, dict):
            return {k: _round_floats(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_floats(v) for v in obj]
        return obj

    if tool_name == "query_warehouse" and "rows" in data:
        return json.dumps(_round_floats(data["rows"][:15]), indent=2)
    if tool_name == "query_geodata" and "documents" in data:
        docs = [
            {k: v for k, v in d.items() if k not in ("geometry", "centroid")}
            for d in data["documents"][:8]
        ]
        return json.dumps(_round_floats(docs), indent=2)
    if tool_name in ("search_reviews", "get_neighbourhood_context") and "results" in data:
        return "\n\n".join(r.get("document", "")[:300] for r in data["results"][:5])
    return data_str[:600]


# ══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════

class LakehouseState(TypedDict):
    question:     str
    route:        str
    intents:      dict
    # Annotated with operator.add so each node appends to the list rather than
    # replacing it — results from Warehouse + Semantic both land in tool_results.
    tool_results: Annotated[list, operator.add]
    answer:       str


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Planner
# ══════════════════════════════════════════════════════════════════════════════

async def planner_node(state: LakehouseState, llm) -> dict:
    """
    Classifies the question into a routing plan using structured JSON output
    (ChatOllama format='json'). Constrained generation prevents the free-text
    JSON problem that affected the single-agent ReAct architecture.

    Returns: route + per-store intent descriptions consumed by executor nodes.
    """
    prompt = (
        "You are a routing agent for the NYC Airbnb data warehouse.\n"
        "Classify the question below and output ONLY valid JSON — no other text.\n\n"
        f"Question: {state['question']}\n\n"
        "DATA SOURCES:\n"
        "  warehouse  — SQL database: Airbnb listing counts, occupancy rates, review\n"
        "               scores, host stats, noise complaints, seasonal trends\n"
        "  geodata    — MongoDB: places with types=['restaurant','bar','lodging',\n"
        "               'tourist_attraction','transit_station','cafe','park','museum',...]\n"
        "               Fields: name, rating (1-5), user_ratings_total, neighbourhood, borough\n"
        "  semantic   — Vector search: guest review text, Wikipedia neighbourhood descriptions\n\n"
        "ROUTING EXAMPLES (follow these exactly):\n"
        '  "best rated restaurants by borough"          → {"route": "geodata", "geo_intent": "average restaurant rating per borough"}\n'
        '  "top bars in Brooklyn"                       → {"route": "geodata", "geo_intent": "bars in Brooklyn sorted by rating"}\n'
        '  "which borough has most listings"            → {"route": "warehouse", "warehouse_intent": "listing count by borough"}\n'
        '  "what do guests say about noise"             → {"route": "semantic", "semantic_intent": "noise complaints in reviews"}\n'
        '  "restaurant quality and occupancy by borough"→ {"route": "geodata+warehouse", "geo_intent": "avg restaurant rating per borough", "warehouse_intent": "avg occupancy per borough"}\n'
        '  "listing stats and guest impressions"        → {"route": "warehouse+semantic", "warehouse_intent": "listing stats", "semantic_intent": "guest experience"}\n\n'
        'Output this JSON structure:\n'
        '{"route": "warehouse|geodata|semantic|geodata+warehouse|warehouse+semantic",\n'
        ' "warehouse_intent": "SQL task description or null",\n'
        ' "geo_intent": "place/location query description or null",\n'
        ' "semantic_intent": "review/context search description or null"}'
    )

    _start_spinner("Planning...")

    response = await llm.ainvoke(prompt)
    _stop_spinner()

    _VALID_ROUTES = ("warehouse", "geodata", "semantic", "geodata+warehouse", "warehouse+semantic")
    try:
        plan  = json.loads(response.content)
        route = plan.get("route", "warehouse")
        if route not in _VALID_ROUTES:
            route = "warehouse"
    except Exception:
        plan  = {}
        route = "warehouse"

    # Routing correction — data-driven safety net.
    # Uses _GEO_PLACE_TYPES (derived from actual MongoDB types field values)
    # rather than hand-picked keywords, so it covers the full category set.
    # Also catches generic place phrases ("places near", "best rated venues").
    q = state["question"].lower()
    _WAREHOUSE_SIGNALS = frozenset({
        "occupancy", "listing", "price", "host", "review score",
        "noise", "complaint", "season", "how many", "average rating",
    })
    _GEO_PHRASES = ("places near", "near me", "best rated", "top rated",
                    "highest rated", "where to eat", "where to drink",
                    "venues in", "spots in")

    has_geo = (
        any(pt in q for pt in _GEO_PLACE_TYPES)
        or any(ph in q for ph in _GEO_PHRASES)
    )
    has_warehouse = any(sig in q for sig in _WAREHOUSE_SIGNALS)

    if has_geo and not has_warehouse and route not in ("geodata", "geodata+warehouse"):
        route = "geodata"
        if not plan.get("geo_intent"):
            plan["geo_intent"] = state["question"]
    elif has_geo and has_warehouse and route not in ("geodata+warehouse",):
        route = "geodata+warehouse"
        if not plan.get("geo_intent"):
            plan["geo_intent"] = state["question"]

    route_display = {
        "warehouse":          "warehouse",
        "geodata":            "geodata",
        "semantic":           "semantic",
        "geodata+warehouse":  "geodata + warehouse",
        "warehouse+semantic": "warehouse + semantic",
    }.get(route, route)
    console.print(f"\n[dim]Route →[/dim] [bold green]{route_display}[/bold green]")
    _start_spinner()
    return {"route": route, "intents": plan, "tool_results": []}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Warehouse executor
# ══════════════════════════════════════════════════════════════════════════════

async def warehouse_node(state: LakehouseState, llm, tool, schema: str) -> dict:
    """
    Generates a SQL SELECT query from the warehouse intent and executes it
    via the query_warehouse MCP tool. Retries once on SQL error with the
    error message included in the prompt.

    Separation from the planner means the LLM only needs to solve SQL
    generation — it already knows which tool to use.
    """
    intent = (state["intents"].get("warehouse_intent") or state["question"]).strip()

    sql_prompt = (
        "Write a single complete SQL SELECT query for PostgreSQL.\n\n"
        f"Task: {intent}\n\n"
        "Schema (live columns — use ONLY these):\n"
        f"{schema}\n\n"
        "STRICT RULES — violating any of these causes an error:\n"
        "  1. Every alias (l, n, h, d) MUST be defined in a FROM or JOIN clause\n"
        "     before it is used. Never reference an alias that has no FROM/JOIN.\n"
        "  2. dim_listing has NO borough/neighbourhood name — only neighbourhood_id.\n"
        "     Always JOIN dim_neighbourhood n ON l.neighbourhood_id = n.neighbourhood_id\n"
        "     to get n.borough or n.neighbourhood.\n"
        "  3. Use only columns that appear in the schema above. Never invent columns.\n"
        "  4. Structure: SELECT ... FROM <table> [alias] [JOIN ...] [WHERE ...] "
        "[GROUP BY ...] [ORDER BY ...] [LIMIT N]\n\n"
        "Correct example:\n"
        "  SELECT n.borough, COUNT(*) AS listings,\n"
        "         AVG(l.estimated_occupancy_l365d) AS avg_occupancy\n"
        "  FROM dim_listing l\n"
        "  JOIN dim_neighbourhood n ON l.neighbourhood_id = n.neighbourhood_id\n"
        "  GROUP BY n.borough ORDER BY listings DESC\n\n"
        "Return ONLY the SQL query — no markdown fences, no explanation."
    )

    _stop_spinner()
    response = await llm.ainvoke(sql_prompt)
    sql = response.content.strip().strip("```sql").strip("```").strip()

    _print_tool_call("query_warehouse", {"sql": sql})
    _start_spinner("Querying Warehouse...")

    result = await tool.ainvoke({"sql": sql})
    _stop_spinner()

    # Retry once if the SQL produced an error
    data, _ = _extract_result_data(result)
    if data and "error" in data:
        console.print(f"  [yellow]SQL error — retrying:[/yellow] {data['error'][:120]}")
        retry_prompt = (
            sql_prompt
            + f"\n\nPrevious attempt failed with: {data['error']}\n"
            "Fix the SQL and return only the corrected query."
        )
        _start_spinner("Querying Warehouse...")
        response2 = await llm.ainvoke(retry_prompt)
        sql = response2.content.strip().strip("```sql").strip("```").strip()
        _stop_spinner()
        _print_tool_call("query_warehouse", {"sql": sql})
        _start_spinner("Querying Warehouse...")
        result = await tool.ainvoke({"sql": sql})
        _stop_spinner()

    _print_tool_result("query_warehouse", result)
    _start_spinner()

    return {"tool_results": [{"tool": "query_warehouse", "query": sql, "result": result}]}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Geo executor
# ══════════════════════════════════════════════════════════════════════════════

async def geo_node(state: LakehouseState, llm, tool) -> dict:
    """
    Generates a MongoDB filter from the geo intent and executes query_geodata.
    The LLM outputs JSON which is parsed and sanitised before the tool call —
    this isolates parameter-generation errors from execution errors.
    """
    intent = (state["intents"].get("geo_intent") or state["question"]).strip()

    # Build the types string for the prompt from the canonical set
    types_list = ", ".join(sorted(_GEO_PLACE_TYPES - {"food", "establishment", "point_of_interest"}))

    # Single unified prompt — the LLM decides whether to use find or aggregate
    # based on the task semantics. No hardcoded keyword detection.
    geo_prompt = (
        "Generate a MongoDB query for the 'places' collection to answer this task.\n\n"
        f"Task: {intent}\n\n"
        "Document structure:\n"
        f"  types        : ARRAY of strings, e.g. ['restaurant','food']\n"
        f"                 Known values: {types_list}\n"
        "  rating       : float (1.0–5.0), may be missing\n"
        "  borough      : 'Manhattan','Brooklyn','Queens','Bronx','Staten Island'\n"
        "  neighbourhood: string\n"
        "  name, vicinity, price_level (0–4)\n\n"
        "Array matching: {\"types\": \"restaurant\"} matches any doc whose types array "
        "contains 'restaurant'.\n\n"
        "Choose the right output format:\n\n"
        "  FIND (looking up specific places, listing top N, filtering by location/type):\n"
        '  {"filter": {"types": "bar", "neighbourhood": "Williamsburg"}, '
        '"sort": [["rating", -1]], "limit": 10}\n\n'
        "  AGGREGATE (grouping, averaging, ranking across boroughs/neighbourhoods, "
        "counting, comparing):\n"
        '  {"pipeline": [\n'
        '    {"$match": {"types": "restaurant", "rating": {"$exists": true, "$gt": 0}}},\n'
        '    {"$group": {"_id": "$borough", "avg_rating": {"$avg": "$rating"}, '
        '"place_count": {"$sum": 1}}},\n'
        '    {"$sort": {"avg_rating": -1}}\n'
        "  ]}\n\n"
        "Output ONLY valid JSON with either a 'pipeline' key or a 'filter' key. "
        "No explanation."
    )

    _stop_spinner()
    response = await llm.ainvoke(geo_prompt)
    _start_spinner()

    params: dict = {"collection": "places"}

    try:
        text   = re.sub(r"```json\s*|\s*```", "", response.content).strip()
        parsed = json.loads(text)

        if "pipeline" in parsed:
            # Aggregate mode — the LLM decided grouping is needed
            params["pipeline"] = parsed["pipeline"]
        else:
            # Find mode — sanitise filter/sort/limit
            params["filter"] = parsed.get("filter", {})
            params["limit"]  = min(int(parsed.get("limit", 10)), 50)
            raw_sort = parsed.get("sort")
            if raw_sort:
                clean = [
                    [item[0], item[1] if isinstance(item[1], int) else -1]
                    for item in raw_sort
                    if isinstance(item, list) and len(item) == 2
                ]
                if clean:
                    params["sort"] = clean
    except Exception:
        params["filter"] = {}
        params["limit"]  = 10

    _stop_spinner()
    _print_tool_call("query_geodata", params)
    _start_spinner("Querying Geodata...")

    result = await tool.ainvoke(params)
    _stop_spinner()

    # Retry once if aggregation pipeline failed
    data, _ = _extract_result_data(result)
    if data and "error" in data and "pipeline" in params:
        console.print(f"  [yellow]Pipeline error — retrying:[/yellow] {data['error'][:120]}")
        retry_prompt = (
            geo_prompt
            + f"\n\nPrevious attempt failed: {data['error']}\n"
            "Fix the pipeline and return only the corrected JSON."
        )
        _start_spinner("Querying Geodata...")
        response2 = await llm.ainvoke(retry_prompt)
        _stop_spinner()
        try:
            text2   = re.sub(r"```json\s*|\s*```", "", response2.content).strip()
            parsed2 = json.loads(text2)
            if "pipeline" in parsed2:
                params["pipeline"] = parsed2["pipeline"]
                params.pop("filter", None)
                params.pop("limit", None)
                params.pop("sort", None)
        except Exception:
            pass
        _print_tool_call("query_geodata", params)
        _start_spinner("Querying Geodata...")
        result = await tool.ainvoke(params)
        _stop_spinner()

    _print_tool_result("query_geodata", result)
    _start_spinner()

    query_desc = json.dumps(params.get("pipeline", params.get("filter", {})))
    return {"tool_results": [{"tool": "query_geodata",
                               "query": query_desc,
                               "result": result}]}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Semantic executor
# ══════════════════════════════════════════════════════════════════════════════

async def semantic_node(
    state: LakehouseState,
    review_tool,
    context_tool,
) -> dict:
    """
    Runs search_reviews and get_neighbourhood_context using the semantic intent
    as the query string. No LLM call needed — the planner already distilled
    what to search for, so the intent maps directly to a vector query.
    """
    intent  = (state["intents"].get("semantic_intent") or state["question"]).strip()
    results = []

    # ── search_reviews ──────────────────────────────────────────────────────
    _stop_spinner()
    _print_tool_call("search_reviews", {"query": intent})
    _start_spinner("Searching Reviews...")
    reviews = await review_tool.ainvoke({"query": intent, "n_results": 5})
    _stop_spinner()
    _print_tool_result("search_reviews", reviews)
    results.append({"tool": "search_reviews", "query": intent, "result": reviews})

    # ── get_neighbourhood_context ───────────────────────────────────────────
    _print_tool_call("get_neighbourhood_context", {"query": intent})
    _start_spinner("Searching Context...")
    context = await context_tool.ainvoke({"query": intent, "n_results": 3})
    _stop_spinner()
    _print_tool_result("get_neighbourhood_context", context)
    results.append({"tool": "get_neighbourhood_context", "query": intent, "result": context})

    _start_spinner()
    return {"tool_results": results}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Synthesizer
# ══════════════════════════════════════════════════════════════════════════════

async def synthesizer_node(state: LakehouseState, llm) -> dict:
    """
    Composes the final answer from all accumulated tool results.
    Receives clean summaries (not raw JSON blobs) so the LLM can focus on
    writing a coherent, data-driven response.
    """
    sections = []
    for tr in state.get("tool_results", []):
        data, _ = _extract_result_data(tr["result"])
        if data and "error" in data:
            continue   # skip failed tool calls — don't feed errors to synthesizer
        summary = _summarise_for_synthesizer(tr["tool"], tr["result"])
        sections.append(f"[{tr['tool']}]\n{summary}")

    # Guard: if every tool call failed, return an honest message instead of
    # letting the LLM hallucinate data it doesn't have.
    if not sections:
        return {"answer": (
            "I was unable to retrieve data for this question — all tool calls "
            "returned errors. Please try rephrasing, or check that the databases "
            "are running."
        )}

    synth_prompt = (
        f"Answer this question about NYC Airbnb data.\n\n"
        f"Question: {state['question']}\n\n"
        "Data retrieved from the lakehouse:\n\n"
        + "\n\n---\n\n".join(sections)
        + "\n\nWrite a clear, data-driven answer using ONLY the data above. "
        "Include specific numbers rounded to 2 decimal places. "
        "Do not add information not present in the data."
    )

    _start_spinner("Composing answer...")
    response = await llm.ainvoke(synth_prompt)
    _stop_spinner()

    return {"answer": response.content}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _route_from_planner(state: LakehouseState) -> str:
    """First node to visit based on planner route."""
    route = state.get("route", "warehouse")
    if route in ("geodata", "geodata+warehouse"):
        return "geodata"
    if route == "semantic":
        return "semantic"
    return "warehouse"   # warehouse, warehouse+semantic


def _route_after_warehouse(state: LakehouseState) -> str:
    """After warehouse: continue to semantic for warehouse+semantic, else synthesize."""
    return "semantic" if state.get("route") == "warehouse+semantic" else "synthesizer"


def _route_after_geodata(state: LakehouseState) -> str:
    """After geodata: continue to warehouse for geodata+warehouse, else synthesize."""
    return "warehouse" if state.get("route") == "geodata+warehouse" else "synthesizer"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph(llm, planner_llm, tools_by_name: dict, schema: str):
    """
    Assemble the LangGraph StateGraph.

    Graph topology:
      planner ──(conditional)──► warehouse ──(conditional)──► synthesizer
                              │                            └──► semantic ──► synthesizer
                              ├──► geodata ──(conditional)──► synthesizer
                              │                            └──► warehouse ──► synthesizer
                              └──► semantic ─────────────────────────────► synthesizer

    Routes:
      warehouse         → warehouse → synthesizer
      geodata           → geodata   → synthesizer
      semantic          → semantic  → synthesizer
      warehouse+semantic→ warehouse → semantic → synthesizer
      geodata+warehouse → geodata   → warehouse → synthesizer
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(LakehouseState)

    graph.add_node("planner",     partial(planner_node, llm=planner_llm))
    graph.add_node("warehouse",   partial(
        warehouse_node,
        llm=llm,
        tool=tools_by_name["query_warehouse"],
        schema=schema,
    ))
    graph.add_node("geodata",     partial(
        geo_node,
        llm=llm,
        tool=tools_by_name["query_geodata"],
    ))
    graph.add_node("semantic",    partial(
        semantic_node,
        review_tool=tools_by_name["search_reviews"],
        context_tool=tools_by_name["get_neighbourhood_context"],
    ))
    graph.add_node("synthesizer", partial(synthesizer_node, llm=llm))

    graph.set_entry_point("planner")

    graph.add_conditional_edges("planner", _route_from_planner, {
        "warehouse": "warehouse",
        "geodata":   "geodata",
        "semantic":  "semantic",
    })
    graph.add_conditional_edges("warehouse", _route_after_warehouse, {
        "semantic":    "semantic",
        "synthesizer": "synthesizer",
    })
    graph.add_conditional_edges("geodata", _route_after_geodata, {
        "warehouse":   "warehouse",
        "synthesizer": "synthesizer",
    })
    graph.add_edge("semantic",    "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def run_graph_agent(question: str) -> str:
    """
    Multi-agent Planner → Executor(s) → Synthesizer pipeline.

    LLM selection:
      Groq (llama-3.3-70b-versatile) when GROQ_API_KEY is set;
      local Ollama otherwise — no API key required.

    The planner LLM uses constrained JSON output (format='json' on Ollama,
    temperature=0 on Groq) to produce a routing plan. Executor nodes each
    solve a single sub-problem (SQL generation, MongoDB filter, vector query).
    The synthesizer composes the final answer from all tool results.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        groq_model  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        llm         = ChatGroq(model=groq_model, temperature=0)
        planner_llm = ChatGroq(model=groq_model, temperature=0)
    else:
        from langchain_ollama import ChatOllama
        model_name  = os.getenv("OLLAMA_MODEL", "llama3.2")
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        llm         = ChatOllama(model=model_name, base_url=ollama_host,
                                  temperature=0, num_predict=2048)
        planner_llm = ChatOllama(model=model_name, base_url=ollama_host,
                                  temperature=0, format="json", num_predict=512)

    client = MultiServerMCPClient({
        "nyc-airbnb-lakehouse": {
            "command":   sys.executable,
            "args":      [str(SERVER_PATH)],
            "transport": "stdio",
            "env":       dict(os.environ),
        }
    })
    tools         = await client.get_tools()
    tools_by_name = {t.name: t for t in tools}
    schema        = _fetch_pg_schema()
    graph         = _build_graph(llm, planner_llm, tools_by_name, schema)

    try:
        result = await graph.ainvoke({
            "question":     question,
            "route":        "",
            "intents":      {},
            "tool_results": [],
            "answer":       "",
        })
        _stop_spinner()
        return result.get("answer") or "(no answer produced)"

    except Exception as e:
        _stop_spinner()
        err = str(e)
        if "rate_limit_exceeded" in err or "Rate limit" in err:
            wait = re.search(r"try again in ([^\.]+)", err)
            wait_str = f" Try again in {wait.group(1)}." if wait else ""
            return f"[Rate limit reached] Groq daily token quota exhausted.{wait_str}"
        if "Failed to call a function" in err or "failed_generation" in err:
            return "[Model error] The model produced a malformed tool call. Please rephrase your question and try again."
        raise


def main():
    parser = argparse.ArgumentParser(
        description="NYC Airbnb Lakehouse Agent — multi-agent LangGraph + Ollama"
    )
    parser.add_argument("-q", "--question", help="Single question (non-interactive)")
    args = parser.parse_args()

    if os.getenv("GROQ_API_KEY"):
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        backend = "groq"
    else:
        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        backend = "ollama"
    console.print(f"Agent initialising — model: [bold]{model}[/bold] ({backend}) | server: {SERVER_PATH.name}")

    if args.question:
        console.print(f"\n[dim]Question:[/dim] {args.question}")
        console.print("─" * 70)
        answer = asyncio.run(run_graph_agent(args.question))
        console.print(f"\n{'─' * 70}")
        console.print(answer)
        return

    # Interactive REPL
    console.print("What would you like to know about Airbnb in New York City?\n")
    first = True
    while True:
        try:
            prompt   = "Ask anything about NYC Airbnb → " if first else "Next question → "
            question = input(prompt).strip()
            first    = False
        except (KeyboardInterrupt, EOFError):
            console.print("\nBye.")
            break
        if not question or question.lower() in ("exit", "quit"):
            break
        answer = asyncio.run(run_graph_agent(question))
        console.print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
