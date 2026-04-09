"""
evaluate_agent.py
-----------------
Runs the 20 test questions against the active branch's agent and scores each
answer on three metrics:

  faithfulness    — are all claims in the answer grounded in the retrieved
                    tool results? (0.0–1.0)

                    Scoring strategy:
                      warehouse / geodata questions → heuristic: what fraction
                        of numbers extracted from tool_results appear in the
                        answer? Deterministic, no LLM needed.
                      semantic questions            → Gemini Flash as
                        independent LLM judge. Requires GEMINI_API_KEY in .env
                        (free at aistudio.google.com). Gemini is independent
                        from both Groq and Ollama — no circular self-grading.

  route_accuracy  — did the agent take the expected route? (0 or 1)
  tool_coverage   — fraction of expected tools actually called (0.0–1.0)

Run on each candidate branch to produce a per-branch results file:

  feature/mcp           → results_feature_mcp.json
  feature/multi-agent   → results_feature_multi_agent.json
  feature/react-agent   → results_feature_react_agent.json

Usage:
  python3 evaluation/evaluate_agent.py                  # all 20 questions
  python3 evaluation/evaluate_agent.py --ids 1 3 5      # specific questions
  python3 evaluation/evaluate_agent.py --difficulty 1   # filter by difficulty
  python3 evaluation/evaluate_agent.py --dry-run        # print questions only

Output: evaluation/results_<branch>.json
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

EVAL_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
REPO_ROOT   = PROJECT_DIR.parent
AGENT_PATH  = PROJECT_DIR / "agent"

sys.path.insert(0, str(AGENT_PATH))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

QUESTIONS_FILE = EVAL_DIR / "test_questions.json"

# ── Branch detection ──────────────────────────────────────────────────────────

def _current_branch() -> str:
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ── Faithfulness scoring ──────────────────────────────────────────────────────

# Semantic routes — Gemini judge used. All others → heuristic.
_SEMANTIC_ROUTES = {"semantic", "warehouse+semantic"}

_GEMINI_JUDGE_PROMPT = (
    "You are evaluating whether an AI answer is grounded in retrieved data.\n\n"
    "Question: {question}\n\n"
    "Retrieved context (tool results):\n{context}\n\n"
    "Answer: {answer}\n\n"
    "Score the faithfulness of the answer: what fraction of the factual claims "
    "in the answer are directly supported by the retrieved context above?\n"
    "- 1.0 = every claim is traceable to the context\n"
    "- 0.5 = roughly half the claims are supported\n"
    "- 0.0 = the answer contains claims not present in the context (hallucination)\n\n"
    "Output ONLY a single float between 0.0 and 1.0. No explanation."
)


def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric tokens, normalising thousands separators (36,445 → 36445)."""
    text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


def score_faithfulness_heuristic(context: str, answer: str) -> float:
    """
    Precision-based faithfulness check for warehouse / geodata answers.

    Asks: what fraction of numbers stated in the ANSWER also appear in the
    tool results (context)? This measures whether the answer invents numbers
    not grounded in retrieved data — the correct direction for faithfulness.

    Previous (wrong) direction was recall: fraction of context numbers in the
    answer, which penalised correct answers for not repeating metadata like
    row_count or query limit.

    Returns -1.0 if the answer contains no numbers (nothing to evaluate).
    """
    answer_nums = _extract_numbers(answer)
    if not answer_nums:
        return -1.0
    context_nums = _extract_numbers(context)
    matched = answer_nums & context_nums
    return round(len(matched) / len(answer_nums), 2)


async def score_faithfulness_llm_judge(question: str, context: str, answer: str) -> float:
    """
    GPT-4o-mini as independent LLM judge for semantic answers.

    OpenAI GPT-4o-mini is independent from both Groq (llama-3.3-70b) and
    Ollama (llama3.2), eliminating the circular self-grading problem.
    Requires OPENAI_API_KEY in .env.
    Returns -1.0 if the key is missing or the call fails.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key_here":
        print("    [warn] OPENAI_API_KEY not set — semantic faithfulness skipped")
        return -1.0

    try:
        from langchain_openai import ChatOpenAI
        judge = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0,
        )
        prompt = _GEMINI_JUDGE_PROMPT.format(
            question=question,
            context=context[:2000],
            answer=answer,
        )
        response = await judge.ainvoke(prompt)
        return min(1.0, max(0.0, float(response.content.strip())))
    except Exception as e:
        print(f"    [warn] LLM judge failed: {e}")
        return -1.0


async def score_faithfulness(
    question: str, context: str, answer: str, expected_route: str
) -> float:
    """Dispatch to heuristic or Gemini judge based on expected route."""
    if expected_route in _SEMANTIC_ROUTES:
        return await score_faithfulness_llm_judge(question, context, answer)
    return score_faithfulness_heuristic(context, answer)


# ── Instrumented agent run ────────────────────────────────────────────────────

async def run_instrumented(question: str) -> dict:
    """
    Run the agent on a single question and capture:
      - answer
      - tools_called (list of tool names in order)
      - tool_results (raw results, used for faithfulness scoring)
      - route (from planner_node console output, best-effort)
      - elapsed_seconds
    """
    # Import agent internals — works on any branch since agent.py is always
    # at project/agent/agent.py
    try:
        import agent as _agent
    except ImportError as e:
        return {"error": str(e), "answer": "", "tools_called": [], "tool_results": []}

    from langchain_mcp_adapters.client import MultiServerMCPClient

    tools_called  = []
    tool_results  = []
    route_seen    = []

    # Monkey-patch display functions to capture calls without printing
    original_print_call   = _agent._print_tool_call
    original_print_result = _agent._print_tool_result
    original_start        = _agent._start_spinner
    original_stop         = _agent._stop_spinner

    def capture_call(tool_name, args):
        tools_called.append(tool_name)

    def capture_result(tool_name, raw):
        tool_results.append({"tool": tool_name, "result": raw})

    def noop_start(msg=""):
        pass

    def noop_stop():
        pass

    _agent._print_tool_call   = capture_call
    _agent._print_tool_result = capture_result
    _agent._start_spinner     = noop_start
    _agent._stop_spinner      = noop_stop

    t0 = time.perf_counter()
    try:
        # Call whichever run function exists on this branch
        if hasattr(_agent, "run_graph_agent"):
            answer = await _agent.run_graph_agent(question)
        elif hasattr(_agent, "run_agent"):
            answer = await _agent.run_agent(question)
        else:
            answer = "(no run function found)"
    except Exception as e:
        answer = f"[EXCEPTION] {e}"
    finally:
        elapsed = time.perf_counter() - t0
        _agent._print_tool_call   = original_print_call
        _agent._print_tool_result = original_print_result
        _agent._start_spinner     = original_start
        _agent._stop_spinner      = original_stop

    return {
        "answer":           answer,
        "tools_called":     tools_called,
        "tool_results":     tool_results,
        "elapsed_seconds":  round(elapsed, 1),
    }


# ── Scoring helpers ───────────────────────────────────────────────────────────

def score_route(actual_tools: list[str], expected_route: str) -> float:
    """
    Infer actual route from tools called and compare to expected.
    Returns 1.0 if route matches, 0.0 otherwise.
    """
    called = set(actual_tools)
    has_wh  = "query_warehouse" in called
    has_geo = "query_geodata"   in called
    has_sem = "search_reviews"  in called or "get_neighbourhood_context" in called

    if   has_geo and has_wh and not has_sem: actual_route = "geodata+warehouse"
    elif has_wh and has_sem and not has_geo: actual_route = "warehouse+semantic"
    elif has_geo and not has_wh:             actual_route = "geodata"
    elif has_wh  and not has_geo:            actual_route = "warehouse"
    elif has_sem and not has_wh and not has_geo: actual_route = "semantic"
    else:                                    actual_route = "mixed"

    return 1.0 if actual_route == expected_route else 0.0


def score_tool_coverage(actual_tools: list[str], expected_tools: list[str]) -> float:
    if not expected_tools:
        return 1.0
    called = set(actual_tools)
    return len(called & set(expected_tools)) / len(expected_tools)


# ── Main evaluation loop ──────────────────────────────────────────────────────

async def evaluate(question_ids: list[int] | None, difficulty: int | None):
    data = json.loads(QUESTIONS_FILE.read_text())
    questions = data["questions"]

    if question_ids:
        questions = [q for q in questions if q["id"] in question_ids]
    if difficulty is not None:
        questions = [q for q in questions if q["difficulty"] == difficulty]

    if not questions:
        print("No questions match the filter.")
        return

    branch  = _current_branch()
    results = []

    for q in questions:
        print(f"\n[{q['id']:02d}/{len(questions)}] difficulty={q['difficulty']}  "
              f"route={q['expected_route']}")
        print(f"    Q: {q['question']}")

        run = await run_instrumented(q["question"])

        # Build context string for faithfulness scoring
        context = "\n\n".join(
            f"[{tr['tool']}]\n{tr['result'][:500]}"
            for tr in run.get("tool_results", [])
        ) or "(no tool results)"

        faithfulness = await score_faithfulness(
            q["question"], context, run["answer"], q["expected_route"]
        )
        route_acc     = score_route(run["tools_called"], q["expected_route"])
        tool_cov      = score_tool_coverage(run["tools_called"], q["expected_tools"])

        row = {
            "id":               q["id"],
            "question":         q["question"],
            "difficulty":       q["difficulty"],
            "expected_route":   q["expected_route"],
            "expected_tools":   q["expected_tools"],
            "tools_called":     run["tools_called"],
            "route_accuracy":   route_acc,
            "tool_coverage":    round(tool_cov, 2),
            "faithfulness":     round(faithfulness, 2),
            "elapsed_seconds":  run["elapsed_seconds"],
            "answer":           run["answer"][:500],
        }
        results.append(row)

        print(f"    route_accuracy={route_acc:.0f}  "
              f"tool_coverage={tool_cov:.2f}  "
              f"faithfulness={faithfulness:.2f}  "
              f"({run['elapsed_seconds']}s)")

    # ── Summary ────────────────────────────────────────────────────────────────
    valid_faith = [r["faithfulness"] for r in results if r["faithfulness"] >= 0]
    print(f"\n{'─'*60}")
    print(f"Branch : {branch}")
    print(f"N      : {len(results)} questions")
    print(f"Avg route_accuracy  : {sum(r['route_accuracy'] for r in results) / len(results):.2f}")
    print(f"Avg tool_coverage   : {sum(r['tool_coverage']  for r in results) / len(results):.2f}")
    if valid_faith:
        print(f"Avg faithfulness    : {sum(valid_faith) / len(valid_faith):.2f}")
    print(f"Avg elapsed (s)     : {sum(r['elapsed_seconds'] for r in results) / len(results):.1f}")

    # ── Save results ───────────────────────────────────────────────────────────
    safe_branch = re.sub(r"[^a-z0-9_-]", "_", branch.lower())
    out_path    = EVAL_DIR / f"results_{safe_branch}.json"
    out_path.write_text(json.dumps({
        "branch":    branch,
        "questions": results,
        "summary": {
            "n":                  len(results),
            "avg_route_accuracy": round(sum(r["route_accuracy"] for r in results) / len(results), 2),
            "avg_tool_coverage":  round(sum(r["tool_coverage"]  for r in results) / len(results), 2),
            "avg_faithfulness":   round(sum(valid_faith) / len(valid_faith), 2) if valid_faith else None,
            "avg_elapsed_s":      round(sum(r["elapsed_seconds"] for r in results) / len(results), 1),
        },
    }, indent=2))
    print(f"\nResults saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NYC Airbnb agent.")
    parser.add_argument("--ids",        nargs="+", type=int, help="Run specific question IDs")
    parser.add_argument("--difficulty", type=int,  help="Filter by difficulty (1-4)")
    parser.add_argument("--dry-run",    action="store_true", help="Print questions without running")
    args = parser.parse_args()

    if args.dry_run:
        data = json.loads(QUESTIONS_FILE.read_text())
        for q in data["questions"]:
            print(f"  [{q['id']:02d}] diff={q['difficulty']}  route={q['expected_route']:<20}  {q['question']}")
        return

    asyncio.run(evaluate(args.ids, args.difficulty))


if __name__ == "__main__":
    main()
