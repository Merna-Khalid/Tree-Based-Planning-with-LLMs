"""
Evaluation runner.

run_search_evaluation(puzzles, solvers, llm_solvers, repr_keys)

    solvers      — plain solvers:     {"name": fn(board) -> SearchResult}
    llm_solvers  — LLM-aware solvers: {"name": fn(board, repr_key) -> SearchResult}

Plain solvers are called once per puzzle.
LLM solvers are called once per (puzzle, repr_key) — repr_key appears in results.

Typical notebook usage:

    from sokoban.evaluation.runner import run_search_evaluation, print_summary
    from sokoban.llm import LLMPredictor, AnthropicBackend
    from sokoban.search import astar, beam_search, llm_astar
    import pandas as pd

    predictor = LLMPredictor(AnthropicBackend())

    results = run_search_evaluation(
        puzzles = puzzles[:10],
        solvers = {
            "astar":    astar,
            "beam_w5":  lambda b: beam_search(b, beam_width=5),
        },
        llm_solvers = {
            "llm_astar": lambda b, r: llm_astar(b, predictor=predictor, repr_key=r),
        },
        repr_keys = ["01_ASCII_RAW", "04_COORDINATE_LIST"],
    )
    df = pd.DataFrame(results)
    df.groupby(["algorithm", "repr_key"])["solved"].mean()
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from search.astar import astar
from evaluation.metrics import eval_search, aggregate_search_results

if TYPE_CHECKING:
    from core import MicrobanPuzzle
    from search.result import SearchResult

# Type aliases
PlainSolverFn = Callable   # (board) -> SearchResult
LLMSolverFn   = Callable   # (board, repr_key: str) -> SearchResult


def run_search_evaluation(
    puzzles:     List["MicrobanPuzzle"],
    solvers:     Optional[Dict[str, PlainSolverFn]] = None,
    llm_solvers: Optional[Dict[str, LLMSolverFn]]  = None,
    repr_keys:   Optional[List[str]]                = None,
    compute_optimal: bool = True,
    verbose:     bool = True,
) -> List[dict]:
    """
    Run the full evaluation grid and return DataFrame-ready dicts.

    Each result row contains:
        puzzle_name, puzzle_difficulty, num_boxes, height, width,
        algorithm, repr_key, solved, solution_length, optimal_length,
        path_optimality, nodes_expanded, nodes_generated,
        llm_calls, elapsed_seconds

    Args:
        puzzles:         MicrobanPuzzle list to evaluate on
        solvers:         plain solver fns — called as fn(board)
        llm_solvers:     LLM solver fns  — called as fn(board, repr_key)
        repr_keys:       representation keys to iterate for llm_solvers
                         (ignored for plain solvers)
        compute_optimal: run A* to get optimal push count for path_optimality
        verbose:         print a progress line per result
    """
    solvers     = solvers     or {}
    llm_solvers = llm_solvers or {}
    repr_keys   = repr_keys   or ["01_ASCII_RAW"]

    if not solvers and not llm_solvers:
        raise ValueError("Provide at least one solver in `solvers` or `llm_solvers`.")

    rows: List[dict]           = []
    optimal_cache: Dict[str, Optional[int]] = {}

    for puzzle in puzzles:
        if verbose:
            print(f"\n  Puzzle {puzzle.name} "
                  f"({puzzle.height}x{puzzle.width}, "
                  f"boxes={puzzle.num_boxes}, {puzzle.difficulty_proxy})")

        # Compute optimal push count once per puzzle (used for path_optimality)
        if compute_optimal and puzzle.name not in optimal_cache:
            try:
                opt = astar(puzzle.to_board())
                optimal_cache[puzzle.name] = opt.solution_length if opt.solved else None
            except Exception:
                optimal_cache[puzzle.name] = None
        optimal_len = optimal_cache.get(puzzle.name)

        # ---- plain solvers (no repr loop) ----
        for solver_name, solver_fn in solvers.items():
            row = _run_one(
                puzzle, solver_name, solver_fn,
                repr_key=None, optimal_len=optimal_len, verbose=verbose,
            )
            rows.append(row)

        # ---- LLM solvers (one run per repr_key) ----
        for solver_name, solver_fn in llm_solvers.items():
            for repr_key in repr_keys:
                row = _run_one(
                    puzzle, solver_name,
                    lambda b, fn=solver_fn, rk=repr_key: fn(b, rk),
                    repr_key=repr_key, optimal_len=optimal_len, verbose=verbose,
                )
                rows.append(row)

    return rows


def _run_one(
    puzzle,
    solver_name: str,
    solver_fn:   PlainSolverFn,
    repr_key:    Optional[str],
    optimal_len: Optional[int],
    verbose:     bool,
) -> dict:
    """Run a single (puzzle, solver[, repr_key]) combination."""
    board = puzzle.to_board()
    label = f"{solver_name}[{repr_key}]" if repr_key else solver_name

    try:
        result = solver_fn(board)
    except Exception as exc:
        if verbose:
            print(f"    [✗] {label:30s}  ERROR: {exc}")
        return {
            "puzzle_name":       puzzle.name,
            "puzzle_difficulty": puzzle.difficulty_proxy,
            "num_boxes":         puzzle.num_boxes,
            "height":            puzzle.height,
            "width":             puzzle.width,
            "algorithm":         solver_name,
            "repr_key":          repr_key or "",
            "solved":            False,
            "error":             str(exc),
        }

    row = eval_search(puzzle, result, optimal_length=optimal_len)
    row["algorithm"] = solver_name
    row["repr_key"]  = repr_key or ""

    if verbose:
        status = "✓" if result.solved else "✗"
        opt    = f"{row['path_optimality']:.3f}" if row.get("path_optimality") else "—"
        print(f"    [{status}] {label:30s}  "
              f"solved={result.solved}  "
              f"nodes={result.nodes_expanded:6d}  "
              f"time={result.elapsed_seconds:.3f}s  "
              f"opt={opt}")

    return row


def print_summary(rows: List[dict]):
    """Print aggregated summary grouped by algorithm (and repr_key if present)."""
    # Group by (algorithm, repr_key)
    groups: Dict[tuple, List[dict]] = defaultdict(list)
    for r in rows:
        key = (r.get("algorithm", "?"), r.get("repr_key", ""))
        groups[key].append(r)

    header = (f"{'Algorithm':30s}  {'Repr':25s}  "
              f"{'Solve%':>7}  {'AvgNodes':>9}  {'AvgTime':>8}  {'PathOpt':>8}")
    print(header)
    print("-" * len(header))

    for (algo, repr_key), group_rows in sorted(groups.items()):
        agg = aggregate_search_results(group_rows)
        opt = f"{agg['avg_path_optimality']:.3f}" if agg.get("avg_path_optimality") else "     —"
        print(f"{algo:30s}  {repr_key:25s}  "
              f"{agg['solve_rate']*100:6.1f}%  "
              f"{agg['avg_nodes_expanded']:9d}  "
              f"{agg['avg_elapsed_s']:7.3f}s  "
              f"{opt:>8}")