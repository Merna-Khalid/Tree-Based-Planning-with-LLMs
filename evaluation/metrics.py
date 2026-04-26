"""
Evaluation metrics for the Sokoban LLM+search pipeline.

Four evaluation layers (matching the assignment spec):

  1. eval_state_tracking   — can the LLM predict the next board state? (spatial map test)
  2. eval_zero_shot        — zero-shot baseline: valid move rate on simple puzzles
  3. eval_search           — LLM+search solve rate, nodes expanded, path optimality
  4. eval_reasoning        — does the <think> block match the actual move taken?

All functions return plain dicts so they can be collected into a DataFrame.
"""

from __future__ import annotations

import re
import time
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from core.board import SokobanBoard, SokobanState
from core.result import SearchResult
from core.deadlock import precompute_simple_deadlock_squares
from search.astar import astar
from search.heuristics import manhattan
from core.microban import MicrobanPuzzle
from llm.predictor import LLMPredictor


# LURD direction vectors — used for legal-move lookup
_LURD_VECTORS = {
    "u": (-1, 0), "U": (-1, 0),
    "d": ( 1, 0), "D": ( 1, 0),
    "l": ( 0,-1), "L": ( 0,-1),
    "r": ( 0, 1), "R": ( 0, 1),
}


# ---------------------------------------------------------------------------
# 1. State tracking accuracy
# ---------------------------------------------------------------------------

def eval_state_tracking(
    board: SokobanBoard,
    move: str,
    predictor: LLMPredictor,
    repr_key: str = "01_ASCII_RAW",
) -> dict:
    """
    Give the LLM the current board + a single move.
    Ask it to predict the resulting board state.
    Compare predicted player position and box positions to ground truth.

    Returns a dict with:
      move, player_correct, boxes_correct, fully_correct, latency_ms
    """
    from representations import get_repr

    # Ground truth — move is a LURD char
    try:
        next_board, lurd = board.apply_move(move)
    except ValueError:
        return {"move": move, "player_correct": False, "boxes_correct": False,
                "fully_correct": False, "latency_ms": 0.0,
                "error": "illegal move for ground truth"}

    board_text = get_repr(repr_key)(board)
    prompt = (
        "You are a Sokoban simulator. Given a board state and a LURD move, output the new state.\n"
        f"Legend: # wall | @ player | $ box | . goal | * box-on-goal | + player-on-goal\n"
        "Move encoding (LURD): uppercase = push a box, lowercase = walk only.\n"
        "  U=push-up  D=push-down  L=push-left  R=push-right\n"
        "  u=walk-up  d=walk-down  l=walk-left  r=walk-right\n\n"
        f"Current board ({repr_key}):\n{board_text}\n\n"
        f"Move: {lurd}\n\n"
        "Output the new board state in the same format, then on a new line:\n"
        "<PlayerPos>(row,col)</PlayerPos>\n"
        "<BoxPositions>(r1,c1);(r2,c2);...</BoxPositions>"
    )

    t0 = time.time()
    raw, _, err = predictor._call(prompt, max_new_tokens=512)
    latency = (time.time() - t0) * 1000

    # Parse predicted positions from XML tags
    pred_player = _parse_pos_tag(raw, "PlayerPos")
    pred_boxes  = _parse_pos_list_tag(raw, "BoxPositions")

    true_player = next_board.player
    true_boxes  = set(next_board.boxes)

    player_ok = pred_player == true_player if pred_player else False
    boxes_ok  = pred_boxes  == true_boxes  if pred_boxes  else False

    return {
        "repr_key":       repr_key,
        "move":           move.upper(),
        "player_correct": player_ok,
        "boxes_correct":  boxes_ok,
        "fully_correct":  player_ok and boxes_ok,
        "latency_ms":     round(latency, 1),
        "error":          err or "",
        "raw_response":   raw,
    }


def _parse_pos_tag(text: str, tag: str) -> Optional[Tuple[int, int]]:
    m = re.search(rf"<{tag}>\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*</{tag}>",
                  text, re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def _parse_pos_list_tag(text: str, tag: str) -> Optional[set]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    content = m.group(1)
    pairs = re.findall(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", content)
    if not pairs:
        return None
    return {(int(r), int(c)) for r, c in pairs}


# ---------------------------------------------------------------------------
# 2. Zero-shot baseline
# ---------------------------------------------------------------------------

def eval_zero_shot(
    puzzle: MicrobanPuzzle,
    predictor: "LLMPredictor",
    repr_key: str = "01_ASCII_RAW",
    max_steps: int = 50,
) -> dict:
    """
    Run the LLM greedily (no search) on a puzzle for up to max_steps.
    Metrics:
      solved, steps_taken, valid_move_rate, illegal_moves, latency_ms_total
    """
    board = puzzle.to_board()
    valid_moves = 0
    illegal_moves = 0
    total_latency = 0.0
    steps = 0

    for step in range(max_steps):
        if board.is_solved():
            break

        t0 = time.time()
        result = predictor.predict_next_move(board, repr_key)
        total_latency += (time.time() - t0) * 1000
        steps += 1

        if not result.ok:
            illegal_moves += 1
            continue

        # Check if move is physically legal (LURD — match by direction vector)
        from core.board import _LURD_VECTORS
        legal_dirs = {lurd for lurd, _ in board.get_legal_moves()}
        # Accept any case variant that points the same direction
        move_vec = _LURD_VECTORS.get(result.move)
        legal_vecs = {_LURD_VECTORS[l] for l in legal_dirs}
        if move_vec not in legal_vecs:
            illegal_moves += 1
            continue

        valid_moves += 1
        try:
            board, _ = board.apply_move(result.move)
        except ValueError:
            illegal_moves += 1

    total_steps = valid_moves + illegal_moves
    return {
        "puzzle_name":     puzzle.name,
        "repr_key":        repr_key,
        "solved":          board.is_solved(),
        "steps_taken":     steps,
        "valid_moves":     valid_moves,
        "illegal_moves":   illegal_moves,
        "valid_move_rate": round(valid_moves / total_steps, 3) if total_steps else 0.0,
        "latency_ms_total": round(total_latency, 1),
    }


# ---------------------------------------------------------------------------
# 3. Search evaluation
# ---------------------------------------------------------------------------

def eval_search(
    puzzle: MicrobanPuzzle,
    result: SearchResult,
    optimal_length: Optional[int] = None,
) -> dict:
    """
    Evaluate a SearchResult against a puzzle.
    Optionally compare to optimal solution length (from A*).

    path_optimality = optimal_length / result.solution_length
    (1.0 = optimal, < 1.0 = suboptimal)
    """
    board = puzzle.to_board()
    verified = result.verify(board) if result.solved else False

    optimality = None
    if result.solved and optimal_length and result.solution_length > 0:
        optimality = round(optimal_length / result.solution_length, 3)

    return {
        "puzzle_name":      puzzle.name,
        "puzzle_difficulty": puzzle.difficulty_proxy,
        "num_boxes":        puzzle.num_boxes,
        "height":           puzzle.height,
        "width":            puzzle.width,
        "algorithm":        result.algorithm,
        "repr_key":         result.notes,        # notes carries repr info
        "solved":           result.solved,
        "verified":         verified,
        "solution_length":  result.solution_length,
        "optimal_length":   optimal_length,
        "path_optimality":  optimality,
        "nodes_expanded":   result.nodes_expanded,
        "nodes_generated":  result.nodes_generated,
        "llm_calls":        result.llm_calls,
        "elapsed_seconds":  result.elapsed_seconds,
        "deadlock_hit":     False,               # updated by caller if needed
    }


# ---------------------------------------------------------------------------
# 4. Reasoning consistency
# ---------------------------------------------------------------------------

def eval_reasoning_consistency(
    board: "SokobanBoard",
    think_text: str,
    actual_move: str,
    repr_key: str = "01_ASCII_RAW",
) -> dict:
    """
    Check whether the <think> block is consistent with the actual move taken.

    Consistency heuristics (all simple string checks — no second LLM call):
      - Does the think block mention the move direction?
      - Does it mention box/goal positions that exist on the board?
      - Does it NOT contradict the move (e.g. say "go right" but move left)?
    """
    # Normalise to uppercase for direction lookup (u==U, d==D etc)
    move = actual_move.upper()
    think_lower = think_text.lower()

    move_words = {
        "U": ["up", "above", "push up", "walk up"],
        "D": ["down", "below", "push down", "walk down"],
        "L": ["left", "push left", "walk left"],
        "R": ["right", "push right", "walk right"],
    }
    opposites = {"U": "D", "D": "U", "L": "R", "R": "L"}

    # Direction mentioned
    direction_match = any(w in think_lower for w in move_words.get(move, []))

    # Contradiction: opposite direction's words appear, but not the actual direction's
    opp = opposites.get(move, "")
    contradiction = (any(w in think_lower for w in move_words.get(opp, []))
                     and not direction_match)

    # Box/goal position mentioned (any digit pair like "3,4" or "(3, 4)")
    board_numbers = set()
    for pos in list(board.boxes) + list(board.goals) + [board.player]:
        board_numbers.add(str(pos[0]))
        board_numbers.add(str(pos[1]))
    numbers_in_think = re.findall(r"\d+", think_lower)
    spatial_grounding = bool(set(numbers_in_think) & board_numbers)

    consistent = direction_match and not contradiction

    return {
        "repr_key":           repr_key,
        "actual_move":        move,
        "think_length":       len(think_text),
        "direction_match":    direction_match,
        "contradiction":      contradiction,
        "spatial_grounding":  spatial_grounding,
        "consistent":         consistent,
        "think_snippet":      think_text[:120].replace("\n", " "),
    }


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def aggregate_search_results(rows: List[dict]) -> dict:
    """
    Summarise a list of eval_search() dicts into aggregate metrics.
    Useful for printing a quick table per algorithm.
    """
    if not rows:
        return {}

    n = len(rows)
    solved = [r for r in rows if r["solved"]]
    optimalities = [r["path_optimality"] for r in solved if r["path_optimality"] is not None]

    return {
        "algorithm":          rows[0]["algorithm"],
        "n_puzzles":          n,
        "solve_rate":         round(len(solved) / n, 3),
        "avg_nodes_expanded": round(sum(r["nodes_expanded"] for r in rows) / n),
        "avg_llm_calls":      round(sum(r["llm_calls"] for r in rows) / n, 1),
        "avg_elapsed_s":      round(sum(r["elapsed_seconds"] for r in rows) / n, 3),
        "avg_path_optimality": round(sum(optimalities) / len(optimalities), 3) if optimalities else None,
    }


# ---------------------------------------------------------------------------
# Representation × predictor benchmark (full solution)
# ---------------------------------------------------------------------------
def _play_solution(board: "SokobanBoard", solution: str) -> dict:
    """
    Play a solution string on a board move by move.
    Returns a dict of partial-credit metrics regardless of whether it solves.
 
    Metrics:
        valid_moves       — how many moves were legal before the first failure
        invalid_move      — the first move that failed (or None)
        boxes_placed      — max boxes on goals at any point during execution
        best_heuristic    — minimum Manhattan distance reached at any point
        final_heuristic   — Manhattan distance at the last valid state
        solved            — did it fully solve?
        valid_prefix_pct  — valid_moves / total_solution_length
    """
 
    dead_sq = precompute_simple_deadlock_squares(board)
 
    best_boxes   = len(board.boxes_on_goals)
    best_h       = manhattan(board)
    valid_moves  = 0
    invalid_move = None
    b            = board
 
    for ch in solution:
        try:
            b_next, lurd = b.apply_move(ch)
        except ValueError:
            invalid_move = ch
            break
 
        valid_moves += 1
        b = b_next
 
        cur_boxes = len(b.boxes_on_goals)
        cur_h     = manhattan(b)
        if cur_boxes > best_boxes:
            best_boxes = cur_boxes
        if cur_h < best_h:
            best_h = cur_h
 
        if b.is_solved():
            break
 
    total = len(solution)
    return {
        "valid_moves":       valid_moves,
        "invalid_move":      invalid_move,
        "boxes_placed":      best_boxes,
        "total_boxes":       len(board.goals),
        "boxes_placed_pct":  round(best_boxes / len(board.goals), 3) if board.goals else 0.0,
        "best_heuristic":    best_h,
        "final_heuristic":   manhattan(b),
        "solved":            b.is_solved(),
        "valid_prefix_pct":  round(valid_moves / total, 3) if total > 0 else 0.0,
    }
 

def eval_repr_benchmark(
    puzzles: List["MicrobanPuzzle"],
    predictor: "LLMPredictor",
    repr_keys: List[str],
    verbose: bool = True,
) -> List[dict]:
    """
    Representation benchmark with partial-credit metrics.
 
    For every (puzzle, repr_key) pair, ask the LLM for a full solution,
    then replay it move by move to measure partial progress.
 
    Metrics per row:
        puzzle_name, repr_key,
        solved            — fully solved the puzzle
        valid_moves       — moves executed before first illegal move
        valid_prefix_pct  — valid_moves / solution_length  (rule comprehension)
        boxes_placed      — max boxes on goals during execution
        boxes_placed_pct  — boxes_placed / total_boxes  (progress score)
        best_heuristic    — min Manhattan distance reached  (peak progress)
        final_heuristic   — Manhattan distance at last valid state
        solution_length   — length of LLM's attempted solution
        optimal_length    — optimal push count from A*
        path_optimality   — optimal / solution_length  (only if solved)
        latency_ms
 
    Notebook usage:
        rows = eval_repr_benchmark(puzzles[:10], predictor, repr_keys=[...])
        df = pd.DataFrame(rows)
 
        # Rank by partial credit score
        df.groupby("repr_key")["boxes_placed_pct"].mean().sort_values(ascending=False)
 
        # Heatmap: puzzle × repr → boxes_placed_pct
        df.pivot(index="puzzle_name", columns="repr_key", values="boxes_placed_pct")
    """
 
    rows: List[dict] = []
    optimal_cache: Dict[str, Optional[int]] = {}
 
    for puzzle in puzzles:
        if puzzle.name not in optimal_cache:
            try:
                opt = astar(puzzle.to_board())
                optimal_cache[puzzle.name] = opt.solution_length if opt.solved else None
            except Exception:
                optimal_cache[puzzle.name] = None
        optimal_len = optimal_cache[puzzle.name]
 
        if verbose:
            print(f"\n  Puzzle {puzzle.name} "
                  f"({puzzle.height}x{puzzle.width}, "
                  f"boxes={puzzle.num_boxes}, {puzzle.difficulty_proxy})  "
                  f"optimal={optimal_len}P")
 
        for repr_key in repr_keys:
            board = puzzle.to_board()
            t0 = time.time()
 
            try:
                result = predictor.predict_full_solution(board, repr_key)
                print(f"Predicted solution: {result}")
                latency = (time.time() - t0) * 1000
            except Exception as exc:
                rows.append(_error_row(puzzle, repr_key, optimal_len,
                                       round((time.time() - t0) * 1000, 1), str(exc)))
                if verbose:
                    print(f"    [✗] {repr_key:30s}  ERROR: {exc}")
                continue
 
            if result.error or not result.solution:
                rows.append(_error_row(puzzle, repr_key, optimal_len,
                                       round(latency, 1), result.error or "empty solution"))
                if verbose:
                    print(f"    [✗] {repr_key:30s}  "
                          f"error={result.error or 'empty solution'}")
                continue
 
            # Replay the solution move by move
            play = _play_solution(board, result.solution)
 
            optimality = None
            if play["solved"] and optimal_len and len(result.solution) > 0:
                optimality = round(optimal_len / len(result.solution), 3)
 
            row = {
                "puzzle_name":       puzzle.name,
                "puzzle_difficulty": puzzle.difficulty_proxy,
                "num_boxes":         puzzle.num_boxes,
                "repr_key":          repr_key,
                # partial credit
                "solved":            play["solved"],
                "valid_moves":       play["valid_moves"],
                "valid_prefix_pct":  play["valid_prefix_pct"],
                "boxes_placed":      play["boxes_placed"],
                "total_boxes":       play["total_boxes"],
                "boxes_placed_pct":  play["boxes_placed_pct"],
                "best_heuristic":    play["best_heuristic"],
                "final_heuristic":   play["final_heuristic"],
                "invalid_move":      play["invalid_move"] or "",
                # solution info
                "solution_length":   len(result.solution),
                "optimal_length":    optimal_len,
                "path_optimality":   optimality,
                "latency_ms":        round(latency, 1),
                "error":             "",
                "rationale":         result.rationale,
            }
            rows.append(row)
 
            if verbose:
                status = "✓" if play["solved"] else "~"
                print(f"    [{status}] {repr_key:30s}  "
                      f"solved={play['solved']}  "
                      f"boxes={play['boxes_placed']}/{play['total_boxes']}  "
                      f"valid={play['valid_moves']}/{len(result.solution)}  "
                      f"best_h={play['best_heuristic']}  "
                      f"{f'opt={optimality:.3f}' if optimality else 'opt=—'}  "
                      f"latency={latency:.0f}ms")
 
    return rows
 
 
def _error_row(puzzle, repr_key: str, optimal_len, latency_ms: float, error: str) -> dict:
    return {
        "puzzle_name":       puzzle.name,
        "puzzle_difficulty": puzzle.difficulty_proxy,
        "num_boxes":         puzzle.num_boxes,
        "repr_key":          repr_key,
        "solved":            False,
        "valid_moves":       0,
        "valid_prefix_pct":  0.0,
        "boxes_placed":      0,
        "total_boxes":       puzzle.num_boxes,
        "boxes_placed_pct":  0.0,
        "best_heuristic":    None,
        "final_heuristic":   None,
        "invalid_move":      "",
        "solution_length":   0,
        "optimal_length":    optimal_len,
        "path_optimality":   None,
        "latency_ms":        latency_ms,
        "error":             error,
        "rationale":         "",
    }
 
 
def repr_benchmark_summary(rows: List[dict]) -> None:
    """
    Print a ranked summary from eval_repr_benchmark() output.
    Ranked by partial credit score (boxes_placed_pct) not just solve rate,
    so you can distinguish representations even when nothing fully solves.
    """
    from collections import defaultdict
 
    by_repr: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_repr[r["repr_key"]].append(r)
 
    ranked = sorted(
        by_repr.items(),
        key=lambda kv: (
            sum(r["boxes_placed_pct"] for r in kv[1]) / len(kv[1])
        ),
        reverse=True,
    )
 
    header = (f"{'Representation':35s}  "
              f"{'Solve%':>7}  {'BoxPct':>7}  "
              f"{'ValidPfx':>9}  {'BestH':>6}  {'AvgMs':>7}")
    print(header)
    print("-" * len(header))
 
    for repr_key, repr_rows in ranked:
        n          = len(repr_rows)
        solved_pct = sum(r["solved"] for r in repr_rows) / n * 100
        avg_box    = sum(r["boxes_placed_pct"] for r in repr_rows) / n
        avg_valid  = sum(r["valid_prefix_pct"] for r in repr_rows) / n
        best_hs    = [r["best_heuristic"] for r in repr_rows if r["best_heuristic"] is not None]
        avg_h      = sum(best_hs) / len(best_hs) if best_hs else float("nan")
        avg_ms     = sum(r["latency_ms"] for r in repr_rows) / n
 
        print(f"  {repr_key:35s}  "
              f"{solved_pct:6.1f}%  {avg_box:7.3f}  "
              f"{avg_valid:9.3f}  {avg_h:6.1f}  {avg_ms:7.0f}ms")
 

@dataclass
class StepResult:
    """Result of a single step in the benchmark."""
    step: int
    move: Optional[str]
    confidence: float
    think: str
    rationale: str
    latency_ms: float
    board_state_after: str  # brief description
    boxes_on_goals: int
    solved: bool
    error: Optional[str] = None


@dataclass
class PuzzleStepResult:
    """Results for one puzzle with one representation."""
    puzzle_name: str
    repr_key: str
    steps_taken: int
    solved: bool
    final_boxes: int
    total_boxes: int
    step_results: List[StepResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    error: Optional[str] = None
    
    @property
    def boxes_pct(self) -> float:
        return self.final_boxes / self.total_boxes if self.total_boxes > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.steps_taken if self.steps_taken > 0 else 0.0


def eval_repr_step_benchmark(
    puzzles: List[MicrobanPuzzle],
    predictor: LLMPredictor,
    repr_keys: List[str],
    max_steps: int = 50,
    verbose: bool = True,
) -> List[PuzzleStepResult]:
    """
    Step-by-step evaluation: play the game with LLM predictions.
    """
    results = []
    
    for puzzle in puzzles:
        if verbose:
            print(f"\n  Puzzle {puzzle.name} ({puzzle.height}x{puzzle.width}, {puzzle.num_boxes} boxes)")
        
        for repr_key in repr_keys:
            board = puzzle.to_board()
            total_boxes = len(board.goals)
            step_results = []
            total_latency = 0.0
            
            if verbose:
                print(f"    [{repr_key[:20]:20s}] ", end="", flush=True)
            
            solved = False
            steps = 0
            error = None
            
            for step in range(max_steps):
                if board.is_solved():
                    solved = True
                    break
                
                # Get LLM prediction
                t0 = time.time()
                result = predictor.predict_next_move(board, repr_key)
                print(f"Model predicted move {result}")
                latency = (time.time() - t0) * 1000
                total_latency += latency
                
                if not result.ok or result.move is None:
                    error = f"No valid move at step {step}: {result.error}"
                    break
                
                # Apply the move
                try:
                    board, applied_move = board.apply_move(result.move)
                    steps += 1  # Only increment AFTER successful move
                except ValueError as e:
                    error = f"Illegal move '{result.move}' at step {step}: {e}"
                    break
                
                # Record step
                step_results.append(StepResult(
                    step=step,
                    move=result.move,
                    confidence=result.confidence,
                    think=result.think[:200] if result.think else "",
                    rationale=result.rationale,
                    latency_ms=latency,
                    board_state_after=f"player={board.player}, boxes={len(board.boxes_on_goals)}/{total_boxes}",
                    boxes_on_goals=len(board.boxes_on_goals),
                    solved=board.is_solved(),
                ))
            
            final_boxes = len(board.boxes_on_goals) if board else 0
            
            puzzle_result = PuzzleStepResult(
                puzzle_name=puzzle.name,
                repr_key=repr_key,
                steps_taken=steps,
                solved=solved or (board and board.is_solved()),
                final_boxes=final_boxes,
                total_boxes=total_boxes,
                step_results=step_results,
                total_latency_ms=total_latency,
                error=error,
            )
            results.append(puzzle_result)
            
            if verbose:
                status = "✓" if puzzle_result.solved else "✗"
                # FIX: Check if steps > 0 before division
                if steps > 0:
                    avg_latency = total_latency / steps
                    print(f"{status} steps={steps:2d} boxes={final_boxes}/{total_boxes} latency={avg_latency:.0f}ms")
                else:
                    print(f"{status} steps={steps:2d} boxes={final_boxes}/{total_boxes} latency=N/A (no valid moves)")
    
    return results


def summarize_step_benchmark(results: List[PuzzleStepResult]) -> pd.DataFrame:
    """
    Generate summary DataFrame from step benchmark results.
    """
    rows = []
    for r in results:
        rows.append({
            "puzzle": r.puzzle_name,
            "repr": r.repr_key,
            "solved": r.solved,
            "steps": r.steps_taken,
            "boxes_pct": r.boxes_pct,
            "avg_latency_ms": r.avg_latency_ms,
            "total_latency_ms": r.total_latency_ms,
            "error": r.error if r.error else "",
        })
    
    df = pd.DataFrame(rows)
    
    # Add composite score
    df["composite"] = df["solved"].astype(float) * 0.5 + df["boxes_pct"] * 0.3 + (df["steps"] / 50) * 0.2
    df["composite"] = df["composite"].clip(0, 1)
    
    return df


def print_step_benchmark_summary(df: pd.DataFrame):
    """
    Print a clean summary of step benchmark results.
    """
    print("\n" + "="*80)
    print("STEP-BY-STEP REPRESENTATION BENCHMARK SUMMARY")
    print("="*80)
    
    # By representation
    print("\n📊 BY REPRESENTATION:")
    repr_summary = df.groupby("repr").agg({
        "solved": ["mean", "sum"],
        "boxes_pct": "mean",
        "steps": "mean",
        "avg_latency_ms": "mean",
        "composite": "mean",
    }).round(3)
    repr_summary.columns = ["Solve%", "Solved", "Boxes%", "Steps", "Latency(ms)", "Composite"]
    repr_summary["Solve%"] = repr_summary["Solve%"] * 100
    repr_summary = repr_summary.sort_values("Composite", ascending=False)
    print(repr_summary)
    
    # By puzzle
    print("\n📊 BY PUZZLE (best representation each):")
    best_per_puzzle = df.loc[df.groupby("puzzle")["composite"].idxmax()]
    best_per_puzzle = best_per_puzzle[["puzzle", "repr", "solved", "boxes_pct", "steps"]]
    best_per_puzzle["boxes_pct"] = best_per_puzzle["boxes_pct"] * 100
    print(best_per_puzzle.to_string(index=False))
    
    # Top 3 representations recommendation
    print("\n🎯 TOP 3 REPRESENTATIONS:")
    top3 = repr_summary.head(3).index.tolist()
    for i, r in enumerate(top3, 1):
        solve_rate = repr_summary.loc[r, "Solve%"]
        boxes = repr_summary.loc[r, "Boxes%"]
        print(f"   {i}. {r}: Solve={solve_rate:.1f}%, Boxes={boxes:.1f}%")
    
    return top3


def plot_step_benchmark(df: pd.DataFrame):
    """
    Visualize step benchmark results.
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Solve rate by representation
    ax = axes[0, 0]
    solve_by_repr = df.groupby("repr")["solved"].mean().sort_values()
    solve_by_repr.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Solve Rate by Representation")
    ax.set_xlabel("Solve Rate")
    
    # 2. Boxes placed percentage
    ax = axes[0, 1]
    boxes_by_repr = df.groupby("repr")["boxes_pct"].mean().sort_values()
    boxes_by_repr.plot(kind="barh", ax=ax, color="coral")
    ax.set_title("Boxes Placed (Partial Credit)")
    ax.set_xlabel("Boxes %")
    
    # 3. Average steps taken (higher is better until solved)
    ax = axes[1, 0]
    steps_by_repr = df.groupby("repr")["steps"].mean().sort_values()
    steps_by_repr.plot(kind="barh", ax=ax, color="seagreen")
    ax.set_title("Average Steps Taken")
    ax.set_xlabel("Steps")
    
    # 4. Composite score
    ax = axes[1, 1]
    composite_by_repr = df.groupby("repr")["composite"].mean().sort_values()
    composite_by_repr.plot(kind="barh", ax=ax, color="purple")
    ax.set_title("Composite Score (Solve% + Boxes% + Steps)")
    ax.set_xlabel("Composite Score")
    
    plt.tight_layout()
    plt.savefig("step_benchmark_results.png", dpi=150)
    plt.show()



def run_step_benchmark(
    puzzles: List[MicrobanPuzzle],
    predictor: LLMPredictor,
    repr_keys: List[str],
    max_steps: int = 30,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Run step-by-step benchmark and return summary DataFrame + top 3 representations.
    """
    print("\n" + "="*70)
    print("STEP-BY-STEP REPRESENTATION BENCHMARK")
    print("="*70)
    print(f"Puzzles: {len(puzzles)}")
    print(f"Representations: {len(repr_keys)}")
    print(f"Max steps per puzzle: {max_steps}")
    print("="*70)
    
    # Run benchmark
    results = eval_repr_step_benchmark(
        puzzles=puzzles,
        predictor=predictor,
        repr_keys=repr_keys,
        max_steps=max_steps,
        verbose=verbose,
    )
    
    # Summarize
    df = summarize_step_benchmark(results)
    top3 = print_step_benchmark_summary(df)
    
    # Plot
    try:
        plot_step_benchmark(df)
    except Exception as e:
        print(f"\n⚠️ Could not generate plot: {e}")
    
    return df, top3