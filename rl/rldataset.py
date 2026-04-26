"""
Build the GRPO training dataset from Microban puzzles.

Each training example is ONE board state from ONE step of the optimal solution.
The model sees that board state and must predict the correct next move.

Dataset columns (required by GRPOTrainer):
    prompt       : str  — the full prompt sent to the model
    board_ascii  : str  — raw ASCII board for reward functions
    optimal_move : str  — A*-optimal move at this step (LURD uppercase)
    optimal_cost : int  — A* pushes remaining from this state

Usage:
    from sokoban.rl.dataset import build_dataset
    dataset = build_dataset(puzzles, repr_key="01_ASCII_RAW")
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING


from core import MicrobanPuzzle


def build_dataset(
    puzzles: List["MicrobanPuzzle"],
    repr_key: str = "01_ASCII_RAW",
    max_steps_per_puzzle: int = 50,
    verbose: bool = True,
) -> "datasets.Dataset":
    """
    Build a HuggingFace Dataset where each row is one board state from the
    optimal solution path of a Microban puzzle.

    For each puzzle:
      1. Run A* to get the optimal push solution
      2. Expand to full LURD moves (walks + pushes)
      3. For each step along the solution, record:
           - The board state in repr_key format (prompt)
           - The ASCII board (for reward computation)
           - The optimal next move at this step
           - The A* cost from this state (pushes remaining)

    Args:
        puzzles:              list of MicrobanPuzzle
        repr_key:             which board representation to use as prompt
        max_steps_per_puzzle: cap steps per puzzle (avoid very long solutions)
        verbose:              print progress

    Returns:
        HuggingFace Dataset with columns:
            prompt, board_ascii, optimal_move, optimal_cost
    """
    from search.astar import astar
    from representations import get_repr
    from llm.predictor import build_next_move_prompt

    rows = []
    repr_fn = get_repr(repr_key)

    for puzzle in puzzles:
        board = puzzle.to_board()

        # Solve with A*
        result = astar(board)
        if not result.solved:
            if verbose:
                print(f"  [skip] Puzzle {puzzle.name} — A* could not solve")
            continue

        # Expand push-only solution to full walk+push moves
        full_moves = result.expand_to_full_moves(board)

        if verbose:
            print(f"  Puzzle {puzzle.name}: {len(full_moves)} steps "
                  f"({result.push_count} pushes)")

        # Walk through solution, recording each state
        b = board
        pushes_remaining = result.push_count  # starts at total pushes, decreases

        for step_idx, move in enumerate(full_moves[:max_steps_per_puzzle]):
            # Build the prompt for this board state
            prompt = build_next_move_prompt(b, repr_key)
            ascii_board = repr_fn(b) if repr_key == "01_ASCII_RAW" else \
                          get_repr("01_ASCII_RAW")(b)   # always store ASCII for rewards

            rows.append({
                "prompt":        prompt,
                "board_ascii":   ascii_board,
                "optimal_move":  move.upper(),   # always uppercase for reward lookup
                "optimal_cost":  pushes_remaining,
                "puzzle_name":   puzzle.name,
                "step":          step_idx,
                "repr_key":      repr_key,
            })

            # Apply move to advance board state
            try:
                b, lurd = b.apply_move(move)
            except ValueError:
                break

            # Update pushes_remaining when a push happens
            if lurd.isupper():
                pushes_remaining = max(0, pushes_remaining - 1)

            if b.is_solved():
                break

    if verbose:
        print(f"\n  Total training examples: {len(rows)}")

    from datasets import Dataset
    return Dataset.from_list(rows)


def build_dataset_multi_repr(
    puzzles: List["MicrobanPuzzle"],
    repr_keys: List[str],
    max_steps_per_puzzle: int = 50,
    verbose: bool = True,
) -> "datasets.Dataset":
    """
    Build dataset with multiple representations interleaved.
    Each step is duplicated once per repr_key — gives the model
    exposure to different board formats during training.
    """
    from datasets import concatenate_datasets

    all_datasets = []
    for repr_key in repr_keys:
        if verbose:
            print(f"\n  Building dataset for repr: {repr_key}")
        ds = build_dataset(puzzles, repr_key=repr_key,
                           max_steps_per_puzzle=max_steps_per_puzzle,
                           verbose=verbose)
        all_datasets.append(ds)

    combined = concatenate_datasets(all_datasets)
    return combined.shuffle(seed=42)