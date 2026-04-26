"""
Evaluation utilities for GRPO-trained Sokoban models.

Provides:
1. Test/validation split from Microban puzzles
2. Evaluation metrics for trained models
3. Comparison between base model and GRPO-finetuned model
"""

from __future__ import annotations

import re
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from core.microban import MicrobanPuzzle, load_microban
from core.board import SokobanBoard
from search.astar import astar
from llm.predictor import LLMPredictor, BaseBackend
from rl.reward import reward_move_quality, reward_format, _extract_move


@dataclass
class EvalResult:
    """Evaluation result for a single puzzle."""
    puzzle_name: str
    difficulty: str
    num_boxes: int
    model_type: str  # "base" or "grpo"
    solved: bool
    solution_length: int
    optimal_length: int
    pushes: int
    optimal_pushes: int
    llm_calls: int
    time_seconds: float
    reward_score: float
    format_score: float
    move_quality_score: float


def create_train_test_split(
    puzzles: List[MicrobanPuzzle],
    test_size: float = 0.2,
    random_seed: int = 42,
    balance_difficulty: bool = True,
) -> Tuple[List[MicrobanPuzzle], List[MicrobanPuzzle]]:
    """
    Split puzzles into training and test sets.
    
    Args:
        puzzles: List of all puzzles
        test_size: Proportion for test set (0.0-1.0)
        random_seed: For reproducibility
        balance_difficulty: If True, maintain difficulty distribution
    
    Returns:
        (train_puzzles, test_puzzles)
    """
    import random
    random.seed(random_seed)
    
    if balance_difficulty:
        # Split by difficulty to ensure balanced representation
        by_difficulty = {"easy": [], "medium": [], "hard": []}
        for p in puzzles:
            by_difficulty[p.difficulty_proxy].append(p)
        
        train = []
        test = []
        
        for difficulty, puzzle_list in by_difficulty.items():
            n_test = max(1, int(len(puzzle_list) * test_size))
            shuffled = puzzle_list.copy()
            random.shuffle(shuffled)
            test.extend(shuffled[:n_test])
            train.extend(shuffled[n_test:])
    else:
        # Simple random split
        shuffled = puzzles.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - test_size))
        train = shuffled[:split_idx]
        test = shuffled[split_idx:]
    
    return train, test


def evaluate_puzzle(
    puzzle: MicrobanPuzzle,
    predictor: LLMPredictor,
    repr_key: str = "13_ACTION_CENTRIC",
    model_type: str = "base",
    max_steps: int = 100,
    beam_width: int = 3,
) -> EvalResult:
    """
    Evaluate a single puzzle using LLM-guided search.
    
    Args:
        puzzle: Puzzle to evaluate
        predictor: LLM predictor (base or GRPO-finetuned)
        repr_key: Representation to use
        model_type: "base" or "grpo"
        max_steps: Maximum search steps
        beam_width: Beam width for search
    
    Returns:
        EvalResult with metrics
    """
    from search.astar import llm_astar
    
    board = puzzle.to_board()
    
    # Get optimal solution for comparison
    optimal = astar(board)
    optimal_length = optimal.solution_length if optimal.solved else None
    optimal_pushes = optimal.push_count if optimal.solved else None
    
    # Run LLM-guided search
    t0 = time.time()
    result = llm_astar(
        board=board,
        predictor=predictor,
        repr_key=repr_key,
        max_nodes=max_steps,
        beam_width=beam_width,
    )
    elapsed = time.time() - t0
    
    # Compute reward scores for the solution
    move_quality_score = 0.0
    format_score = 0.0
    
    if result.solved and result.moves:
        # Sample first few moves for reward evaluation
        sample_moves = result.moves[:10]
        for move in sample_moves:
            # Create a dummy completion
            completion = f"<NextMove>{move}</NextMove>"
            mq = reward_move_quality(
                prompts=[""],
                completions=[completion],
                board_ascii=[get_board_ascii(board)],
                optimal_cost=[optimal_pushes or 0]
            )[0]
            fmt = reward_format(prompts=[""], completions=[completion])[0]
            move_quality_score += mq
            format_score += fmt
        
        move_quality_score /= len(sample_moves)
        format_score /= len(sample_moves)
    
    return EvalResult(
        puzzle_name=puzzle.name,
        difficulty=puzzle.difficulty_proxy,
        num_boxes=puzzle.num_boxes,
        model_type=model_type,
        solved=result.solved,
        solution_length=result.solution_length if result.solved else 0,
        optimal_length=optimal_length or 0,
        pushes=result.push_count if result.solved else 0,
        optimal_pushes=optimal_pushes or 0,
        llm_calls=result.llm_calls,
        time_seconds=elapsed,
        reward_score=move_quality_score + 0.3 * format_score,
        format_score=format_score,
        move_quality_score=move_quality_score,
    )


def get_board_ascii(board: SokobanBoard) -> str:
    """Get ASCII representation of board for reward functions."""
    from representations import get_repr
    return get_repr("01_ASCII_RAW")(board)


def evaluate_model_on_puzzles(
    predictor: LLMPredictor,
    test_puzzles: List[MicrobanPuzzle],
    repr_key: str = "13_ACTION_CENTRIC",
    model_type: str = "base",
    max_steps: int = 100,
    beam_width: int = 3,
    verbose: bool = True,
) -> List[EvalResult]:
    """
    Evaluate a model on a set of test puzzles.
    
    Args:
        predictor: LLM predictor
        test_puzzles: List of puzzles to evaluate
        repr_key: Representation to use
        model_type: "base" or "grpo"
        max_steps: Maximum search steps per puzzle
        beam_width: Beam width for search
        verbose: Print progress
    
    Returns:
        List of EvalResult objects
    """
    results = []
    
    for i, puzzle in enumerate(test_puzzles):
        if verbose:
            print(f"  [{i+1}/{len(test_puzzles)}] Evaluating {puzzle.name}...", end=" ", flush=True)
        
        try:
            result = evaluate_puzzle(
                puzzle=puzzle,
                predictor=predictor,
                repr_key=repr_key,
                model_type=model_type,
                max_steps=max_steps,
                beam_width=beam_width,
            )
            results.append(result)
            
            if verbose:
                status = "✓" if result.solved else "✗"
                print(f"{status} (solved={result.solved}, pushes={result.pushes}/{result.optimal_pushes})")
        except Exception as e:
            if verbose:
                print(f"❌ Error: {e}")
            continue
    
    return results


def compare_base_vs_grpo(
    base_predictor: LLMPredictor,
    grpo_predictor: LLMPredictor,
    test_puzzles: List[MicrobanPuzzle],
    repr_key: str = "13_ACTION_CENTRIC",
    max_steps: int = 100,
    beam_width: int = 3,
) -> pd.DataFrame:
    """
    Compare base model vs GRPO-finetuned model on test puzzles.
    
    Returns:
        DataFrame with comparison metrics
    """
    print("\n" + "="*60)
    print("EVALUATING BASE MODEL")
    print("="*60)
    
    base_results = evaluate_model_on_puzzles(
        predictor=base_predictor,
        test_puzzles=test_puzzles,
        repr_key=repr_key,
        model_type="base",
        max_steps=max_steps,
        beam_width=beam_width,
    )
    
    print("\n" + "="*60)
    print("EVALUATING GRPO MODEL")
    print("="*60)
    
    grpo_results = evaluate_model_on_puzzles(
        predictor=grpo_predictor,
        test_puzzles=test_puzzles,
        repr_key=repr_key,
        model_type="grpo",
        max_steps=max_steps,
        beam_width=beam_width,
    )
    
    # Convert to DataFrames
    base_df = pd.DataFrame([vars(r) for r in base_results])
    grpo_df = pd.DataFrame([vars(r) for r in grpo_results])
    
    # Add model type
    base_df["model_type"] = "base"
    grpo_df["model_type"] = "grpo"
    
    # Combine
    combined = pd.concat([base_df, grpo_df], ignore_index=True)
    
    return combined


def print_evaluation_summary(df: pd.DataFrame):
    """
    Print summary statistics from evaluation results.
    """
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    # Overall metrics by model type
    summary = df.groupby("model_type").agg({
        "solved": ["mean", "sum"],
        "reward_score": "mean",
        "move_quality_score": "mean",
        "format_score": "mean",
        "llm_calls": "mean",
        "time_seconds": "mean",
        "pushes": "mean",
        "optimal_pushes": "mean",
    }).round(3)
    
    summary.columns = ["Solve Rate", "Solved", "Reward", "Move Quality", "Format", "LLM Calls", "Time (s)", "Avg Pushes", "Optimal Pushes"]
    summary["Solve Rate"] = summary["Solve Rate"] * 100
    
    print("\n📊 Overall Performance:")
    print(summary.to_string())
    
    # By difficulty
    print("\n📊 Performance by Difficulty:")
    by_difficulty = df.groupby(["model_type", "difficulty"]).agg({
        "solved": "mean",
        "reward_score": "mean",
    }).round(3)
    by_difficulty["solved"] = by_difficulty["solved"] * 100
    print(by_difficulty.to_string())
    
    # Improvement calculation
    base_solved = df[df["model_type"] == "base"]["solved"].mean()
    grpo_solved = df[df["model_type"] == "grpo"]["solved"].mean()
    improvement = (grpo_solved - base_solved) * 100
    
    print(f"\n📈 Improvement:")
    print(f"  Base solve rate: {base_solved*100:.1f}%")
    print(f"  GRPO solve rate: {grpo_solved*100:.1f}%")
    print(f"  Improvement: {improvement:+.1f}%")
    
    # Per-puzzle comparison
    print("\n📊 Per-Puzzle Comparison (Base → GRPO):")
    pivot = df.pivot_table(
        index="puzzle_name",
        columns="model_type",
        values="solved",
        aggfunc="first"
    )
    pivot.columns = ["base_solved", "grpo_solved"]
    pivot["improved"] = pivot["grpo_solved"] & ~pivot["base_solved"]
    pivot["regressed"] = ~pivot["grpo_solved"] & pivot["base_solved"]
    
    improved = pivot["improved"].sum()
    regressed = pivot["regressed"].sum()
    
    print(f"  Puzzles where GRPO improved: {improved}")
    print(f"  Puzzles where GRPO regressed: {regressed}")
    print(f"  Puzzles where same: {len(pivot) - improved - regressed}")


def quick_evaluation(
    base_predictor: LLMPredictor,
    grpo_predictor: Optional[LLMPredictor] = None,
    num_test_puzzles: int = 10,
    repr_key: str = "13_ACTION_CENTRIC",
) -> pd.DataFrame:
    """
    Quick evaluation function for notebooks.
    
    Args:
        base_predictor: Base LLM predictor
        grpo_predictor: GRPO-finetuned predictor (optional)
        num_test_puzzles: Number of test puzzles to use
        repr_key: Representation to use
    
    Returns:
        DataFrame with results
    """
    # Load puzzles
    all_puzzles = load_microban("Microban.txt")
    
    # Create test split
    train_puzzles, test_puzzles = create_train_test_split(
        all_puzzles,
        test_size=0.2,
        random_seed=42,
        balance_difficulty=True,
    )
    
    # Use only first N test puzzles
    test_puzzles = test_puzzles[:num_test_puzzles]
    
    print(f"\n📚 Test set: {len(test_puzzles)} puzzles")
    print(f"   Easy: {sum(1 for p in test_puzzles if p.difficulty_proxy == 'easy')}")
    print(f"   Medium: {sum(1 for p in test_puzzles if p.difficulty_proxy == 'medium')}")
    print(f"   Hard: {sum(1 for p in test_puzzles if p.difficulty_proxy == 'hard')}")
    
    if grpo_predictor:
        # Compare base vs GRPO
        results = compare_base_vs_grpo(
            base_predictor=base_predictor,
            grpo_predictor=grpo_predictor,
            test_puzzles=test_puzzles,
            repr_key=repr_key,
            max_steps=50,
            beam_width=3,
        )
        print_evaluation_summary(results)
        return results
    else:
        # Evaluate only base model
        results = evaluate_model_on_puzzles(
            predictor=base_predictor,
            test_puzzles=test_puzzles,
            repr_key=repr_key,
            model_type="base",
            max_steps=50,
            beam_width=3,
        )
        df = pd.DataFrame([vars(r) for r in results])
        print_evaluation_summary(df)
        return df


# if __name__ == "__main__":
#     from llm.predictor import LLMPredictor, LlamaCppBackend
    
#     # Load base model
#     nemotron_path = "/Users/mernahafez/.lmstudio/models/lmstudio-community/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf"
#     backend = LlamaCppBackend(model_path=nemotron_path, n_gpu_layers=-1)
#     base_predictor = LLMPredictor(backend)
    
#     # Quick evaluation
#     results = quick_evaluation(
#         base_predictor=base_predictor,
#         num_test_puzzles=10,
#         repr_key="13_ACTION_CENTRIC",
#     )


def compare_base_vs_rl(
    base_predictor: LLMPredictor,   # Base LLM (will use + A*)
    rl_predictor: LLMPredictor,     # RL-finetuned mode
    test_puzzles: List[MicrobanPuzzle],
    repr_key: str = "13_ACTION_CENTRIC",
    max_steps: int = 50,  # For RL model greedy rollout
    max_nodes: int = 500,
    beam_width: int = 3,
) -> pd.DataFrame:
    """
    Compare:
      1. Base LLM + A* (search)
      2. RL Model ALONE (greedy rollout, no search)
    """
    from search.astar import llm_astar
    
    results = []
    
    for puzzle in test_puzzles:
        board = puzzle.to_board()
        
        # 1. Base LLM + A* (search)
        print(f"\n  Testing {puzzle.name} with BASE LLM + A*...")
        base_result = llm_astar(
            board=board,
            predictor=base_predictor,
            repr_key=repr_key,
            max_nodes=max_nodes,
            beam_width=beam_width,
        )
        
        # 2. RL Model ALONE (greedy rollout, no search)
        print(f"  Testing {puzzle.name} with RL MODEL ALONE...")
        rl_board = puzzle.to_board()
        rl_moves = []
        rl_solved = False
        
        for step in range(max_steps):
            if rl_board.is_solved():
                rl_solved = True
                break
            
            result = rl_predictor.predict_next_move(rl_board, repr_key)
            if not result.ok or result.move is None:
                break
            
            try:
                rl_board, _ = rl_board.apply_move(result.move)
                rl_moves.append(result.move)
            except ValueError:
                break
        
        results.append({
            "puzzle": puzzle.name,
            "difficulty": puzzle.difficulty_proxy,
            "base_solved": base_result.solved,
            "rl_solved": rl_solved,
            "base_llm_calls": base_result.llm_calls,
            "rl_moves": len(rl_moves),
            "base_time_s": base_result.elapsed_seconds,
            "rl_time_s": None,  # Not tracking for now
            "base_pushes": base_result.solution_length if base_result.solved else None,
            "rl_boxes": len(rl_board.boxes_on_goals) if rl_board else 0,
        })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("BASE LLM + A* vs RL MODEL ALONE")
    print("="*60)
    
    base_solved = df["base_solved"].sum()
    rl_solved = df["rl_solved"].sum()
    n = len(df)
    
    print(f"\n  Solve Rate:")
    print(f"    Base LLM + A*:  {base_solved}/{n} ({base_solved/n*100:.1f}%)")
    print(f"    RL Model Alone: {rl_solved}/{n} ({rl_solved/n*100:.1f}%)")
    
    return df