"""
Reward functions for GRPO training of one-step Sokoban predictor.

The model predicts one move given a board state.
Reward is computed by:
  1. Applying the predicted move to the board
  2. Running A* from the resulting state to measure improvement
  3. Optionally calling a judge LLM to score reasoning quality

All reward functions follow the TRL GRPOTrainer signature:
    fn(prompts, completions, **kwargs) -> List[float]

kwargs contains extra dataset columns passed through — we use:
    board_ascii  : str  — ASCII representation of the board state
    optimal_move : str  — the A*-optimal move for this state (LURD char)
    optimal_cost : int  — A* pushes from this state to solution
"""

from __future__ import annotations

import re
import os
from typing import List, Optional

# LURD chars
_VALID_LURD = set("UDLRudlr")


# ---------------------------------------------------------------------------
# Helper: parse move from completion
# ---------------------------------------------------------------------------

def _extract_move(text: str) -> Optional[str]:
    """Extract the predicted move from model completion."""
    # <NextMove>X</NextMove>
    m = re.search(r"<NextMove>\s*([UDLRudlr])\s*</NextMove>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # fallback: last bare LURD char
    letters = re.findall(r"\b([UDLRudlr])\b", text)
    if letters:
        return letters[-1].upper()
    return None


def _board_from_ascii(ascii_str: str):
    """Reconstruct SokobanBoard from stored ASCII string."""
    from core.board import SokobanBoard
    return SokobanBoard(ascii_str)


# ---------------------------------------------------------------------------
# Reward 1: Move quality (A*-based)
# ---------------------------------------------------------------------------

def reward_move_quality(
    prompts: List[str],
    completions: List[str],
    board_ascii: List[str],
    optimal_cost: List[int],
    **kwargs,
) -> List[float]:
    """
    Score the predicted move by comparing A* cost before and after.

    Rewards:
      +2.0  — move solves the puzzle
      +1.0  — move reduces A* cost (progress)
       0.0  — move is neutral (same cost)
      -0.5  — move increases A* cost (regress)
      -1.0  — move is illegal
      -1.0  — move causes deadlock
    """
    from search.astar import astar
    from core.deadlock import is_deadlock, precompute_simple_deadlock_squares

    rewards = []

    for completion, ascii_str, opt_cost in zip(completions, board_ascii, optimal_cost):
        move = _extract_move(completion)

        if move is None:
            rewards.append(-1.0)
            continue

        try:
            board = _board_from_ascii(ascii_str)
        except Exception:
            rewards.append(-1.0)
            continue

        # Try to apply the move
        try:
            new_board, lurd = board.apply_move(move)
        except ValueError:
            rewards.append(-1.0)   # illegal move
            continue

        # Check deadlock
        dead_sq = precompute_simple_deadlock_squares(board)
        if is_deadlock(new_board, dead_sq):
            rewards.append(-1.0)
            continue

        # Solved?
        if new_board.is_solved():
            rewards.append(2.0)
            continue

        # A* cost from new state
        try:
            result = astar(new_board, max_nodes=50_000)
            new_cost = result.solution_length if result.solved else opt_cost + 10
        except Exception:
            new_cost = opt_cost + 10

        # Compare to cost before move (opt_cost = A* pushes from current state)
        # After a push, cost should decrease by at least 1
        if new_cost < opt_cost:
            rewards.append(1.0)
        elif new_cost == opt_cost:
            rewards.append(0.0)
        else:
            rewards.append(-0.5)

    return rewards


# ---------------------------------------------------------------------------
# Reward 2: Format compliance
# ---------------------------------------------------------------------------

def reward_format(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """
    Score whether the model followed the required XML output format.

    Full format (1.0):
        <think>...</think>
        <NextMove>X</NextMove>
        <Confidence>0.0-1.0</Confidence>
        <Rationale>...</Rationale>

    Partial credit for having some tags but not all.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        has_think     = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
        has_move      = bool(re.search(r"<NextMove>[UDLRudlr]</NextMove>", completion, re.IGNORECASE))
        has_confidence= bool(re.search(r"<Confidence>[\d.]+</Confidence>", completion, re.IGNORECASE))
        has_rationale = bool(re.search(r"<Rationale>.+</Rationale>", completion, re.IGNORECASE | re.DOTALL))

        score += 0.25 if has_think      else 0.0
        score += 0.40 if has_move       else 0.0   # most important
        score += 0.10 if has_confidence else 0.0
        score += 0.25 if has_rationale  else 0.0

        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# Reward 3: Reasoning consistency (judge LLM — ByteDance Seed)
# ---------------------------------------------------------------------------

async def reward_reasoning_async(
    prompts: List[str],
    completions: List[str],
    board_ascii: List[str],
    optimal_move: List[str],
    **kwargs,
) -> List[float]:
    """
    Async reward: judge whether <think> block correctly reasoned about the move.

    Calls ByteDance Seed model via OpenAI-compatible API.
    Requires ARK_API_KEY environment variable.

    The judge receives:
      - The board state (ASCII)
      - The optimal move (from A*)
      - The model's <think> block
      - The model's predicted move

    It scores 0.0–1.0 based on:
      - Did the think block correctly identify player/box positions?
      - Did the reasoning align with the predicted move?
      - Was the predicted move the optimal one?
    """
    import asyncio
    try:
        from openai import AsyncOpenAI
    except ImportError:
        # openai not installed — return neutral rewards
        return [0.0] * len(completions)

    api_key = os.environ.get("ARK_API_KEY")
    if not api_key:
        return [0.0] * len(completions)

    client = AsyncOpenAI(
        base_url="https://ark.ap-southeast.bytepluses.com/api/v3",
        api_key=api_key,
    )

    async def judge_one(ascii_str: str, opt_move: str,
                        completion: str) -> float:
        think = ""
        m = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if m:
            think = m.group(1).strip()

        predicted = _extract_move(completion) or "?"

        prompt = (
            "You are evaluating a Sokoban AI's reasoning quality.\n\n"
            f"Board state (ASCII):\n{ascii_str}\n\n"
            f"Optimal next move: {opt_move}\n"
            f"Model's reasoning:\n{think}\n\n"
            f"Model's predicted move: {predicted}\n\n"
            "Score the reasoning quality from 0.0 to 1.0:\n"
            "- 1.0: Reasoning correctly identifies board state, logic is sound, "
            "move matches optimal\n"
            "- 0.5: Reasoning partially correct, move is legal but not optimal\n"
            "- 0.0: Reasoning is wrong or contradicts the predicted move\n\n"
            "Reply with ONLY a number between 0.0 and 1.0."
        )

        try:
            response = await client.chat.completions.create(
                model="seed-2-0-pro-260328",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            text = response.choices[0].message.content.strip()
            score = float(re.search(r"[\d.]+", text).group())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.0

    tasks = [
        judge_one(ascii_str, opt_move, completion)
        for ascii_str, opt_move, completion
        in zip(board_ascii, optimal_move, completions)
    ]
    return list(await asyncio.gather(*tasks))

# ---------------------------------------------------------------------------
# Reward 4: Shaped progress (for curriculum learning)
# ---------------------------------------------------------------------------

def reward_progress(
    prompts: List[str],
    completions: List[str],
    board_ascii: List[str],
    **kwargs,
) -> List[float]:
    """
    Shaped reward: reward progress even if not optimal.
    
    This gives positive feedback for:
    - Moving boxes toward goals (distance reduction)
    - Placing boxes on goals
    - Valid moves (even if not optimal)
    """
    from search.heuristics import manhattan
    
    rewards = []
    
    for completion, ascii_str in zip(completions, board_ascii):
        move = _extract_move(completion)
        
        if move is None:
            rewards.append(0.0)
            continue
        
        try:
            board = _board_from_ascii(ascii_str)
            
            # Check if move is legal first
            legal_moves = [l for l, _ in board.get_legal_moves()]
            move_vec = {"U": (-1,0), "D": (1,0), "L": (0,-1), "R": (0,1),
                       "u": (-1,0), "d": (1,0), "l": (0,-1), "r": (0,1)}
            
            if move not in legal_moves and move.lower() not in legal_moves:
                rewards.append(-0.5)  # Less penalty for exploration
                continue
            
            # Apply move
            new_board, _ = board.apply_move(move)
            
            # Reward 1: Boxes on goals
            boxes_on_goals = len(new_board.boxes_on_goals)
            total_boxes = len(board.goals)
            goal_reward = boxes_on_goals / total_boxes if total_boxes > 0 else 0
            
            # Reward 2: Distance to goals (heuristic improvement)
            old_distance = manhattan(board)
            new_distance = manhattan(new_board)
            distance_improvement = max(0, old_distance - new_distance) / (total_boxes * 5)
            
            # Reward 3: Valid move bonus
            valid_bonus = 0.2 if move in legal_moves or move.lower() in legal_moves else 0
            
            # Combined shaped reward (max 1.0)
            shaped_reward = goal_reward * 0.5 + distance_improvement * 0.3 + valid_bonus * 0.2
            rewards.append(min(1.0, shaped_reward))
            
        except Exception:
            rewards.append(-0.5)
    
    return rewards


def reward_optimal_match(
    prompts: List[str],
    completions: List[str],
    optimal_move: List[str],
    **kwargs,
) -> List[float]:
    """
    Reward for matching the optimal move (supervised signal).
    This is the "teacher forcing" reward that gives immediate positive feedback.
    """
    rewards = []
    
    for completion, opt_move in zip(completions, optimal_move):
        predicted = _extract_move(completion)
        
        if predicted is None:
            rewards.append(0.0)
        elif predicted.upper() == opt_move.upper():
            rewards.append(1.0)  # Positive reward for correct move!
        else:
            rewards.append(0.0)  # Neutral, not negative
    
    return rewards