"""
A* variants for Sokoban — push-based search with Zobrist transposition table.

    astar(board)
    weighted_astar(board, weight)
    llm_astar(board, predictor, repr_key)

llm_astar:
    At every node, use LLM to score ALL possible moves.
    Explore moves in order of LLM confidence (highest first).
    Backtrack when a path fails.
    LLM provides guidance but search explores alternatives.
"""

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.state import SokobanState
from core.deadlock import is_deadlock, precompute_simple_deadlock_squares
from search.heuristics import HEURISTICS
from search.result import SearchResult

from core.board import SokobanBoard
from llm.predictor import LLMPredictor


@dataclass(order=True)
class _Node_Astar:
    f:           float
    g:           int
    state:       SokobanState = field(compare=False)
    zhash:       int          = field(compare=False)
    norm_player: int          = field(compare=False)
    moves:       List[str]    = field(compare=False)
    states:      List         = field(compare=False)
    llm_priority: float = field(default=1.0, compare=False)  # Higher = explore first


def astar(
    board: SokobanBoard,
    heuristic: str = "manhattan",
    max_nodes: int = 500_000,
) -> SearchResult:
    return _run_astar(board, heuristic=heuristic, weight=1.0,
                      max_nodes=max_nodes, algorithm="astar")


def weighted_astar(
    board: SokobanBoard,
    weight: float = 2.0,
    heuristic: str = "manhattan",
    max_nodes: int = 500_000,
) -> SearchResult:
    return _run_astar(board, heuristic=heuristic, weight=weight,
                      max_nodes=max_nodes, algorithm="weighted_astar",
                      notes=f"weight={weight}")


def llm_astar(
    board: SokobanBoard,
    predictor: LLMPredictor,
    repr_key: str = "01_ASCII_RAW",
    max_nodes: int = 50_000,
    beam_width: int = 3,  # Number of LLM-suggested moves to explore per node
) -> SearchResult:
    """
    LLM-guided search with true backtracking.
    
    At each node:
      1. Get ALL legal pushes
      2. Ask LLM for move probabilities (via get_move_probs or multiple calls)
      3. Explore top K moves in order of LLM confidence
      4. Backtrack when a branch fails
    
    This is a true search: LLM guides, but search explores alternatives.
    """
    return _run_llm_search(
        board, predictor=predictor, repr_key=repr_key,
        max_nodes=max_nodes, beam_width=beam_width
    )


# ---------------------------------------------------------------------------
# Pure A* (no LLM)
# ---------------------------------------------------------------------------

def _run_astar(
    board: SokobanBoard,
    heuristic: str,
    weight: float,
    max_nodes: int,
    algorithm: str,
    notes: str = "",
) -> SearchResult:
    t0      = time.time()
    h_fn    = HEURISTICS[heuristic]
    dead_sq = precompute_simple_deadlock_squares(board)
    transposition: Dict[int, int] = {}
    nodes_expanded = nodes_generated = 0

    heap = [_Node_Astar(
        f=weight * h_fn(board), g=0,
        state=board.state, zhash=board.zobrist_hash,
        norm_player=board.norm_player,
        moves=[], states=[],
    )]

    while heap:
        node = heapq.heappop(heap)

        if node.zhash in transposition and transposition[node.zhash] <= node.g:
            continue
        transposition[node.zhash] = node.g
        nodes_expanded += 1

        current = board._make_child(node.state, node.zhash, node.norm_player)
        if current.is_solved():
            return SearchResult(
                algorithm=algorithm, solved=True,
                moves=node.moves, states=node.states,
                nodes_expanded=nodes_expanded, nodes_generated=nodes_generated,
                elapsed_seconds=time.time() - t0, notes=notes,
            )

        if nodes_expanded >= max_nodes:
            break

        for lurd, new_state, new_zhash, new_norm in current.get_legal_pushes():
            new_g = node.g + 1
            if new_zhash in transposition and transposition[new_zhash] <= new_g:
                continue
            child = board._make_child(new_state, new_zhash, new_norm)
            if is_deadlock(child, dead_sq):
                continue
            f = new_g + weight * h_fn(child)
            heapq.heappush(heap, _Node_Astar(
                f=f, g=new_g,
                state=new_state, zhash=new_zhash, norm_player=new_norm,
                moves=node.moves + [lurd],
                states=node.states + [new_state],
            ))
            nodes_generated += 1

    return SearchResult(
        algorithm=algorithm, solved=False, moves=[],
        nodes_expanded=nodes_expanded, nodes_generated=nodes_generated,
        elapsed_seconds=time.time() - t0, notes=notes,
    )


# ---------------------------------------------------------------------------
# LLM-guided search with true backtracking
# ---------------------------------------------------------------------------

def _run_llm_search(
    board: SokobanBoard,
    predictor: LLMPredictor,
    repr_key: str,
    max_nodes: int,
    beam_width: int = 3,
) -> SearchResult:
    """
    LLM-guided search that explores multiple branches.
    ALWAYS calls LLM at each node (that's the point of the assignment!)
    """
    t0 = time.time()
    dead_sq = precompute_simple_deadlock_squares(board)
    transposition: Dict[int, int] = {}
    llm_calls = 0
    nodes_expanded = 0
    nodes_generated = 0
    
    heap = []
    
    # Get LLM probabilities for root node - ALWAYS call
    try:
        move_probs = predictor.get_move_probs(board, repr_key)
        llm_calls += 1
    except Exception as e:
        print(f"Warning: LLM call failed: {e}")
        from llm.predictor import ALL_MOVES
        move_probs = {m: 1/8 for m in ALL_MOVES}
    
    # Get legal pushes
    legal_pushes = []
    for lurd, new_state, new_zhash, new_norm in board.get_legal_pushes():
        child = board._make_child(new_state, new_zhash, new_norm)
        if not is_deadlock(child, dead_sq):
            legal_pushes.append((lurd, new_state, new_zhash, new_norm))
    
    # Score legal pushes by LLM confidence
    scored_pushes = []
    for lurd, ns, nz, nm in legal_pushes:
        conf = move_probs.get(lurd, move_probs.get(lurd.lower(), 0.1))
        penalty = 1.0 - conf
        scored_pushes.append((penalty, lurd, ns, nz, nm))
    
    scored_pushes.sort(key=lambda x: x[0])
    
    # Add top beam_width moves to heap
    for penalty, lurd, ns, nz, nm in scored_pushes[:beam_width]:
        node = _Node_Astar(
            f=1 + penalty,
            g=1,
            state=ns,
            zhash=nz,
            norm_player=nm,
            moves=[lurd],
            states=[ns],
            llm_priority=1.0 - penalty,
        )
        heapq.heappush(heap, node)
        nodes_generated += 1
    
    while heap and nodes_expanded < max_nodes:
        node = heapq.heappop(heap)
        
        if node.zhash in transposition and transposition[node.zhash] <= node.g:
            continue
        transposition[node.zhash] = node.g
        nodes_expanded += 1
        
        current = board._make_child(node.state, node.zhash, node.norm_player)
        
        if current.is_solved():
            return SearchResult(
                algorithm="llm_astar",
                solved=True,
                moves=node.moves,
                states=node.states,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                llm_calls=llm_calls,  # This will now be >0
                elapsed_seconds=time.time() - t0,
                notes=f"repr={repr_key}, beam={beam_width}",
            )
        
        # ALWAYS call LLM at every node (removed the threshold condition)
        try:
            move_probs = predictor.get_move_probs(current, repr_key)
            llm_calls += 1
        except Exception as e:
            # Fallback to uniform if LLM fails
            from llm.predictor import ALL_MOVES
            move_probs = {m: 1/8 for m in ALL_MOVES}
        
        # Get legal pushes
        legal_pushes = []
        for lurd, new_state, new_zhash, new_norm in current.get_legal_pushes():
            if new_zhash in transposition and transposition.get(new_zhash, float('inf')) <= node.g + 1:
                continue
            child = board._make_child(new_state, new_zhash, new_norm)
            if not is_deadlock(child, dead_sq):
                legal_pushes.append((lurd, new_state, new_zhash, new_norm))
        
        if not legal_pushes:
            continue
        
        # Score by LLM confidence
        scored_pushes = []
        for lurd, ns, nz, nm in legal_pushes:
            conf = move_probs.get(lurd, move_probs.get(lurd.lower(), 0.1))
            penalty = 1.0 - conf
            scored_pushes.append((penalty, lurd, ns, nz, nm))
        
        scored_pushes.sort(key=lambda x: x[0])
        
        # Add top beam_width to heap
        for penalty, lurd, ns, nz, nm in scored_pushes[:beam_width]:
            child_node = _Node_Astar(
                f=node.g + 1 + penalty,
                g=node.g + 1,
                state=ns,
                zhash=nz,
                norm_player=nm,
                moves=node.moves + [lurd],
                states=node.states + [ns],
                llm_priority=1.0 - penalty,
            )
            heapq.heappush(heap, child_node)
            nodes_generated += 1
    
    return SearchResult(
        algorithm="llm_astar",
        solved=False,
        moves=[],
        states=[],
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        llm_calls=llm_calls,
        elapsed_seconds=time.time() - t0,
        notes=f"repr={repr_key}, beam={beam_width}",
    )

# ---------------------------------------------------------------------------
# Helper: Get move probabilities from LLM
# ---------------------------------------------------------------------------

def get_move_probs(self, board, repr_key="01_ASCII_RAW"):
    """8-key LURD probability dict for search guidance."""
    print(f"  [LLM CALL] Getting move probabilities for board...")  # Debug
    r = self.predict_next_move(board, repr_key)
    if not r.ok:
        from llm.predictor import ALL_MOVES
        return {m: 1/8 for m in ALL_MOVES}
    probs = {m: (1 - r.confidence) / 7 for m in ALL_MOVES}
    probs[r.move] = r.confidence
    print(f"  [LLM CALL] Got probabilities, top: {r.move}={r.confidence:.2f}")  # Debug
    return probs

# ---------------------------------------------------------------------------
# Add get_move_probs to LLMPredictor if not already present
# ---------------------------------------------------------------------------

def add_get_move_probs_to_predictor():
    """
    Add get_move_probs method to LLMPredictor if missing.
    Call this once before using llm_astar.
    """
    if not hasattr(LLMPredictor, 'get_move_probs'):
        def get_move_probs(self, board, repr_key="01_ASCII_RAW"):
            from llm.predictor import ALL_MOVES
            r = self.predict_next_move(board, repr_key)
            if not r.ok:
                return {m: 1/8 for m in ALL_MOVES}
            probs = {m: (1 - r.confidence) / 7 for m in ALL_MOVES}
            probs[r.move] = r.confidence
            return probs
        LLMPredictor.get_move_probs = get_move_probs


# def _ask_llm(
#     predictor: LLMPredictor,
#     board: SokobanBoard,
#     repr_key: str,
#     valid_pushes: dict,       # lurd → (state, zhash, norm)
# ) -> Optional[Tuple[str, Tuple]]:
#     """
#     Ask LLM for a move. If it picks an invalid direction, tell it that move
#     failed and ask again. No extra information — just "Move X failed."

#     Returns (lurd, (state, zhash, norm)) or None if LLM exhausts all options.
#     """
#     rejected: List[str] = []

#     # At most 4 attempts (one per direction)
#     for _ in range(4):
#         if not rejected:
#             result = predictor.predict_next_move(board, repr_key)
#         else:
#             result = predictor.predict_next_move_with_feedback(
#                 board, repr_key,
#                 rejected=[(m, "failed") for m in rejected],
#             )

#         if not result.ok:
#             return None

#         move = result.move.upper()

#         if move in valid_pushes:
#             return move, valid_pushes[move]

#         # Invalid — record and retry
#         rejected.append(move)

#         # If we've tried all possible directions, give up
#         if set(rejected) >= {"U", "D", "L", "R"}:
#             break

#     return None


def _ask_llm(
    predictor: LLMPredictor,
    board: SokobanBoard,
    repr_key: str,
    valid_pushes: dict,
) -> Optional[Tuple[str, Tuple]]:
    """
    Ask LLM for a move, but if it picks a move that doesn't 
    push an unsolved box, penalize it.
    """
    rejected = []
    unsolved_boxes = board.unsolved_boxes
    
    for attempt in range(4):
        if not rejected:
            result = predictor.predict_next_move(board, repr_key)
        else:
            result = predictor.predict_next_move_with_feedback(
                board, repr_key,
                rejected=[(m, "failed") for m in rejected],
            )
        
        if not result.ok:
            return None
        
        move = result.move.upper()
        
        # Check if this move pushes an unsolved box
        if move in valid_pushes:
            new_state = valid_pushes[move][0]
            # Check if the pushed box was unsolved
            if unsolved_boxes:
                # Find which box moved
                old_boxes = set(board.boxes)
                new_boxes = set(new_state.boxes)
                moved_box = (old_boxes - new_boxes).pop() if old_boxes - new_boxes else None
                
                if moved_box and moved_box in unsolved_boxes:
                    # Pushing an unsolved box - good!
                    return move, valid_pushes[move]
                else:
                    # Pushing an already-solved box - discourage
                    rejected.append(move)
                    continue
            else:
                return move, valid_pushes[move]
        
        rejected.append(move)
    
    return None