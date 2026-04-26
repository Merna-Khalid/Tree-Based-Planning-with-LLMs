"""
Beam search for Sokoban — push-based.

    beam_search(board, beam_width)  — heuristic beam, no LLM
    llm_beam(board, predictor, ...)
        LLM picks ONE move per candidate. If LLM fails → candidate abandoned.
        No heuristic fallback. Beam width = number of LLM-surviving candidates.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Set

from core.state import SokobanState
from core.deadlock import is_deadlock, precompute_simple_deadlock_squares
from search.heuristics import HEURISTICS
from search.result import SearchResult
from search.astar import _ask_llm


from core.board import SokobanBoard
from llm.predictor import LLMPredictor


@dataclass
class _Beam:
    state:       SokobanState
    zhash:       int
    norm_player: int
    moves:       List[str]
    states:      List
    score:       float


def beam_search(
    board: SokobanBoard,
    beam_width: int = 5,
    heuristic: str = "manhattan",
    max_depth: int = 200,
    max_nodes: int = 500_000,
) -> SearchResult:
    return _run_beam(board, beam_width=beam_width, heuristic=heuristic,
                max_depth=max_depth, max_nodes=max_nodes,
                algorithm="beam", notes=f"beam_width={beam_width}")


def llm_beam(
    board: SokobanBoard,
    predictor: LLMPredictor,
    repr_key: str = "01_ASCII_RAW",
    beam_width: int = 5,
    max_depth: int = 200,
    max_nodes: int = 100_000,
) -> SearchResult:
    """
    LLM-guided beam search.

    At every depth level, for every beam candidate:
      - LLM picks ONE move (with feedback on failure)
      - If LLM fails for a candidate → that candidate is dropped
      - Next beam = surviving LLM-chosen children (up to beam_width)

    No heuristic ordering. No rescue. If all candidates fail → search fails.
    """
    return _run_beam(board, beam_width=beam_width, heuristic=None,
                max_depth=max_depth, max_nodes=max_nodes,
                predictor=predictor, repr_key=repr_key,
                algorithm="llm_beam",
                notes=f"beam_width={beam_width},repr={repr_key}")


def _run_beam(
    board: SokobanBoard,
    beam_width: int,
    heuristic: Optional[str],
    max_depth: int,
    max_nodes: int,
    algorithm: str,
    notes: str = "",
    predictor=None,
    repr_key: str = "01_ASCII_RAW",
) -> SearchResult:
    t0      = time.time()
    h_fn    = HEURISTICS[heuristic] if heuristic else None
    dead_sq = precompute_simple_deadlock_squares(board)
    nodes_expanded = nodes_generated = llm_calls = 0

    visited: Set[int] = {board.zobrist_hash}
    beam = [_Beam(
        state=board.state, zhash=board.zobrist_hash,
        norm_player=board.norm_player,
        moves=[], states=[], score=0.0,
    )]

    for _ in range(max_depth):
        if not beam:
            break

        next_beam: List[_Beam] = []

        for candidate in beam:
            nodes_expanded += 1
            if nodes_expanded >= max_nodes:
                break

            current = board._make_child(
                candidate.state, candidate.zhash, candidate.norm_player
            )

            # Valid pushes (not visited, not deadlock)
            valid_pushes = {
                lurd: (ns, nz, nm)
                for lurd, ns, nz, nm in current.get_legal_pushes()
                if nz not in visited
                and not is_deadlock(board._make_child(ns, nz, nm), dead_sq)
            }

            if not valid_pushes:
                continue  # blocked — candidate abandoned

            if predictor is not None:
                # LLM picks one move
                chosen = _ask_llm(predictor, current, repr_key, valid_pushes)
                llm_calls += 1
                if chosen is None:
                    continue  # LLM failed — candidate abandoned
                lurd, (new_state, new_zhash, new_norm) = chosen

                child = board._make_child(new_state, new_zhash, new_norm)
                if child.is_solved():
                    return SearchResult(
                        algorithm=algorithm, solved=True,
                        moves=candidate.moves + [lurd],
                        states=candidate.states + [new_state],
                        nodes_expanded=nodes_expanded,
                        nodes_generated=nodes_generated + 1,
                        llm_calls=llm_calls,
                        elapsed_seconds=time.time() - t0,
                        notes=notes,
                    )

                visited.add(new_zhash)
                nodes_generated += 1
                next_beam.append(_Beam(
                    state=new_state, zhash=new_zhash, norm_player=new_norm,
                    moves=candidate.moves + [lurd],
                    states=candidate.states + [new_state],
                    score=0.0,
                ))

            else:
                # Pure heuristic beam — expand all valid, keep top beam_width
                for lurd, (new_state, new_zhash, new_norm) in valid_pushes.items():
                    child = board._make_child(new_state, new_zhash, new_norm)
                    if child.is_solved():
                        return SearchResult(
                            algorithm=algorithm, solved=True,
                            moves=candidate.moves + [lurd],
                            states=candidate.states + [new_state],
                            nodes_expanded=nodes_expanded,
                            nodes_generated=nodes_generated + 1,
                            llm_calls=0,
                            elapsed_seconds=time.time() - t0,
                            notes=notes,
                        )
                    visited.add(new_zhash)
                    nodes_generated += 1
                    next_beam.append(_Beam(
                        state=new_state, zhash=new_zhash, norm_player=new_norm,
                        moves=candidate.moves + [lurd],
                        states=candidate.states + [new_state],
                        score=h_fn(child),
                    ))

        if not next_beam:
            break

        if predictor is None:
            next_beam.sort(key=lambda c: c.score)
        beam = next_beam[:beam_width]

    return SearchResult(
        algorithm=algorithm, solved=False, moves=[],
        nodes_expanded=nodes_expanded, nodes_generated=nodes_generated,
        llm_calls=llm_calls, elapsed_seconds=time.time() - t0, notes=notes,
    )