"""
Shared result dataclass for all search algorithms.
moves is a list of LURD chars (lowercase = walk, uppercase = push).
solution is just "".join(moves) — no translation needed.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from core.board import SokobanBoard


@dataclass
class SearchResult:
    # ---- identity ----
    algorithm: str

    # ---- outcome ----
    solved: bool
    moves: List[str]        # LURD push chars (uppercase) for push-based search

    # ---- cost counters ----
    nodes_expanded:  int
    nodes_generated: int
    llm_calls: int = 0
    states: List = field(default_factory=list)  # SokobanState after each push

    # ---- timing ----
    elapsed_seconds: float = 0.0

    # ---- metadata ----
    notes: str = ""

    @property
    def solution(self) -> str:
        """LURD solution string — lowercase walk, uppercase push."""
        return "".join(self.moves)

    @property
    def solution_length(self) -> int:
        return len(self.moves)

    @property
    def push_count(self) -> int:
        """Number of box-push moves (uppercase chars)."""
        return sum(1 for m in self.moves if m.isupper())

    @property
    def walk_count(self) -> int:
        """Number of walk-only moves (lowercase chars)."""
        return sum(1 for m in self.moves if m.islower())

    def verify(self, board) -> bool:
        """Play the solution on the board and confirm it solves it."""
        try:
            b = board
            for m in self.moves:
                b, _ = b.apply_move(m)
            return b.is_solved()
        except Exception:
            return False

    def expand_to_full_moves(self, board: "SokobanBoard") -> List[str]:
        """
        Expand a push-only solution into a full LURD move sequence (walks + pushes).

        Uses the stored state sequence to determine exactly where each push happens
        and BFS-pathfinds the player walk between pushes.

        Returns the full LURD list. If no states stored, returns moves as-is.
        """
        if not self.moves:
            return []
        if not self.states or all(m.islower() for m in self.moves):
            return self.moves[:]

        from core.board import _LURD_VECTORS
        full_moves: List[str] = []
        b = board

        for push_char, next_state in zip(self.moves, self.states):
            dr, dc = _LURD_VECTORS[push_char]
            # Player must stand at (next_state.player - direction) to push
            # next_state.player = where box was = where player ends up after push
            player_needed = (next_state.player[0] - dr, next_state.player[1] - dc)

            walk_moves = b.walk_path_to(player_needed)
            if walk_moves is None:
                raise ValueError(
                    f"Cannot walk to {player_needed} for push {push_char!r}. "
                    f"Player at {b.player}, boxes {sorted(b.boxes)}"
                )
            for wm in walk_moves:
                b, wlurd = b.apply_move(wm)
                full_moves.append(wlurd)

            b, plurd = b.apply_move(push_char)
            full_moves.append(plurd)

        return full_moves

    def as_dict(self) -> dict:
        return {
            "algorithm":       self.algorithm,
            "solved":          self.solved,
            "solution":        self.solution,
            "solution_length": self.solution_length,
            "push_count":      self.push_count,
            "walk_count":      self.walk_count,
            "nodes_expanded":  self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "llm_calls":       self.llm_calls,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "notes":           self.notes,
        }

    def __repr__(self):
        status = f"solved in {self.solution_length} moves ({self.push_count} pushes)" \
                 if self.solved else "unsolved"
        return (f"SearchResult({self.algorithm}, {status}, "
                f"expanded={self.nodes_expanded}, "
                f"llm_calls={self.llm_calls}, "
                f"time={self.elapsed_seconds:.2f}s)")