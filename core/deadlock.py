"""
Deadlock detection for Sokoban.

Three levels of increasing cost:
  1. corner_deadlock   — O(boxes): box stuck in non-goal corner
  2. freeze_deadlock   — O(boxes): box immovable horizontally AND vertically
  3. simple_deadlock   — precomputed: squares from which no box can ever reach any goal
"""

from collections import deque
from typing import FrozenSet, Set, Tuple

from core.board import SokobanBoard


def corner_deadlock(board: SokobanBoard) -> bool:
    """
    A box not on a goal is in a corner if two adjacent perpendicular walls surround it.
    """
    for box in board.unsolved_boxes:
        r, c = box
        n = (r - 1, c) in board.walls
        s = (r + 1, c) in board.walls
        e = (r, c + 1) in board.walls
        w = (r, c - 1) in board.walls
        if (n and e) or (n and w) or (s and e) or (s and w):
            return True
    return False


def freeze_deadlock(board: SokobanBoard) -> bool:
    """
    A box is frozen if it cannot move in either axis.
    Propagates: a box blocked by another frozen box is also frozen.
    """
    def is_frozen(box: Tuple[int, int], visited: Set[Tuple[int, int]]) -> bool:
        if box in visited:
            return True  # cycle → treat as frozen
        visited.add(box)

        r, c = box
        blocked_h = (r, c - 1) in board.walls or (r, c + 1) in board.walls
        blocked_v = (r - 1, c) in board.walls or (r + 1, c) in board.walls

        if not blocked_h:
            left, right = (r, c - 1), (r, c + 1)
            left_frozen  = left  in board.boxes and is_frozen(left,  set(visited))
            right_frozen = right in board.boxes and is_frozen(right, set(visited))
            blocked_h = left_frozen or right_frozen

        if not blocked_v:
            up, down = (r - 1, c), (r + 1, c)
            up_frozen   = up   in board.boxes and is_frozen(up,   set(visited))
            down_frozen = down in board.boxes and is_frozen(down, set(visited))
            blocked_v = up_frozen or down_frozen

        return blocked_h and blocked_v

    for box in board.unsolved_boxes:
        if is_frozen(box, set()):
            if box not in board.goals:
                return True
    return False


def precompute_simple_deadlock_squares(board: SokobanBoard) -> FrozenSet[Tuple[int, int]]:
    """
    Reverse BFS from every goal to find all squares a box can legally reach.
    A square is a "dead square" if no box placed there can ever reach any goal.

    Logic: a box at position P can be pushed in direction (dr, dc) if:
      - destination D = (P.r + dr, P.c + dc) is floor (not wall)
      - player position  = (P.r - dr, P.c - dc) is floor (not wall)

    In reverse: to reach square P, a box must have come from some neighbour Q,
    meaning Q = P + (dr,dc) and player was at P - (dr,dc).
    """
    reachable: Set[Tuple[int, int]] = set()
    queue: deque = deque()

    for goal in board.goals:
        reachable.add(goal)
        queue.append(goal)

    while queue:
        pos = queue.popleft()
        r, c = pos
        for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            # Box at `from_pos` would be pushed to `pos`.
            # For that push: player must stand at `player_pos` (opposite side of box).
            from_pos   = (r + dr, c + dc)      # box comes from here
            player_pos = (r + dr + dr, c + dc + dc)  # player stands here to push
            if (
                from_pos   in board.floor
                and player_pos in board.floor
                and from_pos not in reachable
            ):
                reachable.add(from_pos)
                queue.append(from_pos)

    return frozenset(sq for sq in board.floor if sq not in reachable)


def simple_deadlock(board: SokobanBoard, dead_squares: FrozenSet[Tuple[int, int]]) -> bool:
    """Return True if any unsolved box sits on a precomputed dead square."""
    return any(b in dead_squares for b in board.unsolved_boxes)


def is_deadlock(
    board: SokobanBoard,
    dead_squares: FrozenSet[Tuple[int, int]] | None = None,
) -> bool:
    """
    Combined deadlock check in cost order (cheapest first).
    Pass precomputed dead_squares for the full check; omit for fast-only.
    """
    if corner_deadlock(board):
        return True
    if dead_squares is not None and simple_deadlock(board, dead_squares):
        return True
    if freeze_deadlock(board):
        return True
    return False