"""
Heuristics for A* search.

All heuristics take a SokobanBoard and return a non-negative float.
They must be *admissible* (never overestimate) for A* to be optimal.

Available:
    manhattan   — sum of min Manhattan distance from each box to nearest goal  (fast, weak)
    hungarian   — optimal assignment of boxes to goals via Hungarian algorithm  (slower, tighter)
"""

from core.board import SokobanBoard


def manhattan(board: SokobanBoard) -> int:
    """
    Sum of minimum Manhattan distances from each unsolved box to its nearest empty goal.
    Admissible but not tight — ignores walls and box interactions.
    """
    total = 0
    unsolved = board.unsolved_boxes
    empty_goals = [g for g in board.goals if g not in board.boxes]
    for br, bc in unsolved:
        if not empty_goals:
            break
        total += min(abs(br - gr) + abs(bc - gc) for gr, gc in empty_goals)
    return total


def hungarian(board: SokobanBoard) -> int:
    """
    Optimal box-to-goal assignment cost via the Hungarian algorithm.
    Tighter lower bound than manhattan; still admissible.

    Uses a pure-Python O(n³) implementation — fine for typical puzzle sizes (≤10 boxes).
    """
    unsolved = board.unsolved_boxes
    empty_goals = [g for g in board.goals if g not in board.boxes]
    n = len(unsolved)
    if n == 0:
        return 0

    # Cost matrix (boxes × goals); pad to square if needed
    m = max(n, len(empty_goals))
    cost = [[0] * m for _ in range(m)]
    for i, (br, bc) in enumerate(unsolved):
        for j, (gr, gc) in enumerate(empty_goals):
            cost[i][j] = abs(br - gr) + abs(bc - gc)

    return _hungarian(cost, n)


def _hungarian(cost: list, n: int) -> int:
    """Munkres / Hungarian algorithm. Returns minimum assignment cost for top-left n×n."""
    size = len(cost)
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    p = [0] * (size + 1)   # assignment: column j → row p[j]
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minVal = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minVal[j]:
                        minVal[j] = cur
                        way[j] = j0
                    if minVal[j] < delta:
                        delta = minVal[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minVal[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            p[j0] = p[way[j0]]
            j0 = way[j0]

    total = 0
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] <= n and j <= len([g for g in cost[0]]):
            row = p[j] - 1
            col = j - 1
            if row < n and col < len(cost[row]):
                total += cost[row][col]
    return total


# Registry so callers can pick by name
HEURISTICS = {
    "manhattan": manhattan,
    "hungarian": hungarian,
}