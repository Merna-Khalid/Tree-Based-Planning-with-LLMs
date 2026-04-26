"""Coordinate / index / BFS representations: 04, 05, 07, 11."""

from collections import deque
from representations.base import BaseRepr

from core.board import SokobanBoard


class CoordinateList(BaseRepr):
    key = "04_COORDINATE_LIST"

    def render(self, board):
        return (
            f"Board: {board.height}x{board.width}\n"
            f"Player: {board.player}\n"
            f"Boxes: {sorted(board.boxes)}\n"
            f"Goals: {sorted(board.goals)}\n"
            f"Solved: {len(board.boxes_on_goals)}/{len(board.goals)}"
        )


class IndexedState(BaseRepr):
    key = "05_INDEXED_STATE"

    def render(self, board):
        p  = board.tile_index[board.player]
        bs = sorted(board.tile_index[b] for b in board.boxes)
        gs = sorted(board.tile_index[g] for g in board.goals)
        return f"Player: {p}, Boxes: {bs}, Goals: {gs}"


class RelativeSpatial(BaseRepr):
    key = "07_RELATIVE_SPATIAL"

    def render(self, board):
        from core.board import DIRECTIONS
        lines = []
        for (r, c), idx in sorted(board.tile_index.items(), key=lambda x: x[1]):
            entity = board.describe_tile(r, c)
            lines.append(f"Tile {idx} @ ({r},{c}): {entity}")
            for name, dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc
                if (nr, nc) in board.tile_index:
                    lines.append(f"  {name}: tile {board.tile_index[(nr, nc)]} "
                                 f"({board.describe_tile(nr, nc)})")
                elif 0 <= nr < board.height and 0 <= nc < board.width:
                    lines.append(f"  {name}: wall")
                else:
                    lines.append(f"  {name}: edge")
        return "\n".join(lines)


class BFSOrdered(BaseRepr):
    key = "11_BFS_ORDERED"

    def render(self, board):
        visited = {board.player}
        queue = deque([board.player])
        order = []
        while queue:
            r, c = queue.popleft()
            order.append((r, c, board.describe_tile(r, c), board.tile_index[(r, c)]))
            for _, dr, dc in [("u",-1,0),("d",1,0),("l",0,-1),("r",0,1)]:
                pos = (r + dr, c + dc)
                if pos in board.tile_index and pos not in visited and pos not in board.boxes:
                    visited.add(pos)
                    queue.append(pos)
        lines = ["BFS from player (reachable without pushing):"]
        for step, (r, c, entity, tidx) in enumerate(order):
            lines.append(f"  Step {step:2d}: ({r},{c}) tile={tidx} = {entity}")
        return "\n".join(lines)