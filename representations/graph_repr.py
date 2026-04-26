"""Graph adjacency representation: 06."""

from representations.base import BaseRepr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.board import SokobanBoard


class GraphAdjacency(BaseRepr):
    key = "06_GRAPH_ADJACENCY"

    def render(self, board):
        from core.board import DIRECTIONS
        lines = ["Graph:"]
        for (r, c), idx in sorted(board.tile_index.items(), key=lambda x: x[1]):
            entity = board.describe_tile(r, c)
            neighbors = []
            for name, dr, dc in DIRECTIONS:
                nb = (r + dr, c + dc)
                if nb in board.tile_index:
                    neighbors.append(f"{name}->{board.tile_index[nb]}")
            lines.append(
                f"  Node {idx}: {entity} @ ({r},{c}), "
                f"adj=[{', '.join(neighbors) or 'none'}]"
            )
        return "\n".join(lines)