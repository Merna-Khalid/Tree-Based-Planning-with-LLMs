"""Tensor / flat array representation: 08."""

from representations.base import BaseRepr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.board import SokobanBoard

_CH_BITS = {
    "@": (1, 0, 0, 0),
    "+": (1, 0, 0, 1),
    "#": (0, 1, 0, 0),
    "$": (0, 0, 1, 0),
    "*": (0, 0, 1, 1),
    ".": (0, 0, 0, 1),
    " ": (0, 0, 0, 0),
}


class TensorFlatten(BaseRepr):
    key = "08_TENSOR_FLATTEN"

    def render(self, board):
        from representations.ascii_repr import _render_grid
        values = []
        for row in _render_grid(board):
            for ch in row:
                values.extend(_CH_BITS.get(ch, (0, 0, 0, 0)))
        return (
            f"Tensor (C=4, H={board.height}, W={board.width}):\n"
            f"  Channels: [player, wall, box, goal]\n"
            f"  Flattened: {' '.join(map(str, values))}\n"
            f"  Total elements: {len(values)}"
        )