"""Bitboard representations: 14, 15, 16, 17, 18."""

from representations.base import BaseRepr
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from core.board import SokobanBoard


def _bit_masks(board: "SokobanBoard") -> Dict[str, int]:
    wall_m = player_m = box_m = goal_m = 0
    for r in range(board.height):
        for c in range(board.width):
            pos = r * board.width + c
            if (r, c) in board.walls:          wall_m   |= 1 << pos
            if (r, c) == board.player:         player_m |= 1 << pos
            if (r, c) in board.boxes:          box_m    |= 1 << pos
            if (r, c) in board.goals:          goal_m   |= 1 << pos
    return {
        "wall": wall_m, "player": player_m,
        "box": box_m,   "goal": goal_m,
        "total_bits": board.height * board.width,
    }


class BitboardHex(BaseRepr):
    key = "14_BITBOARD_HEX"

    def render(self, board):
        masks = _bit_masks(board)
        nbits = masks["total_bits"]
        hw = (nbits + 3) // 4
        lines = [f"Bitboards (hex, {nbits} bits, row-major LSB=top-left):"]
        for name in ("wall", "player", "box", "goal"):
            lines.append(f"  {name:8s}: 0x{masks[name]:0{hw}x}")
        return "\n".join(lines)


class BitboardBinaryRows(BaseRepr):
    key = "15_BITBOARD_BINARY_ROWS"

    def render(self, board):
        masks = _bit_masks(board)
        lines = ["Bitboards (binary, row-major):"]
        for name in ("wall", "player", "box", "goal"):
            mask = masks[name]
            lines.append(f"\n  {name.upper()}:")
            for r in range(board.height):
                bits = "".join(
                    "1" if (mask >> (r * board.width + c)) & 1 else "0"
                    for c in range(board.width)
                )
                lines.append(f"    Row {r}: {bits}")
        return "\n".join(lines)


class BitboardOverlaid(BaseRepr):
    key = "16_BITBOARD_OVERLAID"

    def render(self, board):
        masks = _bit_masks(board)
        lines = ["Bitboard overlay (P=player, B=box, G=goal, W=wall, .=empty, +=multi):"]
        for r in range(board.height):
            row_str = "    "
            for c in range(board.width):
                pos = r * board.width + c
                chars = (
                    (["W"] if (masks["wall"]   >> pos) & 1 else []) +
                    (["P"] if (masks["player"] >> pos) & 1 else []) +
                    (["B"] if (masks["box"]    >> pos) & 1 else []) +
                    (["G"] if (masks["goal"]   >> pos) & 1 else [])
                )
                row_str += "." if not chars else (chars[0] if len(chars) == 1 else "+")
            lines.append(row_str)
        return "\n".join(lines)


class BitboardMixed(BaseRepr):
    key = "17_BITBOARD_MIXED"

    def render(self, board):
        from representations.ascii_repr import _render_grid
        grid = _render_grid(board)
        lines = ["Bitboard with positions (bit index in parentheses):"]
        for r, row in enumerate(grid):
            row_str = "".join(
                f"{'.' if ch == ' ' else ch}({r * board.width + c:2d})"
                for c, ch in enumerate(row)
            )
            lines.append(f"  Row {r}: {row_str}")
        return "\n".join(lines)


class BitboardCombined(BaseRepr):
    key = "18_BITBOARD_COMBINED"

    def render(self, board):
        combined = 0
        for r in range(board.height):
            for c in range(board.width):
                pos = r * board.width + c
                if   (r, c) in board.walls:  tile_bits = 1
                elif (r, c) in board.boxes:  tile_bits = 2
                elif (r, c) == board.player: tile_bits = 3
                else:                        tile_bits = 0
                combined |= tile_bits << (pos * 2)
        total_bits = board.height * board.width * 2
        hex_str = f"{combined:0{(total_bits + 3) // 4}x}"
        return (
            f"Combined state mask (2 bits/tile):\n"
            f"  Hex: 0x{hex_str}\n"
            f"  Bits: {total_bits}\n"
            f"  Format: 00=floor/goal, 01=wall, 10=box, 11=player"
        )