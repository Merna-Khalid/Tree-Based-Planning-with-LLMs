"""ASCII-based representations: 01, 02, 03, 09, 10, 12, 13."""

from representations.base import BaseRepr
from typing import List

from core.board import SokobanBoard


def _render_grid(board: "SokobanBoard") -> List[str]:
    """Render current board state as a list of strings."""
    rows = []
    for r in range(board.height):
        row = ""
        for c in range(board.width):
            pos = (r, c)
            if   pos == board.player:       row += "+" if pos in board.goals else "@"
            elif pos in board.boxes:        row += "*" if pos in board.goals else "$"
            elif pos in board.goals:        row += "."
            elif pos in board.walls:        row += "#"
            else:                           row += " "
        rows.append(row)
    return rows


class AsciiRaw(BaseRepr):
    key = "01_ASCII_RAW"

    def render(self, board):
        return "\n".join(_render_grid(board))


class AsciiRowMarkers(BaseRepr):
    key = "02_ASCII_ROW_MARKERS"

    def render(self, board):
        return "\n".join(f"<ROW {i}> {line} </ROW>"
                         for i, line in enumerate(_render_grid(board)))


class RLE(BaseRepr):
    key = "03_RLE"

    def render(self, board):
        def encode(line: str) -> str:
            result, i = [], 0
            while i < len(line):
                ch, count = line[i], 1
                while i + count < len(line) and line[i + count] == ch:
                    count += 1
                display = "-" if ch == " " else ch
                result.append(f"{count}{display}" if count >= 2 else display)
                i += count
            return "".join(result)
        return "|".join(encode(row) for row in _render_grid(board))


class SemanticSummary(BaseRepr):
    key = "09_SEMANTIC_SUMMARY"

    def render(self, board):
        from core.deadlock import corner_deadlock
        return (
            f"Sokoban Puzzle:\n"
            f"- Board: {board.height}x{board.width}\n"
            f"- Player at {board.player}\n"
            f"- {len(board.boxes)} boxes: {len(board.boxes_on_goals)} solved, "
            f"{len(board.unsolved_boxes)} to push\n"
            f"- Goals: {sorted(board.goals)}\n"
            f"- Solved: {board.is_solved()}\n"
            f"- Deadlock (corner): {corner_deadlock(board)}"
        )


class CompactSymbolic(BaseRepr):
    key = "10_COMPACT_SYMBOLIC"

    def render(self, board):
        return "|".join(_render_grid(board))


class GridWithIndices(BaseRepr):
    key = "12_GRID_WITH_INDICES"

    def render(self, board):
        lines = ["Grid with tile indices (## = wall):"]
        for r in range(board.height):
            row = ""
            for c in range(board.width):
                if (r, c) in board.walls:
                    row += "[##]"
                elif (r, c) in board.tile_index:
                    row += f"[{board.tile_index[(r, c)]:2d}]"
                else:
                    row += "    "
            lines.append(f"  Row {r}: {row}")
        return "\n".join(lines)


class ActionCentric(BaseRepr):
    key = "13_ACTION_CENTRIC"

    def render(self, board):
        from core.board import DIRECTIONS
        r, c = board.player
        p_idx = board.tile_index[(r, c)]
        lines = [f"Action-centric view:", f"Player at tile {p_idx} ({r},{c})"]
        dir_names = {"u": "Up", "d": "Down", "l": "Left", "r": "Right"}
        for name, dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            neighbor = (nr, nc)
            if neighbor in board.tile_index:
                entity = board.describe_tile(nr, nc)
                t_idx = board.tile_index[neighbor]
                if "box" in entity:
                    behind = (nr + dr, nc + dc)
                    if behind in board.tile_index and behind not in board.boxes:
                        lines.append(f"  {dir_names[name]}: {entity} at tile {t_idx} "
                                     f"-> CAN PUSH to tile {board.tile_index[behind]}")
                    else:
                        lines.append(f"  {dir_names[name]}: {entity} at tile {t_idx} -> BLOCKED")
                else:
                    lines.append(f"  {dir_names[name]}: {entity} at tile {t_idx} -> CAN MOVE")
            else:
                lines.append(f"  {dir_names[name]}: wall -> BLOCKED")
        return "\n".join(lines)