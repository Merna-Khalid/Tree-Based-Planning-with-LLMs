from __future__ import annotations
 
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from core.board import SokobanBoard
 
 
@dataclass
class MicrobanPuzzle:
    name: str           # level number/name as string, e.g. "1"
    index: int          # 0-based index in the file
    raw_lines: List[str]  # original grid lines, no stripping
 
    @property
    def height(self) -> int:
        return len(self.raw_lines)
 
    @property
    def width(self) -> int:
        return max(len(line) for line in self.raw_lines) if self.raw_lines else 0
 
    @property
    def num_boxes(self) -> int:
        grid = "\n".join(self.raw_lines)
        return grid.count("$") + grid.count("*")
 
    @property
    def num_goals(self) -> int:
        grid = "\n".join(self.raw_lines)
        return grid.count(".") + grid.count("*") + grid.count("+")
 
    @property
    def difficulty_proxy(self) -> str:
        """Rough difficulty bucket based on grid size and box count."""
        boxes = self.num_boxes
        area = self.height * self.width
        if boxes <= 1 and area <= 50:
            return "easy"
        if boxes <= 3 and area <= 100:
            return "medium"
        return "hard"
 
    def to_board(self):
        return SokobanBoard("\n".join(self.raw_lines))
 
    def __repr__(self):
        return (f"MicrobanPuzzle(name={self.name!r}, "
                f"{self.height}x{self.width}, "
                f"boxes={self.num_boxes}, "
                f"difficulty={self.difficulty_proxy!r})")
 
 
def load_microban(path: str | Path) -> List[MicrobanPuzzle]:
    """
    Parse a Microban .txt file and return all valid puzzles.
 
    Handles:
      - '; N' separators (with or without spaces)
      - Multiple consecutive blank lines between puzzles
      - Puzzles with no valid grid (skipped with a warning)
      - Windows (\\r\\n) and Unix (\\n) line endings
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
 
    puzzles: List[MicrobanPuzzle] = []
    current_name: Optional[str] = None
    current_lines: List[str] = []
    index = 0
 
    def _flush():
        nonlocal index
        if current_name is None:
            return
        grid = _extract_grid(current_lines)
        if not grid:
            return
        try:
            p = MicrobanPuzzle(name=current_name, index=index, raw_lines=grid)
            # Quick validation: must parse without error
            p.to_board()
            puzzles.append(p)
            index += 1
        except Exception as exc:
            print(f"  [microban] Skipping puzzle {current_name!r}: {exc}")
 
    for line in lines:
        stripped = line.rstrip()
 
        # Detect separator: '; N' or ';N'
        if re.match(r"^\s*;\s*\S", stripped):
            _flush()
            current_name = stripped.lstrip("; \t").strip()
            current_lines = []
        else:
            if current_name is not None:
                current_lines.append(stripped)
 
    _flush()   # catch the last puzzle
    return puzzles
 
 
def _extract_grid(lines: List[str]) -> List[str]:
    """
    From the raw lines after a ';' separator, extract just the puzzle grid.
    Strips leading/trailing blank lines.  Keeps internal blank lines only
    if they appear between grid rows (rare but valid in some editors).
    """
    # Drop leading blank lines
    start = 0
    while start < len(lines) and not lines[start].strip():
        start += 1
 
    # Drop trailing blank lines
    end = len(lines)
    while end > start and not lines[end - 1].strip():
        end -= 1
 
    grid = lines[start:end]
 
    # Must contain at least one wall '#' to be a valid Sokoban grid
    flat = "".join(grid)
    if "#" not in flat:
        return []
 
    return grid
 
 
# ---------------------------------------------------------------------------
# Convenience helpers for notebooks
# ---------------------------------------------------------------------------
 
def select_by_difficulty(
    puzzles: List[MicrobanPuzzle],
    easy: int = 4,
    medium: int = 4,
    hard: int = 2,
) -> List[MicrobanPuzzle]:
    """
    Pick a balanced set of puzzles for evaluation.
    Returns `easy + medium + hard` puzzles in that order.
    """
    by_diff: dict = {"easy": [], "medium": [], "hard": []}
    for p in puzzles:
        by_diff[p.difficulty_proxy].append(p)
 
    selected: List[MicrobanPuzzle] = []
    for bucket, n in [("easy", easy), ("medium", medium), ("hard", hard)]:
        selected.extend(by_diff[bucket][:n])
 
    return selected
 
 
def puzzle_summary(puzzles: List[MicrobanPuzzle]) -> str:
    """Print a quick summary ttable — useful in notebooks."""
    lines = [f"{'#':>4}  {'Name':>6}  {'H':>3}  {'W':>3}  {'Boxes':>5}  {'Difficulty'}"]
    lines.append("-" * 40)
    for p in puzzles:
        lines.append(
            f"{p.index:>4}  {p.name:>6}  {p.height:>3}  {p.width:>3}"
            f"  {p.num_boxes:>5}  {p.difficulty_proxy}"
        )
    return "\n".join(lines)