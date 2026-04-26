from dataclasses import dataclass
from typing import FrozenSet, Tuple


@dataclass(frozen=True)
class SokobanState:
    """Immutable game state. Walls and goals are static — only player and boxes change."""
    player: Tuple[int, int]
    boxes: FrozenSet[Tuple[int, int]]

    def __hash__(self):
        return hash((self.player, self.boxes))

    def __eq__(self, other):
        return (
            isinstance(other, SokobanState)
            and self.player == other.player
            and self.boxes == other.boxes
        )

    def __repr__(self):
        return f"SokobanState(player={self.player}, boxes={sorted(self.boxes)})"