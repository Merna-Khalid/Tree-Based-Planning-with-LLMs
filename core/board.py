import copy
import random
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from core.state import SokobanState

DIRECTIONS: List[Tuple[str, int, int]] = [
    ("u", -1,  0),
    ("d",  1,  0),
    ("l",  0, -1),
    ("r",  0,  1),
]

_LURD_VECTORS: Dict[str, Tuple[int, int]] = {
    "u": (-1, 0), "U": (-1, 0),
    "d": ( 1, 0), "D": ( 1, 0),
    "l": ( 0,-1), "L": ( 0,-1),
    "r": ( 0, 1), "R": ( 0, 1),
}

_PLAYER = 0
_BOX    = 1


class ZobristTable:
    """
    Transposition table key generator.

    table[tile_idx][PLAYER | BOX] = random 64-bit int
    Generated once per board layout, seeded deterministically.

    hash(state) = XOR( table[norm_player][PLAYER],
                       table[box_tile][BOX] for every box )

    Incremental updates:
        walk → same hash (access area invariant, see get_legal_moves)
        push → O(1) XOR for box + O(area) BFS for new norm_player
    """

    def __init__(self, n_tiles: int, seed: int = 42):
        rng = random.Random(seed)
        self.table: List[List[int]] = [
            [rng.getrandbits(64), rng.getrandbits(64)]
            for _ in range(n_tiles)
        ]

    def compute(self, player_tile: int, box_tiles: FrozenSet[int]) -> int:
        """Full hash from scratch — O(n_boxes). Only called during init."""
        h = self.table[player_tile][_PLAYER]
        for bt in box_tiles:
            h ^= self.table[bt][_BOX]
        return h

    def update_player(self, h: int, old_tile: int, new_tile: int) -> int:
        """XOR out old player tile, XOR in new one — O(1)."""
        return h ^ self.table[old_tile][_PLAYER] ^ self.table[new_tile][_PLAYER]

    def update_box(self, h: int, old_tile: int, new_tile: int) -> int:
        """XOR out old box tile, XOR in new one — O(1)."""
        return h ^ self.table[old_tile][_BOX] ^ self.table[new_tile][_BOX]


class SokobanBoard:
    """
    Sokoban board with incremental Zobrist hashing.

    Key properties:
        zobrist_hash  — 64-bit transposition table key, O(1)
        norm_player   — normalised player tile (min reachable), O(1)

    Walk moves:  hash unchanged — walks never change the access area because
                 no box moves, so the reachable tile set is identical.
    Push moves:  O(1) box XOR + O(access_area) BFS for new norm_player.

    Floor is flood-filled from the player (boxes passable) so void tiles
    outside the puzzle walls are excluded from tile_index and hashing.

    LURD encoding:
        u/d/l/r  — player walks, no box pushed
        U/D/L/R  — player walks AND pushes a box

    get_legal_moves  — immediate neighbours only (4 directions, walk or push)
    get_legal_pushes — BFS all reachable pushes from current access area
    """

    def __init__(self, raw_ascii: str):
        lines = raw_ascii.strip().split("\n")
        if lines:
            min_indent = min(len(l) - len(l.lstrip()) for l in lines if l.strip())
            lines = [l[min_indent:] for l in lines]

        self.height = len(lines)
        self.width  = max(len(l) for l in lines)
        padded      = [l.ljust(self.width) for l in lines]

        player = None
        boxes:  List[Tuple[int, int]] = []
        goals:  Set[Tuple[int, int]]  = set()
        self.walls: Set[Tuple[int, int]] = set()

        for r, row in enumerate(padded):
            for c, ch in enumerate(row):
                pos = (r, c)
                if   ch == "#": self.walls.add(pos)
                elif ch == "@": player = pos
                elif ch == "$": boxes.append(pos)
                elif ch == ".": goals.add(pos)
                elif ch == "*": boxes.append(pos); goals.add(pos)
                elif ch == "+": player = pos; goals.add(pos)

        self.goals: FrozenSet[Tuple[int, int]] = frozenset(goals)

        # Flood-fill from player (boxes passable) — excludes void tiles
        self.floor: Set[Tuple[int, int]] = set()
        if player is not None:
            _vis: Set[Tuple[int, int]] = {player}
            _q:   deque = deque([player])
            while _q:
                cr, cc = _q.popleft()
                self.floor.add((cr, cc))
                for _, dr, dc in DIRECTIONS:
                    nb = (cr + dr, cc + dc)
                    if (nb not in _vis
                            and nb not in self.walls
                            and 0 <= nb[0] < self.height
                            and 0 <= nb[1] < self.width):
                        _vis.add(nb)
                        _q.append(nb)

        # Row-major tile index — flood-filled floor only
        self.tile_index:    Dict[Tuple[int, int], int] = {}
        self.index_to_tile: Dict[int, Tuple[int, int]] = {}
        for idx, (r, c) in enumerate(
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) in self.floor
        ):
            self.tile_index[(r, c)] = idx
            self.index_to_tile[idx] = (r, c)

        self._zobrist: ZobristTable = ZobristTable(len(self.tile_index))
        self._state = SokobanState(player=player, boxes=frozenset(boxes))

        # Initial hash — computed from scratch once
        self._norm_player: int = self._bfs_min_tile(self._state)
        self._zhash: int = self._zobrist.compute(
            self._norm_player,
            frozenset(self.tile_index[b] for b in self._state.boxes),
        )

        self._validate()

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def state(self) -> SokobanState:
        return self._state

    @property
    def player(self) -> Tuple[int, int]:
        return self._state.player

    @property
    def boxes(self) -> FrozenSet[Tuple[int, int]]:
        return self._state.boxes

    @property
    def boxes_on_goals(self) -> List[Tuple[int, int]]:
        return [b for b in self._state.boxes if b in self.goals]

    @property
    def unsolved_boxes(self) -> List[Tuple[int, int]]:
        return [b for b in self._state.boxes if b not in self.goals]

    # ------------------------------------------------------------------
    # Hashing
    # ------------------------------------------------------------------

    @property
    def zobrist_hash(self) -> int:
        """64-bit transposition table key. O(1)."""
        return self._zhash

    @property
    def norm_player(self) -> int:
        """Normalised player tile index (min reachable). O(1)."""
        return self._norm_player

    def _bfs_min_tile(self, state: SokobanState) -> int:
        """
        BFS from player, skipping boxes and tiles not in floor.
        Returns the minimum tile index in the reachable area — the canonical
        player representative for this access area.
        O(reachable area).
        """
        start   = state.player
        visited = {start}
        queue   = deque([start])
        min_idx = self.tile_index[start]

        while queue:
            r, c = queue.popleft()
            for _, dr, dc in DIRECTIONS:
                pos = (r + dr, c + dc)
                if (pos not in visited
                        and pos in self.floor          # FIX: was is_valid_pos only
                        and pos not in state.boxes):
                    visited.add(pos)
                    queue.append(pos)
                    idx = self.tile_index[pos]
                    if idx < min_idx:
                        min_idx = idx

        return min_idx

    def _make_child(
        self,
        new_state: SokobanState,
        new_zhash: int,
        new_norm_player: int,
    ) -> "SokobanBoard":
        """
        Fast child board construction — shallow copy, precomputed hash.
        All heavy structures (walls, floor, tile_index, Zobrist table) are shared.
        """
        child = copy.copy(self)
        child._state       = new_state
        child._zhash       = new_zhash
        child._norm_player = new_norm_player
        child._zobrist     = self._zobrist
        return child

    # ------------------------------------------------------------------
    # Game logic
    # ------------------------------------------------------------------

    def is_solved(self) -> bool:
        return self._state.boxes == self.goals

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width and pos not in self.walls

    def get_legal_moves(self) -> List[Tuple[str, SokobanState, int, int]]:
        """
        Returns [(lurd, new_state, new_zhash, new_norm_player), ...].

        Only looks at immediate neighbours (max 4 results).

        Walk moves:
            new_zhash = self._zhash          (unchanged — walk never changes access area)
            new_norm  = self._norm_player    (unchanged — same reasoning)
            Cost: O(1), no BFS.

        Push moves:
            new_zhash = incremental XOR update  (O(1))
            new_norm  = BFS on new state        (O(access_area))
        """
        moves = []
        r, c  = self._state.player

        for base_char, dr, dc in DIRECTIONS:
            new_player = (r + dr, c + dc)

            # Only floor tiles are valid destinations
            if new_player not in self.floor:
                continue

            if new_player in self._state.boxes:
                # ---- PUSH ----
                new_box = (r + 2 * dr, c + 2 * dc)
                if new_box not in self.floor or new_box in self._state.boxes:
                    continue

                new_boxes = (self._state.boxes - {new_player}) | {new_box}
                new_state = SokobanState(player=new_player, boxes=new_boxes)

                # Access area changes after push — BFS required
                new_norm  = self._bfs_min_tile(new_state)
                old_bt    = self.tile_index[new_player]   # box was here
                new_bt    = self.tile_index[new_box]       # box goes here
                h = self._zobrist.update_player(self._zhash, self._norm_player, new_norm)
                h = self._zobrist.update_box(h, old_bt, new_bt)

                moves.append((base_char.upper(), new_state, h, new_norm))

            else:
                # ---- WALK ----
                # Walk moves never change the access area (no box moved, same reachable
                # set) so hash and norm_player are identical to the parent. No BFS.
                new_state = SokobanState(player=new_player, boxes=self._state.boxes)
                moves.append((base_char.lower(), new_state, self._zhash, self._norm_player))

        return moves

    def get_legal_pushes(self) -> List[Tuple[str, SokobanState, int, int]]:
        """
        All reachable push moves via BFS from player.
        Same tuple format as get_legal_moves: (lurd, state, zhash, norm_player).

        The BFS walks the player through the entire access area to find every
        push reachable from any position in that area. Used by push-based search.
        """
        reachable: Set[Tuple[int, int]] = set()
        queue = deque([self._state.player])
        reachable.add(self._state.player)
        while queue:
            rr, cc = queue.popleft()
            for _, dr, dc in DIRECTIONS:
                pos = (rr + dr, cc + dc)
                if (pos not in reachable
                        and pos in self.floor          # FIX: was is_valid_pos only
                        and pos not in self._state.boxes):
                    reachable.add(pos)
                    queue.append(pos)

        pushes: List[Tuple[str, SokobanState, int, int]] = []
        seen:   Set[Tuple] = set()

        for rr, cc in reachable:
            for base_char, dr, dc in DIRECTIONS:
                box_pos  = (rr + dr, cc + dc)
                if box_pos not in self._state.boxes:
                    continue
                box_dest = (rr + 2 * dr, cc + 2 * dc)
                if box_dest not in self.floor or box_dest in self._state.boxes:
                    continue
                key = (box_pos, box_dest)
                if key in seen:
                    continue
                seen.add(key)

                new_boxes = (self._state.boxes - {box_pos}) | {box_dest}
                new_state = SokobanState(player=box_pos, boxes=new_boxes)
                new_norm  = self._bfs_min_tile(new_state)
                old_bt    = self.tile_index[box_pos]
                new_bt    = self.tile_index[box_dest]
                h = self._zobrist.update_player(self._zhash, self._norm_player, new_norm)
                h = self._zobrist.update_box(h, old_bt, new_bt)
                pushes.append((base_char.upper(), new_state, h, new_norm))

        return pushes

    def walk_path_to(self, target: Tuple[int, int]) -> Optional[List[str]]:
        """
        BFS walk path from current player to target without pushing.
        Returns list of lowercase LURD walk chars, or None if unreachable.
        Used by expand_to_full_moves to fill in walks between pushes.
        """
        if self._state.player == target:
            return []

        # visited: pos → (parent_pos, lurd_char)
        visited: Dict[Tuple[int, int], Optional[Tuple]] = {self._state.player: None}
        queue = deque([self._state.player])

        while queue:
            pos = queue.popleft()
            for base_char, dr, dc in DIRECTIONS:
                npos = (pos[0] + dr, pos[1] + dc)
                if (npos not in visited
                        and npos in self.floor          # FIX: was is_valid_pos only
                        and npos not in self._state.boxes):
                    visited[npos] = (pos, base_char.lower())
                    if npos == target:
                        path = []
                        cur  = npos
                        while visited[cur] is not None:
                            prev, ch = visited[cur]
                            path.append(ch)
                            cur = prev
                        return list(reversed(path))
                    queue.append(npos)

        return None

    def apply_move(self, direction: str) -> Tuple["SokobanBoard", str]:
        """
        Apply one move, return (new_board, canonical_lurd_char).
        new_board.zobrist_hash and .norm_player are O(1) — already set.

        FIX vs old version: no longer does a linear scan over get_legal_moves().
        Directly computes the one child for the requested direction.
        """
        if direction not in _LURD_VECTORS:
            raise ValueError(f"Unknown direction: {direction!r}. Use u/d/l/r or U/D/L/R.")

        dr, dc     = _LURD_VECTORS[direction]
        r, c       = self._state.player
        new_player = (r + dr, c + dc)

        if new_player not in self.floor:
            raise ValueError(f"Illegal move: {direction!r}")

        if new_player in self._state.boxes:
            # Push
            new_box = (r + 2 * dr, c + 2 * dc)
            if new_box not in self.floor or new_box in self._state.boxes:
                raise ValueError(f"Illegal move: {direction!r}")
            new_boxes = (self._state.boxes - {new_player}) | {new_box}
            new_state = SokobanState(player=new_player, boxes=new_boxes)
            new_norm  = self._bfs_min_tile(new_state)
            old_bt    = self.tile_index[new_player]
            new_bt    = self.tile_index[new_box]
            h = self._zobrist.update_player(self._zhash, self._norm_player, new_norm)
            h = self._zobrist.update_box(h, old_bt, new_bt)
            lurd = direction.upper()
        else:
            # Walk
            new_state = SokobanState(player=new_player, boxes=self._state.boxes)
            h         = self._zhash
            new_norm  = self._norm_player
            lurd      = direction.lower()

        return self._make_child(new_state, h, new_norm), lurd

    def apply_solution(self, solution: str) -> Tuple[List["SokobanBoard"], str]:
        """Apply a full LURD solution string. Returns (boards, canonical_lurd_string)."""
        boards   = [self]
        lurd_out = []
        for ch in solution:
            new_board, lurd = boards[-1].apply_move(ch)
            boards.append(new_board)
            lurd_out.append(lurd)
        return boards, "".join(lurd_out)

    def with_state(self, state: SokobanState) -> "SokobanBoard":
        """
        Return a board with an arbitrary state.
        Recomputes hash from scratch — intentionally not in the hot search path.
        Use _make_child when you already have zhash and norm_player.
        """
        norm  = self._bfs_min_tile(state)
        zhash = self._zobrist.compute(
            norm, frozenset(self.tile_index[b] for b in state.boxes)
        )
        return self._make_child(state, zhash, norm)

    def get_reachable_floor(self) -> Set[Tuple[int, int]]:
        """
        BFS reachable tiles from player without pushing.
        Shares logic with _bfs_min_tile but returns the full set instead of min index.
        """
        visited = {self._state.player}
        queue   = deque([self._state.player])
        while queue:
            r, c = queue.popleft()
            for _, dr, dc in DIRECTIONS:
                pos = (r + dr, c + dc)
                if (pos not in visited
                        and pos in self.floor           # consistent with _bfs_min_tile
                        and pos not in self._state.boxes):
                    visited.add(pos)
                    queue.append(pos)
        return visited

    def describe_tile(self, r: int, c: int) -> str:
        pos = (r, c)
        if pos in self.walls:         return "wall"
        if pos == self._state.player: return "player-on-goal" if pos in self.goals else "player"
        if pos in self._state.boxes:  return "box-on-goal"    if pos in self.goals else "box"
        if pos in self.goals:         return "goal"
        return "floor"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):
        errors = []
        if self._state.player is None:
            errors.append("No player found")
        elif self._state.player not in self.floor:
            errors.append(f"Player {self._state.player} not on floor")
        for b in self._state.boxes:
            if b not in self.floor:
                errors.append(f"Box {b} not on floor")
        if len(self._state.boxes) != len(self.goals):
            errors.append(f"Box count {len(self._state.boxes)} != goal count {len(self.goals)}")
        if self._state.player in self._state.boxes:
            errors.append("Player on a box")
        if errors:
            raise ValueError("Board validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    def __repr__(self):
        return (f"SokobanBoard({self.height}x{self.width}, "
                f"boxes={len(self._state.boxes)}, solved={self.is_solved()})")