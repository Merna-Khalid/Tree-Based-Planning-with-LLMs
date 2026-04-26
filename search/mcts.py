"""
MCTS for Sokoban — push-based.

    mcts(board, iterations)       — random rollouts, no LLM
    llm_mcts(board, predictor, iterations)
        Expansion: LLM picks one child (PUCT prior 1.0 for that child, 0.1 others).
        Rollout:   LLM picks greedily at each step. No feedback in rollout — if
                   LLM output is invalid, pick random from valid pushes.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.state import SokobanState
from core.deadlock import is_deadlock, precompute_simple_deadlock_squares
from search.heuristics import HEURISTICS
from search.result import SearchResult
from search.astar import _ask_llm

from core.board import SokobanBoard
from llm.predictor import LLMPredictor

UCB_C = math.sqrt(2)


@dataclass
class _MCTSNode:
    """MCTS node with proper state tracking."""
    state: SokobanState
    zhash: int
    norm_player: int
    board_ref: 'SokobanBoard'  # Reference to static board (walls, goals)
    
    # Tree info
    parent: Optional['_MCTSNode'] = field(default=None, repr=False)
    move_from_parent: Optional[str] = None
    moves: List[str] = field(default_factory=list)  # Push moves from root
    states: List[SokobanState] = field(default_factory=list)  # States after each push
    
    # MCTS stats
    visits: int = 0
    total_reward: float = 0.0
    prior: float = 1.0
    children: List['_MCTSNode'] = field(default_factory=list, repr=False)
    untried_moves: List[Tuple[str, SokobanState, int, int]] = field(default_factory=list)
    
    def __post_init__(self):
        # Lazy expansion - store untried moves, don't create children yet
        pass
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0
    
    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
    
    def ucb_score(self, parent_visits: int, c: float = UCB_C) -> float:
        """UCB1 formula for node selection."""
        if self.visits == 0:
            return float('inf')
        return self.value + c * math.sqrt(math.log(parent_visits) / self.visits)
    
    def puct_score(self, parent_visits: int, c: float = 1.0) -> float:
        """PUCT formula with prior."""
        if self.visits == 0:
            return float('inf')
        return self.value + c * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
    
    def best_child(self, use_puct: bool = True, c: float = UCB_C) -> '_MCTSNode':
        """Select best child by UCB/PUCT score."""
        if use_puct:
            return max(self.children, key=lambda ch: ch.puct_score(self.visits, c))
        return max(self.children, key=lambda ch: ch.ucb_score(self.visits, c))


def mcts(
    board: SokobanBoard,
    iterations: int = 1000,
    rollout_depth: int = 50,
    exploration: float = 1.41,
    heuristic: str = "manhattan",
    time_limit: float = 60.0,  # seconds
) -> SearchResult:

    t0 = time.time()
    dead_sq = precompute_simple_deadlock_squares(board)
    h_fn = HEURISTICS[heuristic]
    
    # Create root node
    root = _MCTSNode(
        state=board.state,
        zhash=board.zobrist_hash,
        norm_player=board.norm_player,
        board_ref=board,
        moves=[],
        states=[]
    )
    
    # Initialize untried moves for root
    root.untried_moves = _get_valid_pushes(board, board.state, board.zobrist_hash, 
                                           board.norm_player, dead_sq)
    
    best_solution = None
    best_states = None
    nodes_expanded = 0
    nodes_generated = 1  # root count
    
    for iteration in range(iterations):
        if time.time() - t0 > time_limit:
            break
            
        # 1. Selection - traverse tree
        node = root
        path = [node]
        
        while not node.is_leaf and node.is_fully_expanded:
            node = node.best_child(use_puct=True, c=exploration)
            path.append(node)
        
        # Current board state for this node
        current_board = board._make_child(node.state, node.zhash, node.norm_player)
        
        # 2. Expansion - if not terminal and not fully expanded
        if not current_board.is_solved() and not node.is_fully_expanded:
            # Expand one child (lazy expansion)
            if node.untried_moves:
                lurd, new_state, new_zhash, new_norm = node.untried_moves.pop()
                child = _MCTSNode(
                    state=new_state,
                    zhash=new_zhash,
                    norm_player=new_norm,
                    board_ref=board,
                    parent=node,
                    move_from_parent=lurd,
                    moves=node.moves + [lurd],
                    states=node.states + [new_state],
                    prior=1.0  # Default prior
                )
                # Initialize child's untried moves
                child_board = board._make_child(new_state, new_zhash, new_norm)
                child.untried_moves = _get_valid_pushes(board, new_state, new_zhash,
                                                        new_norm, dead_sq)
                node.children.append(child)
                nodes_generated += 1
                node = child
                path.append(node)
                current_board = board._make_child(node.state, node.zhash, node.norm_player)
        
        # 3. Simulation (rollout) - only if not solved
        if current_board.is_solved():
            reward = 1.0
            if best_solution is None or len(node.moves) < len(best_solution):
                best_solution = node.moves[:]
                best_states = node.states[:]
        else:
            reward, sim_moves, sim_states = _rollout(
                current_board, board, node.moves[:], node.states[:],
                rollout_depth, dead_sq, h_fn
            )
            # Check if rollout found a solution
            if sim_moves is not None and (best_solution is None or 
                                          len(sim_moves) < len(best_solution)):
                best_solution = sim_moves
                best_states = sim_states
        
        # 4. Backpropagation
        for n in reversed(path):
            n.visits += 1
            n.total_reward += reward
        
        nodes_expanded += 1
    
    solved = best_solution is not None
    
    # Verify solution if found
    if solved:
        try:
            # Test the solution
            test_board = board
            for move in best_solution:
                test_board, _ = test_board.apply_move(move)
            solved = test_board.is_solved()
        except:
            solved = False
    
    return SearchResult(
        algorithm="mcts",
        solved=solved,
        moves=best_solution or [],
        states=best_states or [],
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        llm_calls=0,
        elapsed_seconds=time.time() - t0,
        notes=f"iterations={iterations}, rollout_depth={rollout_depth}"
    )


def _get_valid_pushes(
    board: SokobanBoard,
    state: SokobanState,
    zhash: int,
    norm_player: int,
    dead_sq: Set
) -> List[Tuple[str, SokobanState, int, int]]:
    """
    Get all valid pushes from a state, filtered by deadlock.
    """
    current = board._make_child(state, zhash, norm_player)
    valid = []
    
    for lurd, new_state, new_zhash, new_norm in current.get_legal_pushes():
        child = board._make_child(new_state, new_zhash, new_norm)
        if not is_deadlock(child, dead_sq):
            valid.append((lurd, new_state, new_zhash, new_norm))
    
    return valid


def _rollout(
    board: SokobanBoard,
    root_board: SokobanBoard,
    moves_so_far: List[str],
    states_so_far: List[SokobanState],
    max_depth: int,
    dead_sq: Set,
    h_fn,
) -> Tuple[float, Optional[List[str]], Optional[List[SokobanState]]]:
    """
    Improved rollout with heuristic guidance and random exploration.
    """
    b = board
    moves = moves_so_far[:]
    states = states_so_far[:]
    steps = 0
    
    # Track best progress
    best_h = h_fn(b)
    best_boxes = len(b.boxes_on_goals)
    
    while steps < max_depth and not b.is_solved():
        # Get legal pushes
        legal = []
        for lurd, new_state, new_zhash, new_norm in b.get_legal_pushes():
            child = root_board._make_child(new_state, new_zhash, new_norm)
            if not is_deadlock(child, dead_sq):
                legal.append((lurd, new_state, new_zhash, new_norm))
        
        if not legal:
            break
        
        # Choose action with exploration (epsilon-greedy)
        # 70% heuristic best, 30% random
        if random.random() < 0.7:
            # Heuristic-guided: choose move that minimizes heuristic
            best_move = None
            best_score = float('inf')
            for lurd, ns, nz, nm in legal:
                child = root_board._make_child(ns, nz, nm)
                score = h_fn(child)
                if score < best_score:
                    best_score = score
                    best_move = (lurd, ns, nz, nm)
            chosen = best_move or random.choice(legal)
        else:
            # Random exploration
            chosen = random.choice(legal)
        
        lurd, new_state, new_zhash, new_norm = chosen
        b = root_board._make_child(new_state, new_zhash, new_norm)
        moves.append(lurd)
        states.append(new_state)
        steps += 1
        
        # Track progress
        current_h = h_fn(b)
        current_boxes = len(b.boxes_on_goals)
        if current_boxes > best_boxes:
            best_boxes = current_boxes
            best_h = current_h
        elif current_h < best_h:
            best_h = current_h
    
    # Calculate reward based on progress
    if b.is_solved():
        return 1.0, moves, states
    
    # Partial credit based on progress
    boxes_remaining = len(b.unsolved_boxes)
    total_boxes = len(b.goals)
    boxes_solved = total_boxes - boxes_remaining
    
    # Reward formula: 0.7 for boxes solved + 0.3 for heuristic improvement
    box_reward = boxes_solved / total_boxes if total_boxes > 0 else 0
    h_reward = max(0, 1.0 - best_h / (total_boxes * 10 + 1))
    reward = 0.7 * box_reward + 0.3 * h_reward
    
    return reward, None, None


# Fixed LLM MCTS
def llm_mcts(
    board: SokobanBoard,
    predictor: LLMPredictor,
    repr_key: str = "01_ASCII_RAW",
    iterations: int = 500,
    rollout_depth: int = 30,
    exploration: float = 1.0,
    heuristic: str = "manhattan",
    time_limit: float = 120.0,
) -> SearchResult:
    """
    LLM-guided MCTS with proper priors.
    """
    t0 = time.time()
    dead_sq = precompute_simple_deadlock_squares(board)
    h_fn = HEURISTICS[heuristic]
    llm_calls = 0
    
    # Create root node
    root = _MCTSNode(
        state=board.state,
        zhash=board.zobrist_hash,
        norm_player=board.norm_player,
        board_ref=board,
        moves=[],
        states=[]
    )
    
    # Get LLM prior for root moves
    valid_moves = _get_valid_pushes(board, board.state, board.zobrist_hash, 
                                    board.norm_player, dead_sq)
    
    if valid_moves:
        # Ask LLM for preferred move
        result = predictor.predict_next_move(board, repr_key)
        llm_calls += 1
        
        if result.ok and result.move:
            llm_fav = result.move.upper()
            # Set higher prior for LLM's choice
            for lurd, ns, nz, nm in valid_moves:
                if lurd == llm_fav:
                    root.untried_moves.append((lurd, ns, nz, nm))
                else:
                    # Add to untried with lower priority (will be expanded later)
                    root.untried_moves.append((lurd, ns, nz, nm))
            # Shuffle to avoid bias in expansion order
            random.shuffle(root.untried_moves)
        else:
            root.untried_moves = valid_moves
    else:
        root.untried_moves = []
    
    best_solution = None
    best_states = None
    nodes_expanded = 0
    nodes_generated = 1
    
    for iteration in range(iterations):
        if time.time() - t0 > time_limit:
            break
        
        # Selection
        node = root
        path = [node]
        
        while not node.is_leaf and node.is_fully_expanded:
            node = node.best_child(use_puct=True, c=exploration)
            path.append(node)
        
        current_board = board._make_child(node.state, node.zhash, node.norm_player)
        
        # Expansion (only expand if not terminal)
        if not current_board.is_solved() and not node.is_fully_expanded:
            if node.untried_moves:
                # Get LLM prior for this node's children
                llm_fav = None
                if random.random() < 0.8:  # Only call LLM 80% of time to save budget
                    result = predictor.predict_next_move(current_board, repr_key)
                    llm_calls += 1
                    if result.ok and result.move:
                        llm_fav = result.move.upper()
                
                # Expand one child
                lurd, new_state, new_zhash, new_norm = node.untried_moves.pop(0)
                prior = 1.0 if lurd == llm_fav else 0.3
                
                child = _MCTSNode(
                    state=new_state,
                    zhash=new_zhash,
                    norm_player=new_norm,
                    board_ref=board,
                    parent=node,
                    move_from_parent=lurd,
                    moves=node.moves + [lurd],
                    states=node.states + [new_state],
                    prior=prior
                )
                child_board = board._make_child(new_state, new_zhash, new_norm)
                child.untried_moves = _get_valid_pushes(board, new_state, new_zhash,
                                                        new_norm, dead_sq)
                node.children.append(child)
                nodes_generated += 1
                node = child
                path.append(node)
                current_board = child_board
        
        # Simulation
        if current_board.is_solved():
            reward = 1.0
            if best_solution is None or len(node.moves) < len(best_solution):
                best_solution = node.moves[:]
                best_states = node.states[:]
        else:
            reward, sim_moves, sim_states = _llm_rollout(
                current_board, board, predictor, repr_key,
                node.moves[:], node.states[:],
                rollout_depth, dead_sq, h_fn
            )
            llm_calls += 1  # One call per rollout
            if sim_moves is not None and (best_solution is None or 
                                          len(sim_moves) < len(best_solution)):
                best_solution = sim_moves
                best_states = sim_states
        
        # Backpropagation
        for n in reversed(path):
            n.visits += 1
            n.total_reward += reward
        
        nodes_expanded += 1
    
    solved = best_solution is not None
    
    # Verify solution
    if solved:
        try:
            test_board = board
            for move in best_solution:
                test_board, _ = test_board.apply_move(move)
            solved = test_board.is_solved()
        except:
            solved = False
    
    return SearchResult(
        algorithm="llm_mcts",
        solved=solved,
        moves=best_solution or [],
        states=best_states or [],
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        llm_calls=llm_calls,
        elapsed_seconds=time.time() - t0,
        notes=f"iterations={iterations}, repr={repr_key}"
    )


def _llm_rollout(
    board: SokobanBoard,
    root_board: SokobanBoard,
    predictor: LLMPredictor,
    repr_key: str,
    moves_so_far: List[str],
    states_so_far: List[SokobanState],
    max_depth: int,
    dead_sq: Set,
    h_fn,
) -> Tuple[float, Optional[List[str]], Optional[List[SokobanState]]]:
    """
    LLM-guided rollout for MCTS.
    """
    b = board
    moves = moves_so_far[:]
    states = states_so_far[:]
    steps = 0
    
    best_h = h_fn(b)
    best_boxes = len(b.boxes_on_goals)
    
    while steps < max_depth and not b.is_solved():
        # Get legal pushes
        legal = []
        for lurd, ns, nz, nm in b.get_legal_pushes():
            child = root_board._make_child(ns, nz, nm)
            if not is_deadlock(child, dead_sq):
                legal.append((lurd, ns, nz, nm))
        
        if not legal:
            break
        
        # Try LLM first
        result = predictor.predict_next_move(b, repr_key)
        chosen = None
        
        if result.ok and result.move:
            llm_move = result.move.upper()
            # Find matching legal move
            for move in legal:
                if move[0] == llm_move:
                    chosen = move
                    break
        
        # Fallback to heuristic if LLM fails
        if chosen is None:
            # Choose move that minimizes heuristic
            best_move = None
            best_score = float('inf')
            for lurd, ns, nz, nm in legal:
                child = root_board._make_child(ns, nz, nm)
                score = h_fn(child)
                if score < best_score:
                    best_score = score
                    best_move = (lurd, ns, nz, nm)
            chosen = best_move or random.choice(legal)
        
        lurd, new_state, new_zhash, new_norm = chosen
        b = root_board._make_child(new_state, new_zhash, new_norm)
        moves.append(lurd)
        states.append(new_state)
        steps += 1
        
        # Track progress
        current_boxes = len(b.boxes_on_goals)
        if current_boxes > best_boxes:
            best_boxes = current_boxes
            best_h = h_fn(b)
    
    if b.is_solved():
        return 1.0, moves, states
    
    # Partial reward
    total_boxes = len(b.goals)
    box_reward = best_boxes / total_boxes if total_boxes > 0 else 0
    h_reward = max(0, 1.0 - best_h / (total_boxes * 10 + 1))
    reward = 0.7 * box_reward + 0.3 * h_reward
    
    return reward, None, None