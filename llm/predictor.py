"""
LLM predictor for Sokoban. Everything is LURD — no NSEW anywhere.

LURD encoding:
    u/d/l/r  — player walks, no box pushed
    U/D/L/R  — player walks AND pushes a box

Two prediction modes:
    predict_next_move(board, repr_key)
          <NextMove>U</NextMove>
          <Confidence>0.87</Confidence>
          <Rationale>Pushing up moves the box toward the goal.</Rationale>

    predict_full_solution(board, repr_key)
          <Solution>dlURRr...</Solution>
          <Rationale>Push both boxes via the top corridor.</Rationale>

Backends:
    TransformersBackend — local HuggingFace model (GPU or CPU)
    LlamaCppBackend — GGUF models with Metal acceleration
    MLXBackend — Apple Silicon native (fastest)
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.board import SokobanBoard

# ---------------------------------------------------------------------------
# LURD constants
# ---------------------------------------------------------------------------

PUSH_MOVES = ["U", "D", "L", "R"]      # box-push moves (uppercase)
WALK_MOVES = ["u", "d", "l", "r"]      # walk-only moves (lowercase)
ALL_MOVES  = PUSH_MOVES + WALK_MOVES   # all 8 LURD chars

_LURD_CHARS = set(ALL_MOVES)

LEGEND = (
    "# = wall | @ = player | $ = box | . = goal "
    "| * = box-on-goal | + = player-on-goal | (space) = floor"
)

RULES = (
    "Rules:\n"
    "- Push boxes onto ALL goal squares to win.\n"
    "- You PUSH boxes by walking into them — you cannot pull.\n"
    "- A box cannot be pushed into a wall or another box.\n"
    "- Move encoding (LURD):\n"
    "    Uppercase = push a box:  U=push-up  D=push-down  L=push-left  R=push-right\n"
    "    Lowercase = walk only:   u=walk-up  d=walk-down  l=walk-left  r=walk-right"
)


# ---------------------------------------------------------------------------
# Core board block function - ALWAYS includes ASCII visual
# ---------------------------------------------------------------------------

def _board_block(board: SokobanBoard, repr_key: str) -> str:
    """
    Get board representation, ALWAYS including ASCII visual as base.
    This ensures the model can SEE the board layout regardless of repr_key.
    """
    from representations import get_repr
    
    # Always get ASCII visual first (the ground truth)
    ascii_visual = get_repr("01_ASCII_RAW")(board)
    
    # If repr_key is ASCII, just return it
    if repr_key == "01_ASCII_RAW":
        return ascii_visual
    
    # Otherwise, get the specialized representation AND include ASCII
    specialized = get_repr(repr_key)(board)
    
    return f"""VISUAL BOARD (ground truth):
{ascii_visual}

SPECIALIZED VIEW ({repr_key}):
{specialized}"""


# ---------------------------------------------------------------------------
# Prompt builders - ALL include ASCII visual
# ---------------------------------------------------------------------------

def build_next_move_prompt(board: SokobanBoard, repr_key: str) -> str:
    unsolved_count = len(board.unsolved_boxes)
    unsolved_positions = sorted(board.unsolved_boxes)
    solved_boxes = len(board.boxes_on_goals)
    solved_warning = ""
    if solved_boxes > 0:
        solved_warning = f"\n   ⚠️ There are {solved_boxes} box(es) ALREADY ON GOALS. IGNORE THEM. Focus only on UNSOLVED boxes.\n"
    
    return (
        "You are an expert Sokoban solver.\n"
        f"Legend: {LEGEND}\n"
        f"{RULES}\n\n"
        f"IMPORTANT: There are {unsolved_count} UNSOLVED boxes at {unsolved_positions}.{solved_warning}\n\n"
        f"Current board:\n"
        f"{_board_block(board, repr_key)}\n\n"
        "Think step by step inside <think> tags, then output the single best next move.\n"
        "Use LURD encoding: uppercase = push a box, lowercase = walk only.\n"
        "Reply using this format:\n"
        "<NextMove>U</NextMove>\n"
        "<Confidence>0.95</Confidence>\n"
        "<Rationale>One sentence explaining why.</Rationale>"
    )


def build_full_solution_prompt(board: SokobanBoard, repr_key: str) -> str:
    return (
        "You are an expert Sokoban solver.\n"
        f"Legend: {LEGEND}\n"
        f"{RULES}\n\n"
        f"Current board:\n"
        f"{_board_block(board, repr_key)}\n\n"
        "Use LURD encoding: uppercase = push a box, lowercase = walk only.\n"
        "Reply using this format:\n"
        "<Solution>dlURRr...</Solution>\n"
        "<Rationale>One sentence describing the strategy.</Rationale>"
    )


def build_next_move_with_feedback_prompt(
    board: SokobanBoard,
    repr_key: str,
    rejected: list,   # list of (move, reason)
) -> str:
    """
    Build prompt with feedback about failed moves.
    ALWAYS includes ASCII visual.
    """
    board_text = _board_block(board, repr_key)

    failed_line = ""
    if rejected:
        failed_moves = ", ".join(m.upper() for m, _ in rejected)
        failed_line = f"\n❌ The following moves FAILED: {failed_moves}. Choose a different move.\n"

    unsolved_count = len(board.unsolved_boxes)
    unsolved_positions = sorted(board.unsolved_boxes)
    solved_boxes = len(board.boxes_on_goals)
    solved_warning = ""
    if solved_boxes > 0:
        solved_warning = f"\n   ⚠️ There are {solved_boxes} box(es) ALREADY ON GOALS. IGNORE THEM.\n"

    return (
        "You are an expert Sokoban solver.\n"
        f"Legend: {LEGEND}\n"
        f"{RULES}\n\n"
        f"IMPORTANT: There are {unsolved_count} UNSOLVED boxes at {unsolved_positions}.{solved_warning}\n"
        f"{failed_line}\n"
        f"Current board:\n"
        f"{board_text}\n\n"
        "Use LURD encoding: uppercase = push a box, lowercase = walk only.\n"
        "Reply using this format:\n"
        "<NextMove>U</NextMove>\n"
        "<Confidence>0.95</Confidence>\n"
        "<Rationale>One sentence explaining why.</Rationale>"
    )


def build_next_move_with_history_prompt(
    board: SokobanBoard, 
    repr_key: str, 
    history: List[str],
    rejected: List[Tuple[str, str]] = None,
    max_history: int = 5,
) -> str:
    """
    Build prompt with move history and rejected moves.
    ALWAYS includes ASCII visual.
    """
    from representations import get_repr
    
    # Always include ASCII visual
    ascii_visual = get_repr("01_ASCII_RAW")(board)
    
    # Get specialized view if not ASCII
    if repr_key == "01_ASCII_RAW":
        board_text = ascii_visual
    else:
        specialized = get_repr(repr_key)(board)
        board_text = f"""VISUAL BOARD:
{ascii_visual}

{repr_key} VIEW:
{specialized}"""
    
    # Build history string
    recent_moves = history[-max_history:] if history else []
    history_str = ""
    if recent_moves:
        history_str = f"\nRecent moves: {' → '.join(recent_moves)}\n"
        if len(recent_moves) >= 4:
            last_four = recent_moves[-4:]
            if last_four[0] == last_four[2] and last_four[1] == last_four[3]:
                history_str += "⚠️ WARNING: You are oscillating! Try something different.\n"
    
    # Build rejected moves string
    rejected_str = ""
    if rejected:
        failed_moves = ", ".join(m for m, _ in rejected)
        rejected_str = f"\n❌ The following moves FAILED: {failed_moves}. DO NOT try them again.\n"
    
    unsolved_count = len(board.unsolved_boxes)
    unsolved_positions = sorted(board.unsolved_boxes)
    solved_boxes = len(board.boxes_on_goals)
    solved_warning = ""
    if solved_boxes > 0:
        solved_warning = f" (⚠️ {solved_boxes} boxes already on goals - IGNORE them)"
    
    return (
        "You are an expert Sokoban solver.\n"
        f"Legend: {LEGEND}\n"
        f"{RULES}\n\n"
        f"IMPORTANT: {unsolved_count} UNSOLVED boxes at {unsolved_positions}{solved_warning}.\n"
        f"{history_str}{rejected_str}\n"
        f"Current board:\n{board_text}\n\n"
        "Choose the next move (U/D/L/R for pushes, u/d/l/r for walks).\n"
        "Focus ONLY on unsolved boxes.\n"
        "Reply:\n"
        "<NextMove>X</NextMove>\n"
        "<Confidence>0.95</Confidence>\n"
        "<Rationale>Why...</Rationale>"
    )


# ---------------------------------------------------------------------------
# XML tag extractor
# ---------------------------------------------------------------------------

def _tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_next_move_response(text: str) -> "NextMoveResult":
    think     = _tag(text, "think") or ""
    raw_move  = _tag(text, "NextMove") or _tag(text, "Move")
    raw_conf  = _tag(text, "Confidence")
    rationale = _tag(text, "Rationale") or ""

    move: Optional[str] = None
    if raw_move:
        ch = raw_move.strip()[:1]
        move = ch if ch in _LURD_CHARS else None

    # Fallback: last bare LURD char in the text (preserve case)
    if move is None:
        letters = re.findall(r"\b([udlrUDLR])\b", text)
        if letters:
            move = letters[-1]

    confidence = 0.5
    if raw_conf:
        try:
            confidence = max(0.0, min(1.0, float(raw_conf)))
        except ValueError:
            pass

    return NextMoveResult(
        move=move,
        confidence=confidence,
        think=think,
        rationale=rationale,
        raw_response=text,
    )


def parse_full_solution_response(text: str) -> "FullSolutionResult":
    think        = _tag(text, "think") or ""
    solution_raw = _tag(text, "Solution") or ""
    rationale    = _tag(text, "Rationale") or ""
    # Keep only valid LURD chars, preserve case
    solution = re.sub(r"[^udlrUDLR]", "", solution_raw)

    return FullSolutionResult(
        solution=solution,
        think=think,
        rationale=rationale,
        raw_response=text,
    )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NextMoveResult:
    move: Optional[str]   # LURD char: u/d/l/r/U/D/L/R, or None on failure
    confidence: float     # 0.0–1.0
    think: str            # content of <think>...</think>
    rationale: str        # content of <Rationale>...</Rationale>
    raw_response: str
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.move is not None and self.error is None

    @property
    def is_push(self) -> bool:
        """True if the predicted move is a box push (uppercase)."""
        return self.move is not None and self.move.isupper()

    def __repr__(self):
        kind = "push" if self.is_push else "walk"
        return (f"NextMoveResult(move={self.move!r} ({kind}), "
                f"conf={self.confidence:.2f}, ok={self.ok})")


@dataclass
class FullSolutionResult:
    solution: str     # LURD string e.g. "dlURRr..."
    think: str
    rationale: str
    raw_response: str
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.solution) and self.error is None

    @property
    def push_count(self) -> int:
        return sum(1 for c in self.solution if c.isupper())

    @property
    def walk_count(self) -> int:
        return sum(1 for c in self.solution if c.islower())

    def verify(self, board: "SokobanBoard") -> bool:
        """Play the solution on the board and check it solves it."""
        try:
            b = board
            for ch in self.solution:
                b, _ = b.apply_move(ch)
            return b.is_solved()
        except Exception:
            return False

    def __repr__(self):
        return (f"FullSolutionResult(solution={self.solution!r}, "
                f"len={len(self.solution)}, pushes={self.push_count}, ok={self.ok})")


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class BaseBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 400) -> Tuple[str, float]:
        """Return (generated_text, latency_ms)."""


# ---------------------------------------------------------------------------
# Transformers backend
# ---------------------------------------------------------------------------

SUPPORTED_MODELS: Dict[str, dict] = {
    "google/gemma-3-4b":                {"chat": True, "max_new_tokens": 512},
    "LiquidAI/LFM2-1.2B":               {"chat": True, "max_new_tokens": 512},
    "mistralai/ministral-3b-instruct":  {"chat": True, "max_new_tokens": 512},
    "nvidia/Nemotron-Mini-4B-Instruct":  {"chat": True, "max_new_tokens": 512, "manual_tokenizer": True},
}


class TransformersBackend(BaseBackend):
    """
    Local HuggingFace inference.
    Install: pip install transformers accelerate torch
    Optional quantisation: pip install bitsandbytes
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        greedy: bool = True,
    ):
        self.model_id      = model_id
        self._device       = device
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._torch_dtype  = torch_dtype
        self._trust        = trust_remote_code
        self._greedy       = greedy
        self._pipe         = None

        meta = SUPPORTED_MODELS.get(model_id, {})
        self._use_chat         = meta.get("chat", True)
        self._max_new          = meta.get("max_new_tokens", 512)
        self._manual_tokenizer = meta.get("manual_tokenizer", False)

    def _load(self):
        try:
            from transformers import pipeline
            import torch
        except ImportError as e:
            raise ImportError("pip install transformers accelerate torch") from e

        dtype_map = {"auto": "auto", "float16": __import__("torch").float16,
                     "bfloat16": __import__("torch").bfloat16,
                     "float32": __import__("torch").float32}
        device_map = self._device or ("auto" if _has_gpu() else "cpu")
        kw = dict(model=self.model_id, device_map=device_map,
                  torch_dtype=dtype_map.get(self._torch_dtype, "auto"),
                  trust_remote_code=self._trust)
        if self._load_in_8bit: kw["load_in_8bit"] = True
        if self._load_in_4bit: kw["load_in_4bit"] = True
        self._pipe = pipeline("text-generation", **kw)

        # Some models (e.g. Nemotron) require the tokenizer to be assigned manually
        if self._manual_tokenizer:
            from transformers import AutoTokenizer
            self._pipe.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def generate(self, prompt: str, max_new_tokens: int = 0) -> Tuple[str, float]:
        if self._pipe is None:
            self._load()
        n  = max_new_tokens or self._max_new
        kw = {"max_new_tokens": n}
        if self._greedy:
            kw.update(do_sample=False, temperature=None, top_p=None)
        else:
            kw.update(do_sample=True, temperature=0.7)
        t0 = time.time()
        if self._use_chat:
            out  = self._pipe([{"role": "user", "content": prompt}], **kw)
            gen  = out[0]["generated_text"]
            text = gen[-1].get("content", "") if isinstance(gen, list) else str(gen)
        else:
            out  = self._pipe(prompt, return_full_text=False, **kw)
            text = out[0]["generated_text"]
        return text, (time.time() - t0) * 1000


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LlamaCpp backend
# ---------------------------------------------------------------------------

class LlamaCppBackend(BaseBackend):
    """
    Local GGUF model inference using llama-cpp-python.
    Optimized for macOS with Metal GPU acceleration.
    """
    
    def __init__(
        self,
        model_path: str,
        model_family: str = "auto",
        n_ctx: int = 2048,
        n_threads: int = 4,
        n_gpu_layers: int = -1,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        verbose: bool = False,
        use_metal: bool = True,
    ):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("pip install llama-cpp-python")
        
        if use_metal and has_metal():
            print("✓ Metal GPU acceleration enabled")
            if n_gpu_layers == -1:
                n_gpu_layers = 999
        else:
            if n_gpu_layers == -1:
                n_gpu_layers = 0
            print(f"Running on CPU with {n_threads} threads")
        
        self.verbose = verbose
        self.temperature = temperature
        self.model_family = model_family
        self.model_path = model_path
        
        if self.verbose:
            print(f"Loading GGUF model from: {model_path}")
        
        t0 = time.time()
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            verbose=verbose,
        )
        self._load_time_ms = (time.time() - t0) * 1000
        
        if self.verbose:
            print(f"Model loaded in {self._load_time_ms:.1f}ms")

    def _apply_chat_template(self, prompt: str) -> str:
        if self.model_family == "nemotron":
            return f"""<|im_start|>system
You are an expert Sokoban solver.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
        elif self.model_family == "gemma":
            return f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif self.model_family == "ministral":
            return f"<s>[INST] {prompt} [/INST]"
        else:
            if "nemotron" in self.model_path.lower():
                return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            return prompt

    def generate(self, prompt: str, max_new_tokens: int = 512) -> Tuple[str, float]:
        t0 = time.time()
        formatted_prompt = self._apply_chat_template(prompt)
        
        if self.temperature < 0.1:
            response = self.llm(
                formatted_prompt,
                max_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=0.95,
                echo=False,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
        else:
            response = self.llm(
                formatted_prompt,
                max_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=0.95,
                echo=False,
            )
        
        text = response['choices'][0]['text'].strip()
        latency_ms = (time.time() - t0) * 1000
        
        if self.verbose:
            print(f"Generated {len(text)} chars in {latency_ms:.1f}ms")
        
        return text, latency_ms


def has_metal() -> bool:
    try:
        import platform
        import subprocess
        if platform.processor() == "arm":
            result = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True)
            if "Mac" in result.stdout:
                return True
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# MLX backend (Apple Silicon native)
# ---------------------------------------------------------------------------

class MLXBackend(BaseBackend):
    """
    MLX backend for Apple Silicon — fastest option for Mac.
    """
    
    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: int = 42,
        verbose: bool = False,
    ):
        try:
            from mlx_lm import load, generate
        except ImportError:
            raise ImportError("pip install mlx-lm")
        
        self.verbose = verbose
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.seed = seed
        
        if self.verbose:
            print(f"Loading MLX model from: {model_path}")
            if adapter_path:
                print(f"  With LoRA adapter: {adapter_path}")
        
        t0 = time.time()
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self._load_time_ms = (time.time() - t0) * 1000
        
        if self.verbose:
            print(f"Model loaded in {self._load_time_ms:.1f}ms")
    
    def generate(self, prompt: str, max_new_tokens: int = 0) -> Tuple[str, float]:
        from mlx_lm import generate
        
        n = max_new_tokens or self.max_tokens
        
        t0 = time.time()
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=n,
            verbose=self.verbose,
        )
        latency_ms = (time.time() - t0) * 1000
        
        if self.verbose:
            print(f"Generated {len(response)} chars in {latency_ms:.1f}ms")
        
        return response, latency_ms


# ---------------------------------------------------------------------------
# LLMPredictor
# ---------------------------------------------------------------------------

class LLMPredictor:
    """
    Unified Sokoban LLM predictor. Everything is LURD.
    """

    def __init__(self, backend: Optional[BaseBackend] = None):
        self.backend = backend or TransformersBackend("google/gemma-3-4b-it")
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def predict_next_move(
        self,
        board: SokobanBoard,
        repr_key: str = "01_ASCII_RAW",
    ) -> NextMoveResult:
        prompt = build_next_move_prompt(board, repr_key)
        text, latency, error = self._call(prompt, max_new_tokens=512)
        if error:
            return NextMoveResult(
                move=None, confidence=0.0, think="", rationale="",
                raw_response="", latency_ms=latency, error=error,
            )
        result = parse_next_move_response(text)
        result.latency_ms = latency
        return result

    def predict_full_solution(
        self,
        board: SokobanBoard,
        repr_key: str = "01_ASCII_RAW",
    ) -> FullSolutionResult:
        prompt = build_full_solution_prompt(board, repr_key)
        text, latency, error = self._call(prompt, max_new_tokens=1024)
        if error:
            return FullSolutionResult(
                solution="", think="", rationale="",
                raw_response="", latency_ms=latency, error=error,
            )
        result = parse_full_solution_response(text)
        result.latency_ms = latency
        return result

    def predict_next_move_with_feedback(
        self,
        board: SokobanBoard,
        repr_key: str = "01_ASCII_RAW",
        rejected: list = None,
    ) -> NextMoveResult:
        prompt = build_next_move_with_feedback_prompt(board, repr_key, rejected or [])
        text, latency, error = self._call(prompt, max_new_tokens=512)
        if error:
            return NextMoveResult(
                move=None, confidence=0.0, think="", rationale="",
                raw_response="", latency_ms=latency, error=error,
            )
        result = parse_next_move_response(text)
        result.latency_ms = latency
        return result

    def get_move_probs(
        self,
        board: SokobanBoard,
        repr_key: str = "01_ASCII_RAW",
    ) -> Dict[str, float]:
        r = self.predict_next_move(board, repr_key)
        if not r.ok or r.move is None:
            return {m: 1/8 for m in ALL_MOVES}
        probs = {m: (1 - r.confidence) / 7 for m in ALL_MOVES}
        probs[r.move] = r.confidence
        return probs

    def _call(self, prompt: str, max_new_tokens: int) -> Tuple[str, float, Optional[str]]:
        self._call_count += 1
        try:
            text, latency = self.backend.generate(prompt, max_new_tokens=max_new_tokens)
            return text, latency, None
        except Exception as exc:
            return "", 0.0, str(exc)


# ---------------------------------------------------------------------------
# History-aware predictor
# ---------------------------------------------------------------------------

class HistoryAwareLLMPredictor(LLMPredictor):
    """
    LLM Predictor that maintains and uses move history.
    """
    
    def __init__(self, backend: Optional[BaseBackend] = None):
        super().__init__(backend)
        self.history: List[str] = []
        self.max_history = 10
    
    def reset_history(self):
        self.history = []
    
    def predict_next_move(
        self,
        board: SokobanBoard,
        repr_key: str = "01_ASCII_RAW",
        use_history: bool = True,
    ) -> NextMoveResult:
        if use_history and self.history:
            prompt = build_next_move_with_history_prompt(
                board, repr_key, self.history, max_history=self.max_history
            )
        else:
            prompt = build_next_move_prompt(board, repr_key)
        
        text, latency, error = self._call(prompt, max_new_tokens=512)
        
        if error:
            return NextMoveResult(
                move=None, confidence=0.0, think="", rationale="",
                raw_response="", latency_ms=latency, error=error,
            )
        
        result = parse_next_move_response(text)
        result.latency_ms = latency
        
        if result.ok and result.move:
            self.history.append(result.move)
        
        return result
    
    def predict_next_move_with_feedback(
        self,
        board: SokobanBoard,
        repr_key: str = "01_ASCII_RAW",
        rejected: list = None,
        use_history: bool = True,
    ) -> NextMoveResult:
        prompt = build_next_move_with_history_prompt(
            board, repr_key, self.history, rejected, self.max_history
        )
        
        text, latency, error = self._call(prompt, max_new_tokens=512)
        
        if error:
            return NextMoveResult(
                move=None, confidence=0.0, think="", rationale="",
                raw_response="", latency_ms=latency, error=error,
            )
        
        result = parse_next_move_response(text)
        result.latency_ms = latency
        
        if result.ok and result.move:
            self.history.append(result.move)
        
        return result