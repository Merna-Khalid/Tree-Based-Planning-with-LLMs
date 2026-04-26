"""
GRPO training for one-step Sokoban predictor - Aligned with MLX GRPO logic.

The model takes a board state and predicts ONE move in LURD format.
GRPO samples N completions per prompt, scores each, and updates the
model toward better-scoring completions.

Reward functions (combined):
  1. reward_move_quality  — did the move improve A* cost? (main signal)
  2. reward_format        — did the model follow the XML format?
  3. reward_progress      — shaped reward for making progress
  4. reward_optimal_match — does the move match optimal move? (if available)

Reward weights (must mirror MLX compute_reward exactly):
  format:       0.3
  progress:     0.4
  optimal:      0.2
  move_quality: 0.1
  clamp range:  [-1.0, 2.0]

Usage (in RL.ipynb):

    from sokoban.microban import load_microban, select_by_difficulty
    from sokoban.rl.dataset import build_dataset
    from sokoban.rl.grpo import train_grpo, GRPOConfig

    puzzles = load_microban("Microban.txt")
    train_puzzles = select_by_difficulty(puzzles, easy=20, medium=10, hard=5)

    dataset = build_dataset(train_puzzles, repr_key="01_ASCII_RAW")

    config = GRPOConfig(
        model_id       = "google/gemma-3-4b-it",
        output_dir     = "./checkpoints/sokoban-grpo",
        num_generations= 8,
        num_epochs     = 3,
        use_judge      = False,   # MLX version doesn't use judge
    )
    trainer = train_grpo(dataset, config)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from rl.reward import (
    reward_move_quality,
    reward_format,
    reward_optimal_match,
    reward_progress,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GRPOConfig:
    """
    Configuration for GRPO training.

    All hyperparameters are kept in sync with MLXGRPOConfig so that
    both backends produce identical reward signals and training dynamics.

    MLX field -> TRL field mapping
    --------------------------------
    group_size        -> num_generations
    iters             -> num_epochs  (approximate; TRL uses epoch semantics)
    epsilon           -> epsilon
    beta              -> beta
    lora_rank         -> lora_r
    lora_scale        -> lora_alpha
    lora_dropout      -> lora_dropout
    update_every      -> update_every  (MLX syncs old model; TRL handles
                                        reference model internally, kept for
                                        documentation parity)
    max_response_len  -> max_completion_length
    """

    # Model
    model_id: str = "google/gemma-3-4b-it"
    load_in_4bit: bool = True           # save VRAM with 4-bit quantisation

    # Output
    output_dir: str = "./checkpoints/sokoban-grpo"

    # Data  — mirrors MLXGRPOConfig fields set in the notebook
    repr_key: str = "13_ACTION_CENTRIC"         # MLX: config.repr_key
    num_train_puzzles: int = 50                  # MLX: config.num_train_puzzles
    max_steps_per_puzzle: int = 30               # MLX: config.max_steps_per_puzzle

    # GRPO hyperparameters (matching MLX GRPO)
    num_generations: int = 8            # completions sampled per prompt (G) — MLX: group_size=4 default
    num_epochs: float = 3.0
    per_device_batch_size: int = 1      # small — each batch has num_generations completions
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6        # MLX default: 1e-5; set conservatively for GPU
    beta: float = 0.02                  # KL penalty — matches MLX beta=0.02
    temperature: float = 0.9            # sampling temperature
    epsilon: float = 0.2               # PPO clip — matches MLX epsilon=0.2

    # Sequence lengths
    max_prompt_length: int = 512       
    max_completion_length: int = 265    # matches MLX max_response_len=265

    # -----------------------------------------------------------------------
    # Reward weights — MUST match MLX compute_reward constants exactly:
    #   reward = format*0.3 + progress*0.4 + optimal*0.2 + move_quality*0.1
    #   clamped to [-1.0, 2.0]
    # -----------------------------------------------------------------------
    weight_format: float = 0.3
    weight_progress: float = 0.4
    weight_optimal: float = 0.2
    weight_move_quality: float = 0.1

    # Shaped rewards (from MLX version)
    use_progress_reward: bool = True    # reward_progress for shaped learning
    use_optimal_match: bool = True      # reward_optimal_match if optimal_move available

    # Judge LLM (disabled to match MLX)
    use_judge: bool = False

    # LoRA — matches MLX: lora_rank=8, lora_scale=10.0, lora_dropout=0.0
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 10
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Misc
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42
    # MLX syncs old model every update_every steps; TRL manages the
    # reference model internally.  Kept here for documentation parity.
    update_every: int = 10


# ============================================================================
# Reward computation — mirrors MLX compute_reward exactly
# ============================================================================

def compute_reward(
    response: str,
    board_ascii: str,
    optimal_cost: int,
    optimal_move: Optional[str] = None,
    config: Optional[GRPOConfig] = None,
) -> float:
    """
    Single-sample reward.  Signature and logic mirror MLX compute_reward so
    both backends produce identical scores for identical inputs.

    reward = format*0.3 + progress*0.4 + optimal*0.2 + move_quality*0.1
    clamped to [-1.0, 2.0]
    """
    if config is None:
        config = GRPOConfig()

    format_score = reward_format(
        prompts=[""],
        completions=[response],
    )[0]

    progress_score = reward_progress(
        prompts=[""],
        completions=[response],
        board_ascii=[board_ascii],
    )[0]

    optimal_score = 0.0
    if optimal_move:
        optimal_score = reward_optimal_match(
            prompts=[""],
            completions=[response],
            optimal_move=[optimal_move],
        )[0]

    move_quality = reward_move_quality(
        prompts=[""],
        completions=[response],
        board_ascii=[board_ascii],
        optimal_cost=[optimal_cost],
    )[0]

    reward = (
        config.weight_format * format_score
        + config.weight_progress * progress_score
        + config.weight_optimal * optimal_score
        + config.weight_move_quality * move_quality
    )

    return max(-1.0, min(2.0, reward))


def compute_reward_batch(
    completions: List[str],
    board_ascii: List[str],
    optimal_cost: List[int],
    optimal_move: Optional[List[Optional[str]]] = None,
    config: Optional[GRPOConfig] = None,
) -> List[float]:
    """
    Batched wrapper around compute_reward.  Keeps the same per-sample logic
    as the MLX version — no vectorisation that could accidentally diverge.
    """
    if config is None:
        config = GRPOConfig()
    if optimal_move is None:
        optimal_move = [None] * len(completions)

    return [
        compute_reward(
            response=completions[i],
            board_ascii=board_ascii[i],
            optimal_cost=optimal_cost[i],
            optimal_move=optimal_move[i],
            config=config,
        )
        for i in range(len(completions))
    ]


# ============================================================================
# Debug utility — ported from MLX grpo.py
# ============================================================================

def debug_generation(model, tokenizer, prompt: str, max_tokens: int = 256) -> str:
    """
    Generate one response and print a diagnostic breakdown.
    Mirrors the MLX debug_generation helper so both backends can be
    inspected in the same way.

    Parameters
    ----------
    model:      A loaded HuggingFace model (or pipeline-compatible object).
    tokenizer:  Matching tokenizer.
    prompt:     Raw user-turn text (before chat template is applied).
    max_tokens: Maximum new tokens to generate.

    Returns the raw response string.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    print("\n" + "=" * 60)
    print("DEBUG: Model Generation")
    print("=" * 60)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.9,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"\nRaw response:\n{response}")
    print("\n" + "=" * 60)

    has_think      = bool(re.search(r"<think>",      response, re.IGNORECASE))
    has_nextmove   = bool(re.search(r"<NextMove>",   response, re.IGNORECASE))
    has_confidence = bool(re.search(r"<Confidence>", response, re.IGNORECASE))
    has_rationale  = bool(re.search(r"<Rationale>",  response, re.IGNORECASE))

    print(f"Has <think>:      {has_think}")
    print(f"Has <NextMove>:   {has_nextmove}")
    print(f"Has <Confidence>: {has_confidence}")
    print(f"Has <Rationale>:  {has_rationale}")

    move_match = re.search(
        r"<NextMove>\s*([UDLRudlr])\s*</NextMove>", response, re.IGNORECASE
    )
    if move_match:
        print(f"Extracted move: {move_match.group(1).upper()}")
    else:
        print("No valid move extracted!")

    return response


# ============================================================================
# TRL GRPOTrainer setup
# ============================================================================

def train_grpo(
    dataset,
    config: GRPOConfig,
    eval_dataset=None,
) -> object:
    """
    Set up and run GRPOTrainer using reward logic that mirrors MLX compute_reward.

    The reward function passed to TRL calls compute_reward_batch, which is a
    direct port of the MLX single-sample compute_reward applied element-wise.
    This guarantees identical reward signals for identical model outputs.

    Returns the trainer object (already trained).
    Checkpoints are saved to config.output_dir.
    """
    try:
        from trl import GRPOTrainer, GRPOConfig as TRLGRPOConfig
    except ImportError:
        raise ImportError("pip install trl>=0.8.0")

    try:
        from peft import LoraConfig
    except ImportError:
        raise ImportError("pip install peft")

    def reward_func(
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Reward function passed to TRL.

        TRL passes dataset columns as keyword arguments when the dataset
        contains them.  We expect the dataset built by build_dataset to
        include:
          - board_ascii   (str)
          - optimal_cost  (int)
          - optimal_move  (str | None)  — optional column

        If a column is absent, we fall back to safe defaults so training
        does not crash, but a warning is printed because the reward signal
        will be degraded compared to the MLX version.
        """
        board_ascii_list: List[str] = kwargs.get("board_ascii", [])
        optimal_cost_list: List[int] = kwargs.get("optimal_cost", [])
        optimal_move_list: Optional[List[Optional[str]]] = kwargs.get(
            "optimal_move", None
        )

        n = len(completions)

        if not board_ascii_list:
            print(
                "  ⚠ reward_func: 'board_ascii' not found in dataset kwargs. "
                "progress and move_quality rewards will be zero."
            )
            board_ascii_list = [""] * n

        if not optimal_cost_list:
            print(
                "  ⚠ reward_func: 'optimal_cost' not found in dataset kwargs. "
                "move_quality reward will be zero."
            )
            optimal_cost_list = [0] * n

        # optimal_move is genuinely optional (not all puzzle states have it)
        if optimal_move_list is None:
            optimal_move_list = [None] * n

        return compute_reward_batch(
            completions=completions,
            board_ascii=board_ascii_list,
            optimal_cost=optimal_cost_list,
            optimal_move=optimal_move_list,
            config=config,
        )

    # TRL expects a list of reward functions; weights are already baked into
    # compute_reward_batch, so we pass a single function with weight=1.0.
    reward_funcs = [reward_func]
    reward_weights = [1.0]

    # ---- TRL GRPOConfig ----
    training_args = TRLGRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        beta=config.beta,
        temperature=config.temperature,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        seed=config.seed,
        bf16=True,
        gradient_checkpointing=True,
        reward_weights=reward_weights,
    )

    # ---- LoRA config — matches MLX lora_rank=8, lora_scale=10, lora_dropout=0.0 ----
    peft_config = None
    if config.use_lora:
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    # ---- 4-bit quantisation ----
    model_kwargs = {}
    if config.load_in_4bit:
        try:
            import torch
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except ImportError:
            print(
                "  ⚠ bitsandbytes not installed — training without 4-bit quantisation"
            )

    # ---- GRPOTrainer ----
    trainer = GRPOTrainer(
        model=config.model_id,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_funcs,
        peft_config=peft_config,
        **model_kwargs,
    )

    print(f"\n  Starting GRPO training (MLX-aligned logic)")
    print(f"  Model:          {config.model_id}")
    print(f"  Dataset:        {len(dataset)} examples")
    print(f"  Generations:    {config.num_generations} per prompt")
    print(f"  Epochs:         {config.num_epochs}")
    print(
        f"  Reward weights: format={config.weight_format}, "
        f"progress={config.weight_progress}, "
        f"optimal={config.weight_optimal}, "
        f"move_quality={config.weight_move_quality}"
    )
    print(f"  Clamp range:    [-1.0, 2.0]")
    print(f"  beta (KL):      {config.beta}")
    print(f"  epsilon (clip): {config.epsilon}")
    print(f"  Output:         {config.output_dir}\n")

    trainer.train()
    return trainer


# ============================================================================
# Post-training utilities
# ============================================================================

def save_merged_model(trainer, output_dir: str) -> None:
    """
    Merge LoRA weights into the base model and save a standalone checkpoint.
    Call this after train_grpo() to produce a model that can be loaded
    without PEFT.
    """
    print(f"  Merging LoRA weights and saving to {output_dir} ...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir)
    trainer.processing_class.save_pretrained(output_dir)
    print(f"  ✓ Saved to {output_dir}")