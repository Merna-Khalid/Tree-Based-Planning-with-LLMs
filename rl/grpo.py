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
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class GRPOConfig:
    """Configuration for GRPO training - Aligned with MLX GRPO logic."""
    
    # Model
    model_id: str = "google/gemma-3-4b-it"
    load_in_4bit: bool = True          # save VRAM with 4-bit quantisation
    
    # Output
    output_dir: str = "./checkpoints/sokoban-grpo"
    
    # GRPO hyperparameters (matching MLX GRPO)
    num_generations: int = 8           # completions sampled per prompt (G)
    num_epochs: float = 3.0
    per_device_batch_size: int = 1     # small — each batch has num_generations completions
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    beta: float = 0.04                 # KL penalty — keep model close to reference
    temperature: float = 0.9           # sampling temperature
    epsilon: float = 0.2               # PPO clip (from MLX)
    
    # Sequence lengths
    max_prompt_length: int = 1024      # board repr can be long
    max_completion_length: int = 300   # <think> + <NextMove> + <Confidence> + <Rationale>
    
    # Reward weights (matching MLX compute_reward logic)
    weight_format: float = 0.3
    weight_progress: float = 0.4       # shaped reward for progress
    weight_optimal: float = 0.2        # optimal move match
    weight_move_quality: float = 0.1   # sparse A* improvement reward
    
    # Shaped rewards (from MLX version)
    use_progress_reward: bool = True    # reward_progress for shaped learning
    use_optimal_match: bool = True      # reward_optimal_match if optimal_move available
    
    # Judge LLM (disabled by default to match MLX)
    use_judge: bool = False             # MLX doesn't use judge
    
    # LoRA (memory-efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 8                     # matching MLX lora_rank
    lora_alpha: int = 10                # matching MLX lora_scale
    lora_dropout: float = 0.0           # matching MLX lora_dropout
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Misc
    logging_steps: int = 10
    save_steps: int = 100
    seed: int = 42
    update_every: int = 10              # sync old model (from MLX)


def compute_reward_wrapper(
    completions: List[str],
    board_ascii: List[str],
    optimal_cost: List[int],
    optimal_move: Optional[List[str]] = None,
    config: Optional[GRPOConfig] = None,
) -> List[float]:
    """
    Wrapper that computes rewards using the SAME logic as MLX compute_reward.
    This ensures standard GRPO and MLX GRPO produce identical reward signals.
    """
    if config is None:
        config = GRPOConfig()
    
    rewards = []
    for i, response in enumerate(completions):
        # Format reward (always positive if format is correct)
        format_score = reward_format(
            prompts=[""],
            completions=[response]
        )[0]
        
        reward_total = 0.0
        
        # Progress reward (shaped, gives positive feedback for any progress)
        if config.use_progress_reward:
            from rl.reward import reward_progress
            progress_score = reward_progress(
                prompts=[""],
                completions=[response],
                board_ascii=[board_ascii[i]]
            )[0]
            reward_total += config.weight_progress * progress_score
        
        # Optimal match reward (if we have the optimal move)
        if config.use_optimal_match and optimal_move and optimal_move[i]:
            from rl.reward import reward_optimal_match
            optimal_score = reward_optimal_match(
                prompts=[""],
                completions=[response],
                optimal_move=[optimal_move[i]]
            )[0]
            reward_total += config.weight_optimal * optimal_score
        
        # Move quality (sparse, but still useful)
        move_quality = reward_move_quality(
            prompts=[""],
            completions=[response],
            board_ascii=[board_ascii[i]],
            optimal_cost=[optimal_cost[i]]
        )[0]
        reward_total += config.weight_move_quality * move_quality
        
        # Format reward (always add)
        reward_total += config.weight_format * format_score
        
        # Clamp to same range as MLX version
        reward_total = max(-1.0, min(2.0, reward_total))
        rewards.append(reward_total)
    
    return rewards


def train_grpo(
    dataset,
    config: GRPOConfig,
    eval_dataset=None,
) -> object:
    """
    Set up and run GRPOTrainer with MLX-aligned logic.
    
    Returns the trainer object (already trained).
    Checkpoints are saved to config.output_dir.
    """
    try:
        from trl import GRPOTrainer, GRPOConfig as TRLGRPOConfig
    except ImportError:
        raise ImportError("pip install trl")
    
    try:
        from peft import LoraConfig
    except ImportError:
        raise ImportError("pip install peft")
    
    from rl.reward import reward_format, reward_move_quality
    
    # ---- Create a reward function wrapper that matches MLX logic ----
    def reward_func(prompts: List[str], completions: List[str], **kwargs):
        """Reward function that mirrors MLX compute_reward logic."""
        # Extract board_ascii and optimal_cost from dataset
        # This assumes the dataset has these columns
        board_ascii_list = kwargs.get('board_ascii', [])
        optimal_cost_list = kwargs.get('optimal_cost', [])
        optimal_move_list = kwargs.get('optimal_move', [None] * len(completions))
        
        if not board_ascii_list and len(dataset) > 0:
            # Fallback: try to get from dataset based on prompt matching
            # This is simplified - in practice you'd need proper alignment
            board_ascii_list = [None] * len(completions)
            optimal_cost_list = [0] * len(completions)
        
        return compute_reward_wrapper(
            completions=completions,
            board_ascii=board_ascii_list,
            optimal_cost=optimal_cost_list,
            optimal_move=optimal_move_list,
            config=config
        )
    
    # ---- Reward function list (single combined function to match MLX) ----
    # MLX uses a single compute_reward that combines everything
    # So we do the same here instead of multiple separate reward functions
    reward_funcs = [reward_func]
    reward_weights = [1.0]  # Weight already applied inside compute_reward_wrapper
    
    # ---- TRL GRPOConfig ----
    training_args = TRLGRPOConfig(
        output_dir                 = config.output_dir,
        num_train_epochs           = config.num_epochs,
        per_device_train_batch_size= config.per_device_batch_size,
        gradient_accumulation_steps= config.gradient_accumulation_steps,
        learning_rate              = config.learning_rate,
        beta                       = config.beta,
        temperature                = config.temperature,
        num_generations            = config.num_generations,
        max_prompt_length          = config.max_prompt_length,
        max_completion_length      = config.max_completion_length,
        logging_steps              = config.logging_steps,
        save_steps                 = config.save_steps,
        seed                       = config.seed,
        bf16                       = True,    # use bfloat16 if GPU supports it
        gradient_checkpointing     = True,    # save VRAM
        reward_weights             = reward_weights,
    )
    
    # ---- LoRA config (matching MLX settings) ----
    peft_config = None
    if config.use_lora:
        peft_config = LoraConfig(
            r              = config.lora_r,           # matching MLX lora_rank=8
            lora_alpha     = config.lora_alpha,       # matching MLX lora_scale=10
            lora_dropout   = config.lora_dropout,     # matching MLX lora_dropout=0.0
            target_modules = config.lora_target_modules,
            bias           = "none",
            task_type      = "CAUSAL_LM",
        )
    
    # ---- Model loading kwargs ----
    model_kwargs = {}
    if config.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            import torch
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except ImportError:
            print("  ⚠ bitsandbytes not installed — training without 4-bit quantisation")
    
    # ---- GRPOTrainer ----
    trainer = GRPOTrainer(
        model         = config.model_id,
        args          = training_args,
        train_dataset = dataset,
        eval_dataset  = eval_dataset,
        reward_funcs  = reward_funcs,
        peft_config   = peft_config,
        **model_kwargs,
    )
    
    print(f"\n  Starting GRPO training (MLX-aligned logic)")
    print(f"  Model:        {config.model_id}")
    print(f"  Dataset:      {len(dataset)} examples")
    print(f"  Generations:  {config.num_generations} per prompt")
    print(f"  Epochs:       {config.num_epochs}")
    print(f"  Reward weights: format={config.weight_format}, progress={config.weight_progress}, optimal={config.weight_optimal}, move_quality={config.weight_move_quality}")
    print(f"  Clamp range:  [-1.0, 2.0]")
    print(f"  Output:       {config.output_dir}\n")
    
    trainer.train()
    return trainer


def save_merged_model(trainer, output_dir: str):
    """
    Merge LoRA weights into base model and save.
    Call this after training to get a standalone model.
    """
    print(f"  Merging LoRA weights and saving to {output_dir}...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir)
    trainer.processing_class.save_pretrained(output_dir)
    print(f"  ✓ Saved to {output_dir}")