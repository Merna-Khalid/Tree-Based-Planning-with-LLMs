from rl.rldataset import build_dataset, build_dataset_multi_repr
from rl.reward import (
    reward_format,
    reward_move_quality,
    reward_reasoning_async,
    reward_progress,
    reward_optimal_match,
)
from rl.grpo import GRPOConfig, train_grpo, save_merged_model
from rl.rl_evaluate import quick_evaluation, compare_base_vs_rl

try:
    from rl.mlx_grpo import MLXGRPOConfig, setup_models, grpo_train_loop, debug_generation
except ImportError:
    MLXGRPOConfig = None
    setup_models = None
    grpo_train_loop = None
    debug_generation = None

__all__ = [
    "build_dataset",
    "build_dataset_multi_repr",
    "reward_format",
    "reward_move_quality",
    "reward_reasoning_async",
    "reward_progress",
    "reward_optimal_match",
    "GRPOConfig",
    "train_grpo",
    "save_merged_model",
    "quick_evaluation",
    "compare_base_vs_rl",
    # MLX — None on non-Apple platforms
    "MLXGRPOConfig",
    "setup_models",
    "grpo_train_loop",
    "debug_generation",
]