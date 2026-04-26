"""
MLX GRPO Training for Sokoban
Based on: https://github.com/searlion/mlx-finetuning/blob/main/MLX%20LM%20GRPO.ipynb


"""

import json
import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load, generate
from mlx_lm.tuner import linear_to_lora_layers
import tqdm
import time
import os
from pathlib import Path

from rl.reward import reward_move_quality, reward_format, reward_optimal_match, reward_progress

os.environ["TOKENIZERS_PARALLELISM"] = "true"


# ============================================================================
# Configuration
# ============================================================================

class MLXGRPOConfig:
    def __init__(self):
        # Model
        self.model_path = "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit"
        
        # Data (uses your build_dataset)
        self.repr_key = "13_ACTION_CENTRIC"
        self.max_steps_per_puzzle = 30
        self.num_train_puzzles = 30
        
        # GRPO from notebook
        self.learning_rate = 1e-5
        self.output_dir = './rl-output'
        self.iters = 25
        self.group_size = 4      # G responses per prompt
        self.batch_size = 2
        self.epsilon = 0.2       # PPO clip
        self.beta = 0.02         # KL penalty
        self.update_every = 10
        self.max_response_len = 265
        self.lora_dropout = 0.0
        
        # LoRA
        self.lora_layers = 8
        self.lora_rank = 8
        self.lora_scale = 10.0
        
        # Output
        self.adapter_path = Path("adapters")



# ============================================================================
# GRPO Helper Functions (from notebook)
# ============================================================================

def pad_sequences(sequences, pad_token_id):
    if not sequences:
        return mx.array([])
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padding = mx.array([pad_token_id] * (max_len - len(seq)))
            padded.append(mx.concatenate([seq, padding]))
        else:
            padded.append(seq)
    return mx.stack(padded)


def calculate_log_probs(model, sequences, a_toks):
    """Calculate log probabilities of answer tokens."""
    logits = model(sequences)
    log_probs_full = nn.log_softmax(logits, axis=-1)
    
    batch_size, seq_len = sequences.shape
    _, ans_len = a_toks.shape
    start_pos = seq_len - ans_len
    
    answer_log_probs = log_probs_full[:, start_pos:start_pos+ans_len, :]
    indices = a_toks[:, :, None]
    selected = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(-1)
    return mx.sum(selected, axis=-1)


def grpo_loss_fn(model, model_ref, sequences, a_toks, advantages, old_log_probs, beta, epsilon):
    """GRPO loss with PPO-clip + KL penalty."""
    log_probs = calculate_log_probs(model, sequences, a_toks)
    log_probs_ref = calculate_log_probs(model_ref, sequences, a_toks)
    
    # PPO-clip
    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages)
    
    # KL penalty: r - log(r) - 1 where r = π_ref / π_θ
    log_ratio_kl = log_probs_ref - log_probs
    ratio_kl = mx.exp(log_ratio_kl)
    kl_div = ratio_kl - log_ratio_kl - 1
    
    loss = -mx.mean(policy_reward - beta * kl_div)
    return loss, mx.mean(policy_reward), mx.mean(kl_div)


# ============================================================================
# Reward Wrapper 
# ============================================================================

# def compute_reward(response: str, board_ascii: str, optimal_cost: int) -> float:
#     """Wrapper for your existing reward functions."""
#     move_quality = reward_move_quality(
#         prompts=[""],  # Not used
#         completions=[response],
#         board_ascii=[board_ascii],
#         optimal_cost=[optimal_cost]
#     )[0]
    
#     format_score = reward_format(
#         prompts=[""],
#         completions=[response]
#     )[0]
    
#     return move_quality + 0.3 * format_score

def compute_reward(response: str, board_ascii: str, optimal_cost: int, optimal_move: str = None) -> float:
    """Compute reward using shaped rewards for better learning."""
    
    # Format reward (always positive if format is correct)
    format_score = reward_format(
        prompts=[""],
        completions=[response]
    )[0]
    
    # Progress reward (shaped, gives positive feedback for any progress)
    progress_score = reward_progress(
        prompts=[""],
        completions=[response],
        board_ascii=[board_ascii]
    )[0]
    
    # Optimal match reward (if we have the optimal move)
    optimal_score = 0.0
    if optimal_move:
        optimal_score = reward_optimal_match(
            prompts=[""],
            completions=[response],
            optimal_move=[optimal_move]
        )[0]
    
    # Move quality (sparse, but still useful)
    move_quality = reward_move_quality(
        prompts=[""],
        completions=[response],
        board_ascii=[board_ascii],
        optimal_cost=[optimal_cost]
    )[0]
    
    # Combine: shaped rewards for learning, sparse for final tuning
    # This ensures the model always gets SOME positive signal
    reward = (
        format_score * 0.3 +
        progress_score * 0.4 +
        optimal_score * 0.2 +
        move_quality * 0.1
    )
    
    return max(-1.0, min(2.0, reward))  # Clamp


# ============================================================================
# Setup Models (LoRA)
# ============================================================================

def setup_models(config):
    """Load model, add LoRA, create old and ref copies."""
    print(f"\nLoading model: {config.model_path}")
    model, tokenizer = load(config.model_path)

     # CRITICAL: Set pad token if not set (common for Gemma/Nemotron)
    if tokenizer.pad_token_id is None:
        # Use eos_token as pad_token if available, otherwise use a safe default
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        print(f"  Set pad_token_id to {tokenizer.pad_token_id}")
    
    # Reference model (frozen)
    model_ref, _ = load(config.model_path)
    model_ref.freeze()
    
    # Save LoRA config
    config.adapter_path.mkdir(parents=True, exist_ok=True)
    with open(config.adapter_path / "adapter_config.json", "w") as f:
        json.dump({
            "num_layers": config.lora_layers,
            "lora_parameters": {"rank": config.lora_rank, "scale": config.lora_scale, "dropout": getattr(config, 'lora_dropout', 0.0)}
        }, f, indent=4)
    
    # Add LoRA to trainable model
    model.freeze()
    linear_to_lora_layers(model, config.lora_layers, {"rank": config.lora_rank, "scale": config.lora_scale, "dropout": getattr(config, 'lora_dropout', 0.0)})
    
    # Old model for rollouts
    model_old, _ = load(config.model_path)
    linear_to_lora_layers(model_old, config.lora_layers, {"rank": config.lora_rank, "scale": config.lora_scale, "dropout": getattr(config, 'lora_dropout', 0.0)})
    model_old.update(model.parameters())
    model_old.freeze()
    
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable params: {trainable:,}")
    
    return model, model_old, model_ref, tokenizer


# ============================================================================
# GRPO Training Loop (from notebook, adapted for your data)
# ============================================================================

def grpo_train_loop(model, model_old, model_ref, tokenizer, optimizer, train_data, config):
    """Training loop based on the notebook."""
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)
    
    losses = []
    all_rewards = []
    
    # train_data is from your build_dataset (list of dicts)
    print(f"\nStarting GRPO: {len(train_data)} examples, {config.iters} iterations")
    
    pbar = tqdm.tqdm(range(config.iters))
    for it in pbar:
        # 1. Sample batch
        indices = np.random.randint(0, len(train_data), config.batch_size)
        batch = [train_data[i] for i in indices]
        
        rollout_sequences = []
        rollout_rewards = []
        rollout_a_toks = []
        
        # 2. Rollout: Generate G responses per prompt
        for item in batch:
            # Format prompt with chat template
            messages = [{"role": "user", "content": item["prompt"]}]
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            
            group_rewards = []
            for _ in range(config.group_size):
                # Generate response
                response = generate(
                    model_old, tokenizer, prompt_tokens,
                    max_tokens=config.max_response_len,
                )
                answer_tokens = tokenizer.encode(response, add_special_tokens=False)
                
                # Compute reward using YOUR reward functions
                reward = compute_reward(response, item["board_ascii"], item["optimal_cost"])
                group_rewards.append(reward)
                
                # Store
                full_seq = mx.array(prompt_tokens + answer_tokens)
                rollout_sequences.append(full_seq)
                rollout_a_toks.append(mx.array(answer_tokens))
            
            all_rewards.extend(group_rewards)
            rollout_rewards.append(mx.array(group_rewards))
        
        # 3. Compute advantages (group-relative)
        advantages = []
        for rewards in rollout_rewards:
            mean_r = mx.mean(rewards)
            std_r = mx.sqrt(mx.var(rewards)) + 1e-8
            advantages.append((rewards - mean_r) / std_r)
        advantages = mx.concatenate(advantages)
        
        sequences = pad_sequences(rollout_sequences, tokenizer.pad_token_id)
        a_toks = pad_sequences(rollout_a_toks, tokenizer.pad_token_id)
        old_log_probs = calculate_log_probs(model_old, sequences, a_toks)
        
        # 4. Optimization step
        (loss, policy_reward, kl_div), grads = loss_and_grad_fn(
            model, model_ref, sequences, a_toks, advantages, old_log_probs,
            config.beta, config.epsilon
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        
        losses.append(loss.item())
        avg_reward = np.mean(all_rewards[-20:]) if len(all_rewards) >= 20 else np.mean(all_rewards)
        pbar.set_description(f"Loss: {loss.item():.3f} | Reward: {avg_reward:.3f}")
        
        # 5. Sync old model
        if (it + 1) % config.update_every == 0:
            model_old.update(model.parameters())
    
    # Save final adapter
    model.save_weights(str(config.adapter_path / "adapters.safetensors"))
    return losses, all_rewards


def debug_generation(model, tokenizer, prompt, config):
    """Debug what the model is actually generating."""
    
    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    print("\n" + "="*60)
    print("DEBUG: Model Generation")
    print("="*60)
    
    response = generate(
        model, tokenizer, prompt_tokens,
        max_tokens=256, verbose=True
    )
    
    print(f"\nRaw response:\n{response}")
    print("\n" + "="*60)
    
    # Check for XML tags
    import re
    has_think = bool(re.search(r"<think>", response, re.IGNORECASE))
    has_nextmove = bool(re.search(r"<NextMove>", response, re.IGNORECASE))
    has_confidence = bool(re.search(r"<Confidence>", response, re.IGNORECASE))
    has_rationale = bool(re.search(r"<Rationale>", response, re.IGNORECASE))
    
    print(f"Has <think>: {has_think}")
    print(f"Has <NextMove>: {has_nextmove}")
    print(f"Has <Confidence>: {has_confidence}")
    print(f"Has <Rationale>: {has_rationale}")
    
    # Extract move
    move_match = re.search(r"<NextMove>\s*([UDLRudlr])\s*</NextMove>", response, re.IGNORECASE)
    if move_match:
        print(f"Extracted move: {move_match.group(1)}")
    else:
        print("No valid move extracted!")
    
    return response

