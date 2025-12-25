#!/bin/bash
# Modified version for Google Colab PRO/PRO+ (A100 40GB or V100 32GB)
# This script is optimized for single high-end GPU
# Better performance than free tier, but still single GPU

export RAY_DEDUP_LOGS=0
# Disable flash attention (not available on Colab by default)
export VERL_DISABLE_FLASH_ATTN=1

math_train_path=./data/math/train.parquet
math_test_path=./data/math/test.parquet
aime2025_test_path=./data/aime2025/test.parquet
amc23_test_path=./data/amc23/test.parquet

train_files="['$math_train_path']"
test_files="['$math_test_path', '$aime2025_test_path', '$amc23_test_path']"
advantage="positive"   # PSR
# advantage="negative"   # NSR
# advantage="weighted"   # W-REINFORCE
# positive_advantage_weight=0.1   # For W-REINFORCE only
kl_coef=0.0
lr=1e-6
model_name=Qwen/Qwen3-4B
prompt_template_type="qwen3_no_thinking"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=psr_nsr \
    algorithm.advantage=$advantage \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_template_type=$prompt_template_type \
    actor_rollout_ref.model.path=$model_name \
    +actor_rollout_ref.model.attn_implementation=eager \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=6 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.experiment_name="MATH-Qwen3-4B-$advantage-colab-pro" \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='verl-colab-pro' \
    trainer.n_gpus_per_node=1 \
    +trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=7 \
    trainer.test_freq=7 \
    trainer.total_epochs=20 $@
    # algorithm.positive_advantage_weight=$positive_advantage_weight \

# KEY CHANGES FROM ORIGINAL (for Colab Pro with A100/V100):
# =========================================================
# 1. n_gpus_per_node: 8 -> 1 (Colab Pro has 1 GPU)
# 2. tensor_model_parallel_size: 2 -> 1 (No multi-GPU parallelism)
# 3. train_batch_size: 1024 -> 256 (Moderate reduction)
# 4. max_response_length: 31744 -> 8192 (Significant reduction but better than free)
# 5. ppo_max_token_len_per_gpu: 48000 -> 12000 (Balance memory/performance)
# 6. log_prob_max_token_len_per_gpu: 64000 -> 16000 (Balanced)
# 7. ppo_mini_batch_size: 256 -> 64 (Moderate reduction)
# 8. rollout.n: 8 -> 6 (Slightly fewer samples)
# 9. gpu_memory_utilization: 0.7 -> 0.80 (Use more GPU memory)
# 10. param_offload: False -> False (Keep on GPU for speed)
# 11. optimizer_offload: False -> True (CPU offload for optimizer only)
# 12. attn_implementation: flash_attn -> eager (No flash-attn on Colab)
#
# ADVANTAGES OVER FREE TIER:
# - 8x larger batch size (256 vs 32)
# - 4x longer responses (8192 vs 2048)
# - 3x more tokens per batch
# - Better convergence and quality
# - 5-10x faster training
#
# LIMITATIONS VS ORIGINAL:
# - Still single GPU (vs 8 GPUs)
# - Reduced batch size (256 vs 1024)
# - Shorter max response (8192 vs 31744)
# - Will be slower than 8-GPU setup
