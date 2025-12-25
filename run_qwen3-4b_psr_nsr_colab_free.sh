#!/bin/bash
# Modified version for Google Colab FREE tier (T4 GPU ~15GB VRAM)
# This script is HEAVILY optimized for limited resources
# WARNING: Training will be MUCH slower and results may vary

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

# CRITICAL: Consider using smaller model for Colab free
model_name=Qwen/Qwen2.5-0.5B-Instruct  # Much smaller, will fit better (RECOMMENDED FOR COLAB FREE)
# model_name=Qwen/Qwen2.5-1.5B-Instruct  # Good balance
# model_name=Qwen/Qwen3-4B  # Original, TOO LARGE for Colab Free (causes OOM)

prompt_template_type="qwen3_no_thinking"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=psr_nsr \
    algorithm.advantage=$advantage \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_template_type=$prompt_template_type \
    actor_rollout_ref.model.path=$model_name \
    +actor_rollout_ref.model.attn_implementation=eager \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=6000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_num_seqs=512 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.experiment_name="MATH-Qwen3-4B-$advantage-colab-free" \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='verl-colab-free' \
    trainer.n_gpus_per_node=1 \
    +trainer.val_before_train=True \
    trainer.nnodes=1 \
    trainer.save_freq=7 \
    trainer.test_freq=7 \
    trainer.total_epochs=20 $@
    # algorithm.positive_advantage_weight=$positive_advantage_weight \

# KEY CHANGES FROM ORIGINAL (EXTREME OPTIMIZATIONS FOR COLAB FREE):
# ================================================================
# 1. model: Qwen3-4B -> Qwen2.5-0.5B-Instruct (10x smaller)
# 2. n_gpus_per_node: 8 -> 1 (Colab has 1 GPU)
# 3. tensor_model_parallel_size: 2 -> 1 (No multi-GPU parallelism)
# 4. train_batch_size: 1024 -> 8 (128x reduction for memory)
# 5. max_prompt_length: 1024 -> 512 (Reduce context)
# 6. max_response_length: 31744 -> 1536 (21x reduction, necessary for memory)
# 7. ppo_max_token_len_per_gpu: 48000 -> 4000 (12x reduction)
# 8. log_prob_max_token_len_per_gpu: 64000 -> 6000 (10x reduction)
# 9. ppo_mini_batch_size: 256 -> 8 (32x reduction)
# 10. rollout.n: 8 -> 2 (4x fewer rollout samples per prompt)
# 11. gpu_memory_utilization: 0.7 -> 0.3 (Reduce vLLM KV cache to ~3GB)
# 12. max_num_batched_tokens: 16384 -> 2048 (8x reduction for vLLM batch)
# 13. max_num_seqs: 1024 -> 512 (Limit parallel sequences in vLLM)
# 14. param_offload: False -> True (Offload to CPU to save GPU memory)
# 15. optimizer_offload: False -> True (Offload optimizer states)
# 16. free_cache_engine: False -> True (Free cache between iterations)
# 17. attn_implementation: flash_attn -> eager (No flash-attn on Colab)
#
# MEMORY BREAKDOWN (Expected with these settings):
# - Model weights: 0.93GB
# - vLLM KV cache: ~3GB (gpu_memory_utilization=0.3, reduced sequences/tokens)
# - Activations: ~1.2GB (reduced due to smaller batches)
# - Total WorkerDict: ~6-7GB (target: down from 8.87GB)
# - Ray overhead: ~2-3GB
# - Target: ~9-10GB / 12.67GB = 75-80% (safe margin below 95% threshold)
#
# LIMITATIONS (EXTREME - This is bare minimum to fit in Colab free):
# - Training will be 100-150x slower than original (batch size 8 vs 1024)
# - Extremely small batch size (8) will significantly affect convergence quality
# - Very short response length (1536 tokens) severely limits reasoning capability
# - Smaller model (0.5B) has much lower reasoning ability than 4B
# - Fewer rollout samples (n=2) reduces exploration
# - This is the MINIMUM viable configuration for Colab free tier
# - If this still OOMs, you MUST use Colab Pro with more RAM
