#!/bin/bash
# Modified version for Google Colab FREE tier (T4 GPU ~15GB VRAM)
# This script is HEAVILY optimized for limited resources
# WARNING: Training will be MUCH slower and results may vary

export RAY_DEDUP_LOGS=0

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
# model_name=Qwen/Qwen2.5-0.5B-Instruct  # Much smaller, will fit better
# model_name=Qwen/Qwen2.5-1.5B-Instruct  # Good balance
model_name=Qwen/Qwen3-4B  # Original, will be tight on memory

prompt_template_type="qwen3_no_thinking"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=psr_nsr \
    algorithm.advantage=$advantage \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.prompt_template_type=$prompt_template_type \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=6000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=6000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=4 \
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

# KEY CHANGES FROM ORIGINAL:
# ========================
# 1. n_gpus_per_node: 8 -> 1 (Colab has 1 GPU)
# 2. tensor_model_parallel_size: 2 -> 1 (No multi-GPU parallelism)
# 3. train_batch_size: 1024 -> 32 (Drastic reduction for memory)
# 4. max_prompt_length: 1024 -> 512 (Reduce context)
# 5. max_response_length: 31744 -> 2048 (HUGE reduction, but necessary)
# 6. ppo_max_token_len_per_gpu: 48000 -> 4000 (Memory constraint)
# 7. log_prob_max_token_len_per_gpu: 64000 -> 6000 (Memory constraint)
# 8. ppo_mini_batch_size: 256 -> 8 (Reduce batch processing)
# 9. rollout.n: 8 -> 4 (Fewer rollout samples)
# 10. gpu_memory_utilization: 0.7 -> 0.85 (Use more of available memory)
# 11. param_offload: False -> True (Offload to CPU to save GPU memory)
# 12. optimizer_offload: False -> True (Offload optimizer states)
# 13. free_cache_engine: False -> True (Free cache between iterations)
#
# LIMITATIONS:
# - Training will be 20-30x slower
# - Reduced batch size may affect convergence
# - Shortened response length limits reasoning capability
# - May still OOM on T4; consider using Qwen2.5-0.5B or 1.5B instead
