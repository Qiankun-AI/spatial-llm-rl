#!/bin/bash

export http_proxy=http://9.131.113.25:11113
export https_proxy=http://9.131.113.25:11113

# pip install -U cos-python-sdk-v5  
export WANDB_API_KEY=126fa132fe9bf6916604208961f10a607c62d8e2
export WANDB_ENTITY=Qiankun-AI
export WANDB_PROJECT=spatialvl_rl
wandb login

set -x
ENGINE=${1:-vllm}
export HYDRA_FULL_ERROR=1
# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REWARD_FUNCTION_PATH="/mnt/sh/mmvision/home/jonahli/projects/SpatialVL/spatial-llm-rl/examples/grpo_trainer/geoloc_reward_function.py"

cd /mnt/sh/mmvision/home/jonahli/projects/SpatialVL/spatial-llm-rl
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/mnt/sh/mmvision/home/jonahli/data_rl/global-streetscapes/train.parquet \
    data.val_files=/mnt/sh/mmvision/home/jonahli/data_rl/global-streetscapes/test.parquet \
    data.train_batch_size=512 \
    data.max_prompt_length=15000 \
    data.max_response_length=4000 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=/mnt/sh/mmvision/home/jonahli/init_ckpt/vllm/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.agent.num_workers=128 \
    actor_rollout_ref.rollout.name=$ENGINE \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$REWARD_FUNCTION_PATH" \
    custom_reward_function.name=geoloc_haversine_reward_function \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='spatialvl_rl' \
    trainer.experiment_name='qwen2_5_vl_3b_geoloc_haversine' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.default_local_dir=/mnt/sh/mmvision/home/jonahli/save/checkpoints_rl/spatialvl_rl/qwen2_5_vl_3b_geoloc_haversine \
    trainer.test_freq=5 \
    trainer.total_epochs=30 $@ 