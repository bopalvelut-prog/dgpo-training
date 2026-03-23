#!/bin/bash
# DGPO Training Scripts for Different Model Sizes
# Usage: ./train_all_sizes.sh [size]

SIZE=${1:-"all"}

echo "=== DGPO Training ==="
echo "Size: $SIZE"

if [ "$SIZE" = "all" ] || [ "$SIZE" = "1.5b" ]; then
    echo "Training 1.5B model..."
    python train_dgpo.py \
        --teacher_model "Qwen/Qwen2.5-7B-Instruct" \
        --student_model "Qwen/Qwen2.5-1.5B-Instruct" \
        --output_dir "./dgpo-qwen2.5-1.5b" \
        --model_size "1.5b" \
        --kd_epochs 5 \
        --rl_steps 1000 \
        --kd_batch_size 64 \
        --rl_batch_size 512 \
        --use_lora True \
        --lora_r 16 \
        --lora_alpha 32 \
        --use_deepspeed False \
        --num_gpus 1
fi

if [ "$SIZE" = "all" ] || [ "$SIZE" = "3b" ]; then
    echo "Training 3B model..."
    python train_dgpo.py \
        --teacher_model "Qwen/Qwen2.5-14B-Instruct" \
        --student_model "Qwen/Qwen2.5-3B-Instruct" \
        --output_dir "./dgpo-qwen2.5-3b" \
        --model_size "3b" \
        --kd_epochs 8 \
        --rl_steps 3000 \
        --kd_batch_size 32 \
        --rl_batch_size 256 \
        --use_lora True \
        --lora_r 32 \
        --lora_alpha 64 \
        --use_deepspeed False \
        --num_gpus 2
fi

if [ "$SIZE" = "all" ] || [ "$SIZE" = "7b" ]; then
    echo "Training 7B model..."
    python train_dgpo.py \
        --teacher_model "Qwen/Qwen2.5-72B-Instruct" \
        --student_model "Qwen/Qwen2.5-7B-Instruct" \
        --output_dir "./dgpo-qwen2.5-7b" \
        --model_size "7b" \
        --kd_epochs 10 \
        --rl_steps 5000 \
        --kd_batch_size 16 \
        --rl_batch_size 128 \
        --use_lora False \
        --use_deepspeed True \
        --num_gpus 4
fi

if [ "$SIZE" = "all" ] || [ "$SIZE" = "14b" ]; then
    echo "Training 14B model..."
    python train_dgpo.py \
        --teacher_model "Qwen/Qwen2.5-72B-Instruct" \
        --student_model "Qwen/Qwen2.5-14B-Instruct" \
        --output_dir "./dgpo-qwen2.5-14b" \
        --model_size "14b" \
        --kd_epochs 12 \
        --rl_steps 8000 \
        --kd_batch_size 8 \
        --rl_batch_size 64 \
        --use_lora False \
        --use_deepspeed True \
        --num_gpus 8
fi

if [ "$SIZE" = "all" ] || [ "$SIZE" = "32b" ]; then
    echo "Training 32B model..."
    python train_dgpo.py \
        --teacher_model "Qwen/Qwen2.5-72B-Instruct" \
        --student_model "Qwen/Qwen2.5-32B-Instruct" \
        --output_dir "./dgpo-qwen2.5-32b" \
        --model_size "32b" \
        --kd_epochs 15 \
        --rl_steps 10000 \
        --kd_batch_size 4 \
        --rl_batch_size 32 \
        --use_lora False \
        --use_deepspeed True \
        --num_gpus 8 \
        --tensor_parallel 2
fi

if [ "$SIZE" = "all" ] || [ "$SIZE" = "72b" ]; then
    echo "Training 72B model..."
    python train_dgpo.py \
        --teacher_model "Qwen/Qwen2.5-72B-Instruct" \
        --student_model "Qwen/Qwen2.5-72B-Instruct" \
        --output_dir "./dgpo-qwen2.5-72b" \
        --model_size "72b" \
        --kd_epochs 20 \
        --rl_steps 15000 \
        --kd_batch_size 2 \
        --rl_batch_size 16 \
        --use_lora False \
        --use_deepspeed True \
        --num_gpus 8 \
        --tensor_parallel 4 \
        --pipeline_parallel 2
fi

echo "Training complete!"
