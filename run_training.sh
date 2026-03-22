#!/bin/bash
# DGPO Training Script for Qwen2.5-1.5B
# Usage: bash run_training.sh
#
# For Docker: docker compose up --build
# For GPU:   CUDA_VISIBLE_DEVICES=0 bash run_training.sh
# For multi-GPU: accelerate launch --multi_gpu run_training.sh

set -e

echo "=========================================="
echo "DGPO Training: Qwen2.5-1.5B"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.10+"
    exit 1
fi

# Install dependencies if needed
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Set environment variables
export WANDB_PROJECT="${WANDB_PROJECT:-dgpo-qwen2.5-1.5b}"
export PYTHONUNBUFFERED=1

# Run training
echo "Starting DGPO training..."
echo "Teacher: Qwen/Qwen2.5-7B-Instruct"
echo "Student: Qwen/Qwen2.5-1.5B-Instruct"
echo ""

python3 train_dgpo.py \
    --teacher_model "Qwen/Qwen2.5-7B-Instruct" \
    --student_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --output_dir "./dgpo-qwen2.5-1.5b" \
    --kd_epochs 5 \
    --kd_batch_size 32 \
    --kd_learning_rate 2e-5 \
    --rl_steps 1000 \
    --rl_batch_size 512 \
    --kl_coef 0.001 \
    --actor_lr 1e-6 \
    --critic_lr 1e-5 \
    --max_prompt_length 4096 \
    --max_response_length 500 \
    --top_k_docs 3 \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    --train_datasets "nq,triviaqa,popqa,hotpotqa" \
    --wandb_project "${WANDB_PROJECT}"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Model saved to: ./dgpo-qwen2.5-1.5b"
echo ""
echo "To convert to GGUF:"
echo "  bash convert_to_gguf.sh Q4_K_M"
echo "=========================================="
