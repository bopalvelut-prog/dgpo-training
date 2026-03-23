# DGPO Training for Qwen2.5

Distillation-Guided Policy Optimization (DGPO) training implementation for compact language models.

Based on the paper: [Can Compact Language Models Search Like Agents?](https://arxiv.org/abs/2508.20324)

## What is DGPO?

DGPO enables small models to perform agentic RAG/search like larger models using:

- **Cold-Start Knowledge Distillation** - Student learns from teacher outputs
- **Selective KL PPO** - Reward if correct, mimic teacher if wrong

## Supported Model Sizes

| Size | Student | Teacher | GPUs Required | Training Time |
|------|---------|---------|---------------|---------------|
| 1.5B | Qwen2.5-1.5B | Qwen2.5-7B | 1x A100 | ~6 hours |
| 3B | Qwen2.5-3B | Qwen2.5-14B | 2x A100 | ~12 hours |
| 7B | Qwen2.5-7B | Qwen2.5-72B | 4x A100 | ~24 hours |
| 14B | Qwen2.5-14B | Qwen2.5-72B | 8x A100 | ~48 hours |
| 32B | Qwen2.5-32B | Qwen2.5-72B | 8x A100 80GB | ~96 hours |
| 72B | Qwen2.5-72B | Qwen2.5-72B | 16x A100 80GB | ~1 week |

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPUs (see table above for requirements)
- DeepSpeed >= 0.14.0
- Flash Attention 2

### Installation

```bash
pip install -r requirements.txt
```

### Training All Sizes

```bash
# Train all sizes (1.5B through 72B)
bash train_all_sizes.sh

# Train specific size
bash train_all_sizes.sh 7b
bash train_all_sizes.sh 14b
bash train_all_sizes.sh 32b
```

### Training Single Size

```bash
# 7B model with DeepSpeed
deepspeed --num_gpus=4 train_dgpo.py \
    --teacher_model "Qwen/Qwen2.5-72B-Instruct" \
    --student_model "Qwen/Qwen2.5-7B-Instruct" \
    --output_dir "./dgpo-qwen2.5-7b" \
    --model_size "7b" \
    --use_deepspeed True \
    --num_gpus 4
```

### Docker (Multi-GPU)

```bash
docker compose up --build
```

### Convert to GGUF

```bash
# Convert trained model to GGUF for inference
bash convert_to_gguf.sh Q4_K_M dgpo-qwen2.5-7b
```

## Project Structure

```
train_dgpo.py        - DGPO training script (KD + PPO)
train_all_sizes.sh   - Train all model sizes
run_training.sh      - Training launcher
convert_to_gguf.sh   - Convert to GGUF format
test_model.py        - Inference test
Dockerfile           - Docker build (CUDA)
docker-compose.yml   - Docker run config
ds_config.json       - DeepSpeed config
requirements.txt     - Dependencies
```

## Configuration

Edit `train_dgpo.py` or pass arguments:

| Parameter | Default (7B) | Description |
|-----------|--------------|-------------|
| teacher_model | Qwen/Qwen2.5-72B-Instruct | Teacher model |
| student_model | Qwen/Qwen2.5-7B-Instruct | Student model |
| model_size | 7b | Model size |
| kd_epochs | 10 | KD training epochs |
| rl_steps | 5000 | RL training steps |
| kl_coef | 0.005 | KL penalty coefficient |
| use_lora | False | Use LoRA (for small models) |
| use_deepspeed | True | Use DeepSpeed |
| num_gpus | 8 | Number of GPUs |
| tensor_parallel | 1 | Tensor parallelism |
| pipeline_parallel | 1 | Pipeline parallelism |

## Performance

From the paper (Qwen2.5 3B → 0.5B):

| Method | Avg. Score |
|--------|-----------|
| Base (0.5B) | 0.006 |
| Teacher (3B) | 0.353 |
| PPO | 0.238 |
| KD | 0.298 |
| **DGPO** | **0.329** |

~55x improvement over base model!

## Scaling Guidelines

### 1.5B Model
- 1x A100 80GB
- LoRA enabled
- No DeepSpeed needed
- ~6 hours training

### 7B Model
- 4x A100 80GB
- DeepSpeed Stage 3
- Full fine-tuning (no LoRA)
- ~24 hours training

### 72B Model
- 16x A100 80GB
- DeepSpeed Stage 3
- Tensor Parallel 4
- Pipeline Parallel 2
- ~1 week training

## References

- [Paper](https://arxiv.org/abs/2508.20324)
- [HuggingFace Model](https://huggingface.co/omron-sinicx/DGPO-qwen2.5-0.5b)
- [GGUF Version](https://huggingface.co/mradermacher/DGPO-qwen2.5-0.5b-GGUF)

## License

Apache 2.0
