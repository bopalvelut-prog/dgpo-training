# DGPO Training for Qwen2.5

Distillation-Guided Policy Optimization (DGPO) training implementation for compact language models.

Based on the paper: [Can Compact Language Models Search Like Agents?](https://arxiv.org/abs/2508.20324)

## What is DGPO?

DGPO enables small models (0.5B-1.5B) to perform agentic RAG/search like larger models (3B-7B) using:

- **Cold-Start Knowledge Distillation** - Student learns from teacher outputs
- **Selective KL PPO** - Reward if correct, mimic teacher if wrong

## Quick Start

### Requirements

- Python 3.10+
- CUDA GPU (recommended: 24GB+ VRAM)
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
bash run_training.sh
```

### Docker

```bash
docker compose up --build
```

### Convert to GGUF

```bash
bash convert_to_gguf.sh Q4_K_M
```

## Project Structure

```
train_dgpo.py       - DGPO training script (KD + PPO)
run_training.sh     - Training launcher
convert_to_gguf.sh  - Convert to GGUF format
test_model.py       - Inference test
Dockerfile          - Docker build
docker-compose.yml  - Docker run config
requirements.txt    - Dependencies
```

## Configuration

Edit `train_dgpo.py` or pass arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| teacher_model | Qwen/Qwen2.5-7B-Instruct | Teacher model |
| student_model | Qwen/Qwen2.5-1.5B-Instruct | Student model |
| kd_epochs | 5 | KD training epochs |
| rl_steps | 1000 | RL training steps |
| kl_coef | 0.001 | KL penalty coefficient |
| use_lora | True | Use LoRA for efficiency |

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

## References

- [Paper](https://arxiv.org/abs/2508.20324)
- [HuggingFace Model](https://huggingface.co/omron-sinicx/DGPO-qwen2.5-0.5b)
- [GGUF Version](https://huggingface.co/mradermacher/DGPO-qwen2.5-0.5b-GGUF)

## License

Apache 2.0
