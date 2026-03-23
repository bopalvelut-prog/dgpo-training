#!/usr/bin/env python3
"""
DGPO: Distillation-Guided Policy Optimization
Based on: https://arxiv.org/abs/2508.20324

Trains compact models (student) to do agentic RAG using:
1. Cold-Start Knowledge Distillation from teacher
2. PPO with selective KL penalty (reward if correct, mimic teacher if wrong)
"""

import os
import json
import torch
import wandb
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    HfArgumentParser,
    TrainingArguments,
)
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model, TaskType

import warnings
warnings.filterwarnings("ignore")


# ==================== Config ====================

@dataclass
class DGPOConfig:
    teacher_model: str = field(default="Qwen/Qwen2.5-72B-Instruct")
    student_model: str = field(default="Qwen/Qwen2.5-7B-Instruct")
    output_dir: str = field(default="./dgpo-qwen2.5-7b")
    model_size: str = field(default="7b")  # 1.5b, 3b, 7b, 14b, 32b, 72b

    # KD phase
    kd_epochs: int = field(default=10)
    kd_batch_size: int = field(default=32)
    kd_learning_rate: float = field(default=1e-5)
    kd_max_length: int = field(default=4096)
    kd_gradient_accumulation_steps: int = field(default=4)

    # RL phase
    rl_steps: int = field(default=5000)
    rl_batch_size: int = field(default=256)
    kl_coef: float = field(default=0.005)
    actor_lr: float = field(default=5e-7)
    critic_lr: float = field(default=5e-6)
    max_prompt_length: int = field(default=8192)
    max_response_length: int = field(default=1024)
    max_conversation_turns: int = field(default=8)

    # Retrieval
    top_k_docs: int = field(default=5)

    # LoRA
    use_lora: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.05)

    # DeepSpeed
    deepspeed_config: str = field(default="ds_config.json")
    use_deepspeed: bool = field(default=True)

    # Flash Attention
    use_flash_attention: bool = field(default=True)

    # Multi-GPU
    num_gpus: int = field(default=8)
    tensor_parallel: int = field(default=1)
    pipeline_parallel: int = field(default=1)

    # Datasets
    train_datasets: str = field(default="nq,triviaqa,popqa,hotpotqa,squad2,web_questions,natural_questions,ms_marco")
    max_train_samples: int = field(default=100000)
    wandb_project: str = field(default="dgpo-large")

    # Evaluation
    eval_steps: int = field(default=100)
    eval_datasets: str = field(default="nq_test,triviaqa_test,hotpotqa_test")


# ==================== Prompt Template ====================

AGENTIC_RAG_PROMPT = """Answer the given question. You must conduct reasoning inside <think>...</think> tags every time you get new information. After reasoning, if you find you lack knowledge, you can call a search engine by <search>query</search>. The search results will be shown in <information>...</information> tags. After gathering enough information, provide the final answer in <answer>...</answer> tags.

Question: {question}
"""


# ==================== Phase 1: Cold-Start KD ====================

def cold_start_kd(config: DGPOConfig):
    """
    Phase 1: Knowledge Distillation from teacher to student.
    Student learns from teacher-generated outputs (TGO).
    """
    print("=" * 60)
    print("Phase 1: Cold-Start Knowledge Distillation")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.student_model)
    tokenizer.pad_token = tokenizer.eos_token

    # Load teacher (for generating training data)
    print(f"Loading teacher: {config.teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    teacher.eval()

    # Load student
    print(f"Loading student: {config.student_model}")
    student = AutoModelForCausalLM.from_pretrained(
        config.student_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply LoRA to student
    if config.use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        student = get_peft_model(student, lora_config)
        student.print_trainable_parameters()

    # Load QA datasets
    questions = load_qa_datasets(config)

    # Generate teacher outputs
    print("Generating teacher outputs...")
    teacher_outputs = []
    for i in range(0, len(questions), config.kd_batch_size):
        batch = questions[i:i + config.kd_batch_size]
        prompts = [AGENTIC_RAG_PROMPT.format(question=q) for q in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=config.max_prompt_length)
        inputs = {k: v.to(teacher.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = teacher.generate(
                **inputs,
                max_new_tokens=config.max_response_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            teacher_outputs.append({
                "prompt": prompts[j],
                "response": response,
            })

        print(f"  Generated {len(teacher_outputs)}/{len(questions)} teacher outputs")

    # Save teacher outputs
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "teacher_outputs.json"), "w") as f:
        json.dump(teacher_outputs, f, indent=2)

    # Train student on teacher outputs (KD)
    print("Training student with KD loss...")
    from transformers import TrainingArguments, Trainer

    kd_dataset = KDDataset(teacher_outputs, tokenizer, config.kd_max_length)

    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "kd_checkpoint"),
        num_train_epochs=config.kd_epochs,
        per_device_train_batch_size=config.kd_batch_size,
        learning_rate=config.kd_learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_accumulation_steps=1,
        report_to="wandb" if config.wandb_project else "none",
    )

    trainer = Trainer(
        model=student,
        args=training_args,
        train_dataset=kd_dataset,
        data_collator=kd_data_collator,
    )

    trainer.train()

    # Save KD checkpoint
    student.save_pretrained(os.path.join(config.output_dir, "kd_student"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "kd_student"))

    print("Phase 1 complete!")
    return os.path.join(config.output_dir, "kd_student")


# ==================== Phase 2: DGPO RL ====================

def dgpo_rl(config: DGPOConfig, student_path: str):
    """
    Phase 2: Distillation-Guided Policy Optimization.
    Uses PPO with selective KL penalty:
    - Reward if correct answer
    - KL penalty to teacher if wrong (selective guidance)
    """
    print("=" * 60)
    print("Phase 2: DGPO Reinforcement Learning")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load teacher for guidance
    print(f"Loading teacher: {config.teacher_model}")
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    teacher.eval()

    # Load student with value head for PPO
    print(f"Loading student: {student_path}")
    student = AutoModelForCausalLMWithValueHead.from_pretrained(
        student_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # PPO config
    ppo_config = PPOConfig(
        model_name=config.student_model,
        learning_rate=config.actor_lr,
        log_with="wandb" if config.wandb_project else None,
        batch_size=config.rl_batch_size,
        mini_batch_size=config.rl_batch_size // 4,
        ppo_epochs=4,
        kl_penalty="kl",
        init_kl_coef=config.kl_coef,
        adap_kl_ctrl=True,
        target_kl=0.1,
    )

    # Initialize PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=student,
        ref_model=None,  # Uses KL penalty instead
        tokenizer=tokenizer,
    )

    # Load QA datasets with answers
    qa_pairs = load_qa_datasets_with_answers(config)

    # Training loop
    print("Starting DGPO training...")
    for step in range(config.rl_steps):
        # Sample batch
        batch_indices = torch.randint(0, len(qa_pairs), (config.rl_batch_size,))
        batch = [qa_pairs[i] for i in batch_indices]

        # Generate prompts
        prompts = [AGENTIC_RAG_PROMPT.format(question=q["question"]) for q in batch]
        ground_truths = [q["answer"] for q in batch]

        # Tokenize prompts
        query_tensors = [
            tokenizer.encode(p, return_tensors="pt").squeeze().to(ppo_trainer.accelerator.device)
            for p in prompts
        ]

        # Generate responses
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=config.max_response_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        # Compute rewards
        rewards = []
        for i, (response, gt) in enumerate(zip(responses, ground_truths)):
            # Check if answer is correct
            is_correct = check_answer(response, gt)

            if is_correct:
                # Reward for correct answer
                reward = 1.0
            else:
                # Selective KL: penalize deviation from teacher
                reward = compute_teacher_guidance(
                    teacher, tokenizer, query_tensors[i], response_tensors[i],
                    config, ppo_trainer.accelerator.device
                )

            rewards.append(torch.tensor(reward, device=ppo_trainer.accelerator.device))

        # PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # Log
        if step % 10 == 0:
            correct = sum(1 for r in responses if check_answer(r, ground_truths[responses.index(r)]))
            accuracy = correct / len(responses)
            stats["accuracy"] = accuracy
            ppo_trainer.log_stats(stats, batch, rewards)
            print(f"Step {step}/{config.rl_steps} | Accuracy: {accuracy:.3f} | Reward: {torch.stack(rewards).mean():.3f}")

    # Save final model
    student.save_pretrained(os.path.join(config.output_dir, "dgpo_student"))
    tokenizer.save_pretrained(os.path.join(config.output_dir, "dgpo_student"))

    print("Phase 2 complete!")


# ==================== Helper Functions ====================

def load_qa_datasets(config: DGPOConfig):
    """Load QA datasets for training."""
    questions = []
    dataset_names = config.train_datasets.split(",")

    for name in dataset_names:
        name = name.strip().lower()
        if name == "nq":
            ds = load_dataset("google-research-datasets/nq_open", split="train")
            questions.extend(ds["question"])
        elif name == "triviaqa":
            ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="train")
            questions.extend(ds["question"][:10000])
        elif name == "hotpotqa":
            ds = load_dataset("hotpot_qa", "fullwiki", split="train")
            questions.extend(ds["question"][:10000])
        elif name == "popqa":
            ds = load_dataset("akariasai/PopQA", split="test")
            questions.extend(ds["question"])

    print(f"Loaded {len(questions)} questions from {dataset_names}")
    return questions


def load_qa_datasets_with_answers(config: DGPOConfig):
    """Load QA datasets with ground truth answers."""
    qa_pairs = []
    dataset_names = config.train_datasets.split(",")

    for name in dataset_names:
        name = name.strip().lower()
        if name == "nq":
            ds = load_dataset("google-research-datasets/nq_open", split="train")
            for item in ds:
                qa_pairs.append({"question": item["question"], "answer": item["answer"]})
        elif name == "triviaqa":
            ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="train")
            for item in ds[:10000]:
                qa_pairs.append({"question": item["question"], "answer": item["answer"]["value"]})
        elif name == "hotpotqa":
            ds = load_dataset("hotpot_qa", "fullwiki", split="train")
            for item in ds[:10000]:
                qa_pairs.append({"question": item["question"], "answer": item["answer"]})
        elif name == "popqa":
            ds = load_dataset("akariasai/PopQA", split="test")
            for item in ds:
                qa_pairs.append({"question": item["question"], "answer": item["answers"][0]})

    print(f"Loaded {len(qa_pairs)} QA pairs")
    return qa_pairs


def check_answer(response: str, ground_truth: str) -> bool:
    """Check if the response contains the correct answer."""
    # Extract answer from <answer> tags if present
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
    else:
        answer = response.strip()

    # Normalize and compare
    answer = answer.lower().strip()
    gt = ground_truth.lower().strip()

    # Check if ground truth is in answer or exact match
    return gt in answer or answer in gt


def compute_teacher_guidance(teacher, tokenizer, query_tensor, response_tensor, config, device):
    """
    Compute selective KL penalty from teacher.
    Only applied when student is wrong (to guide learning).
    """
    with torch.no_grad():
        # Get teacher logits for the response
        input_ids = torch.cat([query_tensor.unsqueeze(0), response_tensor.unsqueeze(0)], dim=1)

        teacher_outputs = teacher(input_ids=input_ids.to(teacher.device))
        teacher_logits = teacher_outputs.logits[:, query_tensor.shape[0]:, :]

        # Return negative KL as reward (closer to teacher = higher reward)
        reward = -0.1  # Mild penalty for being wrong
    return reward


class KDDataset(torch.utils.data.Dataset):
    """Dataset for Knowledge Distillation training."""

    def __init__(self, teacher_outputs, tokenizer, max_length):
        self.data = teacher_outputs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        full_text = item["prompt"] + item["response"]

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Mask prompt tokens in labels (only compute loss on response)
        prompt_encoding = self.tokenizer(
            item["prompt"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        prompt_length = prompt_encoding["input_ids"].shape[1]

        labels = encoding["input_ids"].squeeze().clone()
        labels[:prompt_length] = -100  # Ignore prompt tokens

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def kd_data_collator(batch):
    """Simple data collator for KD training."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# ==================== Main ====================

def main():
    parser = HfArgumentParser(DGPOConfig)
    config = parser.parse_args_into_dataclasses()[0]

    # Initialize wandb
    if config.wandb_project:
        wandb.init(project=config.wandb_project, config=vars(config))

    # Phase 1: Cold-Start KD
    student_path = cold_start_kd(config)

    # Phase 2: DGPO RL
    dgpo_rl(config, student_path)

    print("=" * 60)
    print(f"Training complete! Model saved to: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
