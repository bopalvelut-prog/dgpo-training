#!/usr/bin/env python3
"""Quick test for DGPO-trained model."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./dgpo-qwen2.5-1.5b/dgpo_student"

PROMPT = """Answer the given question. You must conduct reasoning inside <think>...</think> tags every time you get new information. After reasoning, if you find you lack knowledge, you can call a search engine by <search>query</search>. The search results will be shown in <information>...</information> tags. After gathering enough information, provide the final answer in <answer>...</answer> tags.

Question: Who wrote the novel 1984?
"""

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Generating response...")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("\n" + "=" * 60)
    print("Response:")
    print("=" * 60)
    print(response)


if __name__ == "__main__":
    main()
