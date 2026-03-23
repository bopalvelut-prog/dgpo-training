#!/usr/bin/env python3
"""Login to HuggingFace and push model."""
import sys
from huggingface_hub import login, HfApi, create_repo

# Login
token = sys.argv[1] if len(sys.argv) > 1 else input("Enter token: ")
login(token=token)

api = HfApi()
user = api.whoami()['name']
print(f"Logged in as: {user}")

# Create repo
repo_id = f"{user}/dgpo-tiny-pure"
try:
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"Repo created: {repo_id}")
except Exception as e:
    print(f"Repo: {e}")

# Upload files
files = [
    "dgpo-tiny-pure/dgpo_tiny.pkl",
    "dgpo-tiny-pure.gguf",
    "train_dgpo_pure.py",
    "convert_to_gguf_pure.py",
]

for f in files:
    try:
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=f.split("/")[-1],
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"Uploaded: {f}")
    except Exception as e:
        print(f"Failed {f}: {e}")

# Upload README
readme = """---
license: apache-2.0
tags:
- tiny-llm
- dgpo
- distillation
---

# DGPO Tiny Pure

Tiny 0.43M parameter language model trained with DGPO (Distillation-Guided Policy Optimization).

## Files

- `dgpo_tiny.pkl` - Python pickle model
- `dgpo-tiny-pure.gguf` - GGUF format for llama.cpp

## Usage

```python
import pickle
with open("dgpo_tiny.pkl", "rb") as f:
    model = pickle.load(f)
```
"""

with open("/tmp/README.md", "w") as f:
    f.write(readme)

api.upload_file(
    path_or_fileobj="/tmp/README.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
)
print("Uploaded README")

print(f"\nDone! Model at: https://huggingface.co/{repo_id}")
