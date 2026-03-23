#!/usr/bin/env python3
"""
SDPO: Self-Distillation Policy Optimization (Pure Python)
Based on: https://arxiv.org/abs/2601.20802

Key insight: Use rich text feedback (errors, hints) as dense signal.
Model learns from its own mistakes by re-evaluating with feedback.

Usage: python3 train_sdpo_pure.py
"""

import os
import json
import math
import random
import pickle
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ==================== Tokenizer ====================

class TinyTokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3,
                         "<FEEDBACK>": 4, "<CORRECT>": 5, "<WRONG>": 6}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 7

    def build_vocab(self, texts, max_vocab=2000):
        word_freq = defaultdict(int)
        for text in texts:
            for word in text.lower().split():
                word_freq[word] += 1
        for word, freq in sorted(word_freq.items(), key=lambda x: -x[1])[:max_vocab-7]:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                self.vocab_size = idx + 1

    def encode(self, text, max_length=128):
        tokens = [self.word2idx.get(w, 1) for w in text.lower().split()]
        return tokens[:max_length]

    def decode(self, ids):
        return " ".join(self.idx2word.get(i, "<UNK>") for i in ids if i not in [0, 2, 3])

# ==================== Tiny Transformer ====================

class TinyTransformer:
    def __init__(self, vocab_size=2000, d_model=128, n_heads=4, n_layers=3, max_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len

        scale = 0.02
        self.embed = self._init((vocab_size, d_model), scale)
        self.pos_embed = self._init((max_len, d_model), scale)

        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                "wq": self._init((d_model, d_model), scale),
                "wk": self._init((d_model, d_model), scale),
                "wv": self._init((d_model, d_model), scale),
                "wo": self._init((d_model, d_model), scale),
                "w1": self._init((d_model, d_model * 2), scale),
                "w2": self._init((d_model * 2, d_model), scale),
                "ln1_g": self._init((d_model,), 1.0),
                "ln1_b": self._init((d_model,), 0.0),
                "ln2_g": self._init((d_model,), 1.0),
                "ln2_b": self._init((d_model,), 0.0),
            })

        self.output_proj = self._init((d_model, vocab_size), scale)

        params = sum(p.size if HAS_NUMPY else len(p) * len(p[0]) if isinstance(p[0], list) else len(p)
                     for p in [self.embed, self.pos_embed, self.output_proj])
        for layer in self.layers:
            for p in layer.values():
                params += p.size if HAS_NUMPY else len(p) * len(p[0]) if isinstance(p[0], list) else len(p)
        print(f"Model: {params:,} params ({params/1e6:.2f}M)")

    def _init(self, shape, scale):
        if HAS_NUMPY:
            return np.random.randn(*shape).astype(np.float32) * scale
        return [[random.gauss(0, scale) for _ in range(shape[1])] for _ in range(shape[0])]

    def softmax(self, x):
        if HAS_NUMPY:
            e = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e / e.sum(axis=-1, keepdims=True)
        max_val = max(x)
        e = [math.exp(v - max_val) for v in x]
        s = sum(e)
        return [v / s for v in e]

    def layer_norm(self, x, g, b, eps=1e-5):
        if HAS_NUMPY:
            m = x.mean(axis=-1, keepdims=True)
            v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
            return g * (x - m) / np.sqrt(v + eps) + b
        n = len(x)
        m = sum(x) / n
        v = sum((i - m) ** 2 for i in x) / n
        return [g[j] * (x[j] - m) / math.sqrt(v + eps) + b[j] for j in range(n)]

    def forward(self, tokens):
        seq_len = len(tokens)
        if HAS_NUMPY:
            x = self.embed[tokens] + self.pos_embed[:seq_len]
        else:
            x = [[self.embed[t][d] + self.pos_embed[p][d] for d in range(self.d_model)]
                 for p, t in enumerate(tokens)]

        for layer in self.layers:
            if HAS_NUMPY:
                q = x @ layer["wq"]
                k = x @ layer["wk"]
                v = x @ layer["wv"]
                scores = (q @ k.T) / math.sqrt(self.d_model)
                mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
                attn = self.softmax(scores + mask)
                out = (attn @ v) @ layer["wo"]
                x = self.layer_norm(x + out, layer["ln1_g"], layer["ln1_b"])
                h = np.maximum(x @ layer["w1"], 0)
                h = h @ layer["w2"]
                x = self.layer_norm(x + h, layer["ln2_g"], layer["ln2_b"])
            else:
                x = self.layer_norm(x, layer["ln1_g"], layer["ln1_b"])

        if HAS_NUMPY:
            return x @ self.output_proj
        return [[sum(x[t][d] * self.output_proj[d][v] for d in range(self.d_model))
                 for v in range(self.vocab_size)] for t in range(seq_len)]

    def log_probs(self, tokens):
        """Get log probabilities for next token predictions."""
        logits = self.forward(tokens)
        if HAS_NUMPY:
            lps = []
            for i in range(len(logits) - 1):
                lp = np.log(self.softmax(logits[i]) + 1e-10)
                lps.append(lp)
            return lps
        return []

    def generate(self, tokenizer, prompt, max_new=20, temperature=0.8):
        tokens = tokenizer.encode(prompt, self.max_len - max_new)
        for _ in range(max_new):
            logits = self.forward(tokens)
            next_logits = logits[-1]
            if HAS_NUMPY:
                next_logits = next_logits / temperature
                probs = self.softmax(next_logits)
                nxt = np.random.choice(len(probs), p=probs)
            else:
                nxt = random.randint(0, self.vocab_size - 1)
            if nxt == 3:
                break
            tokens.append(nxt)
        return tokenizer.decode(tokens)

    def save(self, path):
        data = {
            "config": (self.vocab_size, self.d_model, self.n_heads, self.n_layers, self.max_len),
            "embed": self.embed.tolist() if HAS_NUMPY else self.embed,
            "pos_embed": self.pos_embed.tolist() if HAS_NUMPY else self.pos_embed,
            "layers": self.layers,
            "output_proj": self.output_proj.tolist() if HAS_NUMPY else self.output_proj,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(*data["config"])
        model.embed = np.array(data["embed"], dtype=np.float32) if HAS_NUMPY else data["embed"]
        model.pos_embed = np.array(data["pos_embed"], dtype=np.float32) if HAS_NUMPY else data["pos_embed"]
        model.layers = data["layers"]
        model.output_proj = np.array(data["output_proj"], dtype=np.float32) if HAS_NUMPY else data["output_proj"]
        return model

# ==================== SDPO Training ====================

def load_data():
    """QA pairs with feedback hints."""
    data = [
        {"q": "capital of France", "a": "Paris", "hint": "Paris is the capital"},
        {"q": "author of Romeo", "a": "Shakespeare", "hint": "Shakespeare wrote it"},
        {"q": "2 plus 2", "a": "4", "hint": "basic arithmetic 4"},
        {"q": "color of sky", "a": "blue", "hint": "sky appears blue"},
        {"q": "planet we live", "a": "Earth", "hint": "we live on Earth"},
        {"q": "days in week", "a": "7", "hint": "7 days in week"},
        {"q": "largest ocean", "a": "Pacific", "hint": "Pacific is largest"},
        {"q": "painted Mona Lisa", "a": "Da Vinci", "hint": "Da Vinci painted it"},
        {"q": "what is H2O", "a": "water", "hint": "H2O is water"},
        {"q": "WW2 ended year", "a": "1945", "hint": "1945 WW2 ended"},
        {"q": "tallest mountain", "a": "Everest", "hint": "Everest is tallest"},
        {"q": "Brazil language", "a": "Portuguese", "hint": "Portuguese in Brazil"},
        {"q": "speed of light", "a": "fast", "hint": "light is very fast"},
        {"q": "discovered America", "a": "Columbus", "hint": "Columbus discovered"},
        {"q": "water boiling point", "a": "100", "hint": "boils at 100"},
    ] * 5
    random.shuffle(data)
    return data

def sdpo_train():
    """SDPO training: self-distillation with rich feedback."""
    print("=" * 50)
    print("SDPO: Self-Distillation Policy Optimization")
    print("=" * 50)

    os.makedirs("./sdpo-tiny-pure", exist_ok=True)
    data = load_data()

    # Build tokenizer
    tokenizer = TinyTokenizer()
    all_texts = []
    for item in data:
        all_texts.extend([
            f"q: {item['q']} a: {item['a']}",
            f"feedback: {item['hint']}",
        ])
    tokenizer.build_vocab(all_texts, max_vocab=1500)
    print(f"Vocab: {tokenizer.vocab_size}")

    # Create model
    model = TinyTransformer(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=2, n_layers=2, max_len=64)

    # SDPO training loop
    print("\n" + "=" * 50)
    print("SDPO Training")
    print("=" * 50)

    epochs = 3
    lr = 0.001

    for epoch in range(epochs):
        correct = 0
        total_loss = 0
        total = 0

        random.shuffle(data)

        for item in data:
            # Step 1: Generate attempt (student)
            prompt = f"q: {item['q']} a:"
            tokens = tokenizer.encode(prompt + " " + item['a'])

            if len(tokens) < 2:
                continue

            # Step 2: Get model's current prediction
            logits = model.forward(tokens)
            prompt_len = len(tokenizer.encode(prompt))

            if prompt_len >= len(logits):
                continue

            # Step 3: Check if correct (simulate environment feedback)
            target_token = tokenizer.word2idx.get(item['a'].lower().split()[0], 1)
            student_logits = logits[prompt_len - 1]

            if HAS_NUMPY:
                student_probs = model.softmax(student_logits)
                predicted = np.argmax(student_probs)
                is_correct = (predicted == target_token)

                # Step 4: SDPO self-teacher with feedback
                # Create feedback-conditioned "teacher" signal
                if is_correct:
                    # Correct: reinforce current prediction
                    correct += 1
                    grad = student_probs.copy()
                    grad[target_token] -= 1
                    grad = np.clip(grad, -0.5, 0.5)
                else:
                    # Wrong: use feedback as dense signal
                    # SDPO key insight: model can correct itself with feedback
                    feedback_tokens = tokenizer.encode(f"feedback: {item['hint']}")
                    feedback_logits = model.forward(feedback_tokens)
                    if len(feedback_logits) > 0:
                        # Use feedback logits to guide correction
                        feedback_probs = model.softmax(feedback_logits[-1])
                        # Self-teacher signal: blend with feedback
                        teacher_signal = 0.7 * feedback_probs + 0.3 * student_probs
                        grad = teacher_signal - student_probs
                        grad = np.clip(grad, -0.5, 0.5)
                    else:
                        grad = student_probs.copy()
                        grad[target_token] -= 1

                # Update output projection (SDPO gradient)
                model.output_proj[:, target_token] -= lr * grad[target_token]

                # Also update based on feedback condition
                if not is_correct:
                    # Extra update for wrong answers using self-distillation
                    feedback_target = tokenizer.word2idx.get(item['hint'].lower().split()[0], 1)
                    model.output_proj[:, feedback_target] -= lr * 0.5

                loss = -math.log(max(student_probs[target_token], 1e-10))
                total_loss += loss

            total += 1

        acc = correct / max(total, 1)
        avg_loss = total_loss / max(total, 1)
        print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.1%}")

    # Save
    model.save("./sdpo-tiny-pure/sdpo_tiny.pkl")
    print(f"\nSaved: ./sdpo-tiny-pure/sdpo_tiny.pkl")

    # Test
    print("\n" + "=" * 50)
    print("Test Generation")
    print("=" * 50)
    for q in ["capital of France", "author of Romeo", "color of sky"]:
        resp = model.generate(tokenizer, f"q: {q} a:", max_new=5)
        print(f"Q: {q}")
        print(f"A: {resp}\n")

    return model, tokenizer

def convert_to_gguf(model, tokenizer):
    """Convert to GGUF."""
    import struct

    output = "./sdpo-tiny-pure.gguf"

    print("=" * 50)
    print("Converting to GGUF")
    print("=" * 50)

    # Get weights
    if HAS_NUMPY:
        embed = model.embed
        pos = model.pos_embed
        out_proj = model.output_proj
    else:
        embed = np.array(model.embed, dtype=np.float32)
        pos = np.array(model.pos_embed, dtype=np.float32)
        out_proj = np.array(model.output_proj, dtype=np.float32)

    tensors = [
        ("token_embd.weight", embed),
        ("position_embd.weight", pos),
    ]

    for i, layer in enumerate(model.layers):
        p = f"blk.{i}"
        for k in layer:
            if HAS_NUMPY:
                w = layer[k]
            else:
                w = np.array(layer[k], dtype=np.float32)
            if k in ["wq", "wk", "wv", "wo", "w1", "w2"]:
                w = w.T
            name = {
                "ln1_g": f"{p}.ln1.weight", "ln1_b": f"{p}.ln1.bias",
                "ln2_g": f"{p}.ln2.weight", "ln2_b": f"{p}.ln2.bias",
                "wq": f"{p}.attn.q.weight", "wk": f"{p}.attn.k.weight",
                "wv": f"{p}.attn.v.weight", "wo": f"{p}.attn.o.weight",
                "w1": f"{p}.ffn_up.weight", "w2": f"{p}.ffn_down.weight",
            }.get(k, f"{p}.{k}")
            tensors.append((name, w.astype(np.float32)))

    tensors.append(("output.weight", out_proj.T.astype(np.float32)))

    # Write GGUF
    GGUF_MAGIC = 0x46475546
    GGUF_VERSION = 3
    GGML_F32 = 0
    GGML_STRING = 8
    GGML_UINT64 = 10

    metadata = [
        ("general.architecture", GGML_STRING, "gpt2"),
        ("general.name", GGML_STRING, "sdpo-tiny-pure"),
        ("gpt2.context_length", GGML_UINT64, model.max_len),
        ("gpt2.embedding_length", GGML_UINT64, model.d_model),
        ("gpt2.block_count", GGML_UINT64, model.n_layers),
        ("gpt2.head_count", GGML_UINT64, model.n_heads),
        ("gpt2.vocab_size", GGML_UINT64, tokenizer.vocab_size),
    ]

    with open(output, "wb") as f:
        # Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(tensors)))
        f.write(struct.pack('<Q', len(metadata)))

        # Metadata
        for k, vtype, val in metadata:
            f.write(struct.pack('<Q', len(k)))
            f.write(k.encode())
            f.write(struct.pack('<I', vtype))
            if vtype == GGML_STRING:
                encoded = val.encode()
                f.write(struct.pack('<Q', len(encoded)))
                f.write(encoded)
            else:
                f.write(struct.pack('<Q', val))

        # Tensor info + data
        alignment = 32
        header_end = f.tell()
        info_size = 0
        for name, data in tensors:
            info_size += 8 + len(name) + 4 + data.ndim * 8 + 4 + 8

        data_start = ((header_end + info_size + alignment - 1) // alignment) * alignment

        offset = 0
        for name, data in tensors:
            f.write(struct.pack('<Q', len(name)))
            f.write(name.encode())
            f.write(struct.pack('<I', data.ndim))
            for d in data.shape:
                f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', GGML_F32))
            f.write(struct.pack('<Q', data_start + offset))
            offset += data.nbytes

        f.write(b'\x00' * (data_start - f.tell()))

        for _, data in tensors:
            f.write(data.tobytes())

    size = os.path.getsize(output)
    print(f"Written: {output}")
    print(f"Size: {size/1024:.1f} KB")

if __name__ == "__main__":
    model, tokenizer = sdpo_train()
    convert_to_gguf(model, tokenizer)
    print("\n" + "=" * 50)
    print("Done!")
    print("  Model: ./sdpo-tiny-pure/sdpo_tiny.pkl")
    print("  GGUF:  ./sdpo-tiny-pure.gguf")
    print("=" * 50)
