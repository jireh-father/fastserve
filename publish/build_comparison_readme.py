"""Assemble the consolidated 12-model comparison table for the GitHub README
from: the n=30 publish-gate accuracies, the fresh batch-1 speed run
(benchmarks/results/cmp_*.json), and model sizes (model_sizes.json).

Prints a markdown block. `--inject` rewrites the section between the
<!-- COMPARISON:START --> / END markers in ../README.md in place.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RESULTS = os.path.join(ROOT, "benchmarks", "results")

# base id -> (display, params_B, bf16 gate acc n=30, our AWQ gate acc n=30)
MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct", 0.49, 0.333, 0.333),
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct", 1.54, 0.567, 0.567),
    ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct", 7.62, 0.867, 0.867),
    ("Qwen/Qwen2.5-14B-Instruct", "Qwen2.5-14B-Instruct", 14.77, 0.967, 0.933),
    ("Qwen/Qwen2.5-32B-Instruct", "Qwen2.5-32B-Instruct", 32.76, 0.933, 0.933),
    ("Qwen/Qwen3-8B", "Qwen3-8B", 8.19, 0.833, 0.833),
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3", 7.25, 0.400, 0.333),
    ("google/gemma-2-2b-it", "gemma-2-2b-it", 2.61, 0.533, 0.467),
    ("google/gemma-2-9b-it", "gemma-2-9b-it", 9.24, 0.700, 0.633),
    ("01-ai/Yi-1.5-9B-Chat", "Yi-1.5-9B-Chat", 8.83, 0.433, 0.600),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "DeepSeek-R1-Distill-Qwen-7B", 7.62, 0.767, 0.867),
    ("deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "DeepSeek-R1-0528-Qwen3-8B", 8.19, 0.533, 0.733),
]


def load_speed(base):
    p = os.path.join(RESULTS, f"cmp_{base.split('/')[-1]}.json")
    if not os.path.exists(p):
        return {}
    c = json.load(open(p)).get("configs", {})
    return {k: c.get(k, {}).get("tok_s") for k in ("original", "vllm", "fastserve")}


def build() -> str:
    sizes = json.load(open(os.path.join(RESULTS, "model_sizes.json"))) if \
        os.path.exists(os.path.join(RESULTS, "model_sizes.json")) else {}

    lines = [
        "## Benchmark: original vs vLLM vs fastserve",
        "",
        "12 models self-quantized and published to "
        "[huggingface.co/seoilgun](https://huggingface.co/seoilgun) (see `publish/PUBLISHED.md`). "
        "One A100-80GB. **Accuracy** = GSM8K (n=30), greedy. **Speed** = single-stream "
        "(batch-1) decode tok/s, measured identically for all three. **Memory** = weights only "
        "(bf16 vs AWQ 4-bit); vLLM's KV-cache budget is a separate knob. "
        "*original* = naive HF-eager bf16 · *vLLM* = plain vLLM bf16 · "
        "*fastserve* = our AWQ checkpoint + speculative decoding on vLLM.",
        "",
        "| Model | Size | GSM8K acc (orig → ours) | Speed orig / vLLM / **fastserve** | Speedup | Mem bf16 → AWQ |",
        "|---|---|---|---|---|---|",
    ]
    for base, disp, pB, base_acc, our_acc in MODELS:
        sp = load_speed(base)
        o, v, fs = sp.get("original"), sp.get("vllm"), sp.get("fastserve")
        speed = f"{o or '—'} / {v or '—'} / **{fs or '—'}**"
        speedup = f"{round(fs / o, 1)}×" if (o and fs) else "—"
        sz = sizes.get(base, {})
        mem = f"{sz.get('bf16_gib','?')} → {sz.get('awq_gib','?')} GiB" if sz else "—"
        acc = f"{base_acc:.3f} → {our_acc:.3f}"
        lines.append(f"| [{disp}](https://huggingface.co/seoilgun/{disp}-AWQ) | {pB:.1f}B | {acc} | {speed} | {speedup} | {mem} |")

    lines += [
        "",
        "Speed = tok/s. Accuracy within the 10pp gate for every model (small deltas are n=30 noise; "
        "a few models score *higher* quantized — same noise). The two Gemma quants replace "
        "community AWQ repos that were **broken** (looped garbage, GSM8K 0.000) — see `benchmarks/RESULTS.md`.",
        "",
        _large_models_section(),
    ]
    return "\n".join(lines)


def _large_models_section() -> str:
    """Frontier models: what fits on 2xA100-80GB (160GB) and what doesn't."""
    return """### Frontier models — what actually fits on 2xA100-80GB

We tried the same 3-way on much bigger models. Most don't fit this hardware at
all (160GB total) — even at 4-bit — so the honest result is a feasibility
verdict, not a speed number:

| Model | Params | bf16 | 4-bit | Verdict on 2xA100-80GB |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B** | 36B (MoE) | ~67 GiB | ~23 GiB | ✅ **runs** — see below |
| Qwen3.5-122B-A10B | 125B (MoE) | ~233 GiB | ~58 GiB | ❌ bf16 won't fit; multimodal arch breaks the quantized text path |
| DeepSeek-V4-Flash | 158B | ~294 GiB | ~74 GiB | ❌ bf16 won't fit; no real AWQ checkpoint exists |
| GLM-5.2 | 753B | ~1.4 TiB | ~351 GiB | ❌ impossible — 4-bit alone is 2.2x the total VRAM |
| Kimi-K2.6 | ~1T | ~1.9 TiB | ~493 GiB | ❌ impossible — 3x the total VRAM |

**Qwen3.6-35B-A3B** (the one that fits), GSM8K n=8, single-stream:

| | Original (HF bf16, 1 GPU) | vLLM (bf16, **2 GPUs**) | fastserve (AWQ+ngram, **1 GPU**) |
|---|---|---|---|
| GSM8K acc | 1.00 | 0.875 | 0.875 |
| Decode speed | 12.1 tok/s | 144.9 tok/s | **106.9 tok/s** |
| Weights (VRAM) | 67 GiB | 67 GiB | **23 GiB** |
| GPUs needed | 1 | 2 | **1** |

fastserve runs this 36B model on a **single** GPU at ~9x the naive-HF speed and
1/3 the memory. Plain bf16 vLLM is faster in raw tok/s but needs **both** GPUs
(67 GiB weights leave no room for a KV cache on one). Note: fastserve's
auto-detected EAGLE-3 head for this model is incompatible and was skipped
(n-gram used instead) — a detection gap, not a quantization one."""


def main():
    block = build()
    if "--inject" in sys.argv:
        readme = os.path.join(ROOT, "README.md")
        text = open(readme).read()
        marker = re.compile(r"<!-- COMPARISON:START -->.*?<!-- COMPARISON:END -->", re.DOTALL)
        new = f"<!-- COMPARISON:START -->\n{block}\n<!-- COMPARISON:END -->"
        if marker.search(text):
            text = marker.sub(new, text)
        else:  # insert after the intro (before '## Install')
            text = text.replace("## Install", new + "\n\n## Install", 1)
        open(readme, "w").write(text)
        print("injected into README.md")
    else:
        print(block)


if __name__ == "__main__":
    main()
