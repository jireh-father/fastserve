# fastserve

**Zero-config LLM speedup.** Point it at any Hugging Face model ID; it
auto-detects the fastest available serving stack — pre-quantized weights,
speculative decoding — and either serves it as an OpenAI-compatible API or
benchmarks it against a naive baseline so you get a *measured*, not assumed,
speedup number.

Distilled from a real research campaign that took `Qwen/Qwen3-8B` from
277s → 28s (**9.85×**) end-to-end at *better* quality than the naive
baseline, purely from serving-stack changes — no retraining, no distillation.
Then validated across 25 models / 9 families, 0.5B–72B — see
[`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

<!-- COMPARISON:START -->
## Benchmark: fastserve vs the naive baseline

12 models self-quantized and published to [huggingface.co/seoilgun](https://huggingface.co/seoilgun) (see `publish/PUBLISHED.md`). One A100-80GB. **Accuracy** = GSM8K (n=30), greedy. **Speed** = single-stream (batch-1) decode tok/s. **Memory** = weights (bf16 vs AWQ 4-bit). *original* = naive HF-eager bf16 (what you get running a model out of the box) · *fastserve* = its auto-detected AWQ checkpoint + speculative decoding on vLLM.

| Model | Size | GSM8K acc (orig → fastserve) | Speed orig → **fastserve** | Speedup | Mem bf16 → AWQ |
|---|---|---|---|---|---|
| [Qwen2.5-0.5B-Instruct](https://huggingface.co/seoilgun/Qwen2.5-0.5B-Instruct-AWQ) | 0.5B | 0.333 → 0.333 | 70.9 → **437.8** | **6.2×** | 0.92 → 0.68 GiB |
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/seoilgun/Qwen2.5-1.5B-Instruct-AWQ) | 1.5B | 0.567 → 0.567 | 59.6 → **305.9** | **5.1×** | 2.88 → 1.5 GiB |
| [Qwen2.5-7B-Instruct](https://huggingface.co/seoilgun/Qwen2.5-7B-Instruct-AWQ) | 7.6B | 0.867 → 0.867 | 58.2 → **466.5** | **8.0×** | 14.19 → 5.19 GiB |
| [Qwen2.5-14B-Instruct](https://huggingface.co/seoilgun/Qwen2.5-14B-Instruct-AWQ) | 14.8B | 0.967 → 0.933 | 35.3 → **281.7** | **8.0×** | 27.51 → 9.29 GiB |
| [Qwen2.5-32B-Instruct](https://huggingface.co/seoilgun/Qwen2.5-32B-Instruct-AWQ) | 32.8B | 0.933 → 0.933 | 19.8 → **114.1** | **5.8×** | 61.03 → 18.0 GiB |
| [Qwen3-8B](https://huggingface.co/seoilgun/Qwen3-8B-AWQ) | 8.2B | 0.833 → 0.833 | 37.3 → **323.0** | **8.7×** | 15.26 → 5.68 GiB |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/seoilgun/Mistral-7B-Instruct-v0.3-AWQ) | 7.2B | 0.400 → 0.333 | 56.1 → **212.7** | **3.8×** | 27.0 → 3.88 GiB |
| [gemma-2-2b-it](https://huggingface.co/seoilgun/gemma-2-2b-it-AWQ) | 2.6B | 0.533 → 0.467 | 47.1 → **259.3** | **5.5×** | 4.87 → 3.18 GiB |
| [gemma-2-9b-it](https://huggingface.co/seoilgun/gemma-2-9b-it-AWQ) | 9.2B | 0.700 → 0.633 | 29.7 → **147.5** | **5.0×** | 17.21 → 7.45 GiB |
| [Yi-1.5-9B-Chat](https://huggingface.co/seoilgun/Yi-1.5-9B-Chat-AWQ) | 8.8B | 0.433 → 0.600 | 38.6 → **173.2** | **4.5×** | 16.45 → 5.0 GiB |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/seoilgun/DeepSeek-R1-Distill-Qwen-7B-AWQ) | 7.6B | 0.767 → 0.867 | 59.3 → **343.7** | **5.8×** | 14.19 → 5.19 GiB |
| [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/seoilgun/DeepSeek-R1-0528-Qwen3-8B-AWQ) | 8.2B | 0.533 → 0.733 | 37.6 → **194.7** | **5.2×** | 15.26 → 5.68 GiB |

Speed = tok/s. **fastserve is 3.8-8.7x faster than out-of-the-box serving, at ~3x less memory**, with accuracy held inside a 10pp gate (small deltas are n=30 noise; several models score *higher* quantized). The two Gemma quants replace community AWQ repos that were **broken** — looping garbage, GSM8K 0.000 — which is why `publish/` gates every checkpoint on accuracy before uploading it.

### Frontier models — what actually fits on 2xA100-80GB

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

**Qwen3.6-35B-A3B** (the one that fits) — fastserve puts a 36B model on a
**single** GPU, GSM8K n=8, single-stream:

| | Original (HF bf16) | **fastserve (AWQ)** |
|---|---|---|
| GSM8K acc | 1.00 | 0.875 |
| Decode speed | 12.1 tok/s | **106.9 tok/s** (8.8x) |
| Weights (VRAM) | 67 GiB | **23 GiB** |
| GPUs needed | 1 (no KV room left) | **1** |

fastserve runs this 36B MoE at ~9x out-of-the-box speed on a single 80GB GPU
using 1/3 the memory — the un-quantized model's 67 GiB of weights leave no room
for a KV cache on one GPU, so fastserve's AWQ is what makes single-GPU serving
practical here at all.
<!-- COMPARISON:END -->

## Install

One command:

```bash
./install.sh
```

This creates an isolated `.venv`, installs vLLM (which pulls in a matching
torch/CUDA build for you), installs fastserve itself, and drops a `./fastserve`
shim in this directory so you never have to activate the venv by hand.

Requires: Python 3.9+, a CUDA GPU. That's it.

## Use

```bash
# See what fastserve would do, without running anything
./fastserve info Qwen/Qwen3-8B

# Quick throughput benchmark of the auto-detected stack
./fastserve bench Qwen/Qwen3-8B

# ...and compare against the naive baseline for a real speedup number (slower, downloads
# the model twice: once via vLLM, once via plain transformers)
./fastserve bench Qwen/Qwen3-8B --compare-baseline

# Launch an OpenAI-compatible API server on port 8000
./fastserve serve Qwen/Qwen3-8B
curl localhost:8000/v1/completions -d '{"model": "...", "prompt": "Hello,"}'
```

## What it actually does

For a given model ID, fastserve searches the Hugging Face Hub for:

1. **A pre-quantized checkpoint** (AWQ or GPTQ) of the same model, under the
   same org or common community-requant naming patterns. If found, it serves
   the quantized weights instead of full precision — same model, much less
   memory bandwidth per token.
2. **A published EAGLE-3 speculative-decoding draft head** (checked against
   known publishers: AngelSlim, yuhuili, SpecForge, NousResearch). If found,
   it's wired in as `speculative_config`, letting the engine emit more than
   one token per forward pass.
3. If no EAGLE-3 head exists, it **falls back to n-gram (prompt-lookup)
   speculative decoding** — zero extra downloads, still a free win on any
   workload with repeated substrings (code, tool calls, long context).

Everything runs on **vLLM**, chosen because in the underlying research
campaign the serving engine itself — not just the quantization format —
turned out to be the dominant factor in real speedup: an equivalently-
quantized model served through a less-optimized engine measured 3.77×
*slower* than the vLLM stack despite identical bit-width.

## Self-quantization (`publish/`)

Rather than trust whatever community AWQ repo `detect()` happens to find —
two community Gemma-2 quants turned out to be broken, looping garbage at
GSM8K 0.000 — `publish/quantize.py` quantizes a model ourselves, validates
it against its own bf16 baseline, and only uploads it if it passes:

```bash
.venv/bin/python publish/quantize.py --model mistralai/Mistral-7B-Instruct-v0.3
```

Pipeline: bf16 GSM8K baseline (n=30) → quantize with
[llm-compressor](https://github.com/vllm-project/llm-compressor)'s
AWQModifier (4-bit, `ultrachat-200k` calibration) → re-measure GSM8K on the
quantized checkpoint via vLLM → **publish only if accuracy is within
`--max-drop` (default 10pp) of baseline and <30% of responses look
degenerate** → uploads to `seoilgun/<model>-AWQ`. `detect.py` checks that
namespace before any community repo, so once published, `fastserve serve
<original-model-id>` picks it up automatically with zero flags.

Uses llm-compressor rather than AutoAWQ (which this project used
everywhere else) specifically because AutoAWQ is deprecated and maintains a
hardcoded per-architecture wrapper list — it can't even load a brand-new
architecture like Qwen3.5 (`TypeError: qwen3_5 isn't supported yet.`).
llm-compressor quantizes through plain `transformers.AutoModelForCausalLM`,
so it works on anything `transformers` itself supports.

**Known gap**: models that are natively multimodal even when you only want
their text ability (confirmed: Qwen3.5-4B has a real `vision_config`) save
out with a text-only sub-config that breaks vLLM's multimodal wrapper for
that architecture. Stick to checkpoints with no `vision_config` in
`config.json` until this pipeline handles that case properly.

## Project layout

```
fastserve/
├── install.sh              one-command setup
├── src/fastserve/
│   ├── detect.py            HF Hub auto-detection (quant + EAGLE-3)
│   ├── engine.py            vLLM command/config assembly
│   ├── bench.py             throughput measurement
│   └── cli.py               info / serve / bench subcommands
├── benchmarks/              cross-model speed+accuracy sweep (see RESULTS.md)
├── publish/                 self-quantize + accuracy-gate + upload to seoilgun/
└── pyproject.toml
```
