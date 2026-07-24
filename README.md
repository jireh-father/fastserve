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
## Benchmark: original vs vLLM vs fastserve

12 models self-quantized and published to [huggingface.co/glenic](https://huggingface.co/glenic) (see `publish/PUBLISHED.md`). One A100-80GB. **Accuracy** = GSM8K (n=30), greedy. **Speed** = single-stream (batch-1) decode, tok/s. **Memory** = weights (bf16 vs AWQ 4-bit). *original* = naive HF-eager bf16 · *vLLM* = plain vLLM bf16 · *fastserve* = its auto-detected AWQ checkpoint + speculative decoding on vLLM.

| Model | Size | GSM8K acc (orig → fastserve) | Speed, tok/s (orig / vLLM / **fastserve**) | Speedup vs orig | Mem bf16 → AWQ |
|---|---|---|---|---|---|
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/glenic/Qwen2.5-1.5B-Instruct-AWQ) | 1.5B | 0.567 → 0.567 | 59.6 / 259.6 / **305.9** | **5.1×** | 2.88 → 1.5 GiB |
| [Qwen2.5-7B-Instruct](https://huggingface.co/glenic/Qwen2.5-7B-Instruct-AWQ) | 7.6B | 0.867 → 0.867 | 58.2 / 98.1 / **466.5** | **8.0×** | 14.19 → 5.19 GiB |
| [Qwen2.5-14B-Instruct](https://huggingface.co/glenic/Qwen2.5-14B-Instruct-AWQ) | 14.8B | 0.967 → 0.933 | 35.3 / 51.8 / **281.7** | **8.0×** | 27.51 → 9.29 GiB |
| [Qwen2.5-32B-Instruct](https://huggingface.co/glenic/Qwen2.5-32B-Instruct-AWQ) | 32.8B | 0.933 → 0.933 | 19.8 / 24.0 / **114.1** | **5.8×** | 61.03 → 18.0 GiB |
| [Qwen3-8B](https://huggingface.co/glenic/Qwen3-8B-AWQ) | 8.2B | 0.833 → 0.833 | 37.3 / 91.4 / **323.0** | **8.7×** | 15.26 → 5.68 GiB |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/glenic/Mistral-7B-Instruct-v0.3-AWQ) | 7.2B | 0.400 → 0.333 | 56.1 / 96.8 / **212.7** | **3.8×** | 27.0 → 3.88 GiB |
| [gemma-2-2b-it](https://huggingface.co/glenic/gemma-2-2b-it-AWQ) | 2.6B | 0.533 → 0.467 | 47.1 / 194.8 / **259.3** | **5.5×** | 4.87 → 3.18 GiB |
| [gemma-2-9b-it](https://huggingface.co/glenic/gemma-2-9b-it-AWQ) | 9.2B | 0.700 → 0.633 | 29.7 / 70.8 / **147.5** | **5.0×** | 17.21 → 7.45 GiB |
| [Yi-1.5-9B-Chat](https://huggingface.co/glenic/Yi-1.5-9B-Chat-AWQ) | 8.8B | 0.433 → 0.600 | 38.6 / 77.5 / **173.2** | **4.5×** | 16.45 → 5.0 GiB |
| [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/glenic/DeepSeek-R1-Distill-Qwen-7B-AWQ) | 7.6B | 0.767 → 0.867 | 59.3 / 98.2 / **343.7** | **5.8×** | 14.19 → 5.19 GiB |
| [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/glenic/DeepSeek-R1-0528-Qwen3-8B-AWQ) | 8.2B | 0.533 → 0.733 | 37.6 / 90.6 / **194.7** | **5.2×** | 15.26 → 5.68 GiB |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) † | 36B (3B act) | 0.933 → 0.875 | 12.1 / 14.1 / **106.9** | **8.8×** | 67 → 23 GiB |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) ‡ | 125B (10B act) | — → 0.875 | — / — / **77** | — | 233 → 77 GiB |

Speed in tok/s. **fastserve is 3.8-8.7x faster than out-of-the-box serving and beats plain vLLM on every model here, at ~3x less memory** — accuracy held inside a 10pp gate (small deltas are n=30 noise; several models score *higher* quantized). The two Gemma quants replace community AWQ repos that were **broken** — looping garbage, GSM8K 0.000 — which is why `publish/` gates every checkpoint on accuracy before uploading it.

† **Qwen3.6-35B-A3B** — single GPU. Its bf16 vLLM number (14.1) is eager-only: at 67 GiB the weights leave no room for CUDA graphs on one card (see below). AWQ here is the community `cyankiwi` quant; a community W8A8-INT8 reaches ~121 tok/s.  ‡ **Qwen3.5-122B-A10B** — needs **2 GPUs**; its bf16 (233 GiB) doesn't fit even two 80GB cards, so there's no original/vLLM baseline — AWQ (community `QuantTrio`) is the only way it runs, at 77 tok/s across TP=2.

### Frontier models on 2xA100-80GB

We pushed the same AWQ-on-vLLM approach up to genuinely large models. All
numbers below are **measured on this hardware**, not estimated:

| Model | Params | AWQ size | On 2xA100-80GB (measured) |
|---|---|---|---|
| **Qwen3.6-35B-A3B** | 36B (MoE) | 23 GiB | ✅ **1 GPU**, GSM8K 0.875, **106.9 tok/s** (8.8x over bf16) — see below |
| **Qwen3.5-122B-A10B** | 125B (MoE) | 77 GiB | ✅ **runs on TP=2**, GSM8K 0.875, 77 tok/s, 36 GiB/GPU — loads clean on stock vLLM |
| DeepSeek-V4-Flash | 158B | ~98 GiB | ⚠️ loads on TP=2 and answers coherently, but its sparse-MLA attention is **Hopper-only** — on A100 it's eager-only, ~6 tok/s, and crashes at long decode. Proof-of-load, not a usable deployment. |
| GLM-5.2 | 753B | ~351 GiB | ❌ doesn't fit — 4-bit alone is 2.2x the 160 GiB total |
| Kimi-K2.6 | ~1T | ~493 GiB | ❌ doesn't fit — 4-bit is 3x the total |

Two things worth calling out. **Qwen3.5-122B-A10B only runs *because* of the
quantization** — its bf16 weights are ~233 GiB, which doesn't fit 160 GiB of
VRAM, so there is no un-optimized baseline to compare against: AWQ is the
difference between "can't load" and "77 tok/s." DeepSeek-V4-Flash physically
loads but its attention kernels need Hopper, so A100 only gets a slow eager
fallback. GLM-5.2 and Kimi-K2.6 are simply too big for this box even at 4-bit —
they'd need ~8xH100 and up.

**Qwen3.6-35B-A3B**, all on a **single 80GB GPU** (GSM8K n=8, single-stream decode):

| Config | Speed | Weights |
|---|---|---|
| Original (HF-eager bf16) | 12.1 tok/s | 67 GiB |
| plain vLLM bf16 | 14.1 tok/s | 67 GiB |
| **fastserve AWQ** | **106.9 tok/s** | **23 GiB** |
| W8A8-INT8 (community quant) | 121.3 tok/s | ~35 GiB |

The interesting part: bf16 **does** fit on a single 80GB card (weights are
65.5 GiB) — but it fills the GPU too full to capture **CUDA graphs**, and CUDA
graphs are where most of vLLM's speed comes from. So single-GPU bf16 is stuck
in eager mode at ~14 tok/s, barely above raw HF.

Quantization is what breaks that: shrinking the weights (AWQ 23 GiB, W8A8
~35 GiB) leaves plenty of room to run CUDA graphs on one card — which is why
fastserve hits ~107 tok/s (AWQ) where bf16 can only do 14 — **~8x faster than
plain vLLM on the same single GPU**. A community W8A8-INT8 checkpoint goes a bit
further still (121 tok/s), since A100's INT8 tensor cores skip the 4-bit
dequant. (fastserve auto-serves the AWQ; the W8A8 number is a measured
community quant — see the note below on why `publish/` can't re-make it here.)

### These are A100 numbers — H100/H200 shifts them

- **W8A8-INT8 beats AWQ here (121 vs 107)** because A100 has INT8 tensor cores
  but *no* FP8. On **H100/H200** you'd use **FP8** instead (native fast path),
  and the format ranking changes.
- The "bf16 can't fit CUDA graphs on one card" problem is an **80 GiB** limit;
  on **H200 (141 GiB)** bf16 + graphs fit on one GPU comfortably, so that gap
  mostly closes. H100 (80 GiB) is as tight as A100.
- Models with **Hopper-only kernels** (DeepSeek-V4-Flash's sparse-MLA) run
  fully on H100 but only get a slow eager fallback on A100.
- H100 is ~2-3x faster raw, so every absolute number goes up.

### Newer frontier models — self-quantized & published (2026)

Six more recent releases, each quantized with the format that suits its
architecture and published to [glenic](https://huggingface.co/glenic) after
passing the same GSM8K gate. One A100-80GB; GSM8K n=15, greedy.

| Model | Params | Format | GSM8K (bf16 → quant) | Repo |
|---|---|---|---|---|
| [Qwen3.6-27B](https://huggingface.co/glenic/Qwen3.6-27B-AWQ) | 27B dense | **AWQ 4-bit** | 0.80 → 0.80 | `glenic/Qwen3.6-27B-AWQ` |
| [gemma-4-31B-it](https://huggingface.co/glenic/gemma-4-31B-it-W8A8-INT8) | 33B dense (omni) | W8A8-INT8 | 0.80 → 0.80 | `glenic/gemma-4-31B-it-W8A8-INT8` |
| [gemma-4-26B-A4B-it](https://huggingface.co/glenic/gemma-4-26B-A4B-it-W8A8-INT8) | 26B MoE, 128 experts | W8A8-INT8 | 0.53 → 0.60 | `glenic/gemma-4-26B-A4B-it-W8A8-INT8` |
| [gemma-4-12B-it](https://huggingface.co/glenic/gemma-4-12B-it-W8A8-INT8) | 12B dense (omni) | W8A8-INT8 | 0.73 → 0.67 | `glenic/gemma-4-12B-it-W8A8-INT8` |
| [gemma-4-E4B-it](https://huggingface.co/glenic/gemma-4-E4B-it-W8A8-INT8) | 8B elastic | W8A8-INT8 | 0.47 → 0.53 | `glenic/gemma-4-E4B-it-W8A8-INT8` |
| [gemma-4-E2B-it](https://huggingface.co/glenic/gemma-4-E2B-it-W8A8-INT8) | 5B elastic | W8A8-INT8 | 0.47 → 0.47 | `glenic/gemma-4-E2B-it-W8A8-INT8` |

**Format is chosen per architecture, not one-size-fits-all.** Qwen3.x keeps
**4-bit AWQ** — llm-compressor's dynamic, hybrid-attention-aware mappings resolve
it cleanly. Gemma-4 uses **W8A8-INT8**: on A100 (INT8 tensor cores, no FP8) it's
the faster format, and it avoids AWQ's per-layer smooth mappings, which aren't
registered for Gemma's multimodal wrapper. The pipeline auto-keeps at full
precision the parts quantization would break — the **vision and audio towers** on
the omni models, and the **MoE router**: INT8-ing the 128-expert router collapses
routing into repeated-token garbage (GSM8K 0.53 → 0.00) until it's protected.

**Couldn't be published from this box** (documented, not a fastserve limitation):

| Model | Why not |
|---|---|
| Kimi-Linear-48B-A3B | custom tokenizer import + 256-expert MoE + linear attention — the one combination the pipeline can't quantize cleanly (same class as Qwen3.6-35B) |
| Leanstral-1.5-119B-A6B | bf16 weights are ~238 GiB — can't be loaded to quantize on 2×80 GiB, and there's no baseline to gate against |
| gpt-oss-120b / gpt-oss-20b | already ship **native MXFP4** — they're 4-bit out of the box, so re-quantizing them gains nothing |
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
degenerate** → uploads to `glenic/<model>-AWQ`. `detect.py` checks that
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
├── publish/                 self-quantize + accuracy-gate + upload to glenic/
└── pyproject.toml
```
