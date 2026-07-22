# fastserve/publish — self-quantization pipeline

Quantize a model yourself, prove it didn't lose accuracy, and publish it to a
namespace you control — so fastserve serves *your* validated checkpoint
instead of whatever community requant happened to match the name. This exists
because two community AWQ quants fastserve auto-detected during its benchmark
run (`google/gemma-2-2b-it`, `google/gemma-2-9b-it`) were silently broken —
they looped garbage tokens instead of answering (see
`../benchmarks/RESULTS.md`).

## One command

```bash
# writes to $FASTSERVE_HF_NAMESPACE (default seoilgun), needs HF_TOKEN in .env
.venv/bin/python publish/quantize.py --model mistralai/Mistral-7B-Instruct-v0.3
.venv/bin/python publish/quantize.py --model Qwen/Qwen3-8B --dry-run   # gate only, no upload
.venv/bin/python publish/batch.py                                       # the whole curated list
```

## What it does

1. **Baseline** — bf16 GSM8K accuracy (n=30) of the source model (HF eager).
2. **Quantize** — AWQ 4-bit via
   [llm-compressor](https://github.com/vllm-project/llm-compressor),
   `ultrachat-200k` calibration.
3. **Gate** — reload the quantized checkpoint through vLLM, re-measure GSM8K.
   Publish only if accuracy is within `--max-drop` (default 0.10 absolute) of
   baseline **and** fewer than 30% of responses look degenerate (repeated-token
   loops — the exact failure mode of the broken community quants). A checkpoint
   that fails is not uploaded.
4. **Publish** — upload to `<namespace>/<model>-AWQ` with a model card that
   records both accuracy numbers.

Stages 1–2 (plain HF/transformers) and stage 3 (vLLM) run as separate
subprocesses because vLLM needs its own CUDA context.

## Why llm-compressor, not AutoAWQ

AutoAWQ is officially deprecated and keeps a hardcoded per-architecture wrapper
list — it can't even load a 2026 architecture like Qwen3.5
(`TypeError: qwen3_5 isn't supported yet.`). llm-compressor quantizes through
`transformers.AutoModelForCausalLM`, so it works on anything transformers
supports. Its output is a vLLM-native `compressed-tensors` checkpoint.

## Files

- `quantize.py` — orchestrator (runs the two stages, handles `.env`/caches)
- `quantize_local.py` — stage 1+2: baseline + llm-compressor quantize
- `validate_and_publish.py` — stage 3: vLLM gate + upload
- `batch.py` — run the curated model list, pruning disk between models
- `model_card.py` — the uploaded README template

## Known gaps

- **Multimodal models** (a real `vision_config` even if you only want the text
  side — e.g. Qwen3.5/3.6). The pipeline now *detects* these and loads them via
  `AutoModelForImageTextToText`, ignoring the vision tower, so the save keeps the
  full multimodal config and the result loads in vLLM (earlier it saved a
  text-only config vLLM rejected). It also auto-ignores sensitive recurrent /
  linear-attention blocks (`linear_attn`, `mamba`, `conv1d`) that otherwise
  degenerate under quantization.
- **Very large expert-count MoE (e.g. Qwen3.6-35B-A3B, 256 experts) + hybrid
  linear-attention** is the one combination this pipeline can't do well: the
  *fast* method (RTN W8A8) is too crude and produces degenerate output even with
  the recurrent layers ignored, while the *quality* methods (AWQ / GPTQ) run a
  per-expert search → **hours** across 256 experts. For these, use a
  purpose-built quant (e.g. the AMD-Quark community W8A8, ~121 tok/s on A100).
- **Gated bases** (Llama, some Gemma) need the license accepted on the account
  behind `HF_TOKEN`, or the baseline download 403s.
- **Models too big for a bf16 baseline on one GPU** (e.g. Qwen2.5-72B) can't be
  gated this way — there's no accuracy reference to compare against.
- **Custom-modeling-code models** whose code is stale against the installed
  transformers (e.g. Phi-3.5-mini's `DynamicCache.seen_tokens`) fail the
  HF-eager baseline.
