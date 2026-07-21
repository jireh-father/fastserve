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

## Honest limitations

- **Detection is best-effort, name-pattern-based.** It searches known naming
  conventions and publisher orgs; it will miss quantized/EAGLE checkpoints
  that don't follow them. `fastserve info` always shows you exactly what it
  found (or didn't) before you commit to running anything.
- **No quality gate is run automatically — and this is a real, not
  theoretical, risk.** A 15-model sweep (`benchmarks/RESULTS.md`) turned up
  a concrete case: the community AWQ quants fastserve auto-detected for
  both `google/gemma-2-2b-it` and `google/gemma-2-9b-it` are broken (loop a
  single token / short phrase instead of answering) — matching-name repos
  on the Hub aren't guaranteed to be correct. `fastserve info` always shows
  you which repo it picked; check the output quality yourself before
  trusting `--no-quant` off in production. Greedy speculative decoding
  itself is lossless by construction (it just changes wall-clock, not the
  distribution of accepted tokens) — it's specifically the pre-existing
  quantized checkpoint that can be silently bad.
- **Draft-model speculative decoding (a small same-family model as the
  draft) is not auto-detected** — only EAGLE-3 heads and n-gram. Pass
  `--no-spec` and configure it yourself via `--` passthrough args if you
  want that path.
- **Some checkpoints need `--trust-remote-code`** (custom tokenizer/model
  code, e.g. Kimi-Linear) — on by default since you already chose this
  model id; pass `--no-trust-remote-code` to refuse it.
- **The newest model releases are a moving target.** Checked against each
  vendor's actual latest release as of 2026-07: several (GLM-4.7-Flash,
  Llama-4-Scout, Kimi-Linear) fail for reasons outside fastserve's
  control — a vLLM AWQ+attention bug, a `transformers`/checkpoint config
  mismatch, and a custom-tokenizer import error against current
  `transformers`, respectively. `fastserve info` won't warn you about
  these ahead of time; if `serve`/`bench` fails, check
  `benchmarks/RESULTS.md`'s "Real limitations found" section for whether
  it's a known one.
- Validated across 25 models targeted / 21 producing a result, 9 families
  (Qwen2.5, Qwen3, Qwen3.5/3.6, Llama-3.x, Gemma-2/4, Mistral, Phi, Yi,
  DeepSeek-R1-Distill/0528, gpt-oss), 0.5B-72B, spanning both common
  2024-2025 models and each vendor's actual current release as of
  2026-07 — see `benchmarks/RESULTS.md` for the full tables, two real
  fastserve bugs found and fixed along the way (`detect()` returning dead/
  gated repos; missing `trust_remote_code` support), and methodology notes.

## Self-quantization (`publish/`)

Rather than trust whatever community AWQ repo `detect()` happens to find
(see the Gemma-2 case above), `publish/quantize.py` quantizes a model
ourselves, validates it against its own bf16 baseline, and only uploads it
if it passes:

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
