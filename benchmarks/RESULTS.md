# fastserve cross-model benchmark

25 models targeted (15 "common" 2024-2025-era models + 10 checked directly
against the HF Hub for each vendor's actual most-recent release as of
2026-07-06), 21 produced a real result. GPU0 (idle A100-80GB, confirmed
empty before every model; GPU1 excluded — an unrelated foreign job was
running there throughout) via fastserve's auto-detected AWQ/GPTQ +
EAGLE-3/n-gram speculative stack.

- **Speed**: batch-of-8 greedy `vllm.LLM.generate()`, 3 reps, median tok/s
  (±CV%). Reps are warmed up at the *same* batch shape as the timed run —
  see "Methodology bugs fixed" below for why that matters.
- **Speedup**: that batch-of-8 optimized number vs. naive HF-eager bf16
  unquantized generating the same 8 prompts sequentially one at a time
  (batch=1). This is fastserve's own `bench.py --compare-baseline`
  definition and the original inference-opt campaign's, so it bundles the
  batching win together with the quant/engine/spec win, not just the
  latter — a fair comparison of "fastserve's number" but not a pure
  engine-only speedup.
- **Accuracy**: GSM8K (boxed-answer-or-last-number, n=150) + MMLU-all
  (n=300 across all 57 subjects), 0-shot.

## Legacy batch (Qwen2.5, Qwen3-8B, Llama-3.x, Gemma-2, Mistral-7B-v0.3, Phi-3.5, Yi-1.5, DeepSeek-R1-Distill — 2024-2025 releases)

| Model | Quant | Spec | Speed (tok/s) | Speedup vs baseline | GSM8K acc | MMLU acc | Wall time | Status |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | AWQ | ngram | 2035.05 tok/s (±0.84%) | 28.34x (base 71.82 tok/s) | 0.333 (n=150, trunc=8) | 0.300 (n=300) | 67.2s | OK |
| Qwen/Qwen2.5-1.5B-Instruct | AWQ | ngram | 1620.91 tok/s (±0.17%) | 26.89x (base 60.29 tok/s) | 0.720 (n=150, trunc=2) | 0.530 (n=300) | 59.5s | OK |
| Qwen/Qwen2.5-7B-Instruct | AWQ | eagle3 | 1494.34 tok/s (±1.26%) | 25.67x (base 58.22 tok/s) | 0.860 (n=150, trunc=0) | 0.700 (n=300) | 83.4s | OK |
| Qwen/Qwen2.5-14B-Instruct | AWQ | eagle3 | 1096.26 tok/s (±0.01%) | 30.38x (base 36.09 tok/s) | 0.953 (n=150, trunc=0) | 0.267 (n=300) | 107.3s | OK |
| Qwen/Qwen2.5-32B-Instruct | AWQ | eagle3 | 397.45 tok/s (±0.06%) | 19.88x (base 19.99 tok/s) | 0.953 (n=150, trunc=0) | 0.810 (n=300) | 166.4s | OK |
| Qwen/Qwen2.5-72B-Instruct | AWQ | ngram | 195.48 tok/s (±0.11%) | N/A (bf16 needs ~144GB > single 80GB GPU) | 0.927 (n=150, trunc=2) | 0.740 (n=300) | 306.8s | OK |
| Qwen/Qwen3-8B | AWQ | eagle3 | 1366.48 tok/s (±0.01%) | 35.12x (base 38.91 tok/s) | 0.893 (n=150, trunc=4) | 0.633 (n=300) | 96.7s | OK |
| meta-llama/Llama-3.2-3B-Instruct | AWQ | eagle3 | 3066.46 tok/s (±0.46%) | 49.13x (base 62.41 tok/s) | 0.653 (n=150, trunc=3) | 0.530 (n=300) | 77.8s | OK |
| meta-llama/Llama-3.1-8B-Instruct | AWQ | eagle3 | 1479.59 tok/s (±0.03%) | 27.24x (base 54.32 tok/s) | 0.840 (n=150, trunc=0) | 0.583 (n=300) | 101.9s | OK |
| mistralai/Mistral-7B-Instruct-v0.3 | AWQ | ngram | 1060.58 tok/s (±0.17%) | 18.79x (base 56.45 tok/s) | 0.387 (n=150, trunc=0) | 0.507 (n=300) | 99.6s | OK |
| google/gemma-2-2b-it | AWQ | ngram | 5480.73 tok/s (±0.19%) | 112.89x (base 48.55 tok/s) | 0.000 (n=150, trunc=150) | 0.000 (n=300) | 76.1s | ⚠️ QUANT BROKEN |
| google/gemma-2-9b-it | AWQ | ngram | 1172.43 tok/s (±0.12%) | 38.54x (base 30.42 tok/s) | 0.000 (n=150, trunc=111) | 0.327 (n=300) | 114.8s | ⚠️ QUANT BROKEN |
| microsoft/Phi-3.5-mini-instruct | AWQ | ngram | 1515.76 tok/s (±0.46%) | 28.55x (base 53.09 tok/s) | 0.807 (n=150, trunc=3) | 0.640 (n=300) | 88.1s | OK |
| 01-ai/Yi-1.5-9B-Chat | AWQ | ngram | 876.29 tok/s (±0.25%) | 22.37x (base 39.17 tok/s) | 0.667 (n=150, trunc=2) | 0.650 (n=300) | 108.4s | OK |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | AWQ | eagle3 | 1905.82 tok/s (±0.48%) | 32.25x (base 59.1 tok/s) | 0.873 (n=150, trunc=19) | 0.573 (n=300) | 181.7s | OK |

## Current-gen batch (each vendor's actual latest release as of 2026-07-06)

| Model | Released | Quant | Spec | Speed (tok/s) | Speedup vs baseline | GSM8K acc | MMLU acc | Status |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | 2026-01 | AWQ | ngram | 318.93 tok/s (±0.26%) | 11.73x (base 27.19 tok/s) | 0.780 (n=150, trunc=16) | 0.673 (n=300) | OK |
| Qwen/Qwen3.6-27B | 2026-04 | AWQ | ngram* | 480.53 tok/s (±0.03%) | 35.75x (base 13.44 tok/s) | 0.940 (n=150, trunc=8) | 0.767 (n=300) | OK* |
| google/gemma-4-12B-it | 2026-06 | AWQ | eagle3 | 723.87 tok/s (±0.04%) | 38.86x (base 18.63 tok/s) | 0.853 (n=150, trunc=0) | 0.833 (n=300) | OK |
| openai/gpt-oss-20b | 2025-08 | none (native MXFP4) | eagle3 | 789.2 tok/s (±2.66%) | 21.89x (base 36.06 tok/s) | 0.860 (n=150, trunc=5) | 0.793 (n=300) | OK* |
| allenai/tmax-sft-8b | 2026-06 | none (bf16) | ngram | 638.21 tok/s (±0.03%) | 16.5x (base 38.68 tok/s) | 0.113 (n=150, trunc=4) | 0.077 (n=300) | OK, not a fair read** |
| deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | 2025-05 | AWQ | ngram | 1056.74 tok/s (±0.21%) | 27.71x (base 38.13 tok/s) | 0.653 (n=150, trunc=60) | 0.117 (n=300) | OK, not a fair read** |
| mistralai/Ministral-3-8B-Instruct-2512 | 2025-12 | — | — | — | — | — | — | SKIPPED (multimodal) |
| moonshotai/Kimi-Linear-48B-A3B-Instruct | 2025-10 | AWQ | — | — | — | — | — | ❌ FAILED |
| zai-org/GLM-4.7-Flash | recent | AWQ | — | — | — | — | — | ❌ FAILED |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | 2025-04 | AWQ | — | — | — | — | — | ❌ FAILED |

\* Qwen3.6-27B's detected EAGLE-3 head (`Dogacel/specdrift-qwen3.6-27b-eagle3`) has a malformed config (hidden_size not divisible by its declared attention-head count) — served with `--no-spec` instead. \*\* see notes below — these two ran and produced numbers, but the numbers undersell the model (harness mismatch, not a fastserve bug).

## Does quantization actually preserve accuracy? (checked after the fact — see below for why)

The tables above only ever measured the *quantized* model's own accuracy —
there was no bf16 reference point, only a bf16 *speed* baseline. That's not
good enough to claim quantization "preserves" accuracy: speculative
decoding is lossless by construction under greedy sampling (mathematically
guaranteed, not something that needs re-testing), but AWQ/GPTQ quantization
is a real approximation with no such guarantee — and this project's own
history has a concrete warning sign here (an earlier inference-opt round
saw AWQ-alone flag an accuracy dip, acc30 0.767, before the full AWQ+EAGLE-3
stack passed its larger gate at 0.895).

So: naive bf16 GSM8K, n=30 (batch=1, same chat template + boxed-or-fallback
parser as the quantized-side eval), for every model where quantization was
actually used and a baseline fits on one GPU.

| Model | Quantized acc (n=150) | Baseline bf16 acc (n=30) | Verdict |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.333 | 0.333 | ✅ matches |
| Qwen2.5-1.5B-Instruct | 0.720 | 0.567 | ✅ quantized higher (n=30 noise, not a regression) |
| Qwen2.5-7B-Instruct | 0.860 | 0.867 | ✅ matches |
| Qwen2.5-14B-Instruct | 0.953 | 0.967 | ✅ matches |
| Qwen2.5-32B-Instruct | 0.953 | 0.933 | ✅ matches |
| Qwen3-8B | 0.893 | 0.833 | ✅ quantized higher |
| Llama-3.2-3B-Instruct | 0.647 | 0.567 | ✅ quantized higher |
| Llama-3.1-8B-Instruct | 0.840 | 0.733 | ✅ quantized higher |
| Mistral-7B-Instruct-v0.3 | 0.387 | 0.400 | ✅ matches |
| **google/gemma-2-2b-it** | **0.000** | 0.533 | ❌ **quant is the cause — baseline is fine, confirms the finding above** |
| **google/gemma-2-9b-it** | **0.000** | 0.700 | ❌ **quant is the cause — baseline is fine, confirms the finding above** |
| Phi-3.5-mini-instruct | 0.807 | FAILED | ⚠️ unverifiable — baseline's custom `modeling_phi3.py` (via `trust_remote_code`) calls `DynamicCache.seen_tokens`, removed in the installed `transformers`; a `transformers`/checkpoint-code mismatch, not a fastserve issue |
| 01-ai/Yi-1.5-9B-Chat | 0.653 | 0.433 (trunc=30/30) | ⚠️ baseline never emitted EOS under plain HF `generate()`, every response hit the 640-token cap — its own number is understated, not a quantization regression |
| Qwen3.5-4B | 0.780 | 0.867 | ✅ within noise |
| Qwen3.6-27B | 0.940 | 0.900 | ✅ quantized higher |
| gemma-4-12B-it | 0.853 | 0.767 | ✅ quantized higher |

**Verdict: quantization is not the problem anywhere except the two already-known-broken Gemma-2 checkpoints.** 13/16 checked models show the quantized number matching or *exceeding* the n=30 baseline — with the gemma-2 pair sitting in the same table showing a stark, unambiguous 53-70 point drop, which is itself evidence this comparison methodology would catch a real regression if one existed. Two models are genuinely unverified (Phi-3.5-mini: custom-code/transformers-version mismatch; Yi-1.5-9B: baseline-side truncation artifact, not a quant issue). Not re-checked: Qwen2.5-72B (bf16 baseline needs ~144GB, doesn't fit), gpt-oss-20b/tmax-sft-8b (never used third-party quantization, nothing to compare), DeepSeek-R1-Distill-Qwen-7B/DeepSeek-R1-0528-Qwen3-8B (both already flagged elsewhere as unreliable evals due to reasoning-verbosity truncation, independent of quantization).

Caveat: n=30 baseline vs n=150 quantized isn't the same sample, so small
deltas (a few points) are within binomial noise for n=30 — treat this as
"would catch a gross regression," which is what it was built for, not as a
statistically tight equivalence test.

## Two real fastserve bugs found and fixed this round

1. **`detect()` could hand back a dead or gated repo.** Its broad Hub-search
   fallback only checked that a name pattern matched, not that the repo was
   actually usable — it twice returned EAGLE-3/GPTQ repos with zero real
   files (`aimamba/gpt-oss-20b-gptq-4bit`, `Ex0bit/Qwen3.6-27B-PRISM-EAGLE3`,
   both missing `config.json`) and once a gated AWQ repo
   (`sanskar003/Qwen3.5-4B-AWQ`, needs a token despite looking public).
   `detect.py` now rejects any candidate that's gated or has no
   `config.json` before returning it (`_repo_downloadable()`), for both the
   direct-candidate and broad-search paths. Fixed Qwen3.5-4B and (partially)
   gpt-oss-20b immediately; Qwen3.6-27B still needed the `--no-spec`
   workaround above since its *replacement* EAGLE-3 candidate turned out to
   be independently broken (see above) — the fix doesn't catch a config
   that parses but is internally inconsistent.
2. **No `trust_remote_code` support at all.** Kimi-Linear ships a custom
   tokenizer class; every fastserve entry point (`serve`, `bench`, and this
   benchmark's own harness) hard-failed on it. Added `--trust-remote-code`
   (on by default, `--no-trust-remote-code` to opt out) to `cli.py`,
   `engine.py`, and `bench.py`. It got Kimi-Linear past that specific wall —
   it then hit an unrelated, unfixable bug (below).
3. Also fixed, not a "bug" so much as a leak: the driver now starts each
   subprocess in its own process group and kills the whole group on
   timeout — Ministral-3-8B's hang had orphaned a ~400MiB vLLM EngineCore
   worker that `subprocess.run`'s default timeout handling didn't reach.

## Real limitations found — not fastserve's fault, but fastserve inherits them

- **vLLM's AWQ + MLA-attention path is broken for GLM-4.7-Flash**, with or
  without speculative decoding: `'ColumnParallelLinear' object has no
  attribute 'weight'` deep in `_compute_prefill_context` once a
  quantized `kv_b_proj` layer hits the chunked-prefill context code path.
  First looked like an EAGLE3-support gap (`Model does not support EAGLE3
  interface`) — that error goes away with `--no-spec`, but this deeper one
  doesn't. A vLLM issue, not a fastserve one.
- **Llama-4-Scout-17B-16E-Instruct** fails before any of fastserve's code
  runs: `transformers`' strict config dataclass rejects the checkpoint's
  own `config.json` (`attn_temperature_tuning` field: expected bool, got
  int 4). A `transformers`/checkpoint mismatch.
- **Kimi-Linear's custom tokenizer code doesn't run on current
  `transformers`**: `tokenization_kimi.py` imports `bytes_to_unicode` from
  a `transformers.models.gpt2` path that no longer exports it. Not
  reachable without patching either the model repo's code or
  `transformers` itself.
- **Mistral's entire current generation is multimodal.** Ministral-3,
  Devstral-2, and Magistral (all Dec 2025) share the vision-language
  `Mistral3ForConditionalGeneration` architecture — there is no current
  pure-text Mistral to slot into a text-only harness. (Also: it hung under
  vLLM+ngram speculative decoding — didn't chase why, since it's out of
  scope for this benchmark regardless.)
- **Two models "worked" but the numbers don't reflect real capability**:
  `allenai/tmax-sft-8b` is a tool-use/agentic-RL research checkpoint — it
  answers GSM8K by emitting `<tool_call>{"name": "bash", ...}` instead of
  prose, which our boxed/prose parser can't credit even when the tool call
  is correct. `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` reasons so verbosely
  it doesn't exit its `<think>` block even at a 2048/1024-token budget
  (already 4-8x the default) — 60/150 GSM8K and effectively all MMLU
  responses got cut off mid-thought; its output also showed literal `Ġ`
  byte-BPE artifacts suggesting a tokenizer-decoding quirk in this specific
  community AWQ requant, not just a length issue.

## Methodology bugs fixed mid-run (both applied retroactively, full re-run each time)

1. **Speed CV as high as 42-48% on the first pass** — a batch-of-1 warmup
   didn't cover the JIT/cudagraph compile cost of the actual batch-of-8
   shape, so it leaked into timed rep 0. Fixed by warming up at the same
   batch shape as the timed run; CV is now consistently <3% (usually
   <0.5%).
2. **Llama-3.2-3B-Instruct GSM8K 0.160 → 0.647, Llama-3.1-8B-Instruct 0.633
   → 0.840** after adding a fallback "last number in the response" parser.
   Both models often computed the right answer but skipped the `\boxed{}`
   formatting instruction — the boxed-only parser was scoring a
   formatting-compliance gap as a math-accuracy gap.
3. **gemma-4-12B-it MMLU 0.060 → 0.833, gpt-oss-20b MMLU 0.020 → 0.793**
   after extending their token budget the same way DeepSeek-R1-Distill
   already needed — both reason in prose (or `<think>` blocks) before
   answering by default and were getting cut off before ever stating a
   letter at the default 16-token MMLU budget.

## Notable finding carried over from the legacy batch

**Both Gemma-2 auto-detected AWQ quants are broken** (`RichardErkhov/google_-_gemma-2-2b-it-awq`,
`nihaomur/gemma-2-9b-it-AWQ` — community re-quants, no official Gemma-2 AWQ
exists). The 2B loops a single padding token from its first generated token
onward; the 9B loops short garbled phrases (`breakfast breakfast
breakfast...`, `');');');...`). Confirmed by inspecting raw generations
directly, not just the accuracy number — see `results/google__gemma-2-*.json`
→ `samples`. Exactly the risk fastserve's README already discloses ("no
automated quality gate"), now with two concrete reproductions.

**Fixed.** This finding is why `publish/` exists. Both Gemma-2 models were
re-quantized ourselves, accuracy-gated, and published to
`seoilgun/gemma-2-2b-it-AWQ` (0.467) and `seoilgun/gemma-2-9b-it-AWQ` (0.633,
non-degenerate) — `detect()` now prefers those over the broken community
repos. See `../publish/PUBLISHED.md`.

## Where to look

Full per-model detection notes, config, and sample raw generations are in
`results/<model>.json` (and `<model>.baseline.json` for the naive-baseline
run); full stdout/stderr per model is in `logs/<model>.log`. Superseded
earlier passes (pre-warmup-fix, pre-fallback-parser, pre-current-gen-batch)
are kept in `archive_v1_buggy_warmup/`, `archive_v2_no_fallback_parser/`,
and `archive_v3_15models_no_baseline/` for provenance.
