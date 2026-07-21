"""Model card template for self-published AWQ checkpoints."""

GITHUB_URL = "https://github.com/jireh-father/fastserve"


def _comparison_table(comparison: dict | None) -> str:
    """3-way table: original (HF bf16) vs plain vLLM vs fastserve stack."""
    if not comparison:
        return ""
    c = comparison.get("configs", {})

    def cell(cfg, key, suffix=""):
        v = c.get(cfg, {}).get(key)
        return f"{v}{suffix}" if v is not None else "—"

    n = comparison.get("n", 30)
    spec = comparison.get("detected_spec", "ngram")
    return f"""
## Original vs vLLM vs fastserve

Same GSM8K prompts (n={n}), single-stream (batch-1) greedy decode, one A100-80GB.
"fastserve" = this AWQ checkpoint + speculative decoding (`{spec}`) on vLLM.
Memory = weights only (bf16 vs AWQ); vLLM's KV-cache budget is a separate knob.

| | Original (HF bf16) | vLLM (bf16) | **fastserve (AWQ+spec)** |
|---|---|---|---|
| GSM8K acc | {cell('original','acc')} | {cell('vllm','acc')} | **{cell('fastserve','acc')}** |
| Decode speed | {cell('original','tok_s',' tok/s')} | {cell('vllm','tok_s',' tok/s')} | **{cell('fastserve','tok_s',' tok/s')}** |
| Weights (VRAM) | {cell('original','weight_gib',' GiB')} | {cell('vllm','weight_gib',' GiB')} | **{cell('fastserve','weight_gib',' GiB')}** |
"""


def build_model_card(source_model: str, meta: dict, gate: dict, comparison: dict | None = None) -> str:
    q = meta["quant_config"]
    b = meta["baseline"]
    repo_id = "{this_repo_id}"  # substituted by the publisher
    return f"""---
base_model: {source_model}
tags:
- awq
- quantized
- fastserve
---

# {source_model.split("/")[-1]} — AWQ 4-bit

Auto-quantized from [`{source_model}`](https://huggingface.co/{source_model})
by **[fastserve]({GITHUB_URL})**'s self-quantization pipeline
(`publish/`) — so it can be trusted the way a random community AWQ requant
sometimes can't. Two community quants fastserve auto-detected during its own
benchmark run looped garbage tokens instead of answering; every checkpoint
published here passed an accuracy gate against its bf16 baseline first.

## Quantization

- Method: AWQ ({q["scheme"]}), group size {q["group_size"]}, via [llm-compressor](https://github.com/vllm-project/llm-compressor)
- Calibration: {q["calib_samples"]} samples from `{q["calib_dataset"]}`

## Validation (accuracy gate — this is why you can trust it)

| | bf16 baseline | this AWQ checkpoint |
|---|---|---|
| GSM8K accuracy (n={b["n"]}) | {b["acc"]} | {gate["quant_acc"]} |

Within {gate["max_drop"]} (absolute) of the bf16 baseline, <30% degenerate
(repeated-token loops). **A checkpoint that failed this gate would not have
been uploaded.**
{_comparison_table(comparison)}
## Serve it with fastserve

[fastserve]({GITHUB_URL}) auto-detects this checkpoint — point it at the
**original** model id and it finds this AWQ + wires up speculative decoding:

```bash
git clone {GITHUB_URL} && cd fastserve && ./install.sh

# serve an OpenAI-compatible API (auto-picks this AWQ checkpoint)
./fastserve serve {source_model}

# or benchmark the speedup vs the naive baseline
./fastserve bench {source_model} --compare-baseline
```

Then query it like any OpenAI endpoint:

```bash
curl localhost:8000/v1/completions \\
  -d '{{"model": "{repo_id}", "prompt": "Q: What is 17*4?\\nA:", "max_tokens": 64}}'
```

## Or serve directly with vLLM

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model {repo_id}
```

## License

Inherits the base model's license — see
[`{source_model}`](https://huggingface.co/{source_model}) for terms.
"""
