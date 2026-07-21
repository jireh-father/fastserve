"""Model card template for self-published AWQ checkpoints."""


def build_model_card(source_model: str, meta: dict, gate: dict) -> str:
    q = meta["quant_config"]
    b = meta["baseline"]
    return f"""---
base_model: {source_model}
tags:
- awq
- quantized
- fastserve
---

# {source_model.split("/")[-1]} — AWQ 4-bit

Auto-quantized from [`{source_model}`](https://huggingface.co/{source_model})
by [fastserve](https://github.com/)'s self-quantization pipeline
(`fastserve/publish/`), so it can be trusted the way a random community AWQ
requant sometimes can't — two of the community quants fastserve auto-detected
during its own benchmark run turned out to loop garbage tokens instead of
answering (see `fastserve/benchmarks/RESULTS.md`). Every checkpoint here
passed the same accuracy gate before being uploaded.

## Quantization

- Method: AWQ ({q["scheme"]}), group size {q["group_size"]}, via [llm-compressor](https://github.com/vllm-project/llm-compressor)
- Calibration: {q["calib_samples"]} samples from `{q["calib_dataset"]}`

## Validation (accuracy gate — this is why you can trust it)

| | bf16 baseline | this AWQ checkpoint |
|---|---|---|
| GSM8K accuracy (n={b["n"]}) | {b["acc"]} | {gate["quant_acc"]} |

Passed: quantized accuracy is within {gate["max_drop"]} (absolute) of the
bf16 baseline, and less than 30% of responses looked degenerate (repeated-
token loops — the failure mode found in the broken community quants above).
**A checkpoint that failed this gate would not have been uploaded.**

## Use

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model {{this_repo_id}}
```

Or with [fastserve](https://github.com/) — point it at the *original*
`{source_model}` id and it'll find this checkpoint automatically once
`{{this_repo_id}}`'s namespace is registered as a priority search location.

## License

Inherits the base model's license — see
[`{source_model}`](https://huggingface.co/{source_model}) for terms.
"""
