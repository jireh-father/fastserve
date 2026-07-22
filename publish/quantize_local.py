"""Stage 1 of the self-quantization pipeline: bf16 baseline GSM8K accuracy,
then quantize with llm-compressor's AWQModifier (4-bit, group-wise, produces
a vLLM-native `compressed-tensors` checkpoint — functionally the same thing
AutoAWQ produces, but not limited to a hardcoded per-architecture wrapper
list. AutoAWQ is deprecated and confirmed unable to even load brand-new 2026
architectures like Qwen3.5 (`TypeError: qwen3_5 isn't supported yet.`);
llm-compressor uses transformers' AutoModelForCausalLM directly so it works
for anything transformers itself supports).

Both stages share one CUDA context safely (plain HF/transformers-based,
unlike vLLM which needs its own process — see stage 2, validate_and_publish.py).

Writes the quantized checkpoint plus a `_fastserve_quant_meta.json` (source
model, baseline accuracy, quant config) to --out-dir. Does NOT publish
anything — that only happens in stage 2, after the quantized model passes
its own accuracy check.

Known gap: models that are natively multimodal (vision-language) even when
you only care about their text ability — confirmed with Qwen3.5-4B — save
out with a text-only sub-config that vLLM's multimodal wrapper for that
architecture then rejects. Stick to confirmed text-only architectures
(check `config.json` has no `vision_config`) until that's handled properly.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from eval_tasks import is_long_thinker, run_hf_eager_gsm8k  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="source model id to quantize")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=30, help="baseline GSM8K question count")
    ap.add_argument("--w-bit", type=int, default=4)
    ap.add_argument("--q-group-size", type=int, default=128)
    ap.add_argument("--method", choices=["awq", "w8a8"], default="awq",
                     help="awq = W4A16 AWQ (default); w8a8 = INT8 weights+acts via GPTQ "
                          "(A100 INT8 tensor cores, no 4-bit dequant — faster on A100 for MoE)")
    ap.add_argument("--calib-samples", type=int, default=256)
    ap.add_argument("--calib-seq-len", type=int, default=512)
    # ultrachat-200k is chat-formatted (best for instruct models) but has
    # `system`-role messages — some chat templates (Gemma) reject those
    # (`jinja2 TemplateError: System role not supported`). For those, use a
    # raw-text set like wikitext (no chat template applied at all — also the
    # AWQ paper's own calibration style).
    ap.add_argument("--calib-dataset", default="ultrachat-200k")
    ap.add_argument("--calib-split", default=None,
                     help="dataset split expr; defaults per dataset")
    ap.add_argument("--calib-config", default=None,
                     help="dataset config name (e.g. wikitext-2-raw-v1)")
    args = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    long_thinker = is_long_thinker(args.model)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Multimodal models (Qwen3.5/3.6 = *ForConditionalGeneration with a vision
    # tower) must be quantized *as* the multimodal model — quantizing only the
    # text tower via AutoModelForCausalLM saves a text-only config that vLLM's
    # multimodal loader then rejects (`Expected Qwen3_5MoeConfig, found
    # Qwen3_5MoeTextConfig`). Loading the full ConditionalGeneration model and
    # ignoring the vision layers preserves the multimodal config + vision
    # weights so the result loads.
    _cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    is_multimodal = hasattr(_cfg, "vision_config")
    print(f"multimodal: {is_multimodal}", flush=True)

    print(f"=== baseline bf16 GSM8K(n={args.n}) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    baseline = run_hf_eager_gsm8k(model, tok, args.n, long_thinker)
    print(f"baseline acc = {baseline['acc']}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"=== llm-compressor {args.method.upper()} quantize ===", flush=True)
    from llmcompressor import oneshot

    # Never quantize the vision tower — only the language layers. (Ignoring it
    # also keeps its weights in the saved checkpoint at full precision.)
    ignore = ["lm_head"]
    if is_multimodal:
        ignore += ["re:.*visual.*", "re:.*vision.*", "re:.*merger.*"]
    # Recurrent / linear-attention blocks (Gated DeltaNet, Mamba, short-conv) are
    # extremely sensitive to weight quantization — their state accumulates error
    # over the sequence, so a quantized model starts coherent then degenerates
    # (observed on Qwen3.6-35B's `linear_attn`). Keep them full precision; they're
    # a small fraction of the weights. Patterns are no-ops on models without them.
    ignore += ["re:.*linear_attn.*", "re:.*mamba.*", "re:.*conv1d.*", "re:.*\\.gate$"]

    if args.method == "w8a8":
        # RTN (min-max weights + dynamic per-token INT8 activations), NOT GPTQ —
        # GPTQ solves a Hessian per weight matrix, which on a 256-expert MoE means
        # tens of thousands of per-expert solves (~hours). INT8 is forgiving enough
        # that round-to-nearest is fine and takes minutes.
        from llmcompressor.modifiers.quantization import QuantizationModifier
        scheme = "W8A8"
        recipe = [QuantizationModifier(ignore=ignore, scheme=scheme, targets=["Linear"])]
    else:
        from llmcompressor.modifiers.awq import AWQModifier
        scheme = f"W{args.w_bit}A16_ASYM"
        recipe = [AWQModifier(ignore=ignore, scheme=scheme, targets=["Linear"], duo_scaling="both")]

    default_split = {"ultrachat-200k": f"train_sft[:{args.calib_samples}]"}.get(
        args.calib_dataset, f"train[:{args.calib_samples}]")
    split = args.calib_split or default_split

    oneshot_kwargs = dict(
        model=args.model,
        dataset=args.calib_dataset,
        splits=split,
        recipe=recipe,
        max_seq_length=args.calib_seq_len,
        num_calibration_samples=args.calib_samples,
        output_dir=args.out_dir,
        trust_remote_code_model=True,
    )
    if args.calib_config:
        oneshot_kwargs["dataset_config_name"] = args.calib_config
    if is_multimodal:
        # Pass the full multimodal model object (loaded on CPU; oneshot moves
        # layers to GPU sequentially) instead of a string, so the save keeps the
        # ConditionalGeneration wrapper + vision weights.
        from transformers import AutoModelForImageTextToText
        oneshot_kwargs["model"] = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)

    t0 = time.time()
    oneshot(**oneshot_kwargs)
    quant_wall_s = round(time.time() - t0, 1)
    print(f"quantize done ({quant_wall_s}s)", flush=True)

    quant_config = {"backend": "llm-compressor", "scheme": scheme, "group_size": args.q_group_size,
                     "calib_dataset": args.calib_dataset, "calib_samples": args.calib_samples}
    meta = {
        "model": args.model, "baseline": baseline, "quant_config": quant_config,
        "quant_wall_s": quant_wall_s, "long_thinker": long_thinker,
    }
    with open(os.path.join(args.out_dir, "_fastserve_quant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("WROTE", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
