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


def _vllm_baseline(model_id: str, n: int) -> dict | None:
    """Measure the bf16 baseline through vLLM instead of HF-eager.

    Runs as its own subprocess: vLLM needs a clean CUDA context, and this stage
    still has to load the model for quantization afterwards. Returns the same
    shape run_hf_eager_gsm8k does, or None if vLLM can't serve it either.
    """
    import re
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    worker = os.path.join(here, "..", "benchmarks", "compare3_worker.py")
    cmd = [sys.executable, worker, "--mode", "vllm", "--model", model_id, "--n", str(n)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=os.environ)
    m = re.search(r"^RESULT (\{.*\})$", proc.stdout, re.MULTILINE)
    if not m:
        print("vLLM baseline also failed:\n" + (proc.stdout + proc.stderr)[-400:], flush=True)
        return None
    res = json.loads(m.group(1))
    return {"n": n, "acc": res["acc"], "truncated": 0, "max_new_tokens": None,
            "wall_s": None, "samples": [], "source": "vllm-bf16"}


def _install_tied_weights_shim() -> None:
    """Tolerate the old list form of `_tied_weights_keys` when saving.

    transformers changed `_tied_weights_keys` from a list of tied parameter names
    to a dict mapping tied name -> source name, and `save_pretrained` now calls
    `.keys()` on it. Custom modeling code written before that still declares a
    list, so saving a quantized checkpoint dies with "'list' object has no
    attribute 'keys'" *after* the expensive quantization pass (hit by
    upstage/solar-pro-preview-instruct). Normalize lists to dicts at collection
    time — the values are only used to locate the source tensor, and for these
    models nothing is actually tied (tie_word_embeddings=False).
    """
    try:
        from transformers import modeling_utils
    except Exception:
        return
    orig = getattr(modeling_utils, "_get_tied_weight_keys", None)
    if orig is None or getattr(orig, "_fastserve_shim", False):
        return

    def _get_tied_weight_keys(module):  # noqa: ANN001
        keys = []
        for name, sub in module.named_modules():
            tied = getattr(sub, "_tied_weights_keys", None) or {}
            names = tied if isinstance(tied, (list, tuple, set)) else tied.keys()
            keys.extend([f"{name}.{k}" if name else k for k in names])
        return keys

    _get_tied_weight_keys._fastserve_shim = True
    modeling_utils._get_tied_weight_keys = _get_tied_weight_keys


def _install_legacy_cache_shims() -> None:
    """Restore Cache methods that transformers removed, for models whose bundled
    custom modeling code predates the change.

    A model that ships its own `modeling_*.py` is frozen at whatever transformers
    API existed when it was published; loading it under a much newer transformers
    then dies on APIs that no longer exist. Two seen in practice:
    `DynamicCache.get_max_length()` (renamed to `get_max_cache_shape()`, hit by
    upstage/solar-pro-preview-instruct) and `.seen_tokens` (now `get_seq_length()`,
    hit by Phi-3.5-mini). Re-adding them as thin aliases is safe — they're the
    same values under new names — and it's the difference between a model being
    quantizable here or not. Only defines what's actually missing.
    """
    try:
        from transformers.cache_utils import Cache, DynamicCache
    except Exception:
        return
    _install_tied_weights_shim()

    for cls in (Cache, DynamicCache):
        if not hasattr(cls, "get_max_length"):
            def get_max_length(self):  # noqa: ANN001
                fn = getattr(self, "get_max_cache_shape", None)
                v = fn() if fn else None
                # The new API can return a full cache *shape* tuple, but the old
                # one returned a single length that callers compare numerically
                # (`cache_length + seq_len > max_cache_length`) — a tuple there
                # raises TypeError. Take the sequence dimension.
                if isinstance(v, (tuple, list)):
                    v = v[-2] if len(v) >= 2 else (v[0] if v else None)
                # Old callers test `is not None` to mean "bounded cache"; the new
                # API reports an unbounded dynamic cache as -1, which would send
                # them down the bounded path. Normalize it back to None.
                return None if v is None or v < 0 else v
            cls.get_max_length = get_max_length
        if not hasattr(cls, "seen_tokens"):
            cls.seen_tokens = property(lambda self: self.get_seq_length())


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

    _install_legacy_cache_shims()

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

    # Stale custom modeling code vs. new transformers: modern transformers
    # normalizes a *missing* `rope_scaling` into `{"rope_type": "default", ...}`,
    # but older bundled code reads `rope_scaling["type"]` and only skips when the
    # attribute is falsy — so the injected dict makes it KeyError (observed on
    # upstage/solar-pro-preview-instruct). If the model didn't declare any rope
    # scaling itself, hand the model class an explicit None so it takes its
    # no-scaling path.
    load_kwargs = {}
    if getattr(_cfg, "rope_scaling", None) and "type" not in _cfg.rope_scaling:
        import json as _json
        from huggingface_hub import hf_hub_download
        try:
            declared = _json.load(open(hf_hub_download(args.model, "config.json"))).get("rope_scaling")
        except Exception:
            declared = None
        if declared is None:
            load_kwargs["rope_scaling"] = None
            print("compat: clearing transformers-injected rope_scaling "
                  "(model declares none; bundled code expects the legacy 'type' key)", flush=True)

    print(f"=== baseline bf16 GSM8K(n={args.n}) ===", flush=True)
    try:
        if is_multimodal:
            # ConditionalGeneration models don't map to AutoModelForCausalLM; load the
            # full multimodal model (it generates text fine from text-only input).
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                args.model, dtype=torch.bfloat16, trust_remote_code=True,
                **load_kwargs).to("cuda").eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, trust_remote_code=True,
                **load_kwargs).to("cuda").eval()
        baseline = run_hf_eager_gsm8k(model, tok, args.n, long_thinker)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        # A model that ships its own modeling code is frozen at the transformers
        # API of its release; some are too old to run under the installed version
        # at all (upstage/solar-pro-preview-instruct: transformers now calls
        # `prepare_inputs_for_generation` with new kwargs its bundled code doesn't
        # accept). vLLM has its own implementation of these architectures and
        # doesn't execute the bundled code, so it can still produce an honest bf16
        # reference — which is all the gate needs. Falls back to that rather than
        # dropping the model.
        print(f"HF-eager baseline failed ({type(e).__name__}: {str(e)[:120]}); "
              f"falling back to a vLLM bf16 baseline", flush=True)
        # Free the half-loaded model before handing the GPU to vLLM. `del` alone
        # isn't enough: the live exception keeps its traceback — and every frame's
        # locals, including `model` — alive, so the weights would still be resident
        # and vLLM would fail to find memory for its KV cache.
        model = locals().get("model")
        if model is not None:
            model.to("cpu")
            del model
        e.__traceback__ = None
        gc.collect()
        torch.cuda.empty_cache()
        _free = torch.cuda.mem_get_info()[0] / 2**30
        print(f"GPU free before vLLM baseline: {_free:.1f} GiB", flush=True)
        baseline = _vllm_baseline(args.model, args.n)
        if baseline is None:
            raise
    print(f"baseline acc = {baseline['acc']}", flush=True)

    print(f"=== llm-compressor {args.method.upper()} quantize ===", flush=True)
    from llmcompressor import oneshot

    # Never quantize the vision tower — only the language layers. (Ignoring it
    # also keeps its weights in the saved checkpoint at full precision.)
    ignore = ["lm_head"]
    if is_multimodal:
        # Never quantize the non-text towers. vLLM's multimodal loaders expect these
        # sub-projections at full precision — if llm-compressor quantizes e.g.
        # `embed_audio.embedding_projection`, vLLM rejects the checkpoint at load
        # ("no parameter named ...weight_scale"). Gemma-4 *Unified* omni models carry
        # both a vision AND an audio tower, so cover audio too (not just vision).
        ignore += ["re:.*visual.*", "re:.*vision.*", "re:.*merger.*",
                   "re:.*audio.*", "re:.*embed_audio.*", "re:.*embed_vision.*"]
    # Recurrent / linear-attention blocks (Gated DeltaNet, Mamba, short-conv) are
    # extremely sensitive to weight quantization — their state accumulates error
    # over the sequence, so a quantized model starts coherent then degenerates
    # (observed on Qwen3.6-35B's `linear_attn`). Keep them full precision; they're
    # a small fraction of the weights. Patterns are no-ops on models without them.
    ignore += ["re:.*linear_attn.*", "re:.*mamba.*", "re:.*conv1d.*", "re:.*\\.gate$"]
    # The MoE *router* (a tiny Linear that picks which experts each token goes to)
    # must stay full precision — INT8-ing it corrupts routing and the model
    # degenerates into repeated-token garbage (observed on gemma-4-26B-A4B: the
    # `router.proj` got quantized → GSM8K 0.53→0.00, 87% degenerate). The experts'
    # own `gate_proj`/`up_proj`/`down_proj` are ordinary FFN weights and stay
    # quantized; only the routing network is protected. No-op on dense models.
    ignore += ["re:.*router.*", "re:.*\\.gate\\.", "re:.*block_sparse_moe.gate.*"]

    default_split = {"ultrachat-200k": f"train_sft[:{args.calib_samples}]"}.get(
        args.calib_dataset, f"train[:{args.calib_samples}]")
    split = args.calib_split or default_split

    def _build_recipe(method):
        if method == "w8a8":
            # RTN (min-max weights + dynamic per-token INT8 activations), NOT GPTQ —
            # GPTQ solves a Hessian per weight matrix, which on a big-expert MoE means
            # tens of thousands of per-expert solves (~hours). INT8 is forgiving enough
            # that round-to-nearest is fine and takes minutes. It also needs NO
            # smooth-layer mappings, so it's robust to any architecture (see fallback).
            from llmcompressor.modifiers.quantization import QuantizationModifier
            return "W8A8", [QuantizationModifier(ignore=ignore, scheme="W8A8", targets=["Linear"])]
        from llmcompressor.modifiers.awq import AWQModifier
        s = f"W{args.w_bit}A16_ASYM"
        return s, [AWQModifier(ignore=ignore, scheme=s, targets=["Linear"], duo_scaling="both")]

    def _run(method):
        s, recipe = _build_recipe(method)
        kwargs = dict(
            model=args.model, dataset=args.calib_dataset, splits=split, recipe=recipe,
            max_seq_length=args.calib_seq_len, num_calibration_samples=args.calib_samples,
            output_dir=args.out_dir, trust_remote_code_model=True,
        )
        if args.calib_config:
            kwargs["dataset_config_name"] = args.calib_config
        if is_multimodal:
            # Pass the full multimodal model object (loaded fresh each attempt — a
            # failed AWQ pass leaves the module tree half-transformed) so the save
            # keeps the ConditionalGeneration wrapper + vision weights.
            from transformers import AutoModelForImageTextToText
            kwargs["model"] = AutoModelForImageTextToText.from_pretrained(
                args.model, dtype=torch.bfloat16, trust_remote_code=True,
                low_cpu_mem_usage=True, **load_kwargs)
        elif load_kwargs:
            # oneshot() can't forward from_pretrained kwargs when given a model id,
            # so pre-load with the compat kwargs applied.
            kwargs["model"] = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, trust_remote_code=True,
                low_cpu_mem_usage=True, **load_kwargs)
        oneshot(**kwargs)
        return s

    # AWQ needs per-decoder-layer smooth→balance mappings. For multimodal wrappers
    # whose class isn't in llm-compressor's mapping registry (e.g.
    # Gemma4ForConditionalGeneration), it falls back to generic mappings that can't
    # segment the nested `language_model.layers.*` tree and raises. W8A8 RTN needs no
    # mappings and, on A100 (INT8 tensor cores, no FP8), is actually the faster format
    # anyway — so fall back to it rather than failing the model outright.
    t0 = time.time()
    method_used = args.method
    try:
        scheme = _run(args.method)
    except Exception as e:
        emsg = str(e)
        # Two shapes of AWQ mapping failure: it raises about the mapping itself,
        # or — when *every* mapping is skipped as shape-incompatible (Solar's
        # depth-up-scaled layers under the generic mappings: "64 mappings were
        # skipped") — it resolves nothing and then divides by zero averaging
        # error metrics over an empty list. Both mean "AWQ can't map this arch".
        mapping_failure = ("smoothlayer" in emsg or "AWQMapping" in emsg
                           or "single smooth" in emsg or "match_modules" in emsg
                           or isinstance(e, ZeroDivisionError))
        if args.method == "awq" and mapping_failure:
            print(f"AWQ mapping failed on this architecture ({type(e).__name__}); "
                  f"falling back to W8A8 RTN (mapping-free, A100-optimal)", flush=True)
            # The failed AWQ pass left the whole model on the GPU, and the live
            # exception's traceback keeps every frame's locals alive — so a plain
            # gc.collect() frees nothing and the W8A8 retry OOMs. Drop the
            # traceback and any oneshot state before reloading.
            e.__traceback__ = None
            try:
                from llmcompressor.core import reset_session
                reset_session()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            print(f"GPU free before W8A8 retry: "
                  f"{torch.cuda.mem_get_info()[0] / 2**30:.1f} GiB", flush=True)
            method_used = "w8a8"
            scheme = _run("w8a8")
        else:
            raise
    quant_wall_s = round(time.time() - t0, 1)
    print(f"quantize done ({quant_wall_s}s, method={method_used}, scheme={scheme})", flush=True)

    quant_config = {"backend": "llm-compressor", "scheme": scheme, "group_size": args.q_group_size,
                     "calib_dataset": args.calib_dataset, "calib_samples": args.calib_samples,
                     "method": method_used}
    meta = {
        "model": args.model, "baseline": baseline, "quant_config": quant_config,
        "quant_wall_s": quant_wall_s, "long_thinker": long_thinker,
    }
    with open(os.path.join(args.out_dir, "_fastserve_quant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("WROTE", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
