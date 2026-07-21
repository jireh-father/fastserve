"""Per-model fastserve benchmark: detect -> build vLLM engine once -> speed
bench (3 reps, idle-GPU precision) + GSM8K + MMLU accuracy, one JSON out.

Meant to be run as its own subprocess (see driver.py) so each model gets a
clean CUDA context and GPU memory is fully released before the next one.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval_tasks import (  # noqa: E402
    apply_template, extract_gsm8k_answer, extract_mmlu_letter, gsm8k_correct,
    is_long_thinker, load_gsm8k, load_mmlu, GSM8K_INSTR,
)
from fastserve.bench import _prompts  # noqa: E402
from fastserve.detect import detect  # noqa: E402
from fastserve.engine import build_speculative_config, resolve_model  # noqa: E402


def looks_degenerate(text: str, token_ids) -> bool:
    """Catches broken checkpoints that loop a single token or short phrase
    (e.g. all-pad, or "pattern pattern pattern...") rather than genuinely
    answering — acc=0 alone doesn't distinguish "wrong answer" from "quant
    is broken and emits garbage". Looks at unique-token ratio in the tail
    so it catches multi-token repeat cycles, not just single-token loops.
    """
    if not text.strip():
        return True
    ids = list(token_ids)
    if len(ids) >= 20:
        tail = ids[-100:] if len(ids) > 100 else ids
        if len(set(tail)) / len(tail) < 0.3:
            return True
    return False


def speed_bench(llm, sp_cls, n_prompts: int, max_new: int, reps: int) -> dict:
    prompts = _prompts(n_prompts)
    sp = sp_cls(temperature=0.0, max_tokens=max_new)
    # Warm up at the SAME batch shape as the timed runs — a batch-of-1 warmup
    # leaves batch-of-N-specific kernel/cudagraph JIT compilation to leak into
    # rep 0, which otherwise reads up to 4x slower than reps 1-2 (observed:
    # 41-48% CV on several models before this fix).
    for _ in range(2):
        llm.generate(prompts, sp, use_tqdm=False)

    runs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = llm.generate(prompts, sp, use_tqdm=False)
        dt = time.perf_counter() - t0
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        runs.append({"e2e_s": round(dt, 3), "gen_tokens": ntok, "tok_s": round(ntok / dt, 2)})

    tok_s_vals = [r["tok_s"] for r in runs]
    mean = statistics.mean(tok_s_vals)
    cv_pct = round(100 * statistics.pstdev(tok_s_vals) / mean, 2) if mean else None
    return {
        "n_prompts": n_prompts, "max_new_tokens": max_new, "reps": runs,
        "tok_s_median": sorted(tok_s_vals)[len(tok_s_vals) // 2],
        "tok_s_mean": round(mean, 2), "cv_pct": cv_pct,
    }


def gsm8k_eval(llm, sp_cls, tok, n: int, long_thinker: bool) -> dict:
    max_new = 2048 if long_thinker else 640
    probs = load_gsm8k(n)
    prompts = [apply_template(tok, q + GSM8K_INSTR) for q, _ in probs]
    sp = sp_cls(temperature=0.0, max_tokens=max_new)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    correct = 0
    truncated = 0
    degenerate = 0
    samples = []
    for i, (o, (q, gold)) in enumerate(zip(outs, probs)):
        text = o.outputs[0].text
        ok = gsm8k_correct(extract_gsm8k_answer(text), gold)
        correct += ok
        if len(o.outputs[0].token_ids) >= max_new:
            truncated += 1
        if looks_degenerate(text, o.outputs[0].token_ids):
            degenerate += 1
        if i < 3:
            samples.append({"question": q[:200], "gold": gold, "raw": text[:400], "correct": bool(ok)})
    return {"n": n, "max_new_tokens": max_new, "acc": round(correct / n, 4),
            "truncated": truncated, "degenerate": degenerate, "wall_s": round(dt, 1),
            "samples": samples}


def mmlu_eval(llm, sp_cls, tok, n: int, long_thinker: bool) -> dict:
    max_new = 1024 if long_thinker else 16
    probs = load_mmlu(n)
    prompts = [apply_template(tok, q) for q, _ in probs]
    sp = sp_cls(temperature=0.0, max_tokens=max_new)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    correct = 0
    degenerate = 0
    samples = []
    for i, (o, (q, gold)) in enumerate(zip(outs, probs)):
        text = o.outputs[0].text
        pred = extract_mmlu_letter(text)
        ok = pred == gold
        correct += ok
        if looks_degenerate(text, o.outputs[0].token_ids):
            degenerate += 1
        if i < 3:
            samples.append({"question": q[:200], "gold": gold, "pred": pred, "raw": text[:200], "correct": bool(ok)})
    return {"n": n, "max_new_tokens": max_new, "acc": round(correct / n, 4),
            "degenerate": degenerate, "wall_s": round(dt, 1), "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--speed-n", type=int, default=8)
    ap.add_argument("--speed-max-new", type=int, default=256)
    ap.add_argument("--speed-reps", type=int, default=3)
    ap.add_argument("--gsm8k-n", type=int, default=150)
    ap.add_argument("--mmlu-n", type=int, default=300)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-spec", action="store_true",
                     help="disable speculative decoding — workaround for architectures vLLM's "
                          "EAGLE3 implementation doesn't support (e.g. GLM-4.7-Flash)")
    ap.add_argument("--no-quant", action="store_true",
                     help="serve the original model id as-is — workaround for third-party "
                          "quant repos vLLM's weight loader can't read (e.g. gpt-oss-20b GPTQ)")
    args = ap.parse_args()

    t_start = time.time()
    result = {"model": args.model, "started": t_start}

    det = detect(args.model, skip_quant=args.no_quant, skip_eagle=args.no_spec)
    result["detected"] = {
        "quant": det.quantized_model, "quant_method": det.quant_method,
        "eagle": det.eagle_model, "notes": det.notes,
    }
    resolved = resolve_model(det, use_quant=not args.no_quant)
    spec = None if args.no_spec else build_speculative_config(det)
    result["resolved_model"] = resolved
    result["spec_method"] = "none" if args.no_spec else spec["method"]
    print(json.dumps(result), flush=True)

    from vllm import LLM, SamplingParams

    llm = LLM(model=resolved, tensor_parallel_size=args.tp, gpu_memory_utilization=0.85,
              max_model_len=args.max_model_len, speculative_config=spec, trust_remote_code=True)
    tok = llm.get_tokenizer()
    long_thinker = is_long_thinker(args.model)

    result["speed"] = speed_bench(llm, SamplingParams, args.speed_n, args.speed_max_new, args.speed_reps)
    print("speed:", json.dumps(result["speed"]), flush=True)

    result["gsm8k"] = gsm8k_eval(llm, SamplingParams, tok, args.gsm8k_n, long_thinker)
    print("gsm8k:", json.dumps(result["gsm8k"]), flush=True)

    result["mmlu"] = mmlu_eval(llm, SamplingParams, tok, args.mmlu_n, long_thinker)
    print("mmlu:", json.dumps(result["mmlu"]), flush=True)

    result["total_wall_s"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
