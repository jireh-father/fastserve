"""One config of compare3.py, in its own process. Prints `RESULT {json}`.

Accuracy = GSM8K over n prompts. Speed = single-stream (batch-1) greedy
decode tok/s, measured the SAME way for every config over the first
`speed_prompts` GSM8K prompts (prompt 0 discarded as warmup) so original /
vLLM / fastserve are directly comparable — this is per-request latency, not
vLLM's continuous-batching throughput (a separate axis).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_tasks import (  # noqa: E402
    apply_template, extract_gsm8k_answer, gsm8k_correct, is_long_thinker,
    load_gsm8k, GSM8K_INSTR,
)


def run_hf(model: str, n: int, speed_prompts: int) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    max_new = 2048 if is_long_thinker(model) else 640
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    probs = load_gsm8k(n)
    correct, tot_tok, tot_t = 0, 0, 0.0
    for i, (q, gold) in enumerate(probs):
        enc = tok(apply_template(tok, q + GSM8K_INSTR), return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out[0, enc["input_ids"].shape[1]:]
        correct += gsm8k_correct(extract_gsm8k_answer(tok.decode(gen, skip_special_tokens=True)), gold)
        if 0 < i <= speed_prompts:  # skip prompt 0 (warmup), time the next few
            tot_t += dt
            tot_tok += int(gen.shape[0])
    return {"acc": round(correct / n, 4), "tok_s": round(tot_tok / tot_t, 1) if tot_t else None}


def run_vllm(model: str, n: int, speed_prompts: int, spec: str | None) -> dict:
    from vllm import LLM, SamplingParams

    max_new = 2048 if is_long_thinker(model) else 640
    mem_util = float(os.environ.get("FASTSERVE_GPU_MEM_UTIL", "0.6"))
    tp = int(os.environ.get("FASTSERVE_TP", "1"))
    kwargs = dict(model=model, max_model_len=4096, gpu_memory_utilization=mem_util,
                  tensor_parallel_size=tp, trust_remote_code=True)
    kv_dtype = os.environ.get("FASTSERVE_KV_CACHE_DTYPE")
    if kv_dtype:
        kwargs["kv_cache_dtype"] = kv_dtype
    if os.environ.get("FASTSERVE_ENFORCE_EAGER") == "1":
        kwargs["enforce_eager"] = True
    if spec == "ngram":
        kwargs["speculative_config"] = {"method": "ngram", "num_speculative_tokens": 3,
                                        "prompt_lookup_max": 4, "prompt_lookup_min": 2}
    elif spec:
        kwargs["speculative_config"] = {"method": "eagle3", "model": spec, "num_speculative_tokens": 3}
    llm = LLM(**kwargs)
    tok = llm.get_tokenizer()
    probs = load_gsm8k(n)
    prompts = [apply_template(tok, q + GSM8K_INSTR) for q, _ in probs]

    # accuracy: batched (greedy → identical result, much faster)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    correct = sum(gsm8k_correct(extract_gsm8k_answer(o.outputs[0].text), gold)
                  for o, (_, gold) in zip(outs, probs))

    # speed: single-stream batch-1, same convention as run_hf
    llm.generate(prompts[:1], sp, use_tqdm=False)  # warmup
    tot_tok, tot_t = 0, 0.0
    for p in prompts[1:1 + speed_prompts]:
        t0 = time.perf_counter()
        o = llm.generate([p], sp, use_tqdm=False)
        tot_t += time.perf_counter() - t0
        tot_tok += len(o[0].outputs[0].token_ids)
    return {"acc": round(correct / n, 4), "tok_s": round(tot_tok / tot_t, 1) if tot_t else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf", "vllm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--speed-prompts", type=int, default=5)
    ap.add_argument("--spec", default=None)
    args = ap.parse_args()
    res = (run_hf(args.model, args.n, args.speed_prompts) if args.mode == "hf"
           else run_vllm(args.model, args.n, args.speed_prompts, args.spec))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
