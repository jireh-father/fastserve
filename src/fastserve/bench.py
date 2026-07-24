"""Quick throughput benchmark: measures batch=1 decode speed for the
detected-optimal stack, optionally against a naive HF-eager baseline so the
reported speedup number is measured, not assumed.
"""
from __future__ import annotations

import time

_DEFAULT_PROMPTS = [
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are three tips for staying productive while working from home?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "List five common mistakes people make when learning a new language.",
]


def _prompts(n: int) -> list[str]:
    reps = (n // len(_DEFAULT_PROMPTS)) + 1
    return (_DEFAULT_PROMPTS * reps)[:n]


def bench_vllm(model: str, speculative_config: dict | None, *, n_prompts: int = 5,
               max_new_tokens: int = 256, tp: int = 1, trust_remote_code: bool = True) -> dict:
    from vllm import LLM, SamplingParams

    prompts = _prompts(n_prompts)
    kwargs = dict(model=model, tensor_parallel_size=tp, gpu_memory_utilization=0.85,
                  trust_remote_code=trust_remote_code)
    if speculative_config:
        kwargs["speculative_config"] = speculative_config
    llm = LLM(**kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    llm.generate(prompts[:1], sp, use_tqdm=False)  # warmup

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    dt = time.perf_counter() - t0
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    return {"e2e_s": round(dt, 2), "gen_tokens": ntok, "tok_s": round(ntok / dt, 2)}


def bench_hf_baseline(model: str, *, n_prompts: int = 5, max_new_tokens: int = 256) -> dict:
    """Naive HF transformers eager bf16 generation — the reference point this
    whole project measures speedups against.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()

    prompts = _prompts(n_prompts + 1)  # +1 for a discarded warmup call
    tot_t, tot_tok = 0.0, 0
    for i, p in enumerate(prompts):
        enc = tok(p, return_tensors="pt").to("cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i == 0:
            continue  # warmup, discarded
        tot_t += dt
        tot_tok += out.shape[1] - enc["input_ids"].shape[1]
    return {"e2e_s": round(tot_t, 2), "gen_tokens": tot_tok, "tok_s": round(tot_tok / tot_t, 2)}
