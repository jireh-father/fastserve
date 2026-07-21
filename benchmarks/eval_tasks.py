"""GSM8K + MMLU accuracy tasks for the fastserve benchmark.

GSM8K prompt/parser convention lifted as-is from
../../inference-opt/bench/bench_w1.py (boxed-answer extraction, validated
against truncation false-negatives in that campaign).
"""
from __future__ import annotations

import re

from datasets import load_dataset

GSM8K_INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
MMLU_INSTR_SUFFIX = "\nAnswer with a single letter (A, B, C, or D) only, as the very last character of your response."

# Reasoning-distilled/verbose-by-default models think at length before
# answering; give them enough headroom or every eval reads as a truncation
# failure, not a real one. gemma-4 added after observing MMLU acc=0.060 at
# the default 16-token budget — samples showed genuine (not degenerate) step
# reasoning cut off before ever stating a letter.
_LONG_THINKER_MARKERS = ["r1-distill", "-r1-", "deepseek-r1", "gemma-4", "gpt-oss"]


def is_long_thinker(model_id: str) -> bool:
    low = model_id.lower()
    return any(m in low for m in _LONG_THINKER_MARKERS)


def apply_template(tok, question: str) -> str:
    msgs = [{"role": "user", "content": question}]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def run_hf_eager_gsm8k(model, tok, n: int, long_thinker: bool, sample_limit: int = 3) -> dict:
    """Naive HF-eager batch=1 GSM8K eval — shared by baseline_accuracy.py and
    the self-quantization pipeline's pre/post-quant accuracy gate.
    """
    import time
    import torch

    max_new = 2048 if long_thinker else 640
    probs = load_gsm8k(n)
    correct, truncated, samples = 0, 0, []
    t0 = time.time()
    for i, (q, gold) in enumerate(probs):
        text = apply_template(tok, q + GSM8K_INSTR)
        enc = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
        gen = out[0, enc["input_ids"].shape[1]:]
        resp = tok.decode(gen, skip_special_tokens=True)
        ok = gsm8k_correct(extract_gsm8k_answer(resp), gold)
        correct += ok
        if gen.shape[0] >= max_new:
            truncated += 1
        if i < sample_limit:
            samples.append({"gold": gold, "raw": resp[:400], "correct": bool(ok)})
        print(f"{i+1}/{n} acc_so_far={correct/(i+1):.3f}", flush=True)

    return {
        "n": n, "acc": round(correct / n, 4), "truncated": truncated,
        "max_new_tokens": max_new, "wall_s": round(time.time() - t0, 1), "samples": samples,
    }


def extract_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + 7, 1
    while j < len(text) and depth:
        depth += text[j] == "{"
        depth -= text[j] == "}"
        j += 1
    return text[i + 7:j - 1].strip()


_NUM_RE = re.compile(r"-?\$?[0-9][0-9,]*\.?[0-9]*")


def extract_final_number(text: str) -> str | None:
    """Fallback for models that answer correctly but skip the \\boxed{}
    formatting instruction (common on weaker/smaller instruct models) —
    without this, their score conflates "got the math wrong" with "didn't
    follow the output-format instruction".
    """
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def extract_gsm8k_answer(text: str) -> str | None:
    return extract_boxed(text) or extract_final_number(text)


def _norm(ans: str | None) -> str | None:
    if ans is None:
        return None
    a = ans.strip().rstrip(".").replace(" ", "").replace("\\!", "")
    a = a.replace("\\left", "").replace("\\right", "")
    a = re.sub(r"^\\text\{(.+)\}$", r"\1", a)
    a = re.sub(r"\\dfrac", r"\\frac", a)
    return a.replace(",", "").replace("$", "")


def gsm8k_correct(pred: str | None, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    if p is None:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (TypeError, ValueError):
        return False


def load_gsm8k(n: int):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = list(ds)[:n]
    return [(r["question"], r["answer"].split("####")[-1].strip()) for r in rows]


_LETTER_RE = re.compile(r"\b([ABCD])\b")


def extract_mmlu_letter(text: str) -> str | None:
    m = re.search(r"answer[^A-Za-z]{0,10}([ABCD])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    hits = _LETTER_RE.findall(text)
    return hits[-1] if hits else None


def load_mmlu(n: int, seed: int = 42):
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed).select(range(n))
    letters = "ABCD"
    rows = []
    for r in ds:
        q = r["question"].strip() + "\n" + "\n".join(
            f"{letters[i]}) {c}" for i, c in enumerate(r["choices"])
        ) + MMLU_INSTR_SUFFIX
        rows.append((q, letters[r["answer"]]))
    return rows
