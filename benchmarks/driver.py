"""Sequential driver: runs run_bench.py (optimized fastserve stack) and
baseline_bench.py (naive HF-eager bf16 unquantized) once per model, each as
its own isolated subprocess (clean CUDA context — vLLM doesn't release its
~68GB reservation until the process exits, so the two can't share one),
pinned to GPU0 (confirmed idle; GPU1 has an unrelated foreign job).
Appends each result to RESULTS.md as it finishes so progress is visible
mid-run.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import MODELS, BASELINE_MIRROR, BASELINE_SKIP  # noqa: E402

VENV_PY = os.path.join(HERE, "..", ".venv", "bin", "python")
RESULTS_DIR = os.path.join(HERE, "results")
LOG_DIR = os.path.join(HERE, "logs")
RESULTS_MD = os.path.join(HERE, "RESULTS.md")
TIMEOUT_S = 3600
BASELINE_TIMEOUT_S = 3600
GPU_ID = "0"
# /home/work is a 49GB loop partition (fills up fast); the real disk room is
# the NFS mount at /home/work/source. Point the HF cache there explicitly —
# a prior campaign already hit this exact trap (see WORKLOG.md 2026-06-26).
HF_HOME = os.path.join(HERE, "..", "..", ".hf_cache")
VLLM_CACHE_ROOT = os.path.join(HERE, "..", "..", ".vllm_cache")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU_ID, HF_HOME=HF_HOME, VLLM_CACHE_ROOT=VLLM_CACHE_ROOT)


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def gpu_mem_used(gpu_id: str) -> int:
    out = subprocess.run(
        ["nvidia-smi", f"--id={gpu_id}", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return int(out.splitlines()[0])


def wait_for_idle(gpu_id: str, max_wait_s: int = 90) -> int:
    t0 = time.time()
    mem = gpu_mem_used(gpu_id)
    while mem > 500 and time.time() - t0 < max_wait_s:
        time.sleep(5)
        mem = gpu_mem_used(gpu_id)
    return mem


def run_subprocess(cmd: list[str], log_path: str, timeout: int, pre_note: str = "") -> int:
    """vLLM spawns a separate EngineCore worker process outside the direct
    child — on timeout, killing just the child (subprocess.run's default)
    orphans that worker holding GPU memory (observed: 400+MiB stuck on GPU0
    after a Ministral-3-8B hang). start_new_session + killing the whole
    process group on timeout avoids that leak.
    """
    with open(log_path, "w") as logf:
        if pre_note:
            logf.write(pre_note + "\n")
            logf.flush()
        proc = subprocess.Popen(cmd, env=ENV, stdout=logf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            logf.write(f"\n# TIMEOUT after {timeout}s (process group killed)\n")
            return -1


def run_baseline(model_id: str) -> dict | None:
    """Returns {"tok_s": float, ...} or None if skipped/failed (both logged)."""
    safe = safe_name(model_id)
    if model_id in BASELINE_SKIP:
        print(f"  baseline SKIPPED: {BASELINE_SKIP[model_id]}", flush=True)
        return None
    baseline_model = BASELINE_MIRROR.get(model_id, model_id)
    out_json = os.path.join(RESULTS_DIR, safe + ".baseline.json")
    log_path = os.path.join(LOG_DIR, safe + ".baseline.log")
    cmd = [VENV_PY, os.path.join(HERE, "baseline_bench.py"), "--model", baseline_model, "--out", out_json]
    rc = run_subprocess(cmd, log_path, BASELINE_TIMEOUT_S, f"# baseline model: {baseline_model}")
    if rc == 0 and os.path.exists(out_json):
        with open(out_json) as f:
            return json.load(f)
    tail = "\n".join(open(log_path).read().splitlines()[-8:])
    print(f"  baseline FAILED(rc={rc}): {tail[-300:]}", flush=True)
    return None


def md_row(model_id: str, data: dict | None, baseline: dict | None, status: str, note: str = "") -> str:
    if data is None:
        return f"| {model_id} | - | - | - | - | - | - | - | {status}{(': ' + note) if note else ''} |"
    det = data["detected"]
    quant = det["quant_method"].upper() if det["quant_method"] else "none(bf16)"
    spec = "eagle3" if det["eagle"] else "ngram"
    sp = data["speed"]
    speed_str = f"{sp['tok_s_median']} tok/s (±{sp['cv_pct']}%)"
    if baseline and baseline.get("tok_s"):
        speedup_str = f"{round(sp['tok_s_median'] / baseline['tok_s'], 2)}x (base {baseline['tok_s']} tok/s)"
    elif model_id in BASELINE_SKIP:
        speedup_str = f"N/A ({BASELINE_SKIP[model_id]})"
    else:
        speedup_str = "N/A (baseline failed)"
    g = data["gsm8k"]
    gsm8k_str = f"{g['acc']:.3f} (n={g['n']}, trunc={g['truncated']})"
    m = data["mmlu"]
    mmlu_str = f"{m['acc']:.3f} (n={m['n']})"
    flag = ""
    if g.get("degenerate", 0) / g["n"] > 0.3 or m.get("degenerate", 0) / m["n"] > 0.3:
        flag = " ⚠️QUANT-BROKEN"
    return (f"| {model_id} | {quant} | {spec} | {speed_str} | {speedup_str} | {gsm8k_str} | {mmlu_str} | "
            f"{data['total_wall_s']}s | {status}{flag} |")


def ensure_header():
    if os.path.exists(RESULTS_MD):
        return
    with open(RESULTS_MD, "w") as f:
        f.write(
            "# fastserve cross-model benchmark\n\n"
            "GPU0 (idle A100-80GB, confirmed empty before each model; GPU1 excluded — "
            "unrelated job running there) via fastserve auto-detected AWQ/GPTQ + "
            "EAGLE-3/n-gram speculative stack. Speed = batch-of-8 greedy vLLM generate, "
            "3 reps, median tok/s (±CV%). Speedup = that batch-of-8 optimized number vs "
            "naive HF-eager bf16 unquantized generating the same 8 prompts sequentially "
            "one at a time (batch=1) — this is fastserve's own `bench.py "
            "--compare-baseline` definition and the original inference-opt campaign's, "
            "so it bundles the batching win together with the quant/engine/spec win, "
            "not just the latter. "
            "Accuracy = GSM8K (boxed-or-last-number, n=150) + MMLU-all (n=300), 0-shot.\n\n"
            "| Model | Quant | Spec | Speed (tok/s) | Speedup vs baseline | GSM8K acc | MMLU acc | Wall time | Status |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
        )


def append_row(row: str):
    with open(RESULTS_MD, "a") as f:
        f.write(row + "\n")


def run_one(model_id: str, extra_args: list[str] | None = None, reuse_baseline: bool = False) -> None:
    idle_mem = wait_for_idle(GPU_ID)
    out_json = os.path.join(RESULTS_DIR, safe_name(model_id) + ".json")
    log_path = os.path.join(LOG_DIR, safe_name(model_id) + ".log")
    cmd = [VENV_PY, os.path.join(HERE, "run_bench.py"), "--model", model_id, "--out", out_json]
    if extra_args:
        cmd += extra_args

    print(f"\n=== {model_id} === (gpu0 pre-run mem={idle_mem}MiB)", flush=True)
    rc = run_subprocess(cmd, log_path, TIMEOUT_S, f"# pre-run gpu0 mem: {idle_mem} MiB")
    post_mem = gpu_mem_used(GPU_ID)

    if rc == 0 and os.path.exists(out_json):
        with open(out_json) as f:
            data = json.load(f)
        print(f"OK  {model_id}  post-run mem={post_mem}MiB", flush=True)
        baseline_json = os.path.join(RESULTS_DIR, safe_name(model_id) + ".baseline.json")
        if reuse_baseline and os.path.exists(baseline_json):
            with open(baseline_json) as f:
                baseline = json.load(f)
        else:
            wait_for_idle(GPU_ID)
            baseline = run_baseline(model_id)
        append_row(md_row(model_id, data, baseline, "OK"))
    else:
        tail = "\n".join(open(log_path).read().splitlines()[-15:])
        status = "TIMEOUT" if rc == -1 else f"FAIL(rc={rc})"
        append_row(md_row(model_id, None, None, status, tail.replace("|", "/").replace("\n", " / ")[:300]))
        print(f"{status}  {model_id}  see {log_path}", flush=True)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    ensure_header()
    t0 = time.time()
    for i, model_id in enumerate(MODELS):
        print(f"[{i+1}/{len(MODELS)}] {model_id} (+{round(time.time()-t0)}s elapsed)", flush=True)
        run_one(model_id)
    print(f"\nDONE — {len(MODELS)} models, {round(time.time()-t0)}s total", flush=True)


if __name__ == "__main__":
    main()
