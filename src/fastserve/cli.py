"""fastserve command-line interface."""
from __future__ import annotations

import argparse

from .bench import bench_hf_baseline, bench_vllm
from .detect import detect
from .engine import build_speculative_config, launch_server, resolve_model, serve_command


def _print_notes(det) -> None:
    print(f"\nfastserve: {det.base_model}")
    for n in det.notes:
        print("  -", n)
    print()


def cmd_info(args) -> None:
    det = detect(args.model, skip_quant=args.no_quant, skip_eagle=args.no_spec)
    _print_notes(det)
    print("would launch:")
    print("  " + " ".join(serve_command(det, port=args.port, tp=args.tp,
                                          trust_remote_code=not args.no_trust_remote_code)))


def cmd_serve(args) -> None:
    det = detect(args.model, skip_quant=args.no_quant, skip_eagle=args.no_spec)
    _print_notes(det)
    launch_server(
        det, port=args.port, use_quant=not args.no_quant, use_spec=not args.no_spec,
        tp=args.tp, trust_remote_code=not args.no_trust_remote_code, extra_args=args.vllm_args,
    )


def cmd_bench(args) -> None:
    det = detect(args.model, skip_quant=args.no_quant, skip_eagle=args.no_spec)
    _print_notes(det)

    model = resolve_model(det, use_quant=not args.no_quant)
    spec = build_speculative_config(det) if not args.no_spec else None

    print(f"benchmarking optimized stack ({model}) ...")
    fast = bench_vllm(model, spec, n_prompts=args.n, tp=args.tp,
                       trust_remote_code=not args.no_trust_remote_code)
    print("  optimized:", fast)

    if args.compare_baseline:
        print(f"\nbenchmarking naive HF-eager baseline ({args.model}) ... (this is slow)")
        base = bench_hf_baseline(args.model, n_prompts=args.n)
        print("  baseline: ", base)
        speedup = base["e2e_s"] / fast["e2e_s"]
        print(f"\nmeasured speedup: {speedup:.2f}x")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fastserve",
        description="Zero-config LLM speedup: detects the fastest available vLLM "
                     "stack (quantization + speculative decoding) for any Hugging "
                     "Face model and serves or benchmarks it.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("model", help="Hugging Face model ID, e.g. Qwen/Qwen3-8B")
    common.add_argument("--no-quant", action="store_true", help="disable auto quantization")
    common.add_argument("--no-spec", action="store_true", help="disable speculative decoding")
    common.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    common.add_argument("--no-trust-remote-code", action="store_true",
                         help="refuse to run custom code some checkpoints ship (e.g. Kimi-Linear's "
                              "tokenizer) — on by default since you already chose this model id")

    p_info = sub.add_parser("info", parents=[common], help="show detected optimizations without running anything")
    p_info.add_argument("--port", type=int, default=8000)
    p_info.set_defaults(func=cmd_info)

    p_serve = sub.add_parser("serve", parents=[common], help="launch an OpenAI-compatible API server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("vllm_args", nargs=argparse.REMAINDER,
                          help="extra args passed straight through to vLLM")
    p_serve.set_defaults(func=cmd_serve)

    p_bench = sub.add_parser("bench", parents=[common], help="quick throughput benchmark")
    p_bench.add_argument("-n", type=int, default=5, help="number of prompts")
    p_bench.add_argument("--compare-baseline", action="store_true",
                          help="also measure naive HF-eager baseline for a real speedup number (slow)")
    p_bench.set_defaults(func=cmd_bench)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
