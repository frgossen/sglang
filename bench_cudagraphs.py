#!/usr/bin/env python3
"""Compare CUDA-graph variants across models.

Usage:
  python bench_cudagraphs.py

Variants:
  no_cuda_graph      --disable-cuda-graph
  full_cuda_graph    (default)
  pcg_torch_compile  PiecewiseCudaGraphRunner (torch.compile)
  pcg_standalone     StandalonePiecewiseCudaGraphRunner
                     (piecewise_cuda_graphs package)
  bcg                BreakableCudaGraphRunner
"""

import json
import subprocess
import sys
from pathlib import Path

MODELS = [
    "openai/gpt-oss-120b",  # works for bs=1
    # "deepseek-ai/DeepSeek-V3.2",  # works with from source install of sgl-deep-gemm (bs=1) -- fails again
    "moonshotai/Kimi-K2.6", # works for bs=1
    "zai-org/GLM-4.7", # works for bs=1
    # "MiniMaxAI/MiniMax-M2.7", #fails
    "Qwen/Qwen3.6-35B-A3B", # works for bs=1
    "meta-llama/Meta-Llama-3-8B-Instruct",# works for bs=1
    "meta-llama/Meta-Llama-3-70B-Instruct",# works for bs=1
    "meta-llama/Llama-3.1-405B-Instruct-FP8", # works for bs=1
]
BATCH_SIZES = ["1"] #, "4", "16"]
INPUT_LENS: list[str] = ["512"]
OUTPUT_LENS = ["8"]

COMMON_FLAGS = [
    "--batch-size",
    *BATCH_SIZES,
    "--input-len",
    *INPUT_LENS,
    "--output-len",
    *OUTPUT_LENS,
]
MODEL_FLAGS = {
    "meta-llama/Llama-3.1-405B-Instruct-FP8": [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--tp-size",
        "8",
    ],
    "deepseek-ai/DeepSeek-V3.2": [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.95",
        "--tp-size",
        "8",
    ],
    "moonshotai/Kimi-K2.6": [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--tp-size",
        "8",
    ],
    "zai-org/GLM-4.7": [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--tp-size",
        "8",
    ],
    "MiniMaxAI/MiniMax-M2.7": [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--tp-size",
        "4",
    ],
    "meta-llama/Meta-Llama-3-70B-Instruct": [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.9",
        "--tp-size",
        "4",
    ],
}
PIECEWISE_TOKENS = ["512", "2048", "8192"]  # must cover bs*input_len
OUT = Path(__file__).resolve().parent / "bench_cudagraphs_results"

VARIANT_FLAGS = {
    "no_cuda_graph": [
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
    ],
    # "full_cuda_graph": [],
    # "pcg_torch_compile": [
    #     "--piecewise-cuda-graph-tokens",
    #     *PIECEWISE_TOKENS,
    # ],
    # "pcg_standalone": [
    #     "--enable-standalone-piecewise-cuda-graph",
    #     "--piecewise-cuda-graph-tokens",
    #     *PIECEWISE_TOKENS,
    # ],
    # "bcg": [
    #     "--enable-breakable-cuda-graph",
    #     "--piecewise-cuda-graph-tokens",
    #     *PIECEWISE_TOKENS,
    # ],
}


def main():
    OUT.mkdir(exist_ok=True)
    failures = []
    for model in MODELS:
        model_dir = OUT / model.replace("/", "__")
        model_dir.mkdir(exist_ok=True)
        for name, variant_flags in VARIANT_FLAGS.items():
            result_file = model_dir / f"{name}.jsonl"
            result_file.unlink(missing_ok=True)  # bench_one_batch appends
            model_flags = MODEL_FLAGS.get(model, ["--tp-size", "1"])
            cmd = [
                sys.executable,
                "-m",
                "sglang.bench_one_batch",
                "--model-path",
                model,
                *COMMON_FLAGS,
                *model_flags,
                "--run-name",
                f"{model}:{name}",
                "--result-filename",
                str(result_file),
                *variant_flags,
            ]
            print(f"\n=== {model} / {name}: {' '.join(cmd)}", flush=True)
            # Keep going on failure so one bad variant doesn't waste the whole
            # sweep; record it and report at the end. A run can also "succeed"
            # (exit 0) without producing results, so flag a missing file too.
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                failures.append((model, name, f"exit {ret.returncode}"))
            elif not result_file.exists():
                failures.append((model, name, "no result file written"))

    print(f"\nResults written to {OUT}/<model>/*.jsonl")
    if failures:
        print("\nFailed runs (shown as n/a in the summary):")
        for model, name, why in failures:
            print(f"  {model} / {name}: {why}")
    print_summary()


def print_summary():
    names = list(VARIANT_FLAGS)
    for model in MODELS:
        model_dir = OUT / model.replace("/", "__")
        # (bs, il, ol) -> {variant: row}
        rows = {}
        for name in VARIANT_FLAGS:
            result_file = model_dir / f"{name}.jsonl"
            if not result_file.exists():
                continue  # variant failed, was skipped, or not yet run -> n/a
            for line in result_file.read_text().splitlines():
                r = json.loads(line)
                key = (r["batch_size"], r["input_len"], r["output_len"])
                rows.setdefault(key, {})[name] = r

        for title, field in [
            ("prefill latency (ms)", "prefill_latency"),
            ("decode median latency (ms)", "median_decode_latency"),
        ]:
            print(f"\n{title}   model={model}")
            header = f"{'bs/il/ol':<14}" + "".join(f"{n:>18}" for n in names)
            print(header)
            print("-" * len(header))
            for key in sorted(rows):
                line = f"{'/'.join(map(str, key)):<14}"
                for n in names:
                    v = rows[key].get(n, {}).get(field)
                    line += (
                        f"{v * 1e3:>18.2f}"
                        if isinstance(v, (int, float))
                        else f"{'n/a':>18}"
                    )
                print(line)


if __name__ == "__main__":
    main()
