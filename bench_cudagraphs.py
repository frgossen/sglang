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
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# model -> variants to run (per the "works for" notes; bs=1).
ALL_VARIANTS = [
    "no_cuda_graph",
    "full_cuda_graph",
    "pcg_torch_compile",
    "pcg_standalone",
    "bcg",
]
MODEL_VARIANTS = {
    "openai/gpt-oss-120b": ALL_VARIANTS,
    # "deepseek-ai/DeepSeek-V3.2": ["no_cuda_graph"],
    # "moonshotai/Kimi-K2.6": ["no_cuda_graph"],
    # "zai-org/GLM-4.7": ["no_cuda_graph"],
    "MiniMaxAI/MiniMax-M2.7": ALL_VARIANTS,
    "Qwen/Qwen3.6-35B-A3B": ALL_VARIANTS,
    "meta-llama/Meta-Llama-3-8B-Instruct": ALL_VARIANTS,
    "meta-llama/Meta-Llama-3-70B-Instruct": ALL_VARIANTS,
    "meta-llama/Llama-3.1-405B-Instruct-FP8": ALL_VARIANTS,
}

# Common flags for all models and variants.
BATCH_SIZES = ["1", "4", "16"]
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

# Model-specific flags.
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
        "0.9",
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

# Variant-specific flags.
PIECEWISE_TOKENS = ["512", "2048", "8192"]  # must cover bs*input_len
VARIANT_FLAGS = {
    "no_cuda_graph": [
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
    ],
    "full_cuda_graph": [],
    "pcg_torch_compile": [
        "--piecewise-cuda-graph-tokens",
        *PIECEWISE_TOKENS,
    ],
    "pcg_standalone": [
        "--enable-standalone-piecewise-cuda-graph",
        "--piecewise-cuda-graph-tokens",
        *PIECEWISE_TOKENS,
    ],
    "bcg": [
        "--enable-breakable-cuda-graph",
        "--piecewise-cuda-graph-tokens",
        *PIECEWISE_TOKENS,
    ],
}


OUT = Path(__file__).resolve().parent / "bench_cudagraphs_results"
TIMEOUT_S = 10 * 60  # per-benchmark timeout; bump if runs need longer
RECOVER_TIMEOUT_S = 120  # max wait for our GPU procs to clear between runs
WORKSPACE_MARKER = "workspace-sgl"  # only reap GPU procs belonging to us



def _workspace_gpu_pids():
    """GPU-holding PIDs whose process name references this workspace.

    The conda env is named `workspace-sgl`, so the interpreter path that
    nvidia-smi reports (e.g. .../envs/workspace-sgl/bin/python) carries the
    marker. Filtering on it keeps us from killing unrelated GPU jobs.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.splitlines():
        pid, _, name = line.partition(",")
        if pid.strip().isdigit() and WORKSPACE_MARKER in name:
            pids.append(int(pid.strip()))
    return sorted(set(pids))


def _recover_between_runs(proc):
    """Leave the GPUs clean for the next variant.

    bench_one_batch's TP workers can survive a crash (e.g. hung in NCCL
    teardown after one rank OOMs) and keep ~80 GB/GPU pinned, which then OOMs
    the next run at weight load. Kill the run's process group, then poll
    nvidia-smi until none of this workspace's processes are left on the GPUs,
    reaping stragglers that don't exit on their own.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()

    deadline = time.monotonic() + RECOVER_TIMEOUT_S
    while True:
        pids = _workspace_gpu_pids()
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if time.monotonic() > deadline:
            print(f"  [recover] workspace GPU procs still present: {pids}", flush=True)
            return
        time.sleep(2)


def main():
    OUT.mkdir(exist_ok=True)
    failures = []
    for model, variants in MODEL_VARIANTS.items():
        model_dir = OUT / model.replace("/", "__")
        model_dir.mkdir(exist_ok=True)
        for name in variants:
            variant_flags = VARIANT_FLAGS[name]
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
            # start_new_session puts the child in its own process group so that
            # _recover_between_runs can kill the whole tree (bench_one_batch
            # spawns TP workers); subprocess's own kill only reaps the direct
            # child and would leave those workers orphaned on the GPU.
            proc = subprocess.Popen(cmd, start_new_session=True)
            try:
                returncode = proc.wait(timeout=TIMEOUT_S)
                if returncode != 0:
                    failures.append((model, name, f"exit {returncode}"))
                elif not result_file.exists():
                    failures.append((model, name, "no result file written"))
            except subprocess.TimeoutExpired:
                failures.append((model, name, f"timeout after {TIMEOUT_S}s"))
            finally:
                _recover_between_runs(proc)

    print(f"\nResults written to {OUT}/<model>/*.jsonl")
    if failures:
        print("\nFailed runs (shown as n/a in the summary):")
        for model, name, why in failures:
            print(f"  {model} / {name}: {why}")
    print_summary()


def print_summary():
    for model, variants in MODEL_VARIANTS.items():
        names = list(variants)
        model_dir = OUT / model.replace("/", "__")
        # (bs, il, ol) -> {variant: row}
        rows = {}
        for name in variants:
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
