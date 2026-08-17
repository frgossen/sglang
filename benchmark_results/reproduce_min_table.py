#!/usr/bin/env python3
"""Reproduce min_table.tsv: torch.compile vs eager steady-state denoise latency.

Runs 8 sequential requests per arm, discards request 1, reports the min of 2..8.
See README.md for why (short version: the stock benchmark times one request after a
1-step warmup, so compile cost lands inside the measurement -- up to 6x inflation).

    python3 benchmark_results/reproduce_min_table.py            # all 35 presets
    python3 benchmark_results/reproduce_min_table.py zimage     # subset
    python3 benchmark_results/reproduce_min_table.py --render    # table only

Resumable: cells already in reproduce_results.jsonl are skipped.
"""

import importlib.util, json, os, shutil, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py"
RESULTS = REPO / "benchmark_results" / "reproduce_results.jsonl"
TABLE = REPO / "benchmark_results" / "min_table.tsv"
ITERS_TSV = REPO / "benchmark_results" / "iterations.tsv"
CACHE = Path.home() / ".cache/sgl_diffusion/torch_compile_cache"
ITERS = 8

# Preset repo is 403-gated; the reference issue used this public mirror too.
MIRROR = {"ideogram4-fp8": "cocktailpeanut/ideogram-4-fp8"}
# force_eager: torch.compile changes H3's numerical output, so it has no compile arm.
NO_COMPILE = {"minimax-h3-t2va": "n/a (force_eager)"}

spec = importlib.util.spec_from_file_location("bench", BENCH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
if "iterations" not in bench.build_sglang_cmd.__code__.co_varnames:
    sys.exit("bench_diffusion_denoise.py needs the --iterations patch (see README.md)")
for preset, repo in MIRROR.items():
    bench.MODELS[preset]["path"] = repo   # parent: used for the weight-purge path


def run(model, mode):
    if mode == "compile":
        shutil.rmtree(CACHE, ignore_errors=True)   # measure without a pre-populated cache
    gpus = bench.required_gpus_for_model(model)
    # pick_idle_gpus() rejects large-VRAM cards, so pin explicitly.
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ",".join(map(str, range(gpus))),
                        "NCCL_NVLS_ENABLE": "0", "CI": "1",
                        "SGLANG_DIFFUSION_SYNC_STAGE_PROFILING": "1",
                        "FLASHINFER_DISABLE_VERSION_CHECK": "1"}
    cmd = [sys.executable, str(BENCH), "--model", model, "--iterations", str(ITERS),
           "--label", mode, "--output-dir", str(REPO / "benchmark_results/reproduce")]
    if mode == "eager":
        cmd.append("--no-torch-compile")
    if model in MIRROR:                      # the child re-imports the helper unpatched
        cmd += ["--model-path", MIRROR[model]]
    log = REPO / "benchmark_results/reproduce/logs" / f"{model}_{mode}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as fh:
        try:
            subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                           cwd=REPO, timeout=18000)
        except subprocess.TimeoutExpired:
            pass
    # the helper writes its results dict, including the per-request arrays
    summary = REPO / "benchmark_results/reproduce" / f"results_{mode}.json"
    dn, e2 = [], []
    if summary.exists():
        for r in json.loads(summary.read_text()):
            if r.get("model") == model:
                dn = r.get("denoise_iterations_s") or []
                e2 = r.get("e2e_iterations_s") or []
    rec = {"model": model, "mode": mode, "gpus": gpus, "elapsed_s": round(time.time()-t0, 1),
           "denoise_iterations_s": [round(x, 3) for x in dn],
           "e2e_iterations_s": [round(x, 3) for x in e2]}
    if len(dn) > 1:
        rec |= {"status": "ok", "denoise_min_s": round(min(dn[1:]), 3)}
        if len(e2) > 1:
            rec["e2e_min_s"] = round(min(e2[1:]), 3)
    else:
        rec["status"] = "error"
    return rec


def render():
    data = {}
    for line in RESULTS.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            data[(r["model"], r["mode"])] = r

    def cell(model, mode):
        r = data.get((model, mode)) or {}
        if r.get("status") == "n/a":
            return "n/a", "n/a"
        v = r.get("denoise_iterations_s") or []
        if r.get("status") != "ok" or len(v) < 2:
            return "not run", "not run"
        e2 = r.get("e2e_iterations_s") or []
        return f"{min(v[1:]):.3f}", f"{min(e2[1:]):.3f}" if len(e2) > 1 else "-"

    models = sorted((m for m, mode in data if mode == "eager" and cell(m, "eager")[0] != "not run"),
                    key=lambda m: -float(cell(m, "eager")[0]))
    lines = ["\teager\t\ttorch.compile\t",
             "Model preset\tdenoise latency\te2e\tdenoise latency\te2e"]
    for m in models:
        lines.append("\t".join([m, *cell(m, "eager"), *cell(m, "compile")]))
    TABLE.write_text("\n".join(lines) + "\n")

    # per-request times, so convergence is inspectable without re-measuring
    raw = ["preset\tmode\t" + "\t".join(f"req{i}" for i in range(1, ITERS + 1))]
    for (m, mode), r in sorted(data.items()):
        v = r.get("denoise_iterations_s") or []
        if r.get("status") == "n/a":
            raw.append("\t".join([m, mode] + ["n/a"] * ITERS))
        elif r.get("status") == "ok" and len(v) >= 2:
            raw.append("\t".join([m, mode] + [f"{x:.3f}" for x in v]))
    ITERS_TSV.write_text("\n".join(raw) + "\n")
    print(f"wrote {TABLE.relative_to(REPO)} and {ITERS_TSV.relative_to(REPO)} "
          f"({len(models)} presets)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--render" in sys.argv[1:]:
        return render()
    done = {(r["model"], r["mode"]) for line in
            (RESULTS.read_text().splitlines() if RESULTS.exists() else [])
            if line.strip() and (r := json.loads(line)).get("status") in ("ok", "n/a")}
    presets = args or sorted(bench.MODELS, key=bench.required_gpus_for_model)
    for i, model in enumerate(presets, 1):
        print(f"[{i}/{len(presets)}] {model}", flush=True)
        for mode in ("eager", "compile"):
            if (model, mode) in done:
                continue
            if mode == "compile" and model in NO_COMPILE:
                rec = {"model": model, "mode": mode, "status": "n/a",
                       "reason": NO_COMPILE[model], "elapsed_s": 0.0}
            else:
                rec = run(model, mode)
                if rec["status"] != "ok":                      # one retry, clean cache
                    shutil.rmtree(CACHE, ignore_errors=True)
                    rec = run(model, mode)
            with open(RESULTS, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {mode:8} {rec['status']:6} min={rec.get('denoise_min_s', '-')}", flush=True)
        # weights for all 35 presets are several TB; drop each model after use
        d = Path.home() / ".cache/huggingface/hub" / ("models--" + bench.MODELS[model]["path"].replace("/", "--"))
        shutil.rmtree(d, ignore_errors=True)
    render()


if __name__ == "__main__":
    main()
