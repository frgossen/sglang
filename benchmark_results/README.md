# Reproducing the torch.compile vs eager comparison

`min_table.tsv` reports steady-state denoise latency for all 35 SGLang diffusion
presets, eager vs `torch.compile`. This directory contains everything needed to
regenerate it.

    python3 benchmark_results/reproduce_min_table.py

Roughly 12–16 h on 8×H100 for the full set (weight downloads dominate). Resumable:
completed cells are skipped, so it is safe to interrupt and re-run.

    python3 benchmark_results/reproduce_min_table.py zimage flux2-klein   # subset
    python3 benchmark_results/reproduce_min_table.py --render             # table only

Results stream to `reproduce_results.jsonl` (one JSON line per cell, including the
full per-request array), logs to `reproduce/logs/`. Weights are deleted after each
model — the full set is several TB.

`iterations.tsv` is the reference data behind the committed `min_table.tsv`: the
per-request denoise time for every cell, so the convergence behaviour can be
inspected without re-measuring. `min_table.tsv` is the min of `req2..req8`.
`--render` regenerates both files from existing results.

## Prerequisites

* **SGLang commit.** The published numbers were taken at
  `f6cbdc1dd1c892203b735c8d7f131e2a9ba31f23`, the commit used by the reference
  issue (`BBuf/how-to-optim-algorithm-in-cuda#21`). Any commit works, but numbers
  are only comparable to that issue's table at that commit.
* **Install:** `pip install -e "python[diffusion]"`, plus `sglang-kernel` matching
  the commit's pin.
* **`HF_TOKEN` exported.** Several presets are gated. `flux` needs FLUX.1-dev
  access, `flux2` needs FLUX.2-dev (separate acceptance). Without a token the
  helper fails those cells early with a clear message.
* **Disk:** the full preset set is several TB. The script deletes each model's
  weights after measuring it, so peak usage is bounded by the largest preset.
* **Input images.** 10 presets need `cat.png` and `mova-720p` needs
  `mova_single_person.jpg`. Fetch once:

      ASSETS=inputs/diffusion_benchmark/figs && mkdir -p $ASSETS
      curl -sSL -o $ASSETS/cat.png \
        https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png
      curl -sSL -o $ASSETS/mova_single_person.jpg \
        https://github.com/OpenMOSS/MOVA/raw/main/assets/single_person.jpg

## Required patch (1 change)

The script depends on one addition to the in-repo benchmark helper
`python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile/scripts/bench_diffusion_denoise.py`:

* `run_benchmark_once` → **`run_benchmark`**, gaining an `iterations` parameter and
  a `--iterations` CLI flag. With `iterations > 1` the preset prompt is repeated
  through a prompt file, which the CLI turns into that many *sequential* requests
  in one process; the reported latency is then derived from requests `2..N`.
  `iterations=1` (the default) builds a command byte-identical to the stock one, so
  faithful single-shot reproduction is preserved.
* The results dict is written to `<output-dir>/results_<label>.json`, which is how
  the per-request arrays reach any caller — the perf dumps cannot carry them.

## Why the methodology differs from the reference issue

The stock benchmark measures **one** request after a warmup that runs only
`--warmup-steps` (default **1**) of an N-step denoise schedule. Residual
`torch.compile` work therefore lands inside the timed request. Measured effects:

| preset | stock single-shot | steady state | overstatement |
|---|---:|---:|---:|
| flux2-klein | 1.845 (issue) | 0.275 | **6.7×** |
| zimage | 2.578 (issue) | 0.623 | **4.1×** |
| ltx2 | 25.710 (issue) | 7.408 | **3.5×** |

Three further details, each of which changed a number materially:

1. **Min of requests 2..8.** Convergence is model-dependent — most presets settle
   after 1 request, `zimage-base` after 3. A fixed 8 with `min` is the safe default.
   If the tail still slopes downward, that preset has a recompilation bug rather
   than a warmup shortfall.
2. **`--warmup-steps` alone is not sufficient.** With full-depth warmup and zero
   recompiles, flux2-klein's first request is still 51% above steady state
   (0.411 vs 0.272). Discarding the first measured request is the load-bearing fix.
3. **The compile cache is cleared before each compile arm**, so a cache populated
   by earlier models cannot influence a measurement. The script does this
   automatically.

## Metric definitions

Identical to the reference issue, so numbers are comparable:

* **denoise** — sum of pipeline stages named `*DenoisingStage` / `*RefinementStage`
  (excluding `*BeforeDenoisingStage` setup). Two-stage pipelines such as LTX-2.3
  sum both stages.
* **e2e** — end-to-end time for one request.

## Other files here

| file | contents |
|---|---|
| `min_table.tsv` | the result — steady-state denoise/e2e per preset |
| `iterations.tsv` | per-request denoise times behind `min_table.tsv` |
| `reproduce_min_table.py` | measures every preset and regenerates both files |
