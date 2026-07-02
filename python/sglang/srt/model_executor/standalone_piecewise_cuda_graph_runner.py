# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Standalone piecewise CUDA graph (SPCG) runner.

Identical capture/replay harness to :class:`BreakableCudaGraphRunner`, but the
per-segment capture is delegated to the standalone ``piecewise_cuda_graphs``
package instead of the breakable runner's hand-rolled segment capture. Keeping
the harness shared (via subclassing) means the two runners differ *only* in the
capture mechanism, so they can be benchmarked against each other fairly.

Selected with ``--enable-standalone-piecewise-cuda-graph`` (see
``ModelRunner.init_piecewise_cuda_graphs``).
"""

from __future__ import annotations

from contextlib import nullcontext

from piecewise_cuda_graphs import CUDAGraphSequence, piecewise_graph

from sglang.srt.model_executor.breakable_cuda_graph_runner import (
    BreakableCudaGraphRunner,
)


class StandalonePiecewiseCudaGraphRunner(BreakableCudaGraphRunner):
    """Breakable-style runner backed by the ``piecewise_cuda_graphs`` package."""

    # No sglang-side capture flag: ``piecewise_graph`` (entered in
    # ``_make_capture_context``) sets pcg's own context state, which the
    # attention layers read via ``is_in_piecewise_graph()``.
    _enable_graph_ctx = staticmethod(nullcontext)
    _log_prefix = "[SPCG]"

    def _make_graph(self, pool):
        # pcg attaches the shared memory pool to the sequence at construction.
        return CUDAGraphSequence(pool=pool)

    def _make_capture_context(self, graph, pool, stream):
        # Call the pcg package directly — no adapter. `pool` is already bound to
        # the ``CUDAGraphSequence`` above, so it is not passed to ``piecewise_graph``.
        return piecewise_graph(graph, stream=stream)
