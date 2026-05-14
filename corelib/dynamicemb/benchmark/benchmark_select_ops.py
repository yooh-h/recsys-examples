# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark dyn_emb `select` and `select_index` (CUB DeviceSelect::Flagged).

Dimensions:
  - num_items: total length of flags / inputs.
  - true_ratio: fraction of True in flags (how many rows are kept).
  - value_repeat_ratio: fraction of positions that reuse a smaller value pool
    (inputs = base[arange(N) % unique_count]); simulates skewed keys. Only
    affects `select` inputs pattern; `select_index` does not read values.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

try:
    from dynamicemb_extensions import select, select_index
except ImportError as e:  # pragma: no cover
    print(
        "Import failed: install the package or set PYTHONPATH so that "
        "`dynamicemb_extensions` is importable.",
        file=sys.stderr,
    )
    raise e


@dataclass
class BenchRow:
    op: str
    num_items: int
    true_ratio: float
    value_repeat_ratio: float
    time_ms_median: float
    time_ms_mean: float
    warmup_iters: int
    bench_iters: int


def _unique_count(num_items: int, value_repeat_ratio: float) -> int:
    """Larger repeat_ratio => smaller unique pool (more repeated values)."""
    r = min(max(value_repeat_ratio, 0.0), 0.999999)
    u = max(1, int(num_items * (1.0 - r)))
    return min(u, num_items)


def build_tensors(
    num_items: int,
    true_ratio: float,
    value_repeat_ratio: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flags = torch.rand(num_items, device=device) < true_ratio
    unique_n = _unique_count(num_items, value_repeat_ratio)
    base = torch.arange(unique_n, device=device, dtype=dtype) * 7919
    inputs = base[torch.arange(num_items, device=device) % unique_n]
    outputs = torch.empty_like(inputs)
    output_indices = torch.empty(num_items, device=device, dtype=dtype)
    num_selected = torch.empty(1, device=device, dtype=torch.int64)
    return flags, inputs, outputs, output_indices, num_selected


def time_cuda_kernel(
    fn: callable,
    warmup: int,
    iters: int,
) -> List[float]:
    times_ms: List[float] = []
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def bench_select(
    flags: torch.Tensor,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    num_selected: torch.Tensor,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    def run():
        select(flags, inputs, outputs, num_selected)

    t = time_cuda_kernel(run, warmup, iters)
    return statistics.median(t), statistics.mean(t)


def bench_select_index(
    flags: torch.Tensor,
    output_indices: torch.Tensor,
    num_selected: torch.Tensor,
    warmup: int,
    iters: int,
) -> Tuple[float, float]:
    def run():
        select_index(flags, output_indices, num_selected)

    t = time_cuda_kernel(run, warmup, iters)
    return statistics.median(t), statistics.mean(t)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sizes",
        type=str,
        default="4096,65536,1048576",
        help="Comma-separated num_items list.",
    )
    p.add_argument(
        "--true-ratios",
        type=str,
        default="0.01,0.1,0.5,0.9",
        help="Comma-separated fraction of True in flags.",
    )
    p.add_argument(
        "--repeat-ratios",
        type=str,
        default="0.0,0.5,0.9",
        help="Comma-separated value repeat ratios for input keys (0=no repeat).",
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--dtype", type=str, default="int64", choices=("int32", "int64"))
    p.add_argument("--json", action="store_true", help="Print one JSON object per line.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.int32 if args.dtype == "int32" else torch.int64

    sizes = parse_int_list(args.sizes)
    true_ratios = parse_float_list(args.true_ratios)
    repeat_ratios = parse_float_list(args.repeat_ratios)

    rows: List[BenchRow] = []
    for n in sizes:
        for tr in true_ratios:
            for rr in repeat_ratios:
                flags, inputs, outputs, output_indices, num_selected = build_tensors(
                    n, tr, rr, device, dtype
                )
                m_sel, mean_sel = bench_select(
                    flags, inputs, outputs, num_selected, args.warmup, args.iters
                )
                rows.append(
                    BenchRow(
                        "select",
                        n,
                        tr,
                        rr,
                        m_sel,
                        mean_sel,
                        args.warmup,
                        args.iters,
                    )
                )
                m_idx, mean_idx = bench_select_index(
                    flags, output_indices, num_selected, args.warmup, args.iters
                )
                rows.append(
                    BenchRow(
                        "select_index",
                        n,
                        tr,
                        rr,
                        m_idx,
                        mean_idx,
                        args.warmup,
                        args.iters,
                    )
                )

    if args.json:
        for r in rows:
            d: Dict[str, Any] = {
                "op": r.op,
                "num_items": r.num_items,
                "true_ratio": r.true_ratio,
                "value_repeat_ratio": r.value_repeat_ratio,
                "time_ms_median": r.time_ms_median,
                "time_ms_mean": r.time_ms_mean,
                "warmup_iters": r.warmup_iters,
                "bench_iters": r.bench_iters,
            }
            print(json.dumps(d))
        return

    hdr = f"{'op':<14} {'N':>12} {'true%':>8} {'repeat%':>9} {'med_ms':>10} {'mean_ms':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r.op:<14} {r.num_items:12d} {r.true_ratio * 100:7.1f}% "
            f"{r.value_repeat_ratio * 100:8.1f}% {r.time_ms_median:10.4f} {r.time_ms_mean:10.4f}"
        )


if __name__ == "__main__":
    main()
