"""Process metrics. Never include embeddings or raw frames."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


class Metrics:
    def __init__(self) -> None:
        self.counters: dict[str, float] = defaultdict(float)
        self.gauges: dict[str, float] = {}
        self.timings: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, value: float = 1.0) -> None:
        self.counters[name] += value

    def set(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def time(self, name: str):
        start = time.perf_counter()

        def _done() -> float:
            dt = time.perf_counter() - start
            self.timings[name].append(dt)
            return dt

        return _done

    def snapshot(self) -> dict[str, Any]:
        cpu = psutil.cpu_percent(interval=None) if psutil else None
        mem = psutil.virtual_memory().percent if psutil else None
        gpu = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu = {
                    "device": torch.cuda.get_device_name(0),
                    "mem_allocated": int(torch.cuda.memory_allocated(0)),
                }
        except Exception:
            gpu = None
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timings_mean": {
                k: (sum(v) / len(v) if v else 0.0) for k, v in self.timings.items()
            },
            "cpu_percent": cpu,
            "ram_percent": mem,
            "gpu": gpu,
        }
