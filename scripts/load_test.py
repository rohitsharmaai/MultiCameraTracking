#!/usr/bin/env python3
"""Simple authenticated API load test. Does not send embeddings."""

from __future__ import annotations

import argparse
import os
import statistics
import time

import httpx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--api-key", default=os.environ.get("API_KEY", "change-me"))
    args = p.parse_args()
    times = []
    ok = 0
    with httpx.Client(timeout=10.0) as client:
        for _ in range(args.requests):
            t0 = time.perf_counter()
            r = client.get(f"{args.base}/persons", headers={"X-API-Key": args.api_key})
            times.append(time.perf_counter() - t0)
            if r.status_code == 200:
                ok += 1
    print(f"ok={ok}/{args.requests}")
    print(f"mean={statistics.mean(times):.4f}s p95={statistics.quantiles(times, n=20)[18]:.4f}s")
    return 0 if ok == args.requests else 1


if __name__ == "__main__":
    raise SystemExit(main())
