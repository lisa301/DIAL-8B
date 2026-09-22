#!/usr/bin/env python3
"""Dependency-free concurrent benchmark for DIAL's chat API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * fraction)]


def request_once(url: str, prompt: str, timeout: float) -> tuple[float, int]:
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return time.perf_counter() - started, int(payload.get("generated_tokens", 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--prompt", default="Briefly explain edge inference.")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")

    wall_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(request_once, args.url, args.prompt, args.timeout)
            for _ in range(args.requests)
        ]
        results = [future.result() for future in futures]
    wall_s = time.perf_counter() - wall_started

    latencies = [result[0] for result in results]
    tokens = sum(result[1] for result in results)
    print(
        json.dumps(
            {
                "requests": args.requests,
                "client_concurrency": args.concurrency,
                "wall_s": wall_s,
                "requests_per_second": args.requests / wall_s,
                "tokens": tokens,
                "aggregate_tokens_per_second": tokens / wall_s,
                "latency_mean_s": statistics.fmean(latencies),
                "latency_p50_s": percentile(latencies, 0.50),
                "latency_p95_s": percentile(latencies, 0.95),
                "latency_max_s": max(latencies),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
