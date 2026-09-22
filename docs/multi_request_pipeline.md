# Multi-request pipeline (EdgeShard-style)

`DIAL-8B` can keep several independent inference sessions in the Master and
admit concurrent HTTP requests into those sessions. This enables inter-request
pipeline overlap across the existing layer shards:

```text
time ->
request A: Master shard -> Worker 1 -> Worker 2 -> Master/output
request B:                  Master shard -> Worker 1 -> Worker 2 -> Master/output
request C:                                   Master shard -> Worker 1 -> ...
```

This is request/micro-batch pipeline parallelism. Layers inside one request are
still evaluated in model order; the optimization comes from different requests
occupying different devices at the same time.

## Enable

Add the following option to the Master command:

```bash
--pipeline-concurrency 3
```

`1` is the default and preserves the original fully serialized behavior. A
reasonable starting value is the number of physical pipeline stages. Increase
it only after measuring throughput and tail latency.

Each slot owns:

- an independent local KV cache;
- an independent dialog/token state;
- one independent persistent TCP connection to each Worker;
- an independent Worker-side KV cache, because Worker cache lifetime is scoped
  to a TCP connection;
- isolated per-request distributed timing counters.

The slot queue is bounded. Requests beyond `--pipeline-concurrency` wait for an
available slot rather than allocating unbounded KV caches. Admission and queue
wait are logged as `pipeline admitted ... to slot ...`.

## Correctness and operational notes

- Do not share a conversation across simultaneous requests. Each HTTP request
  is a complete, independent dialog, as in the existing API.
- Weight tensors loaded from safetensors are cheap views in the normal dense
  path, but optional converted/quantized backends may allocate per-slot runtime
  state. Measure RAM/VRAM before choosing a high concurrency value.
- The Worker already accepts multiple Master connections concurrently and
  creates a fresh cache per connection. No protocol change is required.
- Pipeline concurrency mainly improves aggregate throughput under multiple
  simultaneous requests. It does not guarantee lower latency for one isolated
  request and may increase latency under saturation.

## Benchmark

```bash
python3 tools/benchmark_multi_request_pipeline.py \
  --url http://127.0.0.1:8082/api/v1/chat/completions \
  --requests 12 \
  --concurrency 3 \
  --prompt "Briefly explain edge inference."
```

Compare server `--pipeline-concurrency 1` and `3` using the same prompts,
sample length, topology, and warmed-up model. Report both requests/s and
p50/p95 latency.
