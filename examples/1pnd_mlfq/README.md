# 1P-nD MLFQ baseline

This is an experimental SGLang execution path, enabled by `--mlfq-role`.
Send `/generate` or OpenAI requests directly to P; no Rust router or native PD
bootstrap is involved. Existing serving paths are unchanged when the role is off.

## Behavior

- P processes requests in FCFS order, one prefill at a time. Completed prefill
  KV is copied to **P host memory**, and the request enters Q0.
- Each D pulls the first capacity-feasible request in Q0/Q1/Q2 FIFO order.
  Each D runs **one request at a time**, with token-boundary preemption.
- Only decode execution time consumes a quantum. Transfer, restore and RPC waits
  do not. An iteration may overshoot the quantum; prefill is never charged.
- On expiry, D returns the full KV and output-token history to P. Only after P's
  acknowledgement does D release GPU slots. The request enters the next queue;
  Q2 rotates at Q2. Any compatible D can restore it using fresh GPU slots.
- Q1/Q2 requests waiting at least `--mlfq-aging-ms` are promoted to Q0. Running
  and transferring requests are not aged. Promotion starts a fresh Q0 quantum.
- P retains original client/stream state; decoder progress is returned through P.
  Monotonic claim leases reject duplicate/stale returns. Cancellation and worker timeout
  release P's checkpoint reservation and terminate the request.

## Minimal implementation

Only two wire messages are used: `pull` claims a slice; `update` reports progress,
returns KV, or completes the request. There is no separate worker identity,
restore acknowledgement message, or cumulative service counter. The first update
confirms restoration. A returned checkpoint must be acknowledged before D frees KV.
One main loop handles input ingestion, pause/idle handling and either P or D work.

| Class / method | Purpose |
| --- | --- |
| `Job` | Generation state, KV, memory reservation, queue level, lease and timestamps |
| `MLFQPool.__init__` | Validate limits and initialize the three queues |
| `reserve` | Reserve peak checkpoint memory before prefill; apply backpressure when full |
| `publish` | Publish completed prefill to Q0 |
| `_store` | Save KV/output state, clear lease, timestamp and enqueue |
| `dispatch` | Handle claim, progress, return and completion under one lock |
| `tick` / `_expire` / `_age` | Reclaim timed-out requests and promote starved queued requests |
| `cancel` / `_remove` | Remove a request and release its reservation |
| `MLFQRuntime.__init__` | Check supported configuration and construct transport/state |
| `run` | Drive the common P/D event loop and clean up transport |
| `_serve` / `_rpc` | Serve P's background broker and send D's synchronous messages |
| `_prefill` | Process one FCFS request, save CPU KV and release P's GPU slots |
| `_decode` | Restore one claimed request and execute/report one decode slice |
| `_batch` | Allocate a batch and optionally restore CPU KV to fresh GPU slots |
| `_step` | Execute inference, append the sampled token, update finish state and measure time |
| `_checkpoint` | Copy valid KV to CPU and serialize it |
| `_outputs` | Apply D updates to P's original request and stream responses |
| `_state` | Extract generation state; retain client connection state on P |
| `abort` | Apply client cancellation to the pool |
| `reject_unsupported` / `_reject` | Reject requests this experimental path cannot resume correctly |

## Four 24GB GPUs: 1P3D

Run these commands on Linux with a working SGLang CUDA installation. All workers
must use identical model weights, revision, dtype and CPU-copy layout.

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct
COMMON="--model-path $MODEL --dtype float16 --tp 1 --page-size 1 \
--max-running-requests 1 --max-total-tokens 8192 --context-length 4096 \
--disable-radix-cache --disable-overlap-schedule --skip-server-warmup \
--mlfq-endpoint tcp://127.0.0.1:19000"

# Start in separate terminals (or redirect each process to its own log).
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server $COMMON --port 30000 \
  --mlfq-role prefill --mlfq-quanta-ms '[20,40,80]' \
  --mlfq-aging-ms 1000 --mlfq-host-memory-gb 4 --max-queued-requests 256
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server $COMMON --port 30001 --mlfq-role decode
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server $COMMON --port 30002 --mlfq-role decode
CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server $COMMON --port 30003 --mlfq-role decode
```

```bash
curl http://127.0.0.1:30000/generate -H 'Content-Type: application/json' -d \
'{"text":"Explain virtual memory.","sampling_params":{"temperature":0,"max_new_tokens":128},"stream":true}'
```

`MLFQ start`, `MLFQ return` and `MLFQ promote` logs show request ID and queue
transitions. Use `[1,2,4]` for an initial smoke test to force frequent returns;
use measured, longer quanta for throughput experiments.

## Validation

```bash
PYTHONPATH=python python test/registered/unit/disaggregation/test_mlfq.py
```

For inference correctness on four GPUs, initially run **1P2D** on GPUs 0/1/2.
Use GPU 3 for an ordinary reference server with the same model/common flags:

```bash
CUDA_VISIBLE_DEVICES=3 python -m sglang.launch_server $COMMON --port 30004
MLFQ_PREFILL_URL=http://127.0.0.1:30000 \
MLFQ_REFERENCE_URL=http://127.0.0.1:30004 \
PYTHONPATH=python python test/manual/test_mlfq_pd.py
```

This checks concurrent greedy output IDs against uninterrupted generation and
checks streaming prefixes for lost/repeated tokens. Stop the reference server
before using GPU 3 as the third D. CUDA inference tests require actual GPUs;
passing CPU queue tests alone does not validate inference correctness.
For this verification, start P with `--mlfq-quanta-ms '[1,2,4]'` so the requests
exercise repeated checkpoint returns. Queue quanta and aging are controlled by P;
the values configured on D do not override P's tickets.

## Scope and limits

The first baseline supports CUDA, plain full-attention MHA/GQA FP16/BF16 models,
TP/PP/DP/CP=1, page size 1, greedy text generation, no penalties, no prefix cache,
no sessions, LoRA, speculative decoding, SWA/Mamba, or auxiliary outputs.
It deliberately does not yet implement continuous batching within a D. Compare
future policies with **the same batch/concurrency and memory settings**.

P reserves a conservative **peak** checkpoint allowance before prefill, retaining
the reservation even while the request runs on D. When the budget is exhausted,
P stops prefill; D continues consuming queued work. This bounds admitted checkpoint
state, not total process RSS: tensors, serialization/network copies and pending
input requests need additional memory. Keep an input queue limit and RAM headroom.

This correctness-first baseline uses synchronous CPU KV copies and ZMQ transfers,
plus progress RPCs at token boundaries; their overhead is real and must be measured.
It is not Mooncake/RDMA performance and does not claim throughput improvements.
ZMQ messages use Python serialization: expose the endpoint only to trusted workers.
Live weight updates and runtime topology changes are outside this experiment.
