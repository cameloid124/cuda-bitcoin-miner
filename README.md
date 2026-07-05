# Python + CUDA Bitcoin Miner (Proof of Concept)

A Bitcoin (SHA-256d) Stratum V1 miner written in Python with a hand-tuned CUDA
kernel, driven through [CuPy](https://cupy.dev/). This is a **proof of
concept**: GPU mining of SHA-256 has been economically unviable for a decade
(ASICs are ~5 orders of magnitude more efficient), but the project demonstrates
a complete, protocol-correct mining stack in ~600 lines of code.

## Requirements

- NVIDIA GPU (Maxwell or newer; tuned on Ada / compute capability 8.9)
- CUDA 12.x driver
- Python 3.10+ with `pip install -r requirements.txt`
  (`cupy-cuda12x`, `numpy`)

## Usage

```bash
python miner.py -o stratum+tcp://pool.example.com:3333 -u WALLET.WORKER -p x
```

Run the test suites (require a GPU, no network):

```bash
python test_correctness.py   # kernel + header assembly vs. real block #125552 and hashlib
python test_e2e.py           # full Stratum session against a built-in mock pool
```

## Architecture

```
┌────────────────────────────  miner.py  ────────────────────────────┐
│  asyncio event loop                                                │
│                                                                    │
│  listen_for_jobs()          mine_job_loop()  (one task per job)    │
│  ├─ subscribe/authorize     ├─ per extranonce2:                    │
│  ├─ set_difficulty          │    coinbase → merkle root → header   │
│  ├─ mining.notify ──────────┤    prepare_job()  (upload midstate)  │
│  └─ submit results          └─ per 2^24 nonces:                    │
│                                  scan_nonce_range() in a thread    │
│                                  CPU-verify candidates → submit    │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │  (asyncio.to_thread)
┌──────────────────────────────────▼──────────────────  cuda_kernel.py ─┐
│  precompute_midstate  – 1 thread, once per extranonce2 (2^32 hashes) │
│  sha256d_mine         – 32768 blocks × 512 threads = 2^24 nonces     │
│                         per launch; 256 launches cover the full      │
│                         32-bit nonce space                           │
└──────────────────────────────────────────────────────────────────────┘
```

### Stratum V1 layer (`miner.py`)

- Single asyncio TCP connection with automatic reconnect (5 s backoff);
  authorization failure is treated as fatal instead of retrying.
- Requests are matched to responses through a pending-request map keyed by
  JSON-RPC id, so ids stay correct across reconnects.
- Handles `mining.notify`, `mining.set_difficulty` and
  `mining.set_extranonce`. Every `mining.notify` cancels the running job task
  and starts a fresh one, so the GPU always works on the newest job.
- Difficulty changes are picked up mid-job: only the 32-byte target is
  re-uploaded, no work is thrown away.
- Every GPU candidate nonce is re-verified on the CPU with `hashlib` before
  `mining.submit` — a wrong share is never sent to the pool.

### Block header construction

The 76-byte header prefix (everything except the nonce) is rebuilt per
extranonce2. The endianness rules here are the classic source of "all shares
rejected" bugs, so they are worth spelling out:

| Field | Wire format (notify) | Header format |
|---|---|---|
| `version`, `ntime`, `nbits` | big-endian hex | packed little-endian |
| `prevhash` | each 4-byte word byte-swapped | each word reversed individually (**not** a full 32-byte reversal) |
| merkle root | computed locally | raw `sha256d` output, no reversal |
| nonce | — | little-endian in header, submitted as **big-endian** hex |

The share check itself interprets the `sha256d` digest as a 256-bit
little-endian integer and compares it against the target derived from pool
difficulty (`target = diff1_target / difficulty`, computed with exact rational
arithmetic — float division would truncate a 256-bit value to 53 bits).

Header assembly is validated in `test_correctness.py` by byte-exact
reconstruction of mainnet block #125552 from Stratum-formatted inputs.

### CUDA kernel (`cuda_kernel.py`)

The kernel is compiled at import time via NVRTC (`cp.RawModule`) for the
native architecture of the active GPU. Key implementation traits:

- **Midstate caching.** The first 64 bytes of the header (version, prevhash,
  28 bytes of merkle root) are constant across the whole nonce space, so their
  SHA-256 compression runs once per extranonce2. Each mining thread performs
  exactly 2 compressions per nonce (tail chunk + second hash of SHA-256d)
  instead of 3.
- **16-word rolling message schedule.** The message schedule keeps only a
  16-entry window that lives entirely in registers after full unrolling
  (40 registers/thread, zero local-memory spills). Benchmarked ~15% faster
  than the naive flat `W[64]` layout, which spills to local memory.
- **Hardware intrinsics.** Rotations use the funnel-shift instruction
  (`__funnelshift_r`), byte swaps use a single `PRMT` (`__byte_perm`), and
  `CH`/`MAJ` are single `LOP3` truth-table instructions via inline PTX.
  SHA-256 round constants live in `__constant__` memory.
- **Scalar work parameters.** The 12 header tail bytes (merkle tail, ntime,
  nbits) are passed as three pre-swapped scalar kernel arguments; threads only
  read global memory for the 8-word midstate (broadcast via the read-only
  cache). There is **no** per-thread global "found" flag polling — the
  previous design issued ~16 M serialized atomic reads per launch.
- **Full-precision target compare.** The digest is compared word-by-word
  (most-significant first, byte-swapped to match the little-endian hash
  interpretation) against the full 256-bit target, so fractional difficulties
  work correctly.
- **Multi-result reporting.** Candidates are appended to a 16-slot buffer via
  `atomicAdd`, so several shares found in one batch are all captured; the
  batch is never aborted early (finding a share doesn't invalidate the rest of
  the nonce range).

### Host-side scheduling

- Grid: 32768 blocks × 512 threads = exactly 2²⁴ nonces per launch
  (~38 ms per launch on an RTX 4060 Laptop); 256 launches sweep the whole
  32-bit nonce space before extranonce2 is incremented. Block/thread shape was
  benchmark-tuned (512 threads/block beat 128 and 256 on Ada).
- All device buffers are allocated once at import and reused; per-launch
  traffic is a 4-byte memset, the kernel launch, and a 4-byte result read.
- Kernel launches run in `asyncio.to_thread`, so the event loop keeps
  servicing the Stratum socket while the GPU works (CuPy's synchronize
  releases the GIL). The ~38 ms batch size keeps job-switch latency low.

## Measured performance

RTX 4060 Laptop GPU (power-limited to ~50 W, SM clock ~2.0 GHz):
**~435 MH/s sustained**, GPU utilization 100%.

For scale: at current network difficulty this is ~10⁹× short of an expected
block per year — which is exactly why this is a proof of concept.

## Possible further optimizations

- Precompute the first 3 rounds of the tail-chunk compression (nonce enters
  the schedule at `W[3]`) and parts of the message-schedule expansion on the
  host, cgminer-style (~2–4%).
- Double-buffer launches on two CUDA streams to hide the (already small)
  launch/readback gap.
- `ntime` rolling to extend the search space without touching the coinbase.
