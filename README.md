# Python + CUDA Bitcoin Miner (Proof of Concept)

A Bitcoin (SHA-256d) Stratum V1 miner written in Python with a hand-tuned CUDA
kernel, driven through [CuPy](https://cupy.dev/). This is a **proof of
concept**: GPU mining of SHA-256 has been economically unviable for a decade
(ASICs are ~5 orders of magnitude more efficient), but the project demonstrates
a complete, protocol-correct mining stack in ~600 lines of code.

## Requirements

- NVIDIA GPU with CUDA support (Maxwell or newer)
  - **Jetson (default):** Orin Nano / Orin NX / AGX Orin, JetPack **7.2**
    (L4T 39.2, CUDA 13.2)
  - **x86_64:** desktop or server GPU, CUDA 12.x driver
- Python 3.10+
- CuPy (platform-specific — see [Native install](#native-install) or [Docker](#docker))

## Usage

### Native install

```bash
pip install -r requirements.txt -r requirements-jetson.txt   # Jetson / JetPack 7.2
# or
pip install -r requirements.txt -r requirements-x86.txt        # x86_64 / CUDA 12.x

python miner.py -o stratum+tcp://pool.example.com:3333 -u WALLET.WORKER -p x
```

### Docker

The default Compose stack targets **Jetson Orin (ARM64, JetPack 7.2)**.
An x86_64 override is provided for desktop/server GPUs.

**Prerequisites**

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host
- On Jetson: JetPack 7.2 with the `nvidia-container` runtime (`runtime: nvidia`)

**Jetson Orin Nano (default)**

```bash
cp .env.example .env    # set STRATUM_URL, STRATUM_USER, STRATUM_PASSWORD
docker compose up --build
```

Builds `Dockerfile.jetson` (baseline: `nvcr.io/nvidia/pytorch:25.06-py3`, CuPy
CUDA 13.x for JetPack 7.2).

**x86_64 desktop / server**

```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.x86.yml up --build
```

Builds `Dockerfile.x86` (CUDA 12.6 runtime, CuPy CUDA 12.x).

Pool credentials are read from `.env` (`STRATUM_URL`, `STRATUM_USER`,
`STRATUM_PASSWORD`). GPU passthrough is enabled via `runtime: nvidia` and
`gpus: all`.

| File | Platform | Base image | CuPy |
|---|---|---|---|
| `Dockerfile.jetson` | ARM64 / Jetson | `nvcr.io/nvidia/pytorch:25.06-py3` | `cupy-cuda13x[ctk]` |
| `Dockerfile.x86` | x86_64 | `nvidia/cuda:12.6.3-runtime-ubuntu22.04` | `cupy-cuda12x[ctk]` |

Optional `.env` overrides for the default (Jetson) stack:

```env
MINER_DOCKERFILE=Dockerfile.jetson
MINER_IMAGE_TAG=jetson
DOCKER_PLATFORM=linux/arm64
```

### Tests

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
│  ├─ set_difficulty          │    coinbase → merkle root           │
│  ├─ mining.notify ──────────┤    └─ per rolled ntime:              │
│  └─ submit results          │         scan_header() over 2^32      │
│                             │         nonces, 2 batches in flight  │
│                             └─ CPU-verify candidates → submit      │
└──────────────────────────────────┬─────────────────────────────────┘
                                   │  (asyncio.to_thread)
┌──────────────────────────────────▼──────────────────  cuda_kernel.py ─┐
│  GpuScanner            – double-buffered over 2 CUDA streams          │
│  _precompute_job_words – host midstate + tail rounds 0-2 + schedule   │
│                          constants, once per (extranonce2, ntime)     │
│  sha256d_mine          – 32768 blocks × 512 threads = 2^24 nonces     │
│                          per launch; 256 launches cover the full      │
│                          32-bit nonce space                           │
└──────────────────────────────────────────────────────────────────────┘
```

### Stratum V1 layer (`miner.py`)

- Single asyncio TCP connection with automatic reconnect (5 s backoff);
  authorization failure is treated as fatal instead of retrying.
- Requests are matched to responses through a pending-request map keyed by
  JSON-RPC id, so ids stay correct across reconnects.
- Handles `mining.notify`, `mining.set_difficulty` and
  `mining.set_extranonce`. Every `mining.notify` cancels the running job task
  and starts a fresh one, so the GPU always works on the newest job. On
  cancellation the GPU pipeline is drained before the next job is prepared.
- Difficulty changes are picked up mid-job: only the 32-byte target is
  re-uploaded, no work is thrown away.
- Every GPU candidate nonce is re-verified on the CPU with `hashlib` before
  `mining.submit` — a wrong share is never sent to the pool. Accept/reject
  responses are logged with the full share context (job, nonce, extranonce2,
  ntime), carried through the pending-request map.

### Block header construction

The 76-byte header prefix (everything except the nonce) is rebuilt for each
(extranonce2, rolled ntime) pair. The endianness rules here are the classic
source of "all shares rejected" bugs, so they are worth spelling out:

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
  SHA-256 compression runs once per job. Each mining thread performs exactly 2
  compressions per nonce (tail chunk + second hash of SHA-256d) instead of 3.
- **Host-side early-round precomputation.** The nonce first enters the header
  tail at word `W[3]`, so everything before it is nonce-independent. A small
  pure-Python SHA-256 (in `_precompute_job_words`, unit-tested against
  `hashlib`) computes, once per job, a 20-word constant buffer: the chunk-1
  midstate, the working registers *after* rounds 0–2 of the tail chunk, and
  the constant parts of message-schedule words `W[16..19]`. Each of ~16 M
  threads therefore skips 3 compression rounds plus part of the schedule
  expansion and just adds its per-nonce contribution (`SIG0(nonce)` into
  `W[18]`, the nonce word into `W[19]`).
- **16-word rolling message schedule.** The message schedule keeps only a
  16-entry window that lives entirely in registers after full unrolling
  (zero local-memory spills). Benchmarked ~15% faster than the naive flat
  `W[64]` layout, which spills to local memory.
- **Hardware intrinsics.** Rotations use the funnel-shift instruction
  (`__funnelshift_r`), byte swaps use a single `PRMT` (`__byte_perm`), and
  `CH`/`MAJ` are single `LOP3` truth-table instructions via inline PTX.
  SHA-256 round constants live in `__constant__` memory.
- **Minimal global traffic.** The per-job constants and target are read from
  global memory and broadcast through the read-only cache; there is **no**
  per-thread global "found" flag polling (an earlier design issued ~16 M
  serialized atomic reads per launch).
- **Full-precision target compare.** The digest is compared word-by-word
  (most-significant first, byte-swapped to match the little-endian hash
  interpretation) against the full 256-bit target, so fractional difficulties
  work correctly.
- **Multi-result reporting.** Candidates are appended to a 16-slot buffer via
  `atomicAdd`, so several shares found in one batch are all captured; the
  batch is never aborted early (finding a share doesn't invalidate the rest of
  the nonce range).

### Host-side scheduling (`GpuScanner`)

- Grid: 32768 blocks × 512 threads = exactly 2²⁴ nonces per launch; 256
  launches sweep the whole 32-bit nonce space before the search advances.
  Block/thread shape was benchmark-tuned (512 threads/block beat 128 and 256
  on Ada).
- **Double buffering.** Two CUDA streams each carry an independent result
  buffer; the scheduler keeps two batches in flight (submit the next, then
  collect the oldest) so kernel execution overlaps the launch/readback of its
  neighbour and the GPU never idles between launches. Results are copied into
  pinned host memory via `memcpyAsync` for a truly asynchronous readback.
- All device buffers are allocated once and reused; per-launch traffic is a
  4-byte memset, the kernel launch, and a 68-byte result read.
- `submit`/`collect` run in `asyncio.to_thread`, so the event loop keeps
  servicing the Stratum socket while the GPU works (CuPy's synchronize
  releases the GIL). Stream ordering against the null-stream job/target
  uploads is preserved by draining the pipeline before preparing a new job.

### Search space

For a fixed coinbase (extranonce2), the miner sweeps the full 2³² nonce space
across a small window of incremented `ntime` values (`NTIME_ROLL`, default 8)
before advancing extranonce2. ntime rolling extends the searchable space
without recomputing the coinbase or merkle root and keeps block timestamps
fresh; the roll is kept small since pools only accept timestamps within a
bounded forward window.

### Platform notes

- **JetPack 7.2** ships CUDA **13.2** and an ARM SBSA toolkit. Containers built
  for JetPack 6.x / CUDA 12.6 are **not** compatible with JP 7.2 hosts; the
  Jetson image therefore installs `cupy-cuda13x[ctk]` even though its NGC
  PyTorch baseline (`25.06-py3`) predates JP 7.2 — the CuPy wheel brings the
  matching CUDA 13.x runtime and NVRTC components needed for kernel JIT.
- **Jetson Orin** is compute capability **8.7**. The CUDA kernel is compiled at
  import time via NVRTC for the active device, so no architecture-specific
  binary is baked into the image.
- **x86_64** hosts use a separate Dockerfile and CuPy CUDA 12.x stack; select
  it with the `docker-compose.x86.yml` override rather than the default Jetson
  compose file.

## Possible further optimizations

- Fuse the two SHA-256 blocks of the second hash and precompute its first
  round (the digest's first word is known before the second block starts).
- Persistent-kernel / grid-stride design to amortize launch overhead across
  the whole nonce space in fewer launches.
- Multi-GPU fan-out (one `GpuScanner` per device) behind a single Stratum
  connection.
