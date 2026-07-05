# Python + CUDA Bitcoin Miner (Proof of Concept)

A Bitcoin (SHA-256d) Stratum V1 miner written in Python with a hand-tuned CUDA
kernel loaded via the CUDA Driver API (ctypes — **no CuPy, no NumPy**). This
is a **proof of concept**: GPU mining of SHA-256 has been economically unviable
for a decade (ASICs are ~5 orders of magnitude more efficient), but the project
demonstrates a complete, protocol-correct mining stack in ~600 lines of Python
plus a precompiled PTX module.

## Requirements

- NVIDIA GPU with CUDA support (Turing / sm_75 or newer for the shipped PTX)
  - **Jetson (default):** Orin Nano / Orin NX / AGX Orin, JetPack **7.2**
    (L4T 39.2, CUDA 13.2)
  - **x86_64:** desktop or server GPU with a recent NVIDIA driver
- Python 3.10+ (stdlib only at runtime)
- NVIDIA driver (CUDA toolkit **not** required to run — only to rebuild the kernel)

## Usage

### Native install

```bash
python miner.py -o stratum+tcp://pool.example.com:3333 -u WALLET.WORKER -p x
```

The repo ships `kernels/sha256d_mine.ptx` (plus optional `kernels/cubin/` fallbacks).
The driver JIT-compiles PTX to native code on first launch (~1–3 s once per process).

To rebuild the kernel after editing `kernels/sha256d_mine.cu`:

```bash
python build_kernel.py          # requires CUDA toolkit (nvcc)
# or, dev-only fallback if nvcc is unavailable:
python scripts/dev_build_via_nvrtc.py   # requires CuPy
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

Builds `Dockerfile.jetson`. By default the image uses the **prebuilt** kernel
artifacts committed in `kernels/` — no CUDA devel stage is pulled or built.

**x86_64 desktop / server**

```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.x86.yml up --build
```

Builds `Dockerfile.x86` (same prebuilt-kernels default).

To recompile the kernel from source inside a CUDA devel builder stage instead
(e.g. after editing `kernels/sha256d_mine.cu`), set `KERNEL_SOURCE=build`:

```bash
KERNEL_SOURCE=build docker compose up --build
```

Pool credentials are read from `.env` (`STRATUM_URL`, `STRATUM_USER`,
`STRATUM_PASSWORD`). GPU passthrough is enabled via `runtime: nvidia` and
`gpus: all`.

| File | Platform | Runtime base image | Runtime deps |
|---|---|---|---|
| `Dockerfile.jetson` | ARM64 / Jetson | `ubuntu:24.04` (no in-image CUDA libs) | Python 3 stdlib |
| `Dockerfile.x86` | x86_64 | `nvidia/cuda:12.6.3-base-ubuntu22.04` | Python 3 stdlib |

The runtime images use the CUDA **base** variant on x86 (a few hundred MB).
Jetson uses plain **Ubuntu 24.04** instead: generic `nvcr.io/nvidia/cuda:*`
images trigger a [known `nvidia-cdi-hook` panic on Orin](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1271)
during container start. The miner talks to the GPU through `libcuda.so.1`
injected by the host runtime, so no CUDA packages are needed inside the image.

Optional `.env` overrides for the default (Jetson) stack:

```env
MINER_DOCKERFILE=Dockerfile.jetson
MINER_IMAGE_TAG=jetson
DOCKER_PLATFORM=linux/arm64
KERNEL_SOURCE=prebuilt
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
│  cuda_driver.py        – ctypes → libcuda / nvcuda.dll (Driver API)   │
└───────────────────────────────┬───────────────────────────────────────┘
                                │ loads
┌───────────────────────────────▼───────────────────────────────────────┐
│  kernels/sha256d_mine.ptx  (primary)  or  kernels/cubin/sm_*.cubin    │
│  sha256d_mine            – 32768 blocks × 512 threads = 2^24 nonces    │
└───────────────────────────────────────────────────────────────────────┘
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

### CUDA kernel (`kernels/sha256d_mine.cu`)

The kernel source is compiled offline to portable PTX (`kernels/sha256d_mine.ptx`)
and loaded at startup through the CUDA Driver API (`cuda_driver.py`). If PTX JIT
fails, a matching native cubin from `kernels/cubin/sm_XX.cubin` is tried. Key
implementation traits:

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
  servicing the Stratum socket while the GPU works (stream synchronize
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

- **Kernel artifacts.** PTX is built for virtual arch `compute_75` (Turing+).
  The same `.ptx` file runs on Windows, Linux, and Jetson; the driver JITs it
  per GPU. If the installed driver is older than the toolkit that produced the
  PTX, the JIT fails and the loader automatically falls back to a matching
  native cubin from `kernels/cubin/` (shipped for sm_75, sm_86, sm_87 —
  Jetson Orin, sm_89, sm_90).
- **JetPack 7.2 / Jetson Orin.** Runtime image is `ubuntu:24.04`; optional
  kernel builder (`KERNEL_SOURCE=build`) uses
  `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu24.04`. Orin loads the prebuilt
  `kernels/cubin/sm_87.cubin` if PTX JIT is unavailable. If container start
  still fails with a `cudacompat` / `slice bounds out of range` panic, update
  the host `nvidia-container-toolkit` to a release that includes
  [PR #1804](https://github.com/NVIDIA/nvidia-container-toolkit/pull/1804).
- **Jetson Orin** is compute capability **8.7**; desktop Ada (e.g. RTX 4060) is
  **8.9**. Both JIT the shipped PTX without a rebuild.
- **x86_64** hosts use `docker-compose.x86.yml` (CUDA 12.6 base images).

## Possible further optimizations

- Fuse the two SHA-256 blocks of the second hash and precompute its first
  round (the digest's first word is known before the second block starts).
- Persistent-kernel / grid-stride design to amortize launch overhead across
  the whole nonce space in fewer launches.
- Multi-GPU fan-out (one `GpuScanner` per device) behind a single Stratum
  connection.
