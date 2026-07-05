import re
import struct
import subprocess
from collections import deque
from ctypes import CDLL, create_string_buffer

import numpy as np
import cupy as cp

# =======================================================================
# CUDA C++ KERNEL
# =======================================================================
CUDA_MINER_SRC = r"""
// Rotate-right via the funnel-shift hardware instruction (single SHF.R on Maxwell+)
#define ROTR(x, n) __funnelshift_r((x), (x), (n))

// Byte-swap a 32-bit word in a single PRMT instruction
#define BSWAP32(x) __byte_perm((x), 0, 0x0123)

// CH and MAJ as single LOP3 truth-table instructions.
// (nvcc usually emits LOP3 on its own, the asm just makes it explicit.)
__device__ __forceinline__ unsigned int CH(unsigned int x, unsigned int y, unsigned int z) {
    unsigned int res;
    asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(res) : "r"(x), "r"(y), "r"(z));
    return res;
}

__device__ __forceinline__ unsigned int MAJ(unsigned int x, unsigned int y, unsigned int z) {
    unsigned int res;
    asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(res) : "r"(x), "r"(y), "r"(z));
    return res;
}

#define EP0(x)  (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x)  (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

__constant__ unsigned int K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// -------------------------------------------------------------
// SHA-256 ROUND + FULL COMPRESSION (16-WORD ROLLING SCHEDULE)
// -------------------------------------------------------------
// Keeping only a 16-entry window (instead of the full W[64]) lets the whole
// schedule live in registers after full unrolling; W is consumed in place.
// Benchmarked ~15% faster than the flat W[64] layout on Ada (RTX 4060).
#define ROUND(w, i) { \
    unsigned int T1 = h + EP1(e) + CH(e, f, g) + K[i] + (w); \
    unsigned int T2 = EP0(a) + MAJ(a, b, c); \
    h = g; g = f; f = e; e = d + T1; \
    d = c; c = b; b = a; a = T1 + T2; }

__device__ __forceinline__ void sha256_transform(unsigned int* state, unsigned int* W) {
    unsigned int a = state[0], b = state[1], c = state[2], d = state[3];
    unsigned int e = state[4], f = state[5], g = state[6], h = state[7];

    #pragma unroll
    for (int i = 0; i < 16; ++i) ROUND(W[i], i);

    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        unsigned int w = SIG1(W[(i + 14) & 15]) + W[(i + 9) & 15] + SIG0(W[(i + 1) & 15]) + W[i & 15];
        W[i & 15] = w;
        ROUND(w, i);
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// -------------------------------------------------------------
// MINING KERNEL: SHA256D SCAN OVER A 2^24 NONCE WINDOW
// -------------------------------------------------------------
// All per-job constants arrive in a 20-word buffer precomputed on the host
// (see _precompute_job_words in the Python part):
//
//   job[0..7]   chunk-1 midstate (needed for the final feed-forward addition)
//   job[8..15]  working registers a..h AFTER rounds 0-2 of the tail chunk;
//               those rounds depend only on merkle-tail/ntime/nbits, never on
//               the nonce, so each thread skips them entirely
//   job[16]     W[16] of the tail schedule (fully constant)
//   job[17]     W[17]                      (fully constant)
//   job[18]     W[18] minus SIG0(W[3])     (kernel adds the nonce-dependent part)
//   job[19]     W[19] minus W[3]           (kernel adds the nonce word)
//
// Threads touch global memory only for job/target (broadcast through the
// read-only cache) and, rarely, for reporting a candidate.
#define MAX_RESULTS 16

extern "C" __global__
void sha256d_mine(const unsigned int* __restrict__ job,
                  unsigned int nonce_base,
                  const unsigned int* __restrict__ target,   // 8 words, most-significant first
                  unsigned int* __restrict__ results) {      // [0]=count, [1..16]=nonces

    // Header nonce (as the little-endian integer stored at bytes 76..79)
    unsigned int nonce = nonce_base + blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nw = BSWAP32(nonce);   // header stores the nonce LE; SHA words are BE

    // --- First hash: tail chunk of the 80-byte header, rounds 3..63 ---
    unsigned int W[16];
    W[3] = nw;
    W[4] = 0x80000000u;                 // padding bit
    #pragma unroll
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 640;                        // message length: 80 bytes = 640 bits
    // W[0..2] are never read: rounds 0-2 are pre-applied in job[8..15] and
    // their schedule contributions are folded into job[16..19].

    unsigned int a = job[8],  b = job[9],  c = job[10], d = job[11];
    unsigned int e = job[12], f = job[13], g = job[14], h = job[15];

    #pragma unroll
    for (int i = 3; i < 16; ++i) ROUND(W[i], i);

    // Rounds 16-19: schedule words with host-precomputed constant parts.
    unsigned int w;
    w = job[16];            W[0] = w; ROUND(w, 16);
    w = job[17];            W[1] = w; ROUND(w, 17);
    w = job[18] + SIG0(nw); W[2] = w; ROUND(w, 18);
    w = job[19] + nw;       W[3] = w; ROUND(w, 19);

    #pragma unroll
    for (int i = 20; i < 64; ++i) {
        w = SIG1(W[(i + 14) & 15]) + W[(i + 9) & 15] + SIG0(W[(i + 1) & 15]) + W[i & 15];
        W[i & 15] = w;
        ROUND(w, i);
    }

    unsigned int state[8];
    state[0] = job[0] + a; state[1] = job[1] + b; state[2] = job[2] + c; state[3] = job[3] + d;
    state[4] = job[4] + e; state[5] = job[5] + f; state[6] = job[6] + g; state[7] = job[7] + h;

    // --- Second hash: 32-byte digest as a single padded chunk ---
    #pragma unroll
    for (int i = 0; i < 8; i++) W[i] = state[i];
    W[8] = 0x80000000u;
    #pragma unroll
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 256;                        // 32 bytes = 256 bits

    state[0] = 0x6a09e667; state[1] = 0xbb67ae85; state[2] = 0x3c6ef372; state[3] = 0xa54ff53a;
    state[4] = 0x510e527f; state[5] = 0x9b05688c; state[6] = 0x1f83d9ab; state[7] = 0x5be0cd19;

    sha256_transform(state, W);

    // --- Target check ---
    // Bitcoin interprets the digest as a 256-bit little-endian number, so
    // BSWAP32(state[7]) is its most significant 32-bit word. Compare word by
    // word against the big-endian target (target[0] = most significant).
    int cmp = 0;    // -1: hash < target, 0: equal so far, 1: hash > target
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        if (cmp == 0) {
            unsigned int hw = BSWAP32(state[7 - j]);
            unsigned int tw = target[j];
            if (hw < tw) cmp = -1;
            else if (hw > tw) cmp = 1;
        }
    }

    if (cmp <= 0) {
        unsigned int idx = atomicAdd(&results[0], 1u);
        if (idx < MAX_RESULTS) results[1 + idx] = nonce;
    }
}
"""

# =======================================================================
# HOST-SIDE SHA-256 (midstate + early-round precomputation)
# =======================================================================
# Runs once per (extranonce2, ntime) -- i.e. once per 2^32 GPU hashes -- so
# pure Python is more than fast enough, and it is directly testable against
# hashlib.

_M32 = 0xFFFFFFFF

_SHA256_K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

_SHA256_IV = (0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19)


def _rotr(x, n):
    return ((x >> n) | (x << (32 - n))) & _M32

def _sig0(x):
    return _rotr(x, 7) ^ _rotr(x, 18) ^ (x >> 3)

def _sig1(x):
    return _rotr(x, 17) ^ _rotr(x, 19) ^ (x >> 10)

def _ep0(x):
    return _rotr(x, 2) ^ _rotr(x, 13) ^ _rotr(x, 22)

def _ep1(x):
    return _rotr(x, 6) ^ _rotr(x, 11) ^ _rotr(x, 25)

def _ch(x, y, z):
    return z ^ (x & (y ^ z))

def _maj(x, y, z):
    return (x & y) | (z & (x | y))


def _sha256_compress(state, block: bytes):
    """One SHA-256 compression of a 64-byte block. Returns the new state."""
    w = list(struct.unpack('>16I', block))
    for i in range(16, 64):
        w.append((_sig1(w[i - 2]) + w[i - 7] + _sig0(w[i - 15]) + w[i - 16]) & _M32)

    a, b, c, d, e, f, g, h = state
    for i in range(64):
        t1 = (h + _ep1(e) + _ch(e, f, g) + _SHA256_K[i] + w[i]) & _M32
        t2 = (_ep0(a) + _maj(a, b, c)) & _M32
        h, g, f, e = g, f, e, (d + t1) & _M32
        d, c, b, a = c, b, a, (t1 + t2) & _M32

    return tuple((s + v) & _M32 for s, v in zip(state, (a, b, c, d, e, f, g, h)))


def _precompute_job_words(header_prefix: bytes) -> np.ndarray:
    """
    Builds the 20-word per-job constant buffer for the mining kernel:
    chunk-1 midstate, tail-chunk state after rounds 0-2, and the constant
    parts of tail schedule words W[16..19] (the nonce first appears at W[3],
    so all of this is nonce-independent).
    """
    midstate = _sha256_compress(_SHA256_IV, header_prefix[:64])
    w0, w1, w2 = struct.unpack('>3I', header_prefix[64:76])

    # Rounds 0-2 of the tail-chunk compression (no feed-forward addition;
    # the kernel adds the midstate at the end of all 64 rounds).
    a, b, c, d, e, f, g, h = midstate
    for i, w in enumerate((w0, w1, w2)):
        t1 = (h + _ep1(e) + _ch(e, f, g) + _SHA256_K[i] + w) & _M32
        t2 = (_ep0(a) + _maj(a, b, c)) & _M32
        h, g, f, e = g, f, e, (d + t1) & _M32
        d, c, b, a = c, b, a, (t1 + t2) & _M32
    state3 = (a, b, c, d, e, f, g, h)

    # Tail-chunk schedule: W[3]=nonce, W[4]=0x80000000, W[5..14]=0, W[15]=640.
    #   W[16] = SIG1(W[14]) + W[9]  + SIG0(W[1]) + W[0]  -> fully constant
    #   W[17] = SIG1(W[15]) + W[10] + SIG0(W[2]) + W[1]  -> fully constant
    #   W[18] = SIG1(W[16]) + W[11] + SIG0(W[3]) + W[2]  -> constant part + SIG0(nonce_w)
    #   W[19] = SIG1(W[17]) + W[12] + SIG0(W[4]) + W[3]  -> constant part + nonce_w
    w16 = (_sig0(w1) + w0) & _M32
    w17 = (_sig1(640) + _sig0(w2) + w1) & _M32
    c18 = (_sig1(w16) + w2) & _M32
    c19 = (_sig1(w17) + _sig0(0x80000000)) & _M32

    return np.array(midstate + state3 + (w16, w17, c18, c19), dtype=np.uint32)


# =======================================================================
# GPU ORCHESTRATION
# =======================================================================

module = cp.RawModule(code=CUDA_MINER_SRC, options=('--std=c++17',))
_mine_kernel = module.get_function('sha256d_mine')

# Grid configuration: 32768 blocks x 512 threads = exactly 2^24 nonces per
# launch, so the full 32-bit nonce space is covered in 256 launches.
# 512 threads/block benchmarked fastest on Ada (vs 128/256).
BLOCKS = 32768
THREADS_PER_BLOCK = 512
NONCES_PER_LAUNCH = BLOCKS * THREADS_PER_BLOCK
MAX_RESULTS = 16

_RESULT_WORDS = 1 + MAX_RESULTS
_RESULT_BYTES = _RESULT_WORDS * 4
_PIPELINE_DEPTH = 2


class GpuScanner:
    """
    Double-buffered SHA256d scanner. submit() enqueues one 2^24-nonce batch
    on one of two CUDA streams (memset + kernel + async readback into pinned
    host memory); collect() blocks until the oldest in-flight batch finishes
    and returns its candidates. Keeping two batches in flight hides the
    launch/readback gap so the GPU never idles.

    Streams are created as legacy-blocking, so job/target uploads (which go
    through the null stream) are correctly ordered against in-flight kernels.
    prepare_job() must only be called with an empty pipeline.
    """

    def __init__(self):
        self._streams = [cp.cuda.Stream() for _ in range(_PIPELINE_DEPTH)]
        self._d_job = cp.zeros(20, dtype=cp.uint32)
        self._d_target = cp.zeros(8, dtype=cp.uint32)
        self._d_results = [cp.zeros(_RESULT_WORDS, dtype=cp.uint32)
                           for _ in range(_PIPELINE_DEPTH)]
        # Pinned host buffers make the device-to-host result copy asynchronous.
        self._pinned = [cp.cuda.alloc_pinned_memory(_RESULT_BYTES)
                        for _ in range(_PIPELINE_DEPTH)]
        self._h_results = [np.frombuffer(m, dtype=np.uint32, count=_RESULT_WORDS)
                           for m in self._pinned]
        self._in_flight = deque()   # slots in submission order
        self._next_slot = 0

    @property
    def in_flight(self) -> int:
        return len(self._in_flight)

    def prepare_job(self, header_prefix: bytes, target_bytes: bytes) -> None:
        """Uploads the precomputed constants for a new 76-byte header prefix.
        Called once per (extranonce2, ntime). Drains any leftover in-flight
        batches first so the null-stream upload can't race a running kernel."""
        self.drain()
        self._d_job.set(_precompute_job_words(header_prefix))
        self.set_target(target_bytes)

    def drain(self) -> None:
        """Synchronizes and discards every in-flight batch."""
        while self._in_flight:
            slot, _ = self._in_flight.popleft()
            self._streams[slot].synchronize()

    def set_target(self, target_bytes: bytes) -> None:
        """Uploads a new 32-byte big-endian share target (e.g. after
        set_difficulty). Null-stream ordering makes this safe mid-pipeline."""
        self._d_target.set(np.frombuffer(target_bytes, dtype='>u4').astype(np.uint32))

    def submit(self, nonce_base: int) -> None:
        """Enqueues a batch scanning NONCES_PER_LAUNCH nonces from nonce_base.
        Returns immediately; the GPU works in the background."""
        assert len(self._in_flight) < _PIPELINE_DEPTH, "pipeline full"
        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % _PIPELINE_DEPTH
        stream = self._streams[slot]
        d_res = self._d_results[slot]

        with stream:
            d_res.fill(0)
            _mine_kernel(
                (BLOCKS,), (THREADS_PER_BLOCK,),
                (self._d_job, np.uint32(nonce_base), self._d_target, d_res)
            )
        cp.cuda.runtime.memcpyAsync(
            self._h_results[slot].ctypes.data, d_res.data.ptr, _RESULT_BYTES,
            cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)

        self._in_flight.append((slot, nonce_base))

    def collect(self):
        """Blocks until the oldest in-flight batch completes. Returns
        (nonce_base, candidate_nonces). Candidates must be re-verified on the
        CPU before submission."""
        slot, nonce_base = self._in_flight.popleft()
        self._streams[slot].synchronize()
        res = self._h_results[slot]
        count = min(int(res[0]), MAX_RESULTS)
        return nonce_base, [int(n) for n in res[1:1 + count]]


def _format_cuda_version(ver_int: int) -> str:
    return f"{ver_int // 1000}.{(ver_int % 1000) // 10}"


def _query_driver_version_nvidia_smi() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        )
        version = out.strip().splitlines()[0].strip()
        return version or None
    except Exception:
        return None


def _query_driver_version_nvml() -> str | None:
    """Uses NVML (libnvidia-ml), injected by the NVIDIA container runtime on
    Jetson and desktop hosts when GPU passthrough is enabled."""
    try:
        nvml = CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit() != 0:
            return None
        try:
            buf = create_string_buffer(80)
            if nvml.nvmlSystemGetDriverVersion(buf, 80) != 0:
                return None
            return buf.value.decode().strip() or None
        finally:
            nvml.nvmlShutdown()
    except Exception:
        return None


def _query_driver_version_proc() -> str | None:
    try:
        with open("/proc/driver/nvidia/version", encoding="utf-8") as f:
            match = re.search(r"Kernel Module\s+([\d.]+)", f.read())
            return match.group(1) if match else None
    except Exception:
        return None


def _get_nvidia_driver_version() -> str:
    for query in (_query_driver_version_nvidia_smi,
                  _query_driver_version_nvml,
                  _query_driver_version_proc):
        version = query()
        if version:
            return version
    return "Unknown"


def get_gpu_info():
    """Returns (gpu_name, cuda_version, driver_version).

    cuda_version  – max CUDA version supported by the installed driver
                    (driverGetVersion).
    driver_version – NVIDIA driver package version (e.g. 610.62).
    """
    try:
        device_id = cp.cuda.Device().id
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        gpu_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']
        cuda_version = _format_cuda_version(cp.cuda.runtime.driverGetVersion())
        driver_version = _get_nvidia_driver_version()
        return gpu_name, cuda_version, driver_version
    except Exception:
        return "Unknown GPU", "Unknown", "Unknown"
