import time

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
// SHA-256 COMPRESSION, 16-WORD ROLLING MESSAGE SCHEDULE
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
// KERNEL 1: MIDSTATE OF THE FIRST 64-BYTE HEADER CHUNK
// -------------------------------------------------------------
// Runs once per extranonce2 (i.e. once per 2^32 hashes) so a 1-thread
// launch is perfectly adequate.
extern "C" __global__
void precompute_midstate(const unsigned char* __restrict__ chunk1,
                         unsigned int* __restrict__ midstate) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    unsigned int state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    unsigned int W[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = (chunk1[i*4] << 24) | (chunk1[i*4+1] << 16) | (chunk1[i*4+2] << 8) | chunk1[i*4+3];
    }

    sha256_transform(state, W);

    #pragma unroll
    for (int i = 0; i < 8; i++) midstate[i] = state[i];
}

// -------------------------------------------------------------
// KERNEL 2: SHA256D SCAN OVER A 2^24 NONCE WINDOW
// -------------------------------------------------------------
// The 12 tail bytes of the header (merkle tail / ntime / nbits) are passed as
// three pre-swapped scalar words, so threads touch global memory only for the
// midstate (8 words, broadcast through the read-only cache) and, rarely, for
// reporting a candidate.
#define MAX_RESULTS 16

extern "C" __global__
void sha256d_mine(const unsigned int* __restrict__ midstate,
                  unsigned int merkle_tail,
                  unsigned int ntime_w,
                  unsigned int nbits_w,
                  unsigned int nonce_base,
                  const unsigned int* __restrict__ target,   // 8 words, most-significant first
                  unsigned int* __restrict__ result_count,
                  unsigned int* __restrict__ result_nonces) {

    // Header nonce (as the little-endian integer stored at bytes 76..79)
    unsigned int nonce = nonce_base + blockIdx.x * blockDim.x + threadIdx.x;

    // --- First hash: second 64-byte chunk of the 80-byte header ---
    unsigned int W[16];
    W[0] = merkle_tail;
    W[1] = ntime_w;
    W[2] = nbits_w;
    W[3] = BSWAP32(nonce);        // header stores the nonce LE; SHA words are BE
    W[4] = 0x80000000u;           // padding bit
    #pragma unroll
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 640;                  // message length: 80 bytes = 640 bits

    unsigned int state[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) state[i] = midstate[i];

    sha256_transform(state, W);

    // --- Second hash: 32-byte digest as a single padded chunk ---
    #pragma unroll
    for (int i = 0; i < 8; i++) W[i] = state[i];
    W[8] = 0x80000000u;
    #pragma unroll
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 256;                  // 32 bytes = 256 bits

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
        unsigned int idx = atomicAdd(result_count, 1u);
        if (idx < MAX_RESULTS) result_nonces[idx] = nonce;
    }
}
"""

# =======================================================================
# HOST-SIDE ORCHESTRATION
# =======================================================================

module = cp.RawModule(code=CUDA_MINER_SRC, options=('--std=c++17',))
_midstate_kernel = module.get_function('precompute_midstate')
_mine_kernel = module.get_function('sha256d_mine')

# Grid configuration: 32768 blocks x 512 threads = exactly 2^24 nonces per
# launch, so the full 32-bit nonce space is covered in 256 launches.
# 512 threads/block benchmarked fastest on Ada (vs 128/256).
BLOCKS = 32768
THREADS_PER_BLOCK = 512
NONCES_PER_LAUNCH = BLOCKS * THREADS_PER_BLOCK
MAX_RESULTS = 16

# Persistent device buffers -- allocated once, reused for every batch.
_d_chunk1 = cp.zeros(64, dtype=cp.uint8)
_d_midstate = cp.zeros(8, dtype=cp.uint32)
_d_target = cp.zeros(8, dtype=cp.uint32)
_d_result_count = cp.zeros(1, dtype=cp.uint32)
_d_result_nonces = cp.zeros(MAX_RESULTS, dtype=cp.uint32)

# Host-cached scalar words of the header tail (merkle tail, ntime, nbits)
_chunk2_words = (np.uint32(0), np.uint32(0), np.uint32(0))


def prepare_job(header_prefix: bytes, target_bytes: bytes) -> None:
    """
    Uploads a new 76-byte header prefix: computes the chunk-1 midstate on the
    GPU and caches the chunk-2 scalar words. Called once per extranonce2.
    """
    global _chunk2_words
    _d_chunk1.set(np.frombuffer(header_prefix[:64], dtype=np.uint8))
    _midstate_kernel((1,), (1,), (_d_chunk1, _d_midstate))

    words = np.frombuffer(header_prefix[64:76], dtype='>u4').astype(np.uint32)
    _chunk2_words = (words[0], words[1], words[2])

    set_target(target_bytes)


def set_target(target_bytes: bytes) -> None:
    """Uploads a new 32-byte big-endian share target (e.g. after set_difficulty)."""
    _d_target.set(np.frombuffer(target_bytes, dtype='>u4').astype(np.uint32))


def scan_nonce_range(nonce_base: int):
    """
    Hashes NONCES_PER_LAUNCH consecutive nonces starting at nonce_base against
    the currently prepared job. Returns (candidate_nonces, elapsed_seconds).
    Candidates must be re-verified on the CPU before submission.
    """
    start = time.perf_counter()

    _d_result_count.fill(0)
    _mine_kernel(
        (BLOCKS,), (THREADS_PER_BLOCK,),
        (_d_midstate,
         _chunk2_words[0], _chunk2_words[1], _chunk2_words[2],
         np.uint32(nonce_base),
         _d_target, _d_result_count, _d_result_nonces)
    )
    cp.cuda.get_current_stream().synchronize()

    count = int(_d_result_count[0])
    nonces = []
    if count > 0:
        nonces = [int(n) for n in _d_result_nonces[:min(count, MAX_RESULTS)].get()]

    elapsed = time.perf_counter() - start
    return nonces, elapsed


def get_gpu_info():
    """Queries the NVIDIA runtime for the active device name and CUDA API version."""
    try:
        device_id = cp.cuda.Device().id
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        gpu_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']
        cuda_ver_int = cp.cuda.runtime.runtimeGetVersion()
        return gpu_name, f"{cuda_ver_int // 1000}.{(cuda_ver_int % 1000) // 10}"
    except Exception:
        return "Unknown GPU", "Unknown"
