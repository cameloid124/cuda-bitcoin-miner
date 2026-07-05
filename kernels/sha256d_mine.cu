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

#define MAX_RESULTS 16

extern "C" __global__
void sha256d_mine(const unsigned int* __restrict__ job,
                  unsigned int nonce_base,
                  const unsigned int* __restrict__ target,
                  unsigned int* __restrict__ results,
                  unsigned int* __restrict__ best_hash,
                  unsigned int* __restrict__ best_msb) {

    unsigned int nonce = nonce_base + blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int nw = BSWAP32(nonce);

    unsigned int W[16];
    W[3] = nw;
    W[4] = 0x80000000u;
    #pragma unroll
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 640;

    unsigned int a = job[8],  b = job[9],  c = job[10], d = job[11];
    unsigned int e = job[12], f = job[13], g = job[14], h = job[15];

    #pragma unroll
    for (int i = 3; i < 16; ++i) ROUND(W[i], i);

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

    #pragma unroll
    for (int i = 0; i < 8; i++) W[i] = state[i];
    W[8] = 0x80000000u;
    #pragma unroll
    for (int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 256;

    state[0] = 0x6a09e667; state[1] = 0xbb67ae85; state[2] = 0x3c6ef372; state[3] = 0xa54ff53a;
    state[4] = 0x510e527f; state[5] = 0x9b05688c; state[6] = 0x1f83d9ab; state[7] = 0x5be0cd19;

    sha256_transform(state, W);

    int cmp = 0;
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

    // Track the best (numerically lowest) hash in this batch for telemetry.
    // atomicMin arbitrates on the most significant word only; if two hashes
    // tie on that word, concurrent winners may interleave writes to
    // best_hash. This is a deliberate trade-off: the value is telemetry-only
    // (never submitted), and a full 256-bit atomic compare would need a lock.
    unsigned int msb = BSWAP32(state[7]);
    unsigned int prev = atomicMin(best_msb, msb);
    if (msb <= prev) {
        #pragma unroll
        for (int i = 0; i < 8; i++) best_hash[i] = state[i];
    }
}
