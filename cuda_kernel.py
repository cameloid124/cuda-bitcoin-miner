"""GPU orchestration for the SHA-256d mining kernel.

Hosts the per-job precompute (pure Python, hashlib-verified), the
double-buffered GpuScanner used by miner.py, and the startup GPU info query.
The kernel itself lives in kernels/sha256d_mine.cu and is loaded as PTX (or a
native cubin fallback) through cuda_driver.py — no CuPy or NumPy at runtime.
"""
import array
import re
import struct
import subprocess
from collections import deque
from ctypes import CDLL, addressof, c_uint32, create_string_buffer
from pathlib import Path

from cuda_driver import CudaDriver, CUDAError, KernelLoadInfo

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


def _precompute_job_words(header_prefix: bytes) -> array.array:
    """
    Builds the 20-word per-job constant buffer for the mining kernel:
    chunk-1 midstate, tail-chunk state after rounds 0-2, and the constant
    parts of tail schedule words W[16..19] (the nonce first appears at W[3],
    so all of this is nonce-independent).
    """
    midstate = _sha256_compress(_SHA256_IV, header_prefix[:64])
    w0, w1, w2 = struct.unpack('>3I', header_prefix[64:76])

    a, b, c, d, e, f, g, h = midstate
    for i, w in enumerate((w0, w1, w2)):
        t1 = (h + _ep1(e) + _ch(e, f, g) + _SHA256_K[i] + w) & _M32
        t2 = (_ep0(a) + _maj(a, b, c)) & _M32
        h, g, f, e = g, f, e, (d + t1) & _M32
        d, c, b, a = c, b, a, (t1 + t2) & _M32
    state3 = (a, b, c, d, e, f, g, h)

    w16 = (_sig0(w1) + w0) & _M32
    w17 = (_sig1(640) + _sig0(w2) + w1) & _M32
    c18 = (_sig1(w16) + w2) & _M32
    c19 = (_sig1(w17) + _sig0(0x80000000)) & _M32

    return array.array('I', midstate + state3 + (w16, w17, c18, c19))


def _target_bytes_to_words(target_bytes: bytes) -> array.array:
    return array.array('I', struct.unpack('>8I', target_bytes))


# =======================================================================
# GPU ORCHESTRATION
# =======================================================================

# Grid configuration: 32768 blocks x 512 threads = exactly 2^24 nonces per
# launch, so the full 32-bit nonce space is covered in 256 launches.
BLOCKS = 32768
THREADS_PER_BLOCK = 512
NONCES_PER_LAUNCH = BLOCKS * THREADS_PER_BLOCK
MAX_RESULTS = 16

_RESULT_WORDS = 1 + MAX_RESULTS
_RESULT_BYTES = _RESULT_WORDS * 4
_BEST_HASH_WORDS = 8
_PIPELINE_DEPTH = 2
_HASH_MAX_WORD = 0xFFFFFFFF
_DEFAULT_STREAM = 0
_LAUNCH_GRID = (BLOCKS, 1, 1)
_LAUNCH_BLOCK = (THREADS_PER_BLOCK, 1, 1)


def state_words_to_hash_int(state_words) -> int:
    """Converts an 8-word SHA-256 digest (BE words) to a Bitcoin LE integer."""
    digest = b"".join(struct.pack(">I", int(w) & 0xFFFFFFFF) for w in state_words)
    return int.from_bytes(digest, "little")


def _upload_u32_array(cuda: CudaDriver, d_ptr: int, words: array.array) -> None:
    cuda.memcpy_htod(d_ptr, words.buffer_info()[0], len(words) * 4)


class GpuScanner:
    """
    Double-buffered SHA256d scanner backed by the CUDA Driver API.

    submit() enqueues one 2^24-nonce batch on one of two CUDA streams
    (memset + kernel + async readback into pinned host memory); collect()
    blocks until the oldest in-flight batch finishes and returns its
    candidates. Keeping two batches in flight hides the launch/readback gap
    so the GPU never idles.

    Job/target uploads are synchronous copies on the legacy default stream.
    The worker streams are created legacy-blocking (flags=0), so those
    uploads are ordered against in-flight kernels: prepare_job() after
    drain() cannot race a kernel, and a mid-pipeline set_target() serializes
    with running batches instead of tearing the 32-byte target.
    """

    def __init__(self):
        self._cuda = CudaDriver.instance()
        self._streams = [self._cuda.create_stream() for _ in range(_PIPELINE_DEPTH)]
        self._d_job = self._cuda.mem_alloc(20 * 4)
        self._d_target = self._cuda.mem_alloc(8 * 4)
        self._d_results = [self._cuda.mem_alloc(_RESULT_BYTES) for _ in range(_PIPELINE_DEPTH)]
        self._d_best_msb = [self._cuda.mem_alloc(4) for _ in range(_PIPELINE_DEPTH)]
        self._d_best_hash = [self._cuda.mem_alloc(_BEST_HASH_WORDS * 4) for _ in range(_PIPELINE_DEPTH)]
        self._h_results_ptr = [self._cuda.mem_host_alloc(_RESULT_BYTES)
                               for _ in range(_PIPELINE_DEPTH)]
        self._h_results = [(c_uint32 * _RESULT_WORDS).from_address(p)
                           for p in self._h_results_ptr]
        self._in_flight = deque()
        self._next_slot = 0

    @property
    def in_flight(self) -> int:
        return len(self._in_flight)

    def prepare_job(self, header_prefix: bytes, target_bytes: bytes) -> None:
        """Uploads the precomputed constants for a new 76-byte header prefix."""
        self.drain()
        _upload_u32_array(self._cuda, self._d_job, _precompute_job_words(header_prefix))
        self.set_target(target_bytes)

    def drain(self) -> None:
        """Synchronizes and discards every in-flight batch."""
        while self._in_flight:
            slot, _ = self._in_flight.popleft()
            self._cuda.stream_sync(self._streams[slot])

    def set_target(self, target_bytes: bytes) -> None:
        """Uploads a new 32-byte big-endian share target."""
        _upload_u32_array(self._cuda, self._d_target, _target_bytes_to_words(target_bytes))

    def submit(self, nonce_base: int) -> None:
        """Enqueues a batch scanning NONCES_PER_LAUNCH nonces from nonce_base."""
        assert len(self._in_flight) < _PIPELINE_DEPTH, "pipeline full"
        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % _PIPELINE_DEPTH
        stream = self._streams[slot]

        self._cuda.memset_d32_async(self._d_results[slot], 0, _RESULT_WORDS, stream)
        self._cuda.memset_d32_async(self._d_best_msb[slot], _HASH_MAX_WORD, 1, stream)
        self._cuda.memset_d32_async(self._d_best_hash[slot], _HASH_MAX_WORD, _BEST_HASH_WORDS, stream)
        self._cuda.launch_mine(
            stream, _LAUNCH_GRID, _LAUNCH_BLOCK,
            self._d_job, nonce_base, self._d_target,
            self._d_results[slot], self._d_best_hash[slot], self._d_best_msb[slot],
        )
        self._cuda.memcpy_dtoh_async(
            self._h_results_ptr[slot], self._d_results[slot], _RESULT_BYTES, stream)

        self._in_flight.append((slot, nonce_base))

    def collect(self):
        """Blocks until the oldest in-flight batch completes."""
        slot, nonce_base = self._in_flight.popleft()
        self._cuda.stream_sync(self._streams[slot])
        res = self._h_results[slot]
        count = min(int(res[0]), MAX_RESULTS)

        best_msb = c_uint32()
        self._cuda.memcpy_dtoh(addressof(best_msb), self._d_best_msb[slot], 4)
        batch_best = None
        if best_msb.value != _HASH_MAX_WORD:
            best_words = (c_uint32 * _BEST_HASH_WORDS)()
            self._cuda.memcpy_dtoh(addressof(best_words), self._d_best_hash[slot], 32)
            batch_best = state_words_to_hash_int(best_words)

        return nonce_base, [int(res[i]) for i in range(1, 1 + count)], batch_best


# ---------------------------------------------------------------------------
# Test helper: single-nonce launch (used by test_correctness.py)
# ---------------------------------------------------------------------------

class _SingleNonceHarness:
    """Lazy-allocated device buffers for 1×1 kernel launches in tests."""

    def __init__(self):
        self._cuda = CudaDriver.instance()
        self._d_job = self._cuda.mem_alloc(20 * 4)
        self._d_target = self._cuda.mem_alloc(8 * 4)
        self._d_results = self._cuda.mem_alloc(_RESULT_BYTES)
        self._d_best_hash = self._cuda.mem_alloc(_BEST_HASH_WORDS * 4)
        self._d_best_msb = self._cuda.mem_alloc(4)
        self._h_results = (c_uint32 * _RESULT_WORDS)()

    def check(self, header_prefix: bytes, nonce: int, target_int: int) -> bool:
        _upload_u32_array(self._cuda, self._d_job, _precompute_job_words(header_prefix))
        target_bytes = target_int.to_bytes(32, 'big')
        _upload_u32_array(self._cuda, self._d_target, _target_bytes_to_words(target_bytes))

        self._cuda.memset_d32_async(self._d_results, 0, _RESULT_WORDS, _DEFAULT_STREAM)
        self._cuda.memset_d32_async(self._d_best_msb, _HASH_MAX_WORD, 1, _DEFAULT_STREAM)
        self._cuda.memset_d32_async(self._d_best_hash, _HASH_MAX_WORD, _BEST_HASH_WORDS, _DEFAULT_STREAM)
        self._cuda.launch_mine(
            _DEFAULT_STREAM, (1, 1, 1), (1, 1, 1),
            self._d_job, nonce, self._d_target,
            self._d_results, self._d_best_hash, self._d_best_msb,
        )
        self._cuda.stream_sync(_DEFAULT_STREAM)
        self._cuda.memcpy_dtoh(addressof(self._h_results), self._d_results, _RESULT_BYTES)
        return int(self._h_results[0]) == 1


_test_harness: _SingleNonceHarness | None = None


def gpu_check_single(header_prefix: bytes, nonce: int, target_int: int) -> bool:
    """Runs the mining kernel for exactly one nonce; returns target match."""
    global _test_harness
    if _test_harness is None:
        _test_harness = _SingleNonceHarness()
    return _test_harness.check(header_prefix, nonce, target_int)


# ---------------------------------------------------------------------------
# GPU info (startup banner)
# ---------------------------------------------------------------------------

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


def format_kernel_load_info(info: KernelLoadInfo) -> str:
    """Formats the loaded kernel artifact for startup diagnostics."""
    if info.kind == "ptx":
        return f"PTX ({info.artifact}, driver JIT)"
    # e.g. kernels/cubin/sm_89.cubin → sm_89
    stem = Path(info.artifact).stem
    return f"cubin {stem} ({info.artifact})"


def get_gpu_info():
    """Returns (gpu_name, cuda_version, driver_version, kernel_load_desc)."""
    try:
        cuda = CudaDriver.instance()
        gpu_name = cuda.get_device_name()
        cuda_version = _format_cuda_version(cuda.get_driver_version())
        driver_version = _get_nvidia_driver_version()
        kernel_desc = format_kernel_load_info(cuda.kernel_load_info)
        return gpu_name, cuda_version, driver_version, kernel_desc
    except (CUDAError, OSError):
        return "Unknown GPU", "Unknown", "Unknown", "Unknown"
