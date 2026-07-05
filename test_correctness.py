"""
Correctness tests for the CUDA SHA-256d miner.

Validates against real Bitcoin block #125552 and against hashlib on random
headers. Requires a CUDA-capable GPU.

Run: python test_correctness.py
"""
import hashlib
import os
import struct

import numpy as np
import cupy as cp

import cuda_kernel
from cuda_kernel import prepare_job, scan_nonce_range, NONCES_PER_LAUNCH
from miner import StratumMiner, diff_to_target, sha256d, DIFF1_TARGET

# --- Reference data: Bitcoin block #125552 -----------------------------
BLOCK_VERSION = "00000001"
BLOCK_PREVHASH_RPC = "00000000000008a3a41b85b8b29ad444def299fee21793cd8b9e567eab02cd81"
BLOCK_MERKLE_RPC = "2b12fcf1b09288fcaff797d71e950e71ae42b91e8bdb2304758dfcffc2b620e3"
BLOCK_NTIME = "4dd7f5c7"   # 1305998791
BLOCK_NBITS = "1a44b9f2"
BLOCK_NONCE = 0x9546a142   # 2504433986
BLOCK_HASH_RPC = "00000000000000001e8d6829a8a21adc5d38d0a473b144b6765798e61f98bd1d"
BLOCK_HEADER_HEX = (
    "0100000081cd02ab7e569e8bcd9317e2fe99f2de44d49ab2b8851ba4a3080000"
    "00000000e320b6c2fffc8d750423db8b1eb942ae710e951ed797f7affc8892b0"
    "f1fc122bc7f5d74df2b9441a42a14695"
)


def stratum_prevhash(rpc_hex: str) -> str:
    """Converts an RPC (display) block hash into Stratum notify format:
    header byte order, but with every 4-byte word reversed."""
    header_order = bytes.fromhex(rpc_hex)[::-1]
    return b''.join(header_order[i:i+4][::-1] for i in range(0, 32, 4)).hex()


def test_header_packing():
    merkle_root = bytes.fromhex(BLOCK_MERKLE_RPC)[::-1]  # header byte order
    prefix = StratumMiner.pack_header_prefix(
        BLOCK_VERSION, stratum_prevhash(BLOCK_PREVHASH_RPC), merkle_root,
        BLOCK_NTIME, BLOCK_NBITS)
    header = prefix + struct.pack("<I", BLOCK_NONCE)

    assert header.hex() == BLOCK_HEADER_HEX, "packed header mismatch"
    assert sha256d(header)[::-1].hex() == BLOCK_HASH_RPC, "header hash mismatch"
    print("[PASS] Header packing reproduces block #125552 exactly")
    return prefix


def nbits_to_target(nbits_hex: str) -> int:
    nbits = int(nbits_hex, 16)
    return (nbits & 0x007FFFFF) << (8 * ((nbits >> 24) - 3))


def test_gpu_finds_real_block_nonce(prefix: bytes):
    target = nbits_to_target(BLOCK_NBITS).to_bytes(32, 'big')
    prepare_job(prefix, target)

    nonce_base = (BLOCK_NONCE // NONCES_PER_LAUNCH) * NONCES_PER_LAUNCH
    nonces, elapsed = scan_nonce_range(nonce_base)
    assert BLOCK_NONCE in nonces, f"GPU missed the real nonce, got {nonces}"
    print(f"[PASS] GPU found block #125552 nonce 0x{BLOCK_NONCE:08x} at network "
          f"difficulty ({NONCES_PER_LAUNCH / elapsed / 1e6:.0f} MH/s)")

    # Same scan with an impossible target must return nothing
    prepare_job(prefix, (1).to_bytes(32, 'big'))
    nonces, _ = scan_nonce_range(nonce_base)
    assert nonces == [], f"false positives at impossible target: {nonces}"
    print("[PASS] No false positives at an impossible target")


def test_kernel_vs_hashlib_random():
    """Single-thread kernel launches on random headers: the GPU digest must
    match hashlib exactly (target == digest passes, target == digest-1 fails)."""
    kernel = cuda_kernel.module.get_function('sha256d_mine')
    d_count = cp.zeros(1, dtype=cp.uint32)
    d_nonces = cp.zeros(cuda_kernel.MAX_RESULTS, dtype=cp.uint32)

    for trial in range(50):
        prefix = os.urandom(76)
        nonce = int.from_bytes(os.urandom(4), 'little')
        digest_int = int.from_bytes(sha256d(prefix + struct.pack("<I", nonce)), 'little')

        prepare_job(prefix, digest_int.to_bytes(32, 'big'))

        def run_one(target_int):
            cuda_kernel.set_target(target_int.to_bytes(32, 'big'))
            d_count.fill(0)
            kernel((1,), (1,), (
                cuda_kernel._d_midstate,
                cuda_kernel._chunk2_words[0], cuda_kernel._chunk2_words[1],
                cuda_kernel._chunk2_words[2], np.uint32(nonce),
                cuda_kernel._d_target, d_count, d_nonces))
            cp.cuda.get_current_stream().synchronize()
            return int(d_count[0])

        assert run_one(digest_int) == 1, f"trial {trial}: GPU hash != hashlib digest"
        assert run_one(digest_int - 1) == 0, f"trial {trial}: comparison not strict"

    print("[PASS] GPU SHA-256d matches hashlib on 50 random headers "
          "(inclusive target comparison verified)")


def test_diff_to_target():
    assert int.from_bytes(diff_to_target(1.0), 'big') == DIFF1_TARGET
    assert int.from_bytes(diff_to_target(65536), 'big') == DIFF1_TARGET // 65536
    assert int.from_bytes(diff_to_target(0.5), 'big') == DIFF1_TARGET * 2
    print("[PASS] diff_to_target exact for diff 1.0 / 65536 / 0.5")


if __name__ == "__main__":
    test_diff_to_target()
    prefix = test_header_packing()
    test_gpu_finds_real_block_nonce(prefix)
    test_kernel_vs_hashlib_random()
    print("\nAll tests passed.")
