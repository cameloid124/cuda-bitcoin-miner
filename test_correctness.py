"""
Correctness tests for the CUDA SHA-256d miner.

Validates against real Bitcoin block #125552 and against hashlib on random
headers, and checks the host-side midstate/early-round precomputation and the
double-buffered scanner. Requires a CUDA-capable GPU and kernels/sha256d_mine.ptx
(or a matching cubin fallback).

Run: python test_correctness.py
"""
import hashlib
import os
import struct

from cuda_kernel import (
    GpuScanner, NONCES_PER_LAUNCH, gpu_check_single,
    _sha256_compress, _SHA256_IV,
)
from miner import StratumMiner, diff_to_target, sha256d, DIFF1_TARGET, NTIME_ROLL

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


def nbits_to_target(nbits_hex: str) -> int:
    nbits = int(nbits_hex, 16)
    return (nbits & 0x007FFFFF) << (8 * ((nbits >> 24) - 3))


def test_host_sha256_matches_hashlib():
    for length in range(0, 56):
        msg = os.urandom(length)
        block = msg + b'\x80' + b'\x00' * (55 - length) + struct.pack('>Q', length * 8)
        got = struct.pack('>8I', *_sha256_compress(_SHA256_IV, block))
        assert got == hashlib.sha256(msg).digest(), f"host compression != hashlib (len {length})"
    print("[PASS] Host SHA-256 compression matches hashlib on all lengths 0..55")


def test_header_packing():
    merkle_root = bytes.fromhex(BLOCK_MERKLE_RPC)[::-1]
    prefix = StratumMiner.pack_header_prefix(
        BLOCK_VERSION, stratum_prevhash(BLOCK_PREVHASH_RPC), merkle_root,
        BLOCK_NTIME, BLOCK_NBITS)
    header = prefix + struct.pack("<I", BLOCK_NONCE)

    assert header.hex() == BLOCK_HEADER_HEX, "packed header mismatch"
    assert sha256d(header)[::-1].hex() == BLOCK_HASH_RPC, "header hash mismatch"
    print("[PASS] Header packing reproduces block #125552 exactly")
    return prefix


def test_gpu_finds_real_block_nonce(prefix: bytes):
    target_int = nbits_to_target(BLOCK_NBITS)
    assert gpu_check_single(prefix, BLOCK_NONCE, target_int), "GPU missed the real nonce"
    print(f"[PASS] GPU accepts block #125552 nonce 0x{BLOCK_NONCE:08x} at network difficulty")

    assert not gpu_check_single(prefix, BLOCK_NONCE, 1), "false positive at impossible target"
    assert not gpu_check_single(prefix, BLOCK_NONCE ^ 1, target_int), \
        "wrong nonce accepted at network difficulty"
    print("[PASS] GPU rejects impossible target and neighbouring nonce")


def test_scanner_finds_real_block_nonce(prefix: bytes):
    target = nbits_to_target(BLOCK_NBITS).to_bytes(32, 'big')
    scanner = GpuScanner()
    scanner.prepare_job(prefix, target)

    nonce_base = (BLOCK_NONCE // NONCES_PER_LAUNCH) * NONCES_PER_LAUNCH
    scanner.submit(nonce_base)
    _, candidates, batch_best = scanner.collect()
    assert BLOCK_NONCE in candidates, f"scanner missed the real nonce, got {candidates}"
    assert batch_best is not None, "scanner did not report batch best hash"
    print(f"[PASS] Double-buffered scanner found nonce 0x{BLOCK_NONCE:08x}")


def test_scanner_pipeline_covers_space():
    prefix = os.urandom(76)
    target = (0x7fffffff << 224).to_bytes(32, 'big')
    scanner = GpuScanner()
    scanner.prepare_job(prefix, target)

    bases = [0, NONCES_PER_LAUNCH]
    scanner.submit(bases[0])
    scanner.submit(bases[1])
    assert scanner.in_flight == 2

    seen = []
    for expected_base in bases:
        base, candidates, batch_best = scanner.collect()
        assert base == expected_base, "pipeline returned batches out of order"
        assert batch_best is not None, "pipeline batch missing best hash"
        for n in candidates:
            assert expected_base <= n < expected_base + NONCES_PER_LAUNCH, \
                "candidate nonce outside its batch range"
            h = int.from_bytes(sha256d(prefix + struct.pack('<I', n)), 'little')
            assert h <= int.from_bytes(target, 'big'), "scanner reported a non-qualifying nonce"
        seen.extend(candidates)
    assert scanner.in_flight == 0
    print(f"[PASS] Dual-stream pipeline: ordered, in-range, CPU-verified "
          f"({len(seen)} candidates across 2 batches)")


def test_kernel_vs_hashlib_random():
    for trial in range(50):
        prefix = os.urandom(76)
        nonce = int.from_bytes(os.urandom(4), 'little')
        digest_int = int.from_bytes(sha256d(prefix + struct.pack("<I", nonce)), 'little')

        assert gpu_check_single(prefix, nonce, digest_int), \
            f"trial {trial}: GPU hash != hashlib digest"
        assert not gpu_check_single(prefix, nonce, digest_int - 1), \
            f"trial {trial}: comparison not strict"
    print("[PASS] GPU SHA-256d matches hashlib on 50 random headers "
          "(inclusive target comparison verified)")


def test_ntime_roll_headers_distinct():
    prevhash = stratum_prevhash(BLOCK_PREVHASH_RPC)
    merkle_root = bytes.fromhex(BLOCK_MERKLE_RPC)[::-1]
    base = int(BLOCK_NTIME, 16)

    headers = set()
    for roll in range(NTIME_ROLL):
        ntime_hex = f"{base + roll:08x}"
        prefix = StratumMiner.pack_header_prefix(
            BLOCK_VERSION, prevhash, merkle_root, ntime_hex, BLOCK_NBITS)
        assert struct.unpack('<I', prefix[68:72])[0] == base + roll, "ntime not packed"
        headers.add(prefix)
    assert len(headers) == NTIME_ROLL, "ntime roll produced duplicate headers"
    print(f"[PASS] ntime rolling yields {NTIME_ROLL} distinct headers")


def test_diff_to_target():
    assert int.from_bytes(diff_to_target(1.0), 'big') == DIFF1_TARGET
    assert int.from_bytes(diff_to_target(65536), 'big') == DIFF1_TARGET // 65536
    assert int.from_bytes(diff_to_target(0.5), 'big') == DIFF1_TARGET * 2
    print("[PASS] diff_to_target exact for diff 1.0 / 65536 / 0.5")


if __name__ == "__main__":
    test_diff_to_target()
    test_host_sha256_matches_hashlib()
    prefix = test_header_packing()
    test_ntime_roll_headers_distinct()
    test_gpu_finds_real_block_nonce(prefix)
    test_scanner_finds_real_block_nonce(prefix)
    test_scanner_pipeline_covers_space()
    test_kernel_vs_hashlib_random()
    print("\nAll tests passed.")
