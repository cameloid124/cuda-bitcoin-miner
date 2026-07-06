"""
End-to-end test: runs the miner against a local mock Stratum pool.

The mock pool serves a realistic job at difficulty 0.001, then independently
reconstructs the block header from every mining.submit and checks the share
against the target. Requires a CUDA-capable GPU.

Run: python test_e2e.py
"""
import asyncio
import json
import struct
import logging
import sys

from miner import StratumMiner, sha256d, diff_to_target

DIFFICULTY = 0.001
EXTRANONCE1 = "08000002"
EXTRANONCE2_SIZE = 4

JOB = {
    "job_id": "e2e1",
    "prevhash": "e2f61c3f71d1defd3fa999dfa36953755c690689799962b48bebd836974e8cf9",
    "coinb1": "01000000010000000000000000000000000000000000000000000000000000000000000000ffffffff20020862062f503253482f04b8864e5008",
    "coinb2": "072f736c7573682f000000000100f2052a010000001976a914d23fcdf86f7e756a64a7a9688ef9903327048ed988ac00000000",
    "merkle_branch": [
        "9b5f1c3e0a3fdcb4d1c0d1a2b3c4d5e6f708192a3b4c5d6e7f8090a1b2c3d4e5",
        "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
    ],
    "version": "20000000",
    "nbits": "1a44b9f2",
    "ntime": "665f2f00",
}

shares = {"accepted": 0, "rejected": 0}
share_event = asyncio.Event()


def verify_share(extranonce2, ntime, nonce_hex) -> bool:
    """Independent share reconstruction, the way a real pool would do it."""
    coinbase = bytes.fromhex(JOB["coinb1"] + EXTRANONCE1 + extranonce2 + JOB["coinb2"])
    root = sha256d(coinbase)
    for branch in JOB["merkle_branch"]:
        root = sha256d(root + bytes.fromhex(branch))

    prevhash_raw = bytes.fromhex(JOB["prevhash"])
    header = (
        struct.pack("<I", int(JOB["version"], 16))
        + b''.join(prevhash_raw[i:i+4][::-1] for i in range(0, 32, 4))
        + root
        + struct.pack("<I", int(ntime, 16))
        + struct.pack("<I", int(JOB["nbits"], 16))
        + struct.pack("<I", int(nonce_hex, 16))   # big-endian hex -> int -> LE bytes
    )
    hash_int = int.from_bytes(sha256d(header), 'little')
    target_int = int.from_bytes(diff_to_target(DIFFICULTY), 'big')
    return hash_int <= target_int


async def handle_client(reader, writer):
    try:
        await serve_client(reader, writer)
    except ConnectionError:
        pass   # miner disconnected during test teardown


async def serve_client(reader, writer):
    def send(obj):
        writer.write((json.dumps(obj) + "\n").encode())

    while not reader.at_eof():
        line = await reader.readline()
        if not line:
            break
        msg = json.loads(line)
        method = msg.get("method")

        if method == "mining.subscribe":
            send({"id": msg["id"], "result": [
                [["mining.set_difficulty", "s1"], ["mining.notify", "s1"]],
                EXTRANONCE1, EXTRANONCE2_SIZE], "error": None})
        elif method == "mining.authorize":
            send({"id": msg["id"], "result": True, "error": None})
            send({"id": None, "method": "mining.set_difficulty", "params": [DIFFICULTY]})
            send({"id": None, "method": "mining.notify", "params": [
                JOB["job_id"], JOB["prevhash"], JOB["coinb1"], JOB["coinb2"],
                JOB["merkle_branch"], JOB["version"], JOB["nbits"], JOB["ntime"], True]})
        elif method == "mining.submit":
            worker, job_id, extranonce2, ntime, nonce_hex = msg["params"]
            ok = job_id == JOB["job_id"] and verify_share(extranonce2, ntime, nonce_hex)
            shares["accepted" if ok else "rejected"] += 1
            send({"id": msg["id"], "result": ok,
                  "error": None if ok else [23, "low difficulty share", None]})
            if shares["accepted"] + shares["rejected"] >= 3:
                share_event.set()
        elif method == "mining.ping":
            send({"id": msg["id"], "result": "pong", "error": None})
        await writer.drain()


async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')
    server = await asyncio.start_server(handle_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    print(f"[mock pool] listening on 127.0.0.1:{port}, difficulty {DIFFICULTY}")

    miner = StratumMiner("127.0.0.1", port, "testworker", "x")
    miner_task = asyncio.create_task(miner.run())

    try:
        await asyncio.wait_for(share_event.wait(), timeout=120)
    finally:
        await miner.cancel_current_job()
        miner_task.cancel()
        if miner.writer:
            miner.writer.close()
        server.close()
        await asyncio.sleep(0.2)   # let transports finish closing

    print(f"\n[mock pool] shares accepted: {shares['accepted']}, rejected: {shares['rejected']}")
    assert shares["accepted"] >= 3, "expected at least 3 valid shares"
    assert shares["rejected"] == 0, "pool-side verification rejected some shares"
    print("[PASS] End-to-end: miner produced pool-verified shares over Stratum")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.TimeoutError:
        print("[FAIL] no shares within timeout")
        sys.exit(1)
