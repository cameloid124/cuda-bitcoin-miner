import argparse
import asyncio
import json
import struct
import logging
import time
import hashlib
from fractions import Fraction
from urllib.parse import urlparse

from cuda_kernel import (
    prepare_job, set_target, scan_nonce_range, get_gpu_info, NONCES_PER_LAUNCH,
)

USER_AGENT = "python-cuda-miner/1.0"

# =======================================================================
# CRYPTOGRAPHY & STRATUM HELPERS
# =======================================================================

def sha256d(data: bytes) -> bytes:
    """Double SHA-256."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

# Stratum difficulty-1 share target: 0x0000_0000_FFFF << 208
DIFF1_TARGET = 0x00000000FFFF0000000000000000000000000000000000000000000000000000

def diff_to_target(difficulty: float) -> bytes:
    """
    Converts a pool difficulty into a 32-byte big-endian target threshold.
    Uses exact rational arithmetic: float division of a 256-bit integer would
    silently truncate to 53 bits of precision.
    """
    if difficulty <= 0:
        difficulty = 1.0
    target_int = int(Fraction(DIFF1_TARGET) / Fraction(difficulty))
    target_int = min(target_int, (1 << 256) - 1)
    return target_int.to_bytes(32, byteorder='big')

def format_hashrate(hr: float) -> str:
    """Converts raw hashes/second into a human-readable string."""
    if hr > 1e12: return f"{hr/1e12:.2f} TH/s"
    if hr > 1e9:  return f"{hr/1e9:.2f} GH/s"
    if hr > 1e6:  return f"{hr/1e6:.2f} MH/s"
    if hr > 1e3:  return f"{hr/1e3:.2f} KH/s"
    return f"{hr:.2f} H/s"


class AuthorizationError(Exception):
    """Raised when the pool refuses worker credentials (fatal, no retry)."""


# =======================================================================
# MAIN MINER ENGINE
# =======================================================================

class StratumMiner:
    def __init__(self, host, port, username, password):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        # Network State
        self.msg_id = 0
        self.pending_requests = {}   # request id -> {"method": str, "context": dict|None}
        self.reader = None
        self.writer = None

        # Pool State
        self.extranonce1 = ""
        self.extranonce2_size = 4
        self.current_difficulty = 1.0

        # Asynchronous Task Management
        self.current_job_task = None

        # Telemetry State
        self.session_start_time = 0
        self.total_nonces_hashed = 0
        self.last_log_time = 0

    async def run(self):
        """Main entry point with reconnect resiliency."""
        while True:
            try:
                await self.connect()
            except asyncio.CancelledError:
                break
            except AuthorizationError as e:
                logging.error(f"[!] {e}")
                break
            except Exception as e:
                logging.error(f"[!] Network connection lost: {e}. Reconnecting in 5 seconds...")
                self.cancel_current_job()
                await asyncio.sleep(5)

    def cancel_current_job(self):
        if self.current_job_task and not self.current_job_task.done():
            self.current_job_task.cancel()

    async def connect(self):
        logging.info(f"[*] Connecting to Stratum pool at {self.host}:{self.port}...")
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)

        # Reset per-connection state
        self.pending_requests.clear()
        self.session_start_time = time.time()
        self.total_nonces_hashed = 0
        self.last_log_time = time.time()

        await self.send_message("mining.subscribe", [USER_AGENT])
        await self.send_message("mining.authorize", [self.username, self.password])
        await self.listen_for_jobs()

    async def send_message(self, method, params, context=None):
        self.msg_id += 1
        self.pending_requests[self.msg_id] = {"method": method, "context": context}
        payload = {"id": self.msg_id, "method": method, "params": params}
        self.writer.write((json.dumps(payload) + '\n').encode('utf-8'))
        await self.writer.drain()

    async def read_message(self):
        try:
            line = await self.reader.readline()
            if not line: return None
            return json.loads(line.decode('utf-8').strip())
        except Exception:
            return None

    def handle_response(self, msg):
        """Dispatches a pool response by the method of the request it answers."""
        pending = self.pending_requests.pop(msg['id'], None)
        if pending is None:
            return
        method = pending["method"]
        context = pending.get("context") or {}
        result = msg.get('result')

        if method == 'mining.subscribe':
            if result and len(result) >= 3:
                self.extranonce1 = result[1]
                self.extranonce2_size = result[2]
            logging.info(f"[*] Subscribed. Extranonce1: {self.extranonce1}, "
                         f"Extranonce2 size: {self.extranonce2_size}")

        elif method == 'mining.authorize':
            if result:
                logging.info(f"[*] Worker '{self.username}' authorized successfully.")
            else:
                raise AuthorizationError(
                    f"Worker authorization failed: {msg.get('error')}. Check wallet/worker name.")

        elif method == 'mining.submit':
            job_id = context.get("job_id", "?")
            job_short = job_id[:8] + "..." if len(job_id) > 8 else job_id
            share_detail = (
                f"job={job_short} nonce={context.get('nonce_hex', '?')} "
                f"extranonce2={context.get('extranonce2', '?')} ntime={context.get('ntime', '?')}"
            )
            if result:
                logging.info(f"[+] SHARE ACCEPTED | {share_detail}")
            else:
                logging.warning(f"[-] SHARE REJECTED | {share_detail} | error={msg.get('error')}")

    async def listen_for_jobs(self):
        logging.info("[*] Awaiting data from pool...")
        while True:
            msg = await self.read_message()
            if not msg:
                raise ConnectionError("Pool closed connection.")

            # --- 1. Responses to our requests (matched by id) ---
            if msg.get('id') in self.pending_requests:
                self.handle_response(msg)
                continue

            # --- 2. Server notifications ---
            method = msg.get('method')

            if method == 'mining.set_difficulty':
                self.current_difficulty = msg['params'][0]
                logging.info(f"[POOL] Difficulty adjusted to: {self.current_difficulty}")

            elif method == 'mining.set_extranonce':
                self.extranonce1 = msg['params'][0]
                self.extranonce2_size = msg['params'][1]
                logging.info(f"[POOL] Extranonce updated: {self.extranonce1}")

            elif method == 'mining.notify':
                job_id = msg['params'][0]
                clean_jobs = msg['params'][8]

                # Always mine the newest job: the previous one is at best
                # sub-optimal and, if clean_jobs is set, outright stale.
                self.cancel_current_job()

                logging.info(f"[+] Received New Job: {job_id[:8]}... (Clean: {clean_jobs})")
                self.current_job_task = asyncio.create_task(self.mine_job_loop(msg['params']))

    # -------------------------------------------------------------------
    # HEADER CONSTRUCTION
    # -------------------------------------------------------------------

    def compute_merkle_root(self, coinb1, coinb2, merkle_branch, extranonce2_hex) -> bytes:
        """Hashes the coinbase and walks the merkle branch. Returns the root
        in internal (little-endian) byte order, as it appears in the header."""
        coinbase = (bytes.fromhex(coinb1) + bytes.fromhex(self.extranonce1)
                    + bytes.fromhex(extranonce2_hex) + bytes.fromhex(coinb2))
        merkle_root = sha256d(coinbase)
        for branch in merkle_branch:
            merkle_root = sha256d(merkle_root + bytes.fromhex(branch))
        return merkle_root

    @staticmethod
    def pack_header_prefix(version, prevhash, merkle_root, ntime, nbits) -> bytes:
        """
        Packs the 76-byte block header prefix (everything except the nonce).

        Endianness notes:
        - version/ntime/nbits arrive as big-endian hex, headers store them LE.
        - Stratum sends prevhash with each 4-byte word byte-swapped relative
          to the header layout, so each word is reversed individually
          (a full 32-byte reversal would be wrong).
        - merkle_root is a raw sha256d output, already in header byte order.
        """
        version_bin = struct.pack("<I", int(version, 16))
        prevhash_raw = bytes.fromhex(prevhash)
        prevhash_bin = b''.join(prevhash_raw[i:i+4][::-1] for i in range(0, 32, 4))
        ntime_bin = struct.pack("<I", int(ntime, 16))
        nbits_bin = struct.pack("<I", int(nbits, 16))
        return version_bin + prevhash_bin + merkle_root + ntime_bin + nbits_bin

    # -------------------------------------------------------------------
    # MINING LOOP
    # -------------------------------------------------------------------

    async def mine_job_loop(self, params):
        """Scans the full 2^32 nonce space per extranonce2, then rolls
        extranonce2, until cancelled by a new job or disconnect."""
        job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, _ = params[:9]

        extranonce2_int = 0
        max_extranonce2 = (2 ** (self.extranonce2_size * 8)) - 1

        try:
            while extranonce2_int <= max_extranonce2:
                extranonce2_hex = f"{extranonce2_int:0{self.extranonce2_size * 2}x}"

                merkle_root = self.compute_merkle_root(coinb1, coinb2, merkle_branch, extranonce2_hex)
                header_prefix = self.pack_header_prefix(version, prevhash, merkle_root, ntime, nbits)

                uploaded_difficulty = self.current_difficulty
                target_bytes = diff_to_target(uploaded_difficulty)
                target_int = int.from_bytes(target_bytes, 'big')

                # Upload midstate + header tail + target once per extranonce2
                await asyncio.to_thread(prepare_job, header_prefix, target_bytes)

                nonce_base = 0
                while nonce_base < 2 ** 32:
                    # Pick up mid-job difficulty changes without re-hashing anything
                    if self.current_difficulty != uploaded_difficulty:
                        uploaded_difficulty = self.current_difficulty
                        target_bytes = diff_to_target(uploaded_difficulty)
                        target_int = int.from_bytes(target_bytes, 'big')
                        await asyncio.to_thread(set_target, target_bytes)

                    # Offload one 2^24-nonce batch to the GPU (yields event loop)
                    candidates, elapsed = await asyncio.to_thread(scan_nonce_range, nonce_base)

                    self.total_nonces_hashed += NONCES_PER_LAUNCH
                    self.log_telemetry(job_id, elapsed)

                    for nonce in candidates:
                        await self.verify_and_submit(
                            header_prefix, target_int, nonce, job_id, extranonce2_hex, ntime)

                    nonce_base += NONCES_PER_LAUNCH

                extranonce2_int += 1

        except asyncio.CancelledError:
            # Expected: pool pushed a new job or the connection dropped
            pass
        except Exception as e:
            logging.error(f"Error in mining loop: {e}")

    async def verify_and_submit(self, header_prefix, target_int, nonce, job_id, extranonce2_hex, ntime):
        """Re-verifies a GPU candidate on the CPU, then submits it."""
        header = header_prefix + struct.pack("<I", nonce)
        hash_int = int.from_bytes(sha256d(header), 'little')

        if hash_int > target_int:
            logging.warning(f"[!] GPU candidate {nonce:08x} failed CPU verification, discarding.")
            return

        # Stratum expects nonce/ntime as big-endian hex strings
        nonce_hex = f"{nonce:08x}"
        logging.info(f"[!] -> VALID NONCE FOUND: {nonce_hex} (hash: {hash_int:064x}) <-")
        await self.send_message("mining.submit", [
            self.username, job_id, extranonce2_hex, ntime, nonce_hex
        ], context={
            "job_id": job_id,
            "extranonce2": extranonce2_hex,
            "ntime": ntime,
            "nonce_hex": nonce_hex,
        })

    def log_telemetry(self, job_id, batch_elapsed):
        """Throttled hashrate logging (at most once every 10 seconds)."""
        current_time = time.time()
        if current_time - self.last_log_time < 10.0:
            return

        session_elapsed = current_time - self.session_start_time
        instant_hr = (NONCES_PER_LAUNCH / batch_elapsed) if batch_elapsed > 0 else 0
        session_hr = (self.total_nonces_hashed / session_elapsed) if session_elapsed > 0 else 0

        logging.info(
            f"Job {job_id[:4]}... | Diff: {self.current_difficulty} | "
            f"Speed: {format_hashrate(instant_hr)} (Avg: {format_hashrate(session_hr)})"
        )
        self.last_log_time = current_time


# =======================================================================
# CLI ENTRY
# =======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Python CUDA Stratum Miner (Proof of Concept)")
    parser.add_argument('-o', '--url', required=True, help='Stratum pool URL')
    parser.add_argument('-u', '--user', required=True, help='Wallet/Worker name')
    parser.add_argument('-p', '--pass', dest='password', default='x', help='Worker password')
    return parser.parse_args()

async def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')

    gpu_name, cuda_version = get_gpu_info()
    logging.info(f"=====================================")
    logging.info(f" Python CUDA Miner Initialized")
    logging.info(f" GPU: {gpu_name} (CUDA {cuda_version})")
    logging.info(f"=====================================")

    parsed_url = urlparse(args.url)
    miner = StratumMiner(parsed_url.hostname, parsed_url.port or 3333, args.user, args.password)

    try:
        await miner.run()
    except KeyboardInterrupt:
        logging.info("\n[*] Mining session explicitly stopped by user.")

if __name__ == "__main__":
    asyncio.run(main())
