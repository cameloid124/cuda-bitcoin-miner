import argparse
import asyncio
import json
import struct
import logging
import socket
import time
import hashlib
from fractions import Fraction
from urllib.parse import urlparse

from cuda_kernel import GpuScanner, get_gpu_info, NONCES_PER_LAUNCH

USER_AGENT = "python-cuda-miner/1.0"

# Full 32-bit header nonce space, swept in NONCES_PER_LAUNCH-sized batches.
NONCE_SPACE = 1 << 32

# ntime rolling: for each extranonce2 (fixed coinbase / merkle root) we also
# scan a small window of incremented ntime values. This extends the search
# space without recomputing the coinbase, and keeps timestamps fresh. Pools
# tolerate small forward rolls; kept conservative here.
NTIME_ROLL = 8

# How often to send an application-level mining.ping to keep the socket active
# (ckpool and many proxies drop idle connections around 60 s).
KEEPALIVE_INTERVAL = 30

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

def hash_to_difficulty(hash_int: int) -> float:
    """Returns the Stratum difficulty implied by a share hash (as a LE integer)."""
    if hash_int <= 0:
        return float("inf")
    return float(Fraction(DIFF1_TARGET) / Fraction(hash_int))

def format_difficulty(difficulty: float) -> str:
    """Formats a difficulty value for log output."""
    if difficulty <= 0 or difficulty == float("inf"):
        return "n/a"
    if difficulty >= 1e12:
        return f"{difficulty / 1e12:.2f}T"
    if difficulty >= 1e9:
        return f"{difficulty / 1e9:.2f}G"
    if difficulty >= 1e6:
        return f"{difficulty / 1e6:.2f}M"
    if difficulty >= 1e3:
        return f"{difficulty / 1e3:.2f}k"
    if difficulty >= 100:
        return f"{difficulty:.0f}"
    if difficulty >= 1:
        return f"{difficulty:.2f}"
    return f"{difficulty:.4f}"

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
        self.authorized = False
        self._pending_job = None
        self._keepalive_task = None

        # Asynchronous Task Management
        self.current_job_task = None

        # Telemetry State
        self.session_start_time = 0
        self.total_nonces_hashed = 0
        self.last_log_time = 0
        self.best_difficulty = 0.0   # best share difficulty seen since startup

        # GPU
        self.scanner = GpuScanner()

    def note_best_hash(self, hash_int: int):
        """Updates session best from any qualifying hash (GPU batch or submit)."""
        share_diff = hash_to_difficulty(hash_int)
        if share_diff > self.best_difficulty and share_diff != float("inf"):
            self.best_difficulty = share_diff

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

    def _enable_tcp_keepalive(self):
        """Enable OS-level TCP keepalive so NAT/middleboxes don't drop idle sockets."""
        sock = self.writer.get_extra_info("socket")
        if sock is None:
            return
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)

    async def connect(self):
        logging.info(f"[*] Connecting to Stratum pool at {self.host}:{self.port}...")
        self.reader, self.writer = await asyncio.open_connection(self.host, self.port)
        self._enable_tcp_keepalive()

        # Reset per-connection state
        self.pending_requests.clear()
        self.authorized = False
        self._pending_job = None
        self.session_start_time = time.time()
        self.total_nonces_hashed = 0
        self.last_log_time = time.time()
        self.best_difficulty = 0.0

        if self._keepalive_task:
            self._keepalive_task.cancel()
            self._keepalive_task = None

        # Handshake: wait for each response before treating the session as live.
        # ckpool (and others) may emit difficulty/job notifications between
        # subscribe and authorize; buffer the job until auth succeeds.
        sub_id = await self.send_message("mining.subscribe", [USER_AGENT])
        auth_id = await self.send_message("mining.authorize", [self.username, self.password])
        await self._complete_handshake({sub_id, auth_id})

        self.authorized = True
        if self._pending_job:
            self._start_job(self._pending_job)
            self._pending_job = None

        self._keepalive_task = asyncio.create_task(self._keepalive_loop())
        try:
            await self.listen_for_jobs()
        finally:
            if self._keepalive_task:
                self._keepalive_task.cancel()
                self._keepalive_task = None

    async def _complete_handshake(self, pending_ids: set):
        while pending_ids:
            msg = await self.read_message()
            if msg is None:
                raise ConnectionError("Pool closed connection during handshake.")
            if msg.get("id") in pending_ids:
                self.handle_response(msg)
                pending_ids.discard(msg["id"])
            else:
                await self.dispatch_notification(msg, allow_mining=False)

    async def send_message(self, method, params, context=None) -> int:
        self.msg_id += 1
        req_id = self.msg_id
        self.pending_requests[req_id] = {"method": method, "context": context}
        payload = {"id": req_id, "method": method, "params": params}
        self.writer.write((json.dumps(payload) + '\n').encode('utf-8'))
        await self.writer.drain()
        return req_id

    async def send_reply(self, msg_id, result):
        """Send a JSON-RPC result without registering a pending request."""
        payload = {"id": msg_id, "result": result, "error": None}
        self.writer.write((json.dumps(payload) + '\n').encode('utf-8'))
        await self.writer.drain()

    async def read_message(self):
        while True:
            try:
                line = await self.reader.readline()
            except Exception as e:
                logging.warning(f"[POOL] Read error: {e}")
                return None
            if not line:
                return None
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError as e:
                logging.warning(f"[POOL] Ignoring non-JSON line: {text[:120]!r} ({e})")
                continue

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

    async def dispatch_notification(self, msg, allow_mining=True):
        """Handle pool-initiated Stratum notifications."""
        method = msg.get('method')

        if method == 'mining.ping':
            await self.send_reply(msg.get('id'), "pong")
            return

        if method == 'client.reconnect':
            raise ConnectionError("Pool requested reconnect.")

        if method == 'mining.set_difficulty':
            self.current_difficulty = msg['params'][0]
            logging.info(f"[POOL] Difficulty adjusted to: {self.current_difficulty}")

        elif method == 'mining.set_extranonce':
            self.extranonce1 = msg['params'][0]
            self.extranonce2_size = msg['params'][1]
            logging.info(f"[POOL] Extranonce updated: {self.extranonce1}")

        elif method == 'mining.notify':
            if allow_mining and self.authorized:
                self._start_job(msg['params'])
            else:
                self._pending_job = msg['params']

    def _start_job(self, params):
        job_id = params[0]
        clean_jobs = params[8]
        self.cancel_current_job()
        logging.info(f"[+] Received New Job: {job_id[:8]}... (Clean: {clean_jobs})")
        self.current_job_task = asyncio.create_task(self.mine_job_loop(params))

    async def _keepalive_loop(self):
        """Periodic mining.ping keeps the TCP session active through NAT/proxies."""
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            try:
                self.msg_id += 1
                payload = {"id": self.msg_id, "method": "mining.ping", "params": []}
                self.writer.write((json.dumps(payload) + '\n').encode('utf-8'))
                await self.writer.drain()
            except Exception:
                break

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

            # --- 2. Responses to untracked requests (e.g. keepalive ping) ---
            if msg.get('method') is None and 'result' in msg:
                continue

            # --- 3. Server notifications ---
            await self.dispatch_notification(msg)

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
        """For each extranonce2, sweeps the full 2^32 nonce space across a
        small window of rolled ntime values, then advances extranonce2, until
        cancelled by a new job or disconnect."""
        job_id, prevhash, coinb1, coinb2, merkle_branch, version, nbits, ntime, _ = params[:9]
        ntime_base = int(ntime, 16)

        self.uploaded_difficulty = self.current_difficulty
        target_bytes = diff_to_target(self.uploaded_difficulty)
        self.target_int = int.from_bytes(target_bytes, 'big')

        extranonce2_int = 0
        max_extranonce2 = (2 ** (self.extranonce2_size * 8)) - 1

        try:
            while extranonce2_int <= max_extranonce2:
                extranonce2_hex = f"{extranonce2_int:0{self.extranonce2_size * 2}x}"
                merkle_root = self.compute_merkle_root(coinb1, coinb2, merkle_branch, extranonce2_hex)

                for roll in range(NTIME_ROLL):
                    ntime_hex = f"{ntime_base + roll:08x}"
                    header_prefix = self.pack_header_prefix(
                        version, prevhash, merkle_root, ntime_hex, nbits)
                    await self.scan_header(header_prefix, job_id, extranonce2_hex, ntime_hex)

                extranonce2_int += 1

        except asyncio.CancelledError:
            # Expected: pool pushed a new job or the connection dropped.
            # Drain the pipeline so the next job starts from a clean scanner.
            await self.drain_pipeline()
            raise
        except Exception as e:
            logging.error(f"Error in mining loop: {e}")

    async def scan_header(self, header_prefix, job_id, extranonce2_hex, ntime_hex):
        """Double-buffered sweep of the full nonce space for one fixed header
        prefix. Two batches are kept in flight so the GPU never idles between
        launches."""
        target_bytes = diff_to_target(self.uploaded_difficulty)
        await asyncio.to_thread(self.scanner.prepare_job, header_prefix, target_bytes)

        nonce_bases = range(0, NONCE_SPACE, NONCES_PER_LAUNCH)
        submitted = 0

        # Prime the pipeline with the first batch.
        await asyncio.to_thread(self.scanner.submit, nonce_bases[0])
        submitted = 1

        while self.scanner.in_flight:
            # Keep two batches in flight whenever more work remains.
            if submitted < len(nonce_bases):
                await asyncio.to_thread(self.scanner.submit, nonce_bases[submitted])
                submitted += 1

            await self.maybe_update_target()

            batch_start = time.perf_counter()
            _, candidates, batch_best = await asyncio.to_thread(self.scanner.collect)
            elapsed = time.perf_counter() - batch_start

            self.total_nonces_hashed += NONCES_PER_LAUNCH
            if batch_best is not None:
                self.note_best_hash(batch_best)
            self.log_telemetry(job_id, elapsed)

            for nonce in candidates:
                await self.verify_and_submit(
                    header_prefix, self.target_int, nonce, job_id, extranonce2_hex, ntime_hex)

    async def maybe_update_target(self):
        """Applies a mid-job difficulty change without discarding any work
        (only the 32-byte target is re-uploaded)."""
        if self.current_difficulty != self.uploaded_difficulty:
            self.uploaded_difficulty = self.current_difficulty
            target_bytes = diff_to_target(self.uploaded_difficulty)
            self.target_int = int.from_bytes(target_bytes, 'big')
            await asyncio.to_thread(self.scanner.set_target, target_bytes)

    async def drain_pipeline(self):
        """Collects any batches still in flight so the scanner is left empty."""
        while self.scanner.in_flight:
            await asyncio.to_thread(self.scanner.collect)

    async def verify_and_submit(self, header_prefix, target_int, nonce, job_id, extranonce2_hex, ntime):
        """Re-verifies a GPU candidate on the CPU, then submits it."""
        header = header_prefix + struct.pack("<I", nonce)
        hash_int = int.from_bytes(sha256d(header), 'little')

        if hash_int > target_int:
            logging.warning(f"[!] GPU candidate {nonce:08x} failed CPU verification, discarding.")
            return

        share_diff = hash_to_difficulty(hash_int)
        prev_best = self.best_difficulty
        self.note_best_hash(hash_int)
        new_best = share_diff > prev_best

        # Stratum expects nonce/ntime as big-endian hex strings
        nonce_hex = f"{nonce:08x}"
        best_note = f", new session best {format_difficulty(share_diff)}" if new_best else ""
        logging.info(
            f"[!] -> VALID NONCE FOUND: {nonce_hex} "
            f"(share diff: {format_difficulty(share_diff)}{best_note}) <-"
        )
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
            f"Best: {format_difficulty(self.best_difficulty)} | "
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

    gpu_name, cuda_version, driver_version, kernel_desc = get_gpu_info()
    logging.info(f"=====================================")
    logging.info(f" Python CUDA Miner Initialized")
    logging.info(f" GPU: {gpu_name} (Driver {driver_version}, CUDA {cuda_version})")
    logging.info(f" Kernel: {kernel_desc}")
    logging.info(f"=====================================")

    parsed_url = urlparse(args.url)
    miner = StratumMiner(parsed_url.hostname, parsed_url.port or 3333, args.user, args.password)

    try:
        await miner.run()
    except KeyboardInterrupt:
        logging.info("\n[*] Mining session explicitly stopped by user.")

if __name__ == "__main__":
    asyncio.run(main())
