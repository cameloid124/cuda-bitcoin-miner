#!/usr/bin/env bash
set -euo pipefail

: "${STRATUM_URL:?STRATUM_URL must be set (e.g. stratum+tcp://pool.example.com:3333)}"
: "${STRATUM_USER:?STRATUM_USER must be set (wallet or wallet.worker)}"

STRATUM_PASSWORD="${STRATUM_PASSWORD:-x}"

exec python3 miner.py \
  -o "${STRATUM_URL}" \
  -u "${STRATUM_USER}" \
  -p "${STRATUM_PASSWORD}"
