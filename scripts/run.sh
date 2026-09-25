#!/usr/bin/env bash
# Start BAYMAX OS locally.  Usage: scripts/run.sh [--with-stt]
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv 2>/dev/null || true
. .venv/bin/activate
pip install -q -r requirements.txt
if [[ "${1:-}" == "--with-stt" ]]; then pip install -q faster-whisper; fi
exec python -m baymax.server.app
