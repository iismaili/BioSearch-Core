#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "requirements.txt" ]; then
    python3 -m pip install -q -r requirements.txt
fi

python3 run.py
