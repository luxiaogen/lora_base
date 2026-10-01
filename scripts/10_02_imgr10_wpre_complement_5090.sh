#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m unittest test.test_wpre_complement test.test_wpre_complement_night test.test_wpre_distill -q
python3 scripts/run_wpre_complement.py "$@"
