#!/usr/bin/env bash
# Kill any llama-server this repo started and left behind.
#
# score.py/api.py stop their own server on a clean exit, but a SIGKILL (or a
# crashed terminal) orphans it -- and an orphaned 27B holds several GB and
# quietly competes with whatever you measure next.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
found=$(pgrep -f "$HERE/bonzi/llama.cpp.*/llama-server" 2>/dev/null || true)
if [ -z "$found" ]; then echo "no stray llama-server from $HERE"; exit 0; fi
ps -o pid,etime,rss,command -p $found | cut -c1-150
echo "$found" | xargs kill 2>/dev/null || true
sleep 2
still=$(pgrep -f "$HERE/bonzi/llama.cpp.*/llama-server" 2>/dev/null || true)
[ -n "$still" ] && echo "$still" | xargs kill -9 2>/dev/null || true
echo "stopped"
