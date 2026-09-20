#!/usr/bin/env bash
# Build PrismML's llama.cpp fork, which is what runs Bonsai 2's ternary weights.
#
# Stock llama.cpp cannot: PTQ1_0 and PQ2_0 are unknown types to it, and Q2_0
# loads but produces gibberish because the Hadamard activation transform never
# runs. Bonsai 1's Q1_0 is merged upstream and does not need this.
set -euo pipefail

REPO=https://github.com/PrismML-Eng/llama.cpp.git
TAG="${LLAMA_FORK_TAG:-prism-b10709-9a9394a}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/bonzi/llama.cpp-prism"

# Pick the accelerator. Override with LLAMA_BACKEND=metal|cuda|vulkan|cpu.
if [ -n "${LLAMA_BACKEND:-}" ]; then
  BACKEND="$LLAMA_BACKEND"
elif [ "$(uname -s)" = "Darwin" ]; then
  BACKEND=metal
elif command -v nvcc >/dev/null 2>&1; then
  BACKEND=cuda
elif command -v vulkaninfo >/dev/null 2>&1; then
  BACKEND=vulkan
else
  BACKEND=cpu
fi

case "$BACKEND" in
  metal)  FLAGS=(-DGGML_METAL=ON) ;;
  cuda)   FLAGS=(-DGGML_CUDA=ON) ;;
  vulkan) FLAGS=(-DGGML_VULKAN=ON) ;;
  cpu)    FLAGS=() ;;
  *) echo "unknown LLAMA_BACKEND=$BACKEND" >&2; exit 1 ;;
esac

echo "building PrismML llama.cpp ($TAG, $BACKEND) into $DEST"

if [ -d "$DEST/.git" ]; then
  git -C "$DEST" fetch --depth 1 origin "$TAG"
  git -C "$DEST" checkout -q FETCH_HEAD
else
  git clone --depth 1 --branch "$TAG" "$REPO" "$DEST"
fi

cmake -S "$DEST" -B "$DEST/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF "${FLAGS[@]}"
cmake --build "$DEST/build" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu)" \
      --target llama-server llama-cli

echo
echo "built: $DEST/build/bin/llama-server"
"$DEST/build/bin/llama-server" --version 2>&1 | head -2
echo
echo "note: the fork reports a plain llama.cpp version string. Confirm it can"
echo "run ternary weights with:  python3 bonzi/chat.py --check"
