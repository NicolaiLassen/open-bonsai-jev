#!/usr/bin/env bash
# Fetch a STOCK llama.cpp release build. This runs Bonsai 1 (Q1_0), which is
# merged upstream. It will NOT correctly run Bonsai 2's ternary weights -- for
# those use build_llama_fork.sh.
set -euo pipefail

TAG="${LLAMA_TAG:-b11050}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/bonzi/llama.cpp"

# Upstream ships .tar.gz for macOS/Linux and .zip only for Windows.
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)   ASSET="llama-$TAG-bin-macos-arm64.tar.gz" ;;
  Darwin-x86_64)  ASSET="llama-$TAG-bin-macos-x64.tar.gz" ;;
  Linux-aarch64)  ASSET="llama-$TAG-bin-ubuntu-vulkan-arm64.tar.gz" ;;
  Linux-x86_64)   ASSET="llama-$TAG-bin-ubuntu-vulkan-x64.tar.gz" ;;
  *) echo "no prebuilt asset for $(uname -s)-$(uname -m); build from source" >&2; exit 1 ;;
esac

URL="https://github.com/ggml-org/llama.cpp/releases/download/$TAG/$ASSET"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "downloading $ASSET"
curl -fL "$URL" -o "$TMP/$ASSET"

mkdir -p "$DEST"
case "$ASSET" in
  *.tar.gz) tar -xzf "$TMP/$ASSET" -C "$DEST" --strip-components=1 ;;
  *.zip)    unzip -oq "$TMP/$ASSET" -d "$DEST" ;;
esac

find "$DEST" -name 'llama-server*' -perm -u+r -exec chmod +x {} \; 2>/dev/null || true
echo "extracted to $DEST"
find "$DEST" -name 'llama-server*' | head
