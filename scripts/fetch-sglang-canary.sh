#!/usr/bin/env bash
# Fetch the SGLang python tree for Dockerfile.canary as a GitHub tarball.
# The serving image's git cannot fetch unauthenticated, so the tree is staged on the
# host and COPYed into the image. Pinned to the commit measured in docs/window-20260916.md.
set -euo pipefail
REF="${SGLANG_CANARY_REF:-f80c91a4b}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/runtime/sglang-canary"
mkdir -p "$DST"
cd "$DST"
rm -rf python src.tgz
echo "fetching sgl-project/sglang @ $REF ..."
curl -fsSL -m 600 "https://github.com/sgl-project/sglang/archive/${REF}.tar.gz" -o src.tgz
tar -xzf src.tgz --strip-components=1 --wildcards "sglang-*/python"
rm -f src.tgz
echo "$REF" > REF
du -sh python
echo "staged $DST/python (ref $REF); now: docker build -f Dockerfile.canary -t dsv41-4x-spark:canary ."
