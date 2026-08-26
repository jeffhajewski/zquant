#!/usr/bin/env bash
# Fetch ANN benchmark datasets into data/. Not checked in: several MB to several GB,
# and redistribution terms vary by corpus.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

fetch() {
  local name="$1" url="$2" out="data/$1"
  if [ -d "$out" ]; then echo "$name: already present"; return; fi
  echo "$name: downloading..."
  curl -sSL --max-time 900 -o "data/$name.tar.gz" "$url"
  tar xzf "data/$name.tar.gz" -C data
  rm -f "data/$name.tar.gz"
  echo "$name: ready"
}

# ANN_SIFT10K: 10k base, 100 queries, 128-dim, top-100 L2 ground truth (~5 MB).
# The standard small sanity corpus.
fetch siftsmall "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"

# nytimes-256-angular: real text embeddings, 290k x 256, with ground truth (~300 MB).
# The dimension and the fact that it comes from a model, rather than a hand-designed
# descriptor, make it the more representative of the two.
if [ ! -f data/nytimes-256-angular.hdf5 ]; then
  echo "nytimes-256: downloading..."
  curl -sSL --max-time 900 -o data/nytimes-256-angular.hdf5 \
    "http://ann-benchmarks.com/nytimes-256-angular.hdf5"
fi
echo "nytimes-256: ready"

# ANN_SIFT1M: 1M base, 10k queries (~160 MB). Uncomment when needed.
# fetch sift "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
