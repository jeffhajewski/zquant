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

# Real embeddings at d >= 768, from VectorDBBench's public bucket: Cohere
# multilingual-22-12 (100k x 768, ~320 MB) and OpenAI text-embedding-ada-002 on C4
# (50k x 1536, ~460 MB), as parquet. Only fetched when asked for, by
# `tools/fetch_datasets.sh highdim`; `python bench/py/prepare.py cohere-768` (or
# openai-1536) then turns them into the same prepared layout as everything else.
if [ "${1:-}" = "highdim" ]; then
  for corpus in cohere_small_100k openai_small_50k; do
    mkdir -p "data/$corpus"
    for part in train test; do
      if [ ! -f "data/$corpus/$part.parquet" ]; then
        echo "$corpus/$part: downloading..."
        curl -sSL --max-time 900 -o "data/$corpus/$part.parquet" \
          "https://assets.zilliz.com/benchmark/$corpus/$part.parquet"
      fi
    done
    echo "$corpus: ready"
  done
fi
