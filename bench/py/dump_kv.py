"""Dump real attention Q/K/V tensors from a language model.

Synthetic data would beg the question here. A KV cache has structure that matters to a
quantizer and is hard to guess: RoPE rotates keys position-dependently, a handful of key
channels carry outsized magnitude, and values look nothing like keys. Measuring on tensors
a real model actually produced is the only way to know whether that structure hurts.

Writes one file per (layer, tensor) in the same fvecs format the other benchmarks read,
with rows ordered [head, position] so a Zig reader can slice a head without a header.
"""
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("KV_MODEL", "HuggingFaceTB/SmolLM2-135M")
LAYERS = [int(x) for x in os.environ.get("KV_LAYERS", "0,7,15").split(",")]
OUT = "data/kv"

TEXT = """The history of computing is a history of abstraction. Each layer that programmers
build lets the next generation forget something that was once essential and think about
something that was once impossible. Assembly language let us forget opcodes. Compilers let
us forget registers. Garbage collection let us forget lifetimes, at a cost that was argued
about for thirty years and is still argued about today, because the cost is real and so is
the benefit. What makes this interesting is not that abstraction is good, which is obvious,
but that the boundary keeps moving and the arguments keep the same shape. Someone points at
the overhead. Someone else points at the programs that would never have been written. Both
are right, and the disagreement is really about which programs matter.
""" * 6


def write_fvecs(path, arr):
    n, d = arr.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = np.ascontiguousarray(arr, dtype=np.float32).view(np.int32)
    out.tofile(path)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(f"loading {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    ids = tok(TEXT, return_tensors="pt").input_ids
    print(f"  {ids.shape[1]} tokens")

    # Queries are not in past_key_values, so capture them from the projection instead.
    captured = {}

    def hook(layer_idx):
        def fn(module, args, output):
            captured[layer_idx] = output.detach()
        return fn

    handles = []
    for li in LAYERS:
        attn = model.model.layers[li].self_attn
        handles.append(attn.q_proj.register_forward_hook(hook(li)))

    with torch.no_grad():
        out = model(ids, use_cache=True)
    for h in handles:
        h.remove()

    cache = out.past_key_values
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    d_head = cfg.hidden_size // cfg.num_attention_heads
    seq = ids.shape[1]
    print(f"  layers={cfg.num_hidden_layers} q_heads={cfg.num_attention_heads} "
          f"kv_heads={n_kv} d_head={d_head}")

    meta = []
    for li in LAYERS:
        # transformers 5 stores the cache as DynamicCache.layers[i].keys/.values rather
        # than an indexable tuple of tensors.
        layer = cache.layers[li]
        k = layer.keys[0]          # [kv_heads, seq, d_head]
        v = layer.values[0]
        q = captured[li][0].view(seq, cfg.num_attention_heads, d_head).permute(1, 0, 2)

        for name, t, heads in (("k", k, n_kv), ("v", v, n_kv), ("q", q, cfg.num_attention_heads)):
            flat = t.reshape(heads * seq, d_head).float().numpy()
            path = f"{OUT}/layer{li}_{name}.fvecs"
            write_fvecs(path, flat)
        meta.append(f"{li} {n_kv} {cfg.num_attention_heads} {seq} {d_head}")
        # Channel magnitude spread is what makes key caches hard; report it so the
        # benchmark's premise is visible rather than assumed.
        kk = k.reshape(-1, d_head).float().numpy()
        sd = kk.std(axis=0)
        print(f"  layer {li}: key channel sd max/mean {sd.max()/sd.mean():.1f}, "
              f"|k| mean {np.linalg.norm(kk, axis=1).mean():.2f}")

    with open(f"{OUT}/meta.txt", "w") as f:
        f.write("\n".join(meta) + "\n")
    print(f"  wrote -> {OUT}")


if __name__ == "__main__":
    main()
