"""Perplexity with a fake-quantized KV cache: the end-to-end check `kv_bench` cannot make.

`kv_bench` measures attention-output error on three layers of one model. That is a proxy:
it says nothing about how errors compound through every layer, or whether they matter to
the next token. This runs a Hugging Face causal LM over wikitext-2 test with every key and
value the cache receives quantized and immediately dequantized, and reports perplexity.

The cache is a `DynamicCache` whose `update` quantizes what it is handed. Keys arrive
post-RoPE, which is what a real KV cache stores. Each window is one forward pass, so every
query attends to quantized keys and values for all earlier tokens and itself — the state a
cache is in during decoding, measured without decoding token by token.

Methods, all at head_dim 64, bytes per token per head for K+V together:

  fp16              the reference every number is relative to.
  int4 per-row      kv_bench's original baseline: symmetric, one f32 scale per row, 15
                    levels. Kept only so the old comparison can be seen end to end.
  int<b> tok g<n>   asymmetric per token, groups of n channels.
  KIVI-<b> g<n>     keys per channel over groups of n tokens, values per token over groups
                    of min(n, head_dim) channels, both asymmetric (Liu et al., 2024).
  zquant b=<b>      `zquant.Codec(dim=head_dim, bits=b)` encode then decode: calibration
                    off, reconstruction rather than the index's estimator.

Scale and zero point are f16 and are rounded to f16 before use; zquant's norm is f32, as
the library stores it. Both are counted in the bytes column.

Two simplifications, both stated rather than hidden:

  * KIVI keeps the most recent tokens (up to one group, plus a residual window) in fp16.
    That window is left out: every token is quantized, at the bytes shown. As deployed,
    KIVI would do somewhat better than its rows here and cost slightly more.
  * KIVI's per-channel key groups are formed over the whole window at once, so a key's
    group statistics include up to n-1 *later* keys. Real KIVI only quantizes a group once
    it is full, and holds the tokens before that in fp16, so this is the less favourable
    of the two for KIVI as well.

    uv venv /tmp/kvppl && VIRTUAL_ENV=/tmp/kvppl uv pip install torch transformers datasets
    VIRTUAL_ENV=/tmp/kvppl uv pip install -e clients/python    # after `zig build lib`
    /tmp/kvppl/bin/python bench/py/kv_ppl.py HuggingFaceTB/SmolLM2-135M

Environment: KV_PPL_WINDOWS (default 0: every full window in the test set), KV_PPL_SEQ
(default 1024), KV_PPL_DEVICE (default cpu), KV_PPL_METHODS (comma-separated labels,
default all).
"""
import math
import os
import sys
import time

# One Accelerate BLAS thread, set before torch loads it, with torch's OpenMP threads (4 in
# the recorded runs) doing the parallel work. It is what the README numbers ran with; it
# is not known to matter. (Rows that took ~1000 s in those logs were the machine sleeping
# in thermal dark-wake cycles, not anything here.)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import zquant

N_WINDOWS = int(os.environ.get("KV_PPL_WINDOWS", "0"))  # 0: the whole test set
SEQ = int(os.environ.get("KV_PPL_SEQ", "1024"))
# CPU by default. zquant encodes on the CPU, so on MPS every layer forces a GPU sync and a
# copy each way; on an M5 that made the zquant rows over 50x slower than the rest.
DEVICE = os.environ.get("KV_PPL_DEVICE", "cpu")


# --- quantizers. Each takes and returns [batch, heads, tokens, head_dim] float32.

def f16(x):
    return x.half().float()


def _asym(x, bits, dim):
    """Asymmetric min/max quantization reducing over `dim`, all 2^bits levels used."""
    lo = x.amin(dim=dim, keepdim=True)
    hi = x.amax(dim=dim, keepdim=True)
    levels = 2 ** bits - 1
    zero = f16(lo)
    scale = f16((hi - lo) / levels)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round((x - zero) / safe), 0, levels)
    return torch.where(scale == 0, zero.expand_as(x), q * scale + zero)


def per_token(bits, group):
    def fn(x):
        b, h, t, d = x.shape
        g = min(group, d)
        return _asym(x.reshape(b, h, t, d // g, g), bits, -1).reshape(b, h, t, d)
    return fn


def per_channel(bits, group):
    def fn(x):
        out = torch.empty_like(x)
        for t0 in range(0, x.shape[2], group):
            out[:, :, t0:t0 + group] = _asym(x[:, :, t0:t0 + group], bits, 2)
        return out
    return fn


def symmetric_row(bits):
    """kv_bench's original baseline: one symmetric f32 scale per row, 2^(bits-1)-1 levels."""
    levels = 2 ** (bits - 1) - 1

    def fn(x):
        amax = x.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(amax > 0, amax / levels, torch.ones_like(amax))
        return torch.round(x / scale) * scale
    return fn


def zquant_codec(bits, head_dim):
    codec = zquant.Codec(dim=head_dim, bits=bits)

    def fn(x):
        rows = x.detach().to("cpu", torch.float32).reshape(-1, head_dim).numpy()
        codes, norms = codec.encode(rows)
        back = codec.decode(codes, norms)
        return torch.from_numpy(back).reshape(x.shape).to(x.device)
    fn.code_bytes = codec.code_bytes
    return fn


def methods(head_dim):
    d = head_dim

    def tok_bytes(bits, g):
        return d * bits // 8 + 4 * math.ceil(d / g)

    def chan_bytes(bits, g):
        return d * bits // 8 + 4 * d / g

    m = [("fp16", 4 * d, f16, f16)]
    m.append(("int4 per-row", 2 * (d * 4 // 8 + 4), symmetric_row(4), symmetric_row(4)))
    for bits in (4, 2):
        for g in ((64, 32) if bits == 4 else (32,)):
            q = per_token(bits, g)
            m.append((f"int{bits} tok g{g}", 2 * tok_bytes(bits, g), q, q))
        for g in (128, 64, 32):
            vg = min(g, d)
            m.append((f"KIVI-{bits} g{g}", chan_bytes(bits, g) + tok_bytes(bits, vg),
                      per_channel(bits, g), per_token(bits, vg)))
    for bits in (2, 3, 4, 5):
        z = zquant_codec(bits, d)
        m.append((f"zquant b={bits}", 2 * (z.code_bytes + 4), z, z))
    # Keys and values are different problems; kv_bench finds KIVI's per-channel keys beat
    # zquant's while zquant's values beat per-token int4. This row checks that end to end.
    z4 = zquant_codec(4, d)
    m.append(("KIVI-4 g64 K + zquant b=4 V", chan_bytes(4, 64) + z4.code_bytes + 4,
              per_channel(4, 64), z4))
    wanted = os.environ.get("KV_PPL_METHODS")
    if wanted:
        keep = set(wanted.split(","))
        m = [x for x in m if x[0] in keep]
    return m


class FakeQuantCache(DynamicCache):
    """Quantizes and dequantizes whatever the attention layers hand the cache."""

    def __init__(self, qk, qv, **kwargs):
        super().__init__(**kwargs)
        self.qk, self.qv = qk, qv

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        dtype = key_states.dtype
        k = self.qk(key_states.float()).to(dtype)
        v = self.qv(value_states.float()).to(dtype)
        return super().update(k, v, layer_idx, *args, **kwargs)


def windows(tok):
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = ids.numel() // SEQ
    n = min(N_WINDOWS, n) if N_WINDOWS else n
    return [ids[i * SEQ:(i + 1) * SEQ] for i in range(n)], ids.numel()


@torch.no_grad()
def perplexity(model, wins, qk, qv):
    nll, count = 0.0, 0
    for w in wins:
        ids = w.unsqueeze(0).to(DEVICE)
        out = model(ids, past_key_values=FakeQuantCache(qk, qv), use_cache=True)
        logits = out.logits[0, :-1].float()
        loss = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum")
        nll += loss.item()
        count += ids.shape[1] - 1
    return math.exp(nll / count)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(DEVICE).eval()
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    wins, total = windows(tok)
    print(f"{name}: layers={cfg.num_hidden_layers} kv_heads={cfg.num_key_value_heads} "
          f"head_dim={head_dim} device={DEVICE}")
    print(f"wikitext-2-raw-v1 test: first {len(wins)} windows of {SEQ} tokens "
          f"({len(wins) * SEQ} of {total} tokens)\n", flush=True)

    print(f"  {'method':<28} {'B/tok/head':>10} {'ppl':>9} {'vs fp16':>8} {'s':>6}")
    base = None
    for label, nbytes, qk, qv in methods(head_dim):
        t0 = time.time()
        ppl = perplexity(model, wins, qk, qv)
        base = base or ppl
        print(f"  {label:<28} {nbytes:>10g} {ppl:>9.3f} {ppl - base:>+8.3f} {time.time() - t0:>6.1f}",
              flush=True)


if __name__ == "__main__":
    main()
