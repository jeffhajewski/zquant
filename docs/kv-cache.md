# KV-cache compression

TurboQuant's paper pitches it for compressing a transformer's key-value cache as well as
for vector search. This page measures that through zquant's `Codec`, against the schemes
people actually use. The short version is that it is not the best choice here, and the
reasons are specific enough to be useful.

**At matched memory, zquant is not the best KV-cache quantizer measured here.** KIVI-style
4-bit quantization gives lower perplexity than zquant b=4 at the same 72 bytes on every
model tried. On Qwen2.5-0.5B zquant's keys fail outright. On the two SmolLM2 models zquant
does beat plain per-token int4. That was the baseline behind the README's earlier "about
2× lower error than int4" claim, and the baseline was too weak to support it.

## Perplexity

`bench/py/kv_ppl.py` runs a model over wikitext-2 (raw test split, consecutive 1024-token
windows). Every key and value is quantized and dequantized as it enters the cache, and keys
are post-RoPE, which is what a cache stores. B/token/head counts K and V together at
head_dim 64. That covers packed codes, every scale and zero point (f16), and zquant's norm
(f32).

| B/token/head | scheme | SmolLM2-135M | SmolLM2-360M | Qwen2.5-0.5B |
|---|---|---|---|---|
| 256 | fp16 (perplexity) | 17.37 | 12.07 | 13.42 |
| | *increase over fp16:* | | | |
| 72 | int4 per-row, symmetric (the old baseline) | +4.31 | +2.30 | +35.0 |
| 72 | int4 per-token, asymmetric, group 64 | +1.46 | +0.94 | +97.3 |
| 80 | int4 per-token, asymmetric, group 32 | +0.59 | +0.33 | +28.5 |
| 72 | **KIVI-4, group 64** | **+0.31** | **+0.16** | **+0.22** |
| 80 | KIVI-4, group 32 | +0.22 | +0.10 | +0.16 |
| 72 | zquant b=4 | +0.79 | +0.47 | +151 |
| 88 | zquant b=5 | +0.20 | +0.15 | +39.9 |
| 72 | zquant b=4, keys mean-centred | +0.67 | +0.43 | +5.25 |
| 88 | zquant b=5, keys mean-centred | +0.16 | +0.11 | +1.13 |
| 72 | KIVI-4 keys + zquant b=4 values | +0.30 | +0.16 | +0.23 |
| 40 | KIVI-2, group 64 | +30.4 | +15.5 | +41.1 |
| 40 | zquant b=2 | +31.1 | +22.2 | +1097 |
| 48 | KIVI-2, group 32 | +11.8 | +6.6 | +13.2 |
| 56 | zquant b=3 | +3.95 | +2.28 | +367 |

KIVI ([Liu et al., 2024](https://arxiv.org/abs/2402.02750)) quantizes keys per *channel*,
grouped over tokens, and values per token, grouped over channels. Both use a scale and a
zero point. SmolLM2-135M is 64 windows (65,536 tokens). The other two are 16 windows
(16,384 tokens), because the machine could not run longer.

What it shows:

- **At 72 B, KIVI-4's perplexity cost is 2.5–3× smaller than zquant b=4's** on both
  SmolLM2 models. zquant needs b=5, 22% more memory, to draw level with KIVI-4 at 72 B.
  Against KIVI-4 at 80 B it is level on 135M and behind on 360M.
- **The loss is in keys.** Pairing KIVI keys with zquant values matches all-KIVI. zquant
  reconstructs values as well as anything here.
- **Qwen2.5 breaks zquant's keys.** Its `k_proj` has a bias, so every key carries a large
  constant offset in some channels. zquant scales by the vector norm and rotates, which
  spreads that offset over every coordinate. Most of the code's range then goes to a
  constant. Subtracting each channel's mean before encoding (the "mean-centred" rows) takes
  b=4 from +151 to +5.25, and b=5 from +39.9 to +1.13. That is still far behind KIVI's
  +0.22. What remains is consistent with outlier channels that differ in *scale* as well
  as offset: KIVI gives each channel its own scale, and a data-oblivious rotation mixes
  them. Centring helps the SmolLM2 models a little too, but leaves KIVI ahead at 72 B.
  The `Codec` has no centring option today.
- **Nothing here is usable at 2 bits.** KIVI's own 2-bit results depend on an fp16 window
  of recent tokens, which is left out here (below).

## Attention-output error

`zig build kv_bench` measures the error in a single attention layer's output against
fp16, per layer. It uses Q, K and V dumped from SmolLM2-135M over 936 tokens
(`bench/py/dump_kv.py`), with the same schemes and byte accounting. This is a diagnostic,
not the verdict. It splits key error from value error, which perplexity cannot.

| B/token/head | scheme | layer 0 | layer 7 | layer 15 |
|---|---|---|---|---|
| 136 | int8 per-row | 0.010 | 0.016 | 0.013 |
| 72 | int4 per-row, symmetric | 0.200 | 0.242 | 0.233 |
| 72 | int4 per-token, asymmetric, group 64 | 0.152 | 0.152 | 0.169 |
| 72 | KIVI-4, group 64 | 0.107 | **0.091** | **0.093** |
| 72 | zquant b=4 | **0.092** | 0.124 | 0.106 |
| 88 | zquant b=5 | 0.044 | 0.062 | 0.055 |

The proxy is kinder to zquant than perplexity is: zquant wins one layer of three, where
perplexity has KIVI ahead on every model. On keys alone KIVI-4 is ahead on all three
layers: 0.024 / 0.074 / 0.056 against zquant's 0.052 / 0.108 / 0.084.

## Using it

Two things differ from the retrieval defaults, and both matter a great deal. **Turn
calibration off**: these tensors have effective rank 6.4 of 64, and calibration costs
accuracy on low-rank data. **Reconstruct rather than estimate**: the index's per-vector
correction is fitted to preserve *ranking*, and attention needs accurate absolute scores.
Using the retrieval defaults instead measures 0.40 at b=5 on layer 15, worse than int4.

```python
codec = zquant.Codec(dim=64, bits=4)
codes, norms = codec.encode(keys)      # keep both; codes are bit-packed
keys_back = codec.decode(codes, norms)
```

Note that `Codec` takes the codebook width directly, while `Index` reserves a bit for its
residual sketch. So `Codec(bits=4)` and `Index(bits=5)` pack the same number of bits per
coordinate.

Caveats:

- **KIVI is shown without its fp16 window.** Real KIVI keeps recent tokens in fp16 until
  a group fills. Every row here quantizes every token at the bytes shown, which is the
  less favourable setup for KIVI.
- **KIVI's key groups see a little of the future.** Each window is one forward pass, so a
  key's group statistics include up to 63 later keys. Real KIVI holds those tokens in fp16
  instead. This favours KIVI slightly, against the window it gives up. The mean-centred
  rows see the whole window's mean in the same way.
- **`kv_bench` pairs pre-RoPE queries with post-RoPE keys.** `dump_kv.py` hooks `q_proj`,
  which is before RoPE, so its scores are not the model's attention logits. That is one
  more reason the perplexity table is the one to trust.
- **Small models, one dataset.** 135M to 494M parameters, wikitext-2 only, context 1024,
  no downstream tasks. Larger models are known to have more extreme outlier channels,
  which is the regime where zquant did worst.
