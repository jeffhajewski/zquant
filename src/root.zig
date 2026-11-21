//! zquant — TurboQuant vector quantization.
//!
//! See docs/DESIGN.md. Reference: Zandieh, Daliri, Hadian, Mirrokni,
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
//! arXiv:2504.19874.

pub const rng = @import("math/rng.zig");
pub const quadrature = @import("math/quadrature.zig");
pub const density = @import("math/density.zig");
pub const lloyd_max = @import("math/lloyd_max.zig");
pub const rotation = @import("math/rotation.zig");
pub const libm = @import("math/libm.zig");

pub const codebook = @import("quant/codebook.zig");
pub const mse = @import("quant/mse.zig");
pub const prod = @import("quant/prod.zig");
pub const packing = @import("quant/packing.zig");

pub const simd_encode = @import("simd/encode.zig");
pub const simd_scan = @import("simd/scan.zig");
pub const simd_bitmask = @import("simd/bitmask.zig");
pub const simd_shuffle = @import("simd/shuffle.zig");
pub const simd_dot = @import("simd/dot.zig");
pub const simd_tiles = @import("simd/tiles.zig");
pub const simd_dispatch = @import("simd/dispatch.zig");
pub const simd_sketch = @import("simd/sketch.zig");
pub const simd_columns = @import("simd/columns.zig");

pub const topk = @import("index/topk.zig");
pub const flat = @import("index/flat.zig");
pub const ivf = @import("index/ivf.zig");
pub const persist = @import("index/persist.zig");
pub const codes = @import("index/codes.zig");
pub const rerank = @import("index/rerank.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
