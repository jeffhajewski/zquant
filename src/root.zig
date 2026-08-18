//! zquant — TurboQuant vector quantization.
//!
//! See docs/DESIGN.md. Reference: Zandieh, Daliri, Hadian, Mirrokni,
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
//! arXiv:2504.19874.

pub const rng = @import("math/rng.zig");
pub const quadrature = @import("math/quadrature.zig");
pub const density = @import("math/density.zig");
pub const lloyd_max = @import("math/lloyd_max.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
