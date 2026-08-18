//! zquant — TurboQuant vector quantization.
//!
//! See docs/DESIGN.md. Reference: Zandieh, Daliri, Hadian, Mirrokni,
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate",
//! arXiv:2504.19874.

pub const rng = @import("math/rng.zig");

test {
    @import("std").testing.refAllDecls(@This());
}
