//! One CPU level's scan kernels, compiled on their own.
//!
//! build.zig compiles this file once per CPU level above the library's own (on x86_64,
//! AVX2 for a library built for x86_64_v2) and links each object into the library. It
//! exports the level's `tiles.Kernels` as `zq_kernels_<level>`, and `simd/dispatch.zig`
//! chooses among them at run time. Nothing in this object runs unless the CPU has every
//! feature it was compiled with.

const tiles = @import("simd/tiles.zig");
const options = @import("kernel_options");

const instance: tiles.Kernels = tiles.kernels;

comptime {
    // Hidden: resolved when the library is linked, never part of its interface.
    @export(&instance, .{ .name = "zq_kernels_" ++ options.name, .visibility = .hidden });
}
