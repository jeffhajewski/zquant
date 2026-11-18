//! Runtime check that this CPU can run the rest of the library.
//!
//! The kernels pick their instructions at compile time (`simd/shuffle.zig`,
//! `simd/dot.zig`), so a release built for a CPU with SSSE3 or SDOT uses them
//! unconditionally, and on a CPU without them the first search dies of an illegal
//! instruction. Building releases for the bare architecture baseline avoids that, but
//! costs 6-10x in throughput (measured on aarch64 without SDOT), which is the trade the
//! releases used to make without saying so.
//!
//! So releases target a CPU with the fast instructions, and this file answers "can that
//! code run here?" It is compiled separately for the architecture *baseline* and linked
//! into the library, so it executes on any CPU of the architecture — including the ones
//! it exists to turn away, which the rest of the library could not promise. Bindings
//! call `zq_cpu_supported` straight after loading the library, before anything else, and
//! fall back to a portable build or explain themselves rather than crash.
//!
//! `required` lists the instruction-set features the main library was compiled with
//! beyond the baseline, as computed by build.zig. Tuning flags and features this file
//! does not know how to detect are left out of it there.

const std = @import("std");
const builtin = @import("builtin");
const options = @import("cpu_options");

/// Features the library needs beyond the architecture baseline, space-separated, using
/// LLVM/Zig feature names. Empty for a native build, which by construction runs where
/// it was built.
const required: []const u8 = options.required;

/// The CPU model the library was compiled for, for diagnostics.
const model = std.fmt.comptimePrint("{s}", .{options.model});

export fn zq_cpu_supported() c_int {
    return @intFromBool(supported());
}

export fn zq_build_cpu() [*:0]const u8 {
    return model;
}

fn supported() bool {
    return @import("cpu_detect.zig").hasAll(required);
}
