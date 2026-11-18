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
    if (required.len == 0) return true;
    var it = std.mem.tokenizeScalar(u8, required, ' ');
    while (it.next()) |name| {
        if (!has(name)) return false;
    }
    return true;
}

fn has(name: []const u8) bool {
    return switch (builtin.cpu.arch) {
        .x86_64 => x86.has(name),
        .aarch64 => arm.has(name),
        // build.zig only lists features this file can detect, so nothing reaches here.
        else => true,
    };
}

const x86 = struct {
    const Regs = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

    fn cpuid(leaf: u32, sub: u32) Regs {
        var eax: u32 = undefined;
        var ebx: u32 = undefined;
        var ecx: u32 = undefined;
        var edx: u32 = undefined;
        asm volatile ("cpuid"
            : [eax] "={eax}" (eax),
              [ebx] "={ebx}" (ebx),
              [ecx] "={ecx}" (ecx),
              [edx] "={edx}" (edx),
            : [leaf] "{eax}" (leaf),
              [sub] "{ecx}" (sub),
        );
        return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
    }

    /// XCR0: which register state the OS saves on a context switch. AVX instructions
    /// fault unless the OS has enabled the YMM state, whatever CPUID says.
    fn xcr0() u32 {
        var lo: u32 = undefined;
        var hi: u32 = undefined;
        asm volatile ("xgetbv"
            : [lo] "={eax}" (lo),
              [hi] "={edx}" (hi),
            : [sel] "{ecx}" (@as(u32, 0)),
        );
        return lo;
    }

    fn bit(word: u32, n: u5) bool {
        return (word >> n) & 1 == 1;
    }

    fn has(name: []const u8) bool {
        const max = cpuid(0, 0).eax;
        const l1 = cpuid(1, 0);
        const l7 = if (max >= 7) cpuid(7, 0) else Regs{ .eax = 0, .ebx = 0, .ecx = 0, .edx = 0 };
        const l71 = if (max >= 7 and l7.eax >= 1) cpuid(7, 1) else Regs{ .eax = 0, .ebx = 0, .ecx = 0, .edx = 0 };
        const ext_max = cpuid(0x80000000, 0).eax;
        const e1 = if (ext_max >= 0x80000001) cpuid(0x80000001, 0) else Regs{ .eax = 0, .ebx = 0, .ecx = 0, .edx = 0 };

        const osxsave = bit(l1.ecx, 27);
        const xcr = if (osxsave) xcr0() else 0;
        const ymm_state = xcr & 0x6 == 0x6;
        const zmm_state = xcr & 0xE6 == 0xE6;
        const avx_ok = bit(l1.ecx, 28) and ymm_state;

        const eql = std.mem.eql;
        if (eql(u8, name, "sse3")) return bit(l1.ecx, 0);
        if (eql(u8, name, "ssse3")) return bit(l1.ecx, 9);
        if (eql(u8, name, "fma")) return bit(l1.ecx, 12) and avx_ok;
        if (eql(u8, name, "cx16")) return bit(l1.ecx, 13);
        if (eql(u8, name, "sse4_1")) return bit(l1.ecx, 19);
        if (eql(u8, name, "sse4_2") or eql(u8, name, "crc32")) return bit(l1.ecx, 20);
        if (eql(u8, name, "movbe")) return bit(l1.ecx, 22);
        if (eql(u8, name, "popcnt")) return bit(l1.ecx, 23);
        if (eql(u8, name, "xsave")) return bit(l1.ecx, 26);
        if (eql(u8, name, "avx")) return avx_ok;
        if (eql(u8, name, "f16c")) return bit(l1.ecx, 29) and avx_ok;
        if (eql(u8, name, "sahf")) return bit(e1.ecx, 0);
        if (eql(u8, name, "lzcnt")) return bit(e1.ecx, 5);
        if (eql(u8, name, "bmi")) return bit(l7.ebx, 3);
        if (eql(u8, name, "avx2")) return bit(l7.ebx, 5) and avx_ok;
        if (eql(u8, name, "bmi2")) return bit(l7.ebx, 8);
        if (eql(u8, name, "avx512f")) return bit(l7.ebx, 16) and zmm_state;
        if (eql(u8, name, "avx512dq")) return bit(l7.ebx, 17) and zmm_state;
        if (eql(u8, name, "avx512cd")) return bit(l7.ebx, 28) and zmm_state;
        if (eql(u8, name, "avx512bw")) return bit(l7.ebx, 30) and zmm_state;
        if (eql(u8, name, "avx512vl")) return bit(l7.ebx, 31) and zmm_state;
        if (eql(u8, name, "avx512vnni")) return bit(l7.ecx, 11) and zmm_state;
        if (eql(u8, name, "avxvnni")) return bit(l71.eax, 4) and avx_ok;
        // build.zig filters to the names above; an unknown one is a build mismatch, and
        // refusing is the side to err on.
        return false;
    }
};

const arm = struct {
    fn has(name: []const u8) bool {
        if (std.mem.eql(u8, name, "dotprod")) return dotprod();
        return false;
    }

    fn dotprod() bool {
        switch (builtin.os.tag) {
            .linux => {
                // AT_HWCAP = 16, HWCAP_ASIMDDP = 1 << 20.
                //
                // libc's getauxval, not std.os.linux's. Zig's reads a copy of the auxiliary
                // vector that only Zig's own `_start` fills in. In a library loaded by a C,
                // Python, Node or Go program that code never runs, so it returned 0 and
                // every arm64 Linux CPU was refused. The Zig tests could not see it, since
                // they run under Zig's `_start`; the C smoke test on an arm64 runner did.
                const hwcap = if (builtin.link_libc) std.c.getauxval(16) else std.os.linux.getauxval(16);
                return hwcap & (1 << 20) != 0;
            },
            // Every Apple arm64 CPU has had dot product since the A11; the macOS
            // baseline model already assumes it.
            .macos, .ios => return true,
            .windows => {
                // PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE = 43.
                return IsProcessorFeaturePresent(43) != 0;
            },
            else => return false,
        }
    }

    extern "kernel32" fn IsProcessorFeaturePresent(feature: u32) callconv(.winapi) c_int;
};
