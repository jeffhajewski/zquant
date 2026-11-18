const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zquant = b.addModule("zquant", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    // C ABI: a static and a shared library, plus the header. This is what the Python,
    // JavaScript and Go clients link against; see include/zquant.h.
    // The shipped library defaults to ReleaseFast, not to the build's default of Debug.
    // A Debug build of this library is ~500x slower - measured, via the Python client
    // reporting 51 QPS where the Zig benchmark reports 27,000 on the same corpus - and
    // nothing about a language binding makes that visible to the person hitting it.
    const lib_optimize = b.option(
        std.builtin.OptimizeMode,
        "lib-opt",
        "Optimization mode for the C ABI library (default ReleaseFast)",
    ) orelse .ReleaseFast;
    const c_api = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = lib_optimize,
    });
    c_api.link_libc = true;

    // The runtime CPU check (src/cpu_probe.zig), compiled for the architecture
    // *baseline* so that it can run on any CPU of the architecture — including one that
    // lacks the instructions the rest of the library was compiled with, which is the
    // CPU it exists to detect. It is told which features those are.
    var probe_query = target.query;
    probe_query.cpu_model = .baseline;
    probe_query.cpu_features_add = .empty;
    probe_query.cpu_features_sub = .empty;
    const cpu_options = b.addOptions();
    cpu_options.addOption([]const u8, "required", requiredCpuFeatures(b, target));
    cpu_options.addOption([]const u8, "model", target.result.cpu.model.name);
    const probe_mod = b.createModule(.{
        .root_source_file = b.path("src/cpu_probe.zig"),
        .target = b.resolveTargetQuery(probe_query),
        .optimize = lib_optimize,
        .link_libc = true,
        .pic = true,
    });
    probe_mod.addOptions("cpu_options", cpu_options);
    c_api.addObject(b.addObject(.{ .name = "zquant_cpu", .root_module = probe_mod }));

    const lib_static = b.addLibrary(.{ .name = "zquant", .root_module = c_api, .linkage = .static });
    const lib_shared = b.addLibrary(.{ .name = "zquant", .root_module = c_api, .linkage = .dynamic });
    lib_static.installHeader(b.path("include/zquant.h"), "zquant.h");
    b.installArtifact(lib_static);
    b.installArtifact(lib_shared);

    // A pkg-config file, so an installed library can be consumed without every
    // downstream build hardcoding paths. The Go client's default build uses it.
    const pc = b.addWriteFiles();
    const pc_file = pc.add("zquant.pc", b.fmt(
        \\prefix={s}
        \\libdir=${{prefix}}/lib
        \\includedir=${{prefix}}/include
        \\
        \\Name: zquant
        \\Description: TurboQuant vector quantization
        \\Version: 0.1.0
        \\Libs: -L${{libdir}} -lzquant
        \\Cflags: -I${{includedir}}
        \\
    , .{b.install_prefix}));
    b.getInstallStep().dependOn(&b.addInstallFileWithDir(pc_file, .{ .custom = "lib/pkgconfig" }, "zquant.pc").step);

    const lib_step = b.step("lib", "Build the C ABI static and shared libraries");
    lib_step.dependOn(&b.addInstallArtifact(lib_static, .{}).step);
    lib_step.dependOn(&b.addInstallArtifact(lib_shared, .{}).step);

    const unit_tests = b.addTest(.{
        .name = "zquant-test",
        .root_module = zquant,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // The C ABI's own smoke test, compiled as C and linked against the library, so the
    // header and the exported symbols are checked the way a binding author meets them
    // rather than through Zig's type system.
    const c_smoke_mod = b.createModule(.{ .target = target, .optimize = optimize });
    c_smoke_mod.addCSourceFile(.{ .file = b.path("tests/c/smoke.c"), .flags = &.{"-std=c11"} });
    c_smoke_mod.addIncludePath(b.path("include"));
    c_smoke_mod.linkLibrary(lib_static);
    c_smoke_mod.link_libc = true;
    const c_smoke = b.addExecutable(.{ .name = "c-smoke", .root_module = c_smoke_mod });
    test_step.dependOn(&b.addRunArtifact(c_smoke).step);

    // Integration tests live outside the module so they exercise the public API
    // across seams, rather than reaching into internals.
    for ([_][]const u8{ "pipeline", "invariants", "golden" }) |name| {
        const t = b.addTest(.{
            .name = b.fmt("{s}-test", .{name}),
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("tests/{s}.zig", .{name})),
                .target = target,
                .optimize = optimize,
                .imports = &.{.{ .name = "zquant", .module = zquant }},
            }),
        });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }

    // Benches default to ReleaseFast; `-Dbench-opt=Debug` turns on safety checks,
    // which is how a crash in a bench gets a usable panic message.
    //
    // `-Doptimize` does *not* affect benchmarks. The imported `zquant` module follows
    // the bench executable's root module, so this is the only knob that matters here,
    // in either direction: `-Doptimize=Debug` still yields an optimized benchmark.
    // Verified rather than assumed — `-Dbench-opt=Debug` measures 97 QPS on quickbench
    // against 10,450, so a benchmark that had silently fallen back to Debug could not
    // be mistaken for one that had not.
    const bench_optimize = b.option(
        std.builtin.OptimizeMode,
        "bench-opt",
        "Optimization mode for benchmarks (default ReleaseFast)",
    ) orelse .ReleaseFast;

    const bench_step = b.step("bench", "Run all benchmarks");
    for ([_][]const u8{ "encode_bench", "scan_bench", "recall_bench", "index_bench", "sift_bench", "sift_verify", "compare_bench", "diagnose", "kernel_bench", "multiq_bench", "ablate_bench", "lowbit_bench", "quickbench", "kv_bench" }) |name| {
        const exe = b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("bench/{s}.zig", .{name})),
                .target = target,
                .optimize = bench_optimize,
                .imports = &.{.{ .name = "zquant", .module = zquant }},
                // libc for the monotonic clock; see bench/timer.zig.
                .link_libc = true,
            }),
        });
        const run = b.addRunArtifact(exe);
        bench_step.dependOn(&run.step);
        // Each bench is also its own step, so one can be run alone.
        b.step(name, b.fmt("Run {s}", .{name})).dependOn(&run.step);
    }
}

/// Instruction-set features the target enables beyond its architecture's baseline, as
/// a space-separated list for src/cpu_probe.zig to verify at load time.
///
/// Only features the probe knows how to detect are listed. The rest of what a CPU model
/// enables is tuning (instruction scheduling, fusion), which changes no instruction the
/// CPU must execute. A native build lists nothing: it runs where it was built.
fn requiredCpuFeatures(b: *std.Build, target: std.Build.ResolvedTarget) []const u8 {
    if (target.query.isNativeCpu()) return "";
    const t = target.result;
    const baseline = std.Target.Cpu.baseline(t.cpu.arch, t.os);
    var out: std.ArrayList(u8) = .empty;
    switch (t.cpu.arch) {
        .x86_64 => {
            const F = std.Target.x86.Feature;
            const known = [_]F{
                .sse3,     .ssse3,    .fma,      .cx16,     .sse4_1,     .sse4_2,
                .crc32,    .movbe,    .popcnt,   .xsave,    .avx,        .f16c,
                .sahf,     .lzcnt,    .bmi,      .avx2,     .bmi2,       .avx512f,
                .avx512dq, .avx512cd, .avx512bw, .avx512vl, .avx512vnni, .avxvnni,
            };
            for (known) |f| {
                if (std.Target.x86.featureSetHas(t.cpu.features, f) and
                    !std.Target.x86.featureSetHas(baseline.features, f))
                {
                    if (out.items.len > 0) out.append(b.allocator, ' ') catch @panic("OOM");
                    out.appendSlice(b.allocator, @tagName(f)) catch @panic("OOM");
                }
            }
        },
        .aarch64 => {
            if (std.Target.aarch64.featureSetHas(t.cpu.features, .dotprod) and
                !std.Target.aarch64.featureSetHas(baseline.features, .dotprod))
            {
                out.appendSlice(b.allocator, "dotprod") catch @panic("OOM");
            }
        },
        else => {},
    }
    return out.items;
}
