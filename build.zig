const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zquant = b.addModule("zquant", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    addKernelLevels(b, zquant, target, optimize, false);

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
    // Releases strip debug info. On Linux it is about 90% of the library (the x86_64
    // release build is 1.96 MB with it, 183 KB without), and every Python wheel and npm
    // install would carry it. macOS keeps it out of the dylib anyway.
    const lib_strip = b.option(bool, "strip", "Strip debug info from the C ABI library") orelse false;
    const c_api = b.createModule(.{
        .root_source_file = b.path("src/c_api.zig"),
        .target = target,
        .optimize = lib_optimize,
        .strip = lib_strip,
    });
    c_api.link_libc = true;
    addKernelLevels(b, c_api, target, lib_optimize, lib_strip);

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

    // Releases build the portable variant as `zquant-portable`, so it can be installed
    // next to the fast one and the bindings can fall back to it by name when
    // zq_cpu_supported() says no. A distinct name rather than a renamed file, so the
    // ELF SONAME and Mach-O install name agree with the file.
    const lib_name = b.option(
        []const u8,
        "lib-name",
        "Base name of the C ABI library (default zquant)",
    ) orelse "zquant";
    const is_windows = target.result.os.tag == .windows;
    // On Windows the DLL's import library is `<name>.lib`, the same file name a static
    // library gets, and whichever was installed second silently replaced the other.
    const static_name = if (is_windows) b.fmt("{s}_static", .{lib_name}) else lib_name;
    const lib_static = b.addLibrary(.{ .name = static_name, .root_module = c_api, .linkage = .static });
    const lib_shared = b.addLibrary(.{ .name = lib_name, .root_module = c_api, .linkage = .dynamic });
    lib_static.installHeader(b.path("include/zquant.h"), "zquant.h");
    b.installArtifact(lib_static);
    b.installArtifact(lib_shared);

    // A pkg-config file, so an installed library can be consumed without every
    // downstream build hardcoding paths. The Go client's default build uses it.
    //
    // Relocatable: `${pcfiledir}` is wherever the file is found, so an unpacked release
    // tarball works from any directory. It used to bake in the absolute install prefix,
    // which was only right on the machine that built it. The rpath lets a binary find
    // the shared library at run time without LD_LIBRARY_PATH; on macOS the library's
    // install name is @rpath-relative, so without it nothing linked against it runs.
    const rpath = if (is_windows) "" else " -Wl,-rpath,${libdir}";
    const pc = b.addWriteFiles();
    const pc_file = pc.add("zquant.pc", b.fmt(
        \\prefix=${{pcfiledir}}/../..
        \\libdir=${{prefix}}/lib
        \\includedir=${{prefix}}/include
        \\
        \\Name: zquant
        \\Description: TurboQuant vector quantization
        \\Version: 0.1.0
        \\Libs: -L${{libdir}}{s} -l{s}
        \\Cflags: -I${{includedir}}
        \\
    , .{ rpath, lib_name }));
    const install_pc = b.addInstallFileWithDir(pc_file, .{ .custom = "lib/pkgconfig" }, b.fmt("{s}.pc", .{lib_name}));
    b.getInstallStep().dependOn(&install_pc.step);

    const lib_step = b.step("lib", "Build the C ABI static and shared libraries");
    lib_step.dependOn(&b.addInstallArtifact(lib_static, .{}).step);
    lib_step.dependOn(&b.addInstallArtifact(lib_shared, .{}).step);
    lib_step.dependOn(&install_pc.step);

    // `zig build test -Dtest-filter=name` runs only the unit tests whose names contain
    // `name`; give it more than once for several.
    const test_filters = b.option([]const []const u8, "test-filter", "Run only unit tests whose names contain this") orelse &.{};
    const unit_tests = b.addTest(.{
        .name = "zquant-test",
        .root_module = zquant,
        .filters = test_filters,
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

    // The benches' own copy of the module, so the kernel levels linked into them are
    // built at `bench_optimize` too: a ReleaseFast bench calling Debug kernels through
    // the dispatch would measure the wrong thing.
    const bench_zquant = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = bench_optimize,
    });
    addKernelLevels(b, bench_zquant, target, bench_optimize, false);

    const bench_step = b.step("bench", "Run all benchmarks");
    for ([_][]const u8{ "encode_bench", "scan_bench", "recall_bench", "index_bench", "sift_bench", "sift_verify", "compare_bench", "diagnose", "kernel_bench", "multiq_bench", "ablate_bench", "lowbit_bench", "quickbench", "kv_bench", "tile_bench", "fastscan_bench", "coarse_bench" }) |name| {
        const exe = b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("bench/{s}.zig", .{name})),
                .target = target,
                .optimize = bench_optimize,
                .imports = &.{.{ .name = "zquant", .module = bench_zquant }},
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

/// CPU levels above the target's whose scan kernels are worth having: their own copy,
/// compiled for that level, which src/simd/dispatch.zig runs when the CPU has it.
///
/// One today: x86_64 builds with SSSE3 but not AVX2, which is what releases target
/// (x86_64_v2), also get the AVX2 kernels. Measured on a Zen 3 runner they are
/// 1.3-1.8x the 128-bit ones, and search through the library is 1.72x at bits=5. A
/// build without SSSE3 is the portable one, used only on CPUs that could not run
/// these either, and a build with AVX2 compiles the wide kernels in already.
fn kernelLevelCpus(target: std.Build.ResolvedTarget) []const *const std.Target.Cpu.Model {
    const t = target.result;
    if (t.cpu.arch != .x86_64) return &.{};
    const has = std.Target.x86.featureSetHas;
    if (!has(t.cpu.features, .ssse3) or has(t.cpu.features, .avx2)) return &.{};
    return &.{&std.Target.x86.cpu.x86_64_v3};
}

/// Compiles src/kernel_entry.zig for each of `kernelLevelCpus` and links it into
/// `module`, and tells src/simd/dispatch.zig what it has. Every module that compiles
/// the index needs this, since the dispatch refers to the levels' symbols.
fn addKernelLevels(
    b: *std.Build,
    module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    strip: bool,
) void {
    var spec: std.ArrayList(u8) = .empty;
    for (kernelLevelCpus(target)) |cpu| {
        var query = target.query;
        query.cpu_model = .{ .explicit = cpu };
        query.cpu_features_add = .empty;
        query.cpu_features_sub = .empty;
        const level_target = b.resolveTargetQuery(query);

        const options = b.addOptions();
        options.addOption([]const u8, "name", cpu.name);
        const entry = b.createModule(.{
            .root_source_file = b.path("src/kernel_entry.zig"),
            .target = level_target,
            .optimize = optimize,
            .strip = strip,
            .pic = true,
        });
        entry.addOptions("kernel_options", options);
        module.addObject(b.addObject(.{
            .name = b.fmt("zquant_kernels_{s}", .{cpu.name}),
            .root_module = entry,
        }));

        if (spec.items.len > 0) spec.append(b.allocator, ';') catch @panic("OOM");
        spec.print(b.allocator, "{s}:{s}", .{ cpu.name, requiredCpuFeatures(b, level_target) }) catch @panic("OOM");
    }
    const options = b.addOptions();
    options.addOption([]const u8, "levels", spec.items);
    module.addOptions("dispatch_options", options);
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
