const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const zquant = b.addModule("zquant", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    const unit_tests = b.addTest(.{
        .name = "zquant-test",
        .root_module = zquant,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Integration tests live outside the module so they exercise the public API
    // across seams, rather than reaching into internals.
    for ([_][]const u8{ "pipeline", "invariants" }) |name| {
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

    const bench_step = b.step("bench", "Run benchmarks");
    for ([_][]const u8{ "encode_bench", "scan_bench", "recall_bench" }) |name| {
        const exe = b.addExecutable(.{
            .name = name,
            .root_module = b.createModule(.{
                .root_source_file = b.path(b.fmt("bench/{s}.zig", .{name})),
                .target = target,
                .optimize = .ReleaseFast,
                .imports = &.{.{ .name = "zquant", .module = zquant }},
            }),
        });
        bench_step.dependOn(&b.addRunArtifact(exe).step);
    }
}
