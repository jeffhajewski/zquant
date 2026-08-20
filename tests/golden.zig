//! Golden vectors: fixed inputs whose encodings are pinned by hash.
//!
//! Two jobs. First, cross-optimization determinism — Debug and ReleaseFast must
//! produce byte-identical codes, or float reassociation is creeping in somewhere.
//! Run with `-Doptimize=ReleaseFast` as well as the default.
//!
//! Second, and the reason the hashes are checked in rather than computed: these are
//! the conformance vectors the Python, JS, and Go bindings must reproduce (design
//! §7.4). A binding that drifts produces silently different codes, which no
//! per-language test would catch.
//!
//! Inputs are derived from the counter-based RNG rather than stored, so the fixture
//! is a seed rather than a data file — but the *expected output* is a literal, so a
//! change to the encoder cannot quietly update its own expectation.

const std = @import("std");
const testing = std.testing;
const zq = @import("zquant");

/// Deterministic input: integer-derived, so no float reassociation can perturb it
/// before the encoder even sees it.
fn fillInput(buf: []f32, seed: u64) void {
    var stream = zq.rng.Philox.init(seed).stream(.testing);
    for (buf) |*v| {
        // Map 24 random bits into [-1, 1) exactly, avoiding transcendental functions.
        const bits = stream.nextU32() >> 8;
        v.* = @as(f32, @floatFromInt(bits)) * 0x1p-23 - 1.0;
    }
}

fn hash(bytes: []const u8) u64 {
    return std.hash.Wyhash.hash(0, bytes);
}

const Case = struct {
    dim: u32,
    bits: u6,
    seed: u64,
    mse_codes: u64,
    prod_codes: u64,
    prod_sketch: u64,
};

/// Regenerate with `print_golden = true` if the encoder intentionally changes — and
/// treat any unintended change here as a conformance break, not a test to update.
const print_golden = false;

const cases = [_]Case{
    .{ .dim = 64, .bits = 2, .seed = 0x0000, .mse_codes = 0x41FF3623EE0B2BF0, .prod_codes = 0x34459518EB6DD855, .prod_sketch = 0x9597AB384A5B9740 },
    .{ .dim = 128, .bits = 4, .seed = 0x5EED, .mse_codes = 0x633C2947182A864A, .prod_codes = 0xDE0792743302E32E, .prod_sketch = 0xDBB94FCF30B69464 },
    .{ .dim = 256, .bits = 4, .seed = 0xABCD, .mse_codes = 0x6B47F14A5C926498, .prod_codes = 0x6FC03CDB922E86E3, .prod_sketch = 0x457D58AA34212933 },
    .{ .dim = 768, .bits = 3, .seed = 0x1234, .mse_codes = 0xDEE536CD5D4F2EA9, .prod_codes = 0xC33AB1BE40F09B4A, .prod_sketch = 0xD80300BD4BC98CE2 },
    .{ .dim = 1024, .bits = 4, .seed = 0xFEED, .mse_codes = 0xA1F06845B9EA72EB, .prod_codes = 0x8F88C8B1FE0462C5, .prod_sketch = 0x81009704C69D33BA },
};

test "golden encodings are stable" {
    const allocator = testing.allocator;

    for (cases) |c| {
        const input = try allocator.alloc(f32, c.dim);
        defer allocator.free(input);
        fillInput(input, c.seed);

        var mse = try zq.mse.Mse.init(allocator, .{ .dim = c.dim, .bits = c.bits, .seed = c.seed });
        defer mse.deinit();
        var mse_ws = try zq.mse.Workspace.init(allocator, mse);
        defer mse_ws.deinit();
        const mse_codes = try allocator.alloc(u8, mse.codeLen());
        defer allocator.free(mse_codes);
        _ = mse.encode(input, mse_codes, &mse_ws);

        var prod = try zq.prod.Prod.init(allocator, .{ .dim = c.dim, .bits = c.bits, .seed = c.seed });
        defer prod.deinit();
        var prod_ws = try zq.prod.Workspace.init(allocator, prod);
        defer prod_ws.deinit();
        const prod_codes = try allocator.alloc(u8, prod.codeLen());
        defer allocator.free(prod_codes);
        const sketch = try allocator.alloc(u8, prod.sketchLen());
        defer allocator.free(sketch);
        _ = prod.encode(input, prod_codes, sketch, &prod_ws);

        const got_mse = hash(mse_codes);
        const got_prod = hash(prod_codes);
        const got_sketch = hash(sketch);

        if (print_golden) {
            std.debug.print(
                "    .{{ .dim = {d}, .bits = {d}, .seed = 0x{X:0>4}, .mse_codes = 0x{X}, .prod_codes = 0x{X}, .prod_sketch = 0x{X} }},\n",
                .{ c.dim, c.bits, c.seed, got_mse, got_prod, got_sketch },
            );
            continue;
        }

        try testing.expectEqual(c.mse_codes, got_mse);
        try testing.expectEqual(c.prod_codes, got_prod);
        try testing.expectEqual(c.prod_sketch, got_sketch);
    }
}

test "packing is stable too" {
    // Codes being stable is not enough: the bindings read packed storage, so the
    // byte layout is part of the conformance surface.
    const allocator = testing.allocator;
    const dim: u32 = 256;

    var mse = try zq.mse.Mse.init(allocator, .{ .dim = dim, .bits = 4, .seed = 0xABCD });
    defer mse.deinit();
    var ws = try zq.mse.Workspace.init(allocator, mse);
    defer ws.deinit();

    const input = try allocator.alloc(f32, dim);
    defer allocator.free(input);
    fillInput(input, 0xABCD);

    const codes = try allocator.alloc(u8, mse.codeLen());
    defer allocator.free(codes);
    _ = mse.encode(input, codes, &ws);

    const layout = zq.packing.Layout.init(mse.padded, 4);
    const stored = try allocator.alloc(u8, layout.stride());
    defer allocator.free(stored);
    layout.pack(codes, stored);

    if (print_golden) {
        std.debug.print("    packed hash = 0x{X}\n", .{hash(stored)});
        return;
    }
    try testing.expectEqual(@as(u64, 0x03317D8F36AA7D8F), hash(stored));
}
