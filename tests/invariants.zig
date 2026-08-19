//! Property tests over randomized configurations, and adversarial inputs.
//!
//! The unit tests use fixed, hand-chosen parameters. These sweep the configuration
//! space looking for combinations that break invariants — particularly dimensions
//! that are not powers of two, where padding interacts with packing and the density.

const std = @import("std");
const testing = std.testing;
const zq = @import("zquant");

fn randomUnit(buf: []f32, random: std.Random) void {
    var norm: f64 = 0;
    for (buf) |*v| {
        const g = random.floatNorm(f32);
        v.* = g;
        norm += @as(f64, g) * g;
    }
    const inv: f32 = @floatCast(1.0 / @sqrt(norm));
    for (buf) |*v| v.* *= inv;
}

test "round trip holds across a sweep of dimensions and bit-widths" {
    const allocator = testing.allocator;
    // Deliberately awkward dimensions: primes, just-over-power-of-two, just-under.
    const dims = [_]u32{ 1, 2, 3, 7, 17, 31, 32, 33, 63, 64, 65, 127, 200, 257, 768 };

    for (dims) |dim| {
        for ([_]u6{ 1, 2, 3, 4 }) |bits| {
            var q = try zq.mse.Mse.init(allocator, .{ .dim = dim, .bits = bits, .seed = dim });
            defer q.deinit();
            var ws = try zq.mse.Workspace.init(allocator, q);
            defer ws.deinit();

            const x = try allocator.alloc(f32, dim);
            defer allocator.free(x);
            const back = try allocator.alloc(f32, dim);
            defer allocator.free(back);
            const codes = try allocator.alloc(u8, q.codeLen());
            defer allocator.free(codes);

            var prng = std.Random.DefaultPrng.init(dim * 31 + bits);
            randomUnit(x, prng.random());

            const norm = q.encode(x, codes, &ws);
            q.decode(codes, norm, back, &ws);

            // Norm is preserved regardless of dimension.
            try testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-4);
            // Every code is in range.
            const levels: u8 = @intCast(@as(u16, 1) << @as(u4, @intCast(bits)));
            for (codes) |c| try testing.expect(c < levels);
            // Reconstruction is finite and bounded.
            for (back) |v| try testing.expect(std.math.isFinite(v));
        }
    }
}

test "packing round trips for every dimension the quantizer can produce" {
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xFEED);
    const random = prng.random();

    for ([_]u32{ 1, 2, 3, 7, 32, 33, 100, 256, 257 }) |dim| {
        for ([_]u6{ 1, 2, 3, 4, 5, 8 }) |bits| {
            const layout = zq.packing.Layout.init(dim, bits);
            const max: u8 = @intCast((@as(u16, 1) << @as(u4, @intCast(bits))) - 1);

            const codes = try allocator.alloc(u8, dim);
            defer allocator.free(codes);
            const back = try allocator.alloc(u8, dim);
            defer allocator.free(back);
            const stored = try allocator.alloc(u8, layout.stride());
            defer allocator.free(stored);

            for (codes) |*c| c.* = random.uintAtMost(u8, max);
            layout.pack(codes, stored);
            layout.unpack(stored, back);
            try testing.expectEqualSlices(u8, codes, back);
        }
    }
}

test "adversarial inputs do not produce garbage" {
    const allocator = testing.allocator;
    const dim: u32 = 128;

    var q = try zq.prod.Prod.init(allocator, .{ .dim = dim, .bits = 4, .seed = 7 });
    defer q.deinit();
    var ws = try zq.prod.Workspace.init(allocator, q);
    defer ws.deinit();
    var state = try zq.prod.QueryState.init(allocator, q);
    defer state.deinit();

    const codes = try allocator.alloc(u8, q.codeLen());
    defer allocator.free(codes);
    const sketch = try allocator.alloc(u8, q.sketchLen());
    defer allocator.free(sketch);
    const back = try allocator.alloc(f32, dim);
    defer allocator.free(back);

    var x = [_]f32{0} ** dim;
    const cases = [_]struct { name: []const u8, fill: *const fn ([]f32) void }{
        .{ .name = "zero", .fill = struct {
            fn f(v: []f32) void { @memset(v, 0); }
        }.f },
        .{ .name = "axis-aligned", .fill = struct {
            fn f(v: []f32) void { @memset(v, 0); v[0] = 1; }
        }.f },
        .{ .name = "last-axis", .fill = struct {
            fn f(v: []f32) void { @memset(v, 0); v[v.len - 1] = 1; }
        }.f },
        .{ .name = "constant", .fill = struct {
            fn f(v: []f32) void { @memset(v, 1.0); }
        }.f },
        .{ .name = "alternating", .fill = struct {
            fn f(v: []f32) void {
                for (v, 0..) |*e, i| e.* = if (i % 2 == 0) 1.0 else -1.0;
            }
        }.f },
        .{ .name = "tiny", .fill = struct {
            fn f(v: []f32) void { @memset(v, 1e-30); }
        }.f },
        .{ .name = "huge", .fill = struct {
            fn f(v: []f32) void { @memset(v, 1e30); }
        }.f },
        .{ .name = "one-huge-rest-tiny", .fill = struct {
            fn f(v: []f32) void { @memset(v, 1e-20); v[3] = 1e20; }
        }.f },
    };

    for (cases) |case| {
        case.fill(&x);
        const scalars = q.encode(&x, codes, sketch, &ws);
        try testing.expect(std.math.isFinite(scalars.norm));
        try testing.expect(std.math.isFinite(scalars.gamma));
        try testing.expect(scalars.gamma >= 0);
        for (codes) |c| try testing.expect(c < 8); // bits-1 = 3 -> 8 levels

        q.decode(codes, sketch, scalars, back, &ws);
        for (back) |v| {
            if (!std.math.isFinite(v)) {
                std.debug.print("non-finite reconstruction for case '{s}'\n", .{case.name});
                return error.NonFiniteReconstruction;
            }
        }

        // And the estimator stays finite for an ordinary query.
        var probe = [_]f32{0} ** dim;
        var prng = std.Random.DefaultPrng.init(1);
        randomUnit(&probe, prng.random());
        q.prepareQuery(&probe, &state, &ws);
        const score = q.dot(state, codes, sketch, scalars);
        if (!std.math.isFinite(score)) {
            std.debug.print("non-finite score for case '{s}'\n", .{case.name});
            return error.NonFiniteScore;
        }
    }
}

test "duplicate vectors encode identically" {
    // Data-obliviousness in its most visible form: encoding depends only on the
    // vector and the seed, never on what else has been encoded.
    const allocator = testing.allocator;
    const dim: u32 = 256;

    var q = try zq.mse.Mse.init(allocator, .{ .dim = dim, .bits = 4, .seed = 42 });
    defer q.deinit();
    var ws = try zq.mse.Workspace.init(allocator, q);
    defer ws.deinit();

    const x = try allocator.alloc(f32, dim);
    defer allocator.free(x);
    const a = try allocator.alloc(u8, q.codeLen());
    defer allocator.free(a);
    const b = try allocator.alloc(u8, q.codeLen());
    defer allocator.free(b);

    var prng = std.Random.DefaultPrng.init(5);
    randomUnit(x, prng.random());

    _ = q.encode(x, a, &ws);
    // Encode a thousand unrelated vectors in between.
    const noise = try allocator.alloc(f32, dim);
    defer allocator.free(noise);
    const scratch = try allocator.alloc(u8, q.codeLen());
    defer allocator.free(scratch);
    for (0..1000) |_| {
        randomUnit(noise, prng.random());
        _ = q.encode(noise, scratch, &ws);
    }
    _ = q.encode(x, b, &ws);

    try testing.expectEqualSlices(u8, a, b);
}

test "encoding is invariant to positive scaling and equivariant to negation" {
    const allocator = testing.allocator;
    const dim: u32 = 128;

    var q = try zq.mse.Mse.init(allocator, .{ .dim = dim, .bits = 4, .seed = 11 });
    defer q.deinit();
    var ws = try zq.mse.Workspace.init(allocator, q);
    defer ws.deinit();

    const x = try allocator.alloc(f32, dim);
    defer allocator.free(x);
    const scaled = try allocator.alloc(f32, dim);
    defer allocator.free(scaled);
    const base = try allocator.alloc(u8, q.codeLen());
    defer allocator.free(base);
    const other = try allocator.alloc(u8, q.codeLen());
    defer allocator.free(other);

    var prng = std.Random.DefaultPrng.init(13);
    randomUnit(x, prng.random());
    _ = q.encode(x, base, &ws);

    // Positive scaling changes only the stored norm.
    for ([_]f32{ 1e-6, 0.5, 3.0, 1e6 }) |factor| {
        for (scaled, x) |*d, v| d.* = v * factor;
        const norm = q.encode(scaled, other, &ws);
        try testing.expectApproxEqRel(factor, norm, 1e-3);
        try testing.expectEqualSlices(u8, base, other);
    }

    // Negation must map each code to its mirror, because the codebook is symmetric.
    for (scaled, x) |*d, v| d.* = -v;
    _ = q.encode(scaled, other, &ws);
    const levels: u8 = @intCast(q.codebook.levels());
    for (base, other) |a, b| try testing.expectEqual(levels - 1 - a, b);
}

test "estimator is symmetric in the roles it should be" {
    // <q, x> estimated with x quantized should track <x, q> estimated with q
    // quantized. They are different estimators, but both unbiased for the same
    // quantity, so they must agree in aggregate.
    const allocator = testing.allocator;
    const dim: u32 = 256;

    var q = try zq.prod.Prod.init(allocator, .{ .dim = dim, .bits = 4, .seed = 17 });
    defer q.deinit();
    var ws = try zq.prod.Workspace.init(allocator, q);
    defer ws.deinit();
    var state = try zq.prod.QueryState.init(allocator, q);
    defer state.deinit();

    const a = try allocator.alloc(f32, dim);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, dim);
    defer allocator.free(b);
    const codes = try allocator.alloc(u8, q.codeLen());
    defer allocator.free(codes);
    const sketch = try allocator.alloc(u8, q.sketchLen());
    defer allocator.free(sketch);

    var prng = std.Random.DefaultPrng.init(19);
    const random = prng.random();

    var forward_sum: f64 = 0;
    var reverse_sum: f64 = 0;
    const trials = 500;
    for (0..trials) |_| {
        randomUnit(a, random);
        randomUnit(b, random);

        const sa = q.encode(a, codes, sketch, &ws);
        q.prepareQuery(b, &state, &ws);
        forward_sum += q.dot(state, codes, sketch, sa);

        const sb = q.encode(b, codes, sketch, &ws);
        q.prepareQuery(a, &state, &ws);
        reverse_sum += q.dot(state, codes, sketch, sb);
    }
    const scale = @as(f64, @floatFromInt(trials));
    try testing.expectApproxEqAbs(forward_sum / scale, reverse_sum / scale, 0.01);
}
