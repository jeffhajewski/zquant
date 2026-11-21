//! `exp`, `log` and `lgamma` in f64, written out in Zig, so a codebook comes out the
//! same bits on every platform and in every build.
//!
//! `@exp` and `@log` compile to calls to `exp` and `log`. Without libc, compiler-rt
//! provides those. The C library links libc, and an optimized build of it called the
//! platform's libm instead, whose last bits differ: one level of the bits=5 codebook at
//! d=128 came out one bit apart between the macOS library and the test suite, enough to
//! change a stored f16. The codebook is solved at run time from these functions, so
//! whatever they do differently, every index built there does too.
//!
//! These are compiler-rt's own `exp` and `log`, and `lgamma` is the standard library's
//! with its calls routed here, so the results are what builds without libc have always
//! computed, and the encoding is unchanged. Nothing here may use a builtin that lowers
//! to a libm call. `@sqrt` is exact everywhere; `@exp`, `@log`, `@sin` and `@cos` are
//! not.
//!
//! Ported from Zig's compiler-rt and standard library (MIT,
//! https://github.com/ziglang/zig/blob/master/LICENSE), which port musl (MIT,
//! https://git.musl-libc.org/cgit/musl/tree/COPYRIGHT) and Go (BSD-3,
//! https://golang.org/LICENSE):
//!
//! https://git.musl-libc.org/cgit/musl/tree/src/math/exp.c
//! https://git.musl-libc.org/cgit/musl/tree/src/math/log.c?h=1b76ff0767d01df72f692806ee5adee13c67ef88
//! https://git.musl-libc.org/cgit/musl/tree/src/math/tgamma.c
//! https://golang.org/src/math/pow.go

const std = @import("std");

pub fn exp(x_: f64) f64 {
    const half = [_]f64{ 0.5, -0.5 };
    const ln2hi: f64 = 6.93147180369123816490e-01;
    const ln2lo: f64 = 1.90821492927058770002e-10;
    const invln2: f64 = 1.44269504088896338700e+00;
    const P1: f64 = 1.66666666666666019037e-01;
    const P2: f64 = -2.77777777770155933842e-03;
    const P3: f64 = 6.61375632143793436117e-05;
    const P4: f64 = -1.65339022054652515390e-06;
    const P5: f64 = 4.13813679705723846039e-08;

    var x = x_;
    const ux: u64 = @bitCast(x);
    var hx = ux >> 32;
    const sign: i32 = @intCast(hx >> 31);
    hx &= 0x7FFFFFFF;

    if (std.math.isNan(x)) {
        return x;
    }

    // |x| >= 708.39 or nan
    if (hx >= 0x4086232B) {
        // nan
        if (hx > 0x7FF00000) {
            return x;
        }
        if (x > 709.782712893383973096) {
            // overflow if x != inf
            return std.math.inf(f64);
        }
        if (x < -708.39641853226410622) {
            // underflow if x != -inf
            if (x < -745.13321910194110842) {
                return 0;
            }
        }
    }

    // argument reduction
    var k: i32 = undefined;
    var hi: f64 = undefined;
    var lo: f64 = undefined;

    // |x| > 0.5 * ln2
    if (hx > 0x3FD62E42) {
        // |x| >= 1.5 * ln2
        if (hx > 0x3FF0A2B2) {
            k = @intFromFloat(invln2 * x + half[@intCast(sign)]);
        } else {
            k = 1 - sign - sign;
        }

        const dk: f64 = @floatFromInt(k);
        hi = x - dk * ln2hi;
        lo = dk * ln2lo;
        x = hi - lo;
    }
    // |x| > 2^(-28)
    else if (hx > 0x3E300000) {
        k = 0;
        hi = x;
        lo = 0;
    } else {
        // inexact if x != 0
        return 1 + x;
    }

    const xx = x * x;
    const c = x - xx * (P1 + xx * (P2 + xx * (P3 + xx * (P4 + xx * P5))));
    const y = 1 + (x * c / (2 - c) - lo + hi);

    if (k == 0) {
        return y;
    } else {
        return std.math.scalbn(y, k);
    }
}

pub fn log(x: f64) f64 {
    const poly1 = [_]f64{
        -0x1p-1,
        0x1.5555555555577p-2,
        -0x1.ffffffffffdcbp-3,
        0x1.999999995dd0cp-3,
        -0x1.55555556745a7p-3,
        0x1.24924a344de3p-3,
        -0x1.fffffa4423d65p-4,
        0x1.c7184282ad6cap-4,
        -0x1.999eb43b068ffp-4,
        0x1.78182f7afd085p-4,
        -0x1.5521375d145cdp-4,
    };

    const poly = [_]f64{
        -0x1.0000000000001p-1,
        0x1.555555551305bp-2,
        -0x1.fffffffeb459p-3,
        0x1.999b324f10111p-3,
        -0x1.55575e506c89fp-3,
    };

    const tab = [128]struct { invc: f64, logc: f64 }{
        .{ .invc = 0x1.734f0c3e0de9fp+0, .logc = -0x1.7cc7f79e69000p-2 },
        .{ .invc = 0x1.713786a2ce91fp+0, .logc = -0x1.76feec20d0000p-2 },
        .{ .invc = 0x1.6f26008fab5a0p+0, .logc = -0x1.713e31351e000p-2 },
        .{ .invc = 0x1.6d1a61f138c7dp+0, .logc = -0x1.6b85b38287800p-2 },
        .{ .invc = 0x1.6b1490bc5b4d1p+0, .logc = -0x1.65d5590807800p-2 },
        .{ .invc = 0x1.69147332f0cbap+0, .logc = -0x1.602d076180000p-2 },
        .{ .invc = 0x1.6719f18224223p+0, .logc = -0x1.5a8ca86909000p-2 },
        .{ .invc = 0x1.6524f99a51ed9p+0, .logc = -0x1.54f4356035000p-2 },
        .{ .invc = 0x1.63356aa8f24c4p+0, .logc = -0x1.4f637c36b4000p-2 },
        .{ .invc = 0x1.614b36b9ddc14p+0, .logc = -0x1.49da7fda85000p-2 },
        .{ .invc = 0x1.5f66452c65c4cp+0, .logc = -0x1.445923989a800p-2 },
        .{ .invc = 0x1.5d867b5912c4fp+0, .logc = -0x1.3edf439b0b800p-2 },
        .{ .invc = 0x1.5babccb5b90dep+0, .logc = -0x1.396ce448f7000p-2 },
        .{ .invc = 0x1.59d61f2d91a78p+0, .logc = -0x1.3401e17bda000p-2 },
        .{ .invc = 0x1.5805612465687p+0, .logc = -0x1.2e9e2ef468000p-2 },
        .{ .invc = 0x1.56397cee76bd3p+0, .logc = -0x1.2941b3830e000p-2 },
        .{ .invc = 0x1.54725e2a77f93p+0, .logc = -0x1.23ec58cda8800p-2 },
        .{ .invc = 0x1.52aff42064583p+0, .logc = -0x1.1e9e129279000p-2 },
        .{ .invc = 0x1.50f22dbb2bddfp+0, .logc = -0x1.1956d2b48f800p-2 },
        .{ .invc = 0x1.4f38f4734ded7p+0, .logc = -0x1.141679ab9f800p-2 },
        .{ .invc = 0x1.4d843cfde2840p+0, .logc = -0x1.0edd094ef9800p-2 },
        .{ .invc = 0x1.4bd3ec078a3c8p+0, .logc = -0x1.09aa518db1000p-2 },
        .{ .invc = 0x1.4a27fc3e0258ap+0, .logc = -0x1.047e65263b800p-2 },
        .{ .invc = 0x1.4880524d48434p+0, .logc = -0x1.feb224586f000p-3 },
        .{ .invc = 0x1.46dce1b192d0bp+0, .logc = -0x1.f474a7517b000p-3 },
        .{ .invc = 0x1.453d9d3391854p+0, .logc = -0x1.ea4443d103000p-3 },
        .{ .invc = 0x1.43a2744b4845ap+0, .logc = -0x1.e020d44e9b000p-3 },
        .{ .invc = 0x1.420b54115f8fbp+0, .logc = -0x1.d60a22977f000p-3 },
        .{ .invc = 0x1.40782da3ef4b1p+0, .logc = -0x1.cc00104959000p-3 },
        .{ .invc = 0x1.3ee8f5d57fe8fp+0, .logc = -0x1.c202956891000p-3 },
        .{ .invc = 0x1.3d5d9a00b4ce9p+0, .logc = -0x1.b81178d811000p-3 },
        .{ .invc = 0x1.3bd60c010c12bp+0, .logc = -0x1.ae2c9ccd3d000p-3 },
        .{ .invc = 0x1.3a5242b75dab8p+0, .logc = -0x1.a45402e129000p-3 },
        .{ .invc = 0x1.38d22cd9fd002p+0, .logc = -0x1.9a877681df000p-3 },
        .{ .invc = 0x1.3755bc5847a1cp+0, .logc = -0x1.90c6d69483000p-3 },
        .{ .invc = 0x1.35dce49ad36e2p+0, .logc = -0x1.87120a645c000p-3 },
        .{ .invc = 0x1.34679984dd440p+0, .logc = -0x1.7d68fb4143000p-3 },
        .{ .invc = 0x1.32f5cceffcb24p+0, .logc = -0x1.73cb83c627000p-3 },
        .{ .invc = 0x1.3187775a10d49p+0, .logc = -0x1.6a39a9b376000p-3 },
        .{ .invc = 0x1.301c8373e3990p+0, .logc = -0x1.60b3154b7a000p-3 },
        .{ .invc = 0x1.2eb4ebb95f841p+0, .logc = -0x1.5737d76243000p-3 },
        .{ .invc = 0x1.2d50a0219a9d1p+0, .logc = -0x1.4dc7b8fc23000p-3 },
        .{ .invc = 0x1.2bef9a8b7fd2ap+0, .logc = -0x1.4462c51d20000p-3 },
        .{ .invc = 0x1.2a91c7a0c1babp+0, .logc = -0x1.3b08abc830000p-3 },
        .{ .invc = 0x1.293726014b530p+0, .logc = -0x1.31b996b490000p-3 },
        .{ .invc = 0x1.27dfa5757a1f5p+0, .logc = -0x1.2875490a44000p-3 },
        .{ .invc = 0x1.268b39b1d3bbfp+0, .logc = -0x1.1f3b9f879a000p-3 },
        .{ .invc = 0x1.2539d838ff5bdp+0, .logc = -0x1.160c8252ca000p-3 },
        .{ .invc = 0x1.23eb7aac9083bp+0, .logc = -0x1.0ce7f57f72000p-3 },
        .{ .invc = 0x1.22a012ba940b6p+0, .logc = -0x1.03cdc49fea000p-3 },
        .{ .invc = 0x1.2157996cc4132p+0, .logc = -0x1.f57bdbc4b8000p-4 },
        .{ .invc = 0x1.201201dd2fc9bp+0, .logc = -0x1.e370896404000p-4 },
        .{ .invc = 0x1.1ecf4494d480bp+0, .logc = -0x1.d17983ef94000p-4 },
        .{ .invc = 0x1.1d8f5528f6569p+0, .logc = -0x1.bf9674ed8a000p-4 },
        .{ .invc = 0x1.1c52311577e7cp+0, .logc = -0x1.adc79202f6000p-4 },
        .{ .invc = 0x1.1b17c74cb26e9p+0, .logc = -0x1.9c0c3e7288000p-4 },
        .{ .invc = 0x1.19e010c2c1ab6p+0, .logc = -0x1.8a646b372c000p-4 },
        .{ .invc = 0x1.18ab07bb670bdp+0, .logc = -0x1.78d01b3ac0000p-4 },
        .{ .invc = 0x1.1778a25efbcb6p+0, .logc = -0x1.674f145380000p-4 },
        .{ .invc = 0x1.1648d354c31dap+0, .logc = -0x1.55e0e6d878000p-4 },
        .{ .invc = 0x1.151b990275fddp+0, .logc = -0x1.4485cdea1e000p-4 },
        .{ .invc = 0x1.13f0ea432d24cp+0, .logc = -0x1.333d94d6aa000p-4 },
        .{ .invc = 0x1.12c8b7210f9dap+0, .logc = -0x1.22079f8c56000p-4 },
        .{ .invc = 0x1.11a3028ecb531p+0, .logc = -0x1.10e4698622000p-4 },
        .{ .invc = 0x1.107fbda8434afp+0, .logc = -0x1.ffa6c6ad20000p-5 },
        .{ .invc = 0x1.0f5ee0f4e6bb3p+0, .logc = -0x1.dda8d4a774000p-5 },
        .{ .invc = 0x1.0e4065d2a9fcep+0, .logc = -0x1.bbcece4850000p-5 },
        .{ .invc = 0x1.0d244632ca521p+0, .logc = -0x1.9a1894012c000p-5 },
        .{ .invc = 0x1.0c0a77ce2981ap+0, .logc = -0x1.788583302c000p-5 },
        .{ .invc = 0x1.0af2f83c636d1p+0, .logc = -0x1.5715e67d68000p-5 },
        .{ .invc = 0x1.09ddb98a01339p+0, .logc = -0x1.35c8a49658000p-5 },
        .{ .invc = 0x1.08cabaf52e7dfp+0, .logc = -0x1.149e364154000p-5 },
        .{ .invc = 0x1.07b9f2f4e28fbp+0, .logc = -0x1.e72c082eb8000p-6 },
        .{ .invc = 0x1.06ab58c358f19p+0, .logc = -0x1.a55f152528000p-6 },
        .{ .invc = 0x1.059eea5ecf92cp+0, .logc = -0x1.63d62cf818000p-6 },
        .{ .invc = 0x1.04949cdd12c90p+0, .logc = -0x1.228fb8caa0000p-6 },
        .{ .invc = 0x1.038c6c6f0ada9p+0, .logc = -0x1.c317b20f90000p-7 },
        .{ .invc = 0x1.02865137932a9p+0, .logc = -0x1.419355daa0000p-7 },
        .{ .invc = 0x1.0182427ea7348p+0, .logc = -0x1.81203c2ec0000p-8 },
        .{ .invc = 0x1.008040614b195p+0, .logc = -0x1.0040979240000p-9 },
        .{ .invc = 0x1.fe01ff726fa1ap-1, .logc = 0x1.feff384900000p-9 },
        .{ .invc = 0x1.fa11cc261ea74p-1, .logc = 0x1.7dc41353d0000p-7 },
        .{ .invc = 0x1.f6310b081992ep-1, .logc = 0x1.3cea3c4c28000p-6 },
        .{ .invc = 0x1.f25f63ceeadcdp-1, .logc = 0x1.b9fc114890000p-6 },
        .{ .invc = 0x1.ee9c8039113e7p-1, .logc = 0x1.1b0d8ce110000p-5 },
        .{ .invc = 0x1.eae8078cbb1abp-1, .logc = 0x1.58a5bd001c000p-5 },
        .{ .invc = 0x1.e741aa29d0c9bp-1, .logc = 0x1.95c8340d88000p-5 },
        .{ .invc = 0x1.e3a91830a99b5p-1, .logc = 0x1.d276aef578000p-5 },
        .{ .invc = 0x1.e01e009609a56p-1, .logc = 0x1.07598e598c000p-4 },
        .{ .invc = 0x1.dca01e577bb98p-1, .logc = 0x1.253f5e30d2000p-4 },
        .{ .invc = 0x1.d92f20b7c9103p-1, .logc = 0x1.42edd8b380000p-4 },
        .{ .invc = 0x1.d5cac66fb5ccep-1, .logc = 0x1.606598757c000p-4 },
        .{ .invc = 0x1.d272caa5ede9dp-1, .logc = 0x1.7da76356a0000p-4 },
        .{ .invc = 0x1.cf26e3e6b2ccdp-1, .logc = 0x1.9ab434e1c6000p-4 },
        .{ .invc = 0x1.cbe6da2a77902p-1, .logc = 0x1.b78c7bb0d6000p-4 },
        .{ .invc = 0x1.c8b266d37086dp-1, .logc = 0x1.d431332e72000p-4 },
        .{ .invc = 0x1.c5894bd5d5804p-1, .logc = 0x1.f0a3171de6000p-4 },
        .{ .invc = 0x1.c26b533bb9f8cp-1, .logc = 0x1.067152b914000p-3 },
        .{ .invc = 0x1.bf583eeece73fp-1, .logc = 0x1.147858292b000p-3 },
        .{ .invc = 0x1.bc4fd75db96c1p-1, .logc = 0x1.2266ecdca3000p-3 },
        .{ .invc = 0x1.b951e0c864a28p-1, .logc = 0x1.303d7a6c55000p-3 },
        .{ .invc = 0x1.b65e2c5ef3e2cp-1, .logc = 0x1.3dfc33c331000p-3 },
        .{ .invc = 0x1.b374867c9888bp-1, .logc = 0x1.4ba366b7a8000p-3 },
        .{ .invc = 0x1.b094b211d304ap-1, .logc = 0x1.5933928d1f000p-3 },
        .{ .invc = 0x1.adbe885f2ef7ep-1, .logc = 0x1.66acd2418f000p-3 },
        .{ .invc = 0x1.aaf1d31603da2p-1, .logc = 0x1.740f8ec669000p-3 },
        .{ .invc = 0x1.a82e63fd358a7p-1, .logc = 0x1.815c0f51af000p-3 },
        .{ .invc = 0x1.a5740ef09738bp-1, .logc = 0x1.8e92954f68000p-3 },
        .{ .invc = 0x1.a2c2a90ab4b27p-1, .logc = 0x1.9bb3602f84000p-3 },
        .{ .invc = 0x1.a01a01393f2d1p-1, .logc = 0x1.a8bed1c2c0000p-3 },
        .{ .invc = 0x1.9d79f24db3c1bp-1, .logc = 0x1.b5b515c01d000p-3 },
        .{ .invc = 0x1.9ae2505c7b190p-1, .logc = 0x1.c2967ccbcc000p-3 },
        .{ .invc = 0x1.9852ef297ce2fp-1, .logc = 0x1.cf635d5486000p-3 },
        .{ .invc = 0x1.95cbaeea44b75p-1, .logc = 0x1.dc1bd3446c000p-3 },
        .{ .invc = 0x1.934c69de74838p-1, .logc = 0x1.e8c01b8cfe000p-3 },
        .{ .invc = 0x1.90d4f2f6752e6p-1, .logc = 0x1.f5509c0179000p-3 },
        .{ .invc = 0x1.8e6528effd79dp-1, .logc = 0x1.00e6c121fb800p-2 },
        .{ .invc = 0x1.8bfce9fcc007cp-1, .logc = 0x1.071b80e93d000p-2 },
        .{ .invc = 0x1.899c0dabec30ep-1, .logc = 0x1.0d46b9e867000p-2 },
        .{ .invc = 0x1.87427aa2317fbp-1, .logc = 0x1.13687334bd000p-2 },
        .{ .invc = 0x1.84f00acb39a08p-1, .logc = 0x1.1980d67234800p-2 },
        .{ .invc = 0x1.82a49e8653e55p-1, .logc = 0x1.1f8ffe0cc8000p-2 },
        .{ .invc = 0x1.8060195f40260p-1, .logc = 0x1.2595fd7636800p-2 },
        .{ .invc = 0x1.7e22563e0a329p-1, .logc = 0x1.2b9300914a800p-2 },
        .{ .invc = 0x1.7beb377dcb5adp-1, .logc = 0x1.3187210436000p-2 },
        .{ .invc = 0x1.79baa679725c2p-1, .logc = 0x1.377266dec1800p-2 },
        .{ .invc = 0x1.77907f2170657p-1, .logc = 0x1.3d54ffbaf3000p-2 },
        .{ .invc = 0x1.756cadbd6130cp-1, .logc = 0x1.432eee32fe000p-2 },
    };

    const tab2 = [128]struct { chi: f64, clo: f64 }{
        .{ .chi = 0x1.61000014fb66bp-1, .clo = 0x1.e026c91425b3cp-56 },
        .{ .chi = 0x1.63000034db495p-1, .clo = 0x1.dbfea48005d41p-55 },
        .{ .chi = 0x1.650000d94d478p-1, .clo = 0x1.e7fa786d6a5b7p-55 },
        .{ .chi = 0x1.67000074e6fadp-1, .clo = 0x1.1fcea6b54254cp-57 },
        .{ .chi = 0x1.68ffffedf0faep-1, .clo = -0x1.c7e274c590efdp-56 },
        .{ .chi = 0x1.6b0000763c5bcp-1, .clo = -0x1.ac16848dcda01p-55 },
        .{ .chi = 0x1.6d0001e5cc1f6p-1, .clo = 0x1.33f1c9d499311p-55 },
        .{ .chi = 0x1.6efffeb05f63ep-1, .clo = -0x1.e80041ae22d53p-56 },
        .{ .chi = 0x1.710000e86978p-1, .clo = 0x1.bff6671097952p-56 },
        .{ .chi = 0x1.72ffffc67e912p-1, .clo = 0x1.c00e226bd8724p-55 },
        .{ .chi = 0x1.74fffdf81116ap-1, .clo = -0x1.e02916ef101d2p-57 },
        .{ .chi = 0x1.770000f679c9p-1, .clo = -0x1.7fc71cd549c74p-57 },
        .{ .chi = 0x1.78ffffa7ec835p-1, .clo = 0x1.1bec19ef50483p-55 },
        .{ .chi = 0x1.7affffe20c2e6p-1, .clo = -0x1.07e1729cc6465p-56 },
        .{ .chi = 0x1.7cfffed3fc9p-1, .clo = -0x1.08072087b8b1cp-55 },
        .{ .chi = 0x1.7efffe9261a76p-1, .clo = 0x1.dc0286d9df9aep-55 },
        .{ .chi = 0x1.81000049ca3e8p-1, .clo = 0x1.97fd251e54c33p-55 },
        .{ .chi = 0x1.8300017932c8fp-1, .clo = -0x1.afee9b630f381p-55 },
        .{ .chi = 0x1.850000633739cp-1, .clo = 0x1.9bfbf6b6535bcp-55 },
        .{ .chi = 0x1.87000204289c6p-1, .clo = -0x1.bbf65f3117b75p-55 },
        .{ .chi = 0x1.88fffebf57904p-1, .clo = -0x1.9006ea23dcb57p-55 },
        .{ .chi = 0x1.8b00022bc04dfp-1, .clo = -0x1.d00df38e04b0ap-56 },
        .{ .chi = 0x1.8cfffe50c1b8ap-1, .clo = -0x1.8007146ff9f05p-55 },
        .{ .chi = 0x1.8effffc918e43p-1, .clo = 0x1.3817bd07a7038p-55 },
        .{ .chi = 0x1.910001efa5fc7p-1, .clo = 0x1.93e9176dfb403p-55 },
        .{ .chi = 0x1.9300013467bb9p-1, .clo = 0x1.f804e4b980276p-56 },
        .{ .chi = 0x1.94fffe6ee076fp-1, .clo = -0x1.f7ef0d9ff622ep-55 },
        .{ .chi = 0x1.96fffde3c12d1p-1, .clo = -0x1.082aa962638bap-56 },
        .{ .chi = 0x1.98ffff4458a0dp-1, .clo = -0x1.7801b9164a8efp-55 },
        .{ .chi = 0x1.9afffdd982e3ep-1, .clo = -0x1.740e08a5a9337p-55 },
        .{ .chi = 0x1.9cfffed49fb66p-1, .clo = 0x1.fce08c19bep-60 },
        .{ .chi = 0x1.9f00020f19c51p-1, .clo = -0x1.a3faa27885b0ap-55 },
        .{ .chi = 0x1.a10001145b006p-1, .clo = 0x1.4ff489958da56p-56 },
        .{ .chi = 0x1.a300007bbf6fap-1, .clo = 0x1.cbeab8a2b6d18p-55 },
        .{ .chi = 0x1.a500010971d79p-1, .clo = 0x1.8fecadd78793p-55 },
        .{ .chi = 0x1.a70001df52e48p-1, .clo = -0x1.f41763dd8abdbp-55 },
        .{ .chi = 0x1.a90001c593352p-1, .clo = -0x1.ebf0284c27612p-55 },
        .{ .chi = 0x1.ab0002a4f3e4bp-1, .clo = -0x1.9fd043cff3f5fp-57 },
        .{ .chi = 0x1.acfffd7ae1ed1p-1, .clo = -0x1.23ee7129070b4p-55 },
        .{ .chi = 0x1.aefffee510478p-1, .clo = 0x1.a063ee00edea3p-57 },
        .{ .chi = 0x1.b0fffdb650d5bp-1, .clo = 0x1.a06c8381f0ab9p-58 },
        .{ .chi = 0x1.b2ffffeaaca57p-1, .clo = -0x1.9011e74233c1dp-56 },
        .{ .chi = 0x1.b4fffd995badcp-1, .clo = -0x1.9ff1068862a9fp-56 },
        .{ .chi = 0x1.b7000249e659cp-1, .clo = 0x1.aff45d0864f3ep-55 },
        .{ .chi = 0x1.b8ffff987164p-1, .clo = 0x1.cfe7796c2c3f9p-56 },
        .{ .chi = 0x1.bafffd204cb4fp-1, .clo = -0x1.3ff27eef22bc4p-57 },
        .{ .chi = 0x1.bcfffd2415c45p-1, .clo = -0x1.cffb7ee3bea21p-57 },
        .{ .chi = 0x1.beffff86309dfp-1, .clo = -0x1.14103972e0b5cp-55 },
        .{ .chi = 0x1.c0fffe1b57653p-1, .clo = 0x1.bc16494b76a19p-55 },
        .{ .chi = 0x1.c2ffff1fa57e3p-1, .clo = -0x1.4feef8d30c6edp-57 },
        .{ .chi = 0x1.c4fffdcbfe424p-1, .clo = -0x1.43f68bcec4775p-55 },
        .{ .chi = 0x1.c6fffed54b9f7p-1, .clo = 0x1.47ea3f053e0ecp-55 },
        .{ .chi = 0x1.c8fffeb998fd5p-1, .clo = 0x1.383068df992f1p-56 },
        .{ .chi = 0x1.cb0002125219ap-1, .clo = -0x1.8fd8e64180e04p-57 },
        .{ .chi = 0x1.ccfffdd94469cp-1, .clo = 0x1.e7ebe1cc7ea72p-55 },
        .{ .chi = 0x1.cefffeafdc476p-1, .clo = 0x1.ebe39ad9f88fep-55 },
        .{ .chi = 0x1.d1000169af82bp-1, .clo = 0x1.57d91a8b95a71p-56 },
        .{ .chi = 0x1.d30000d0ff71dp-1, .clo = 0x1.9c1906970c7dap-55 },
        .{ .chi = 0x1.d4fffea790fc4p-1, .clo = -0x1.80e37c558fe0cp-58 },
        .{ .chi = 0x1.d70002edc87e5p-1, .clo = -0x1.f80d64dc10f44p-56 },
        .{ .chi = 0x1.d900021dc82aap-1, .clo = -0x1.47c8f94fd5c5cp-56 },
        .{ .chi = 0x1.dafffd86b0283p-1, .clo = 0x1.c7f1dc521617ep-55 },
        .{ .chi = 0x1.dd000296c4739p-1, .clo = 0x1.8019eb2ffb153p-55 },
        .{ .chi = 0x1.defffe54490f5p-1, .clo = 0x1.e00d2c652cc89p-57 },
        .{ .chi = 0x1.e0fffcdabf694p-1, .clo = -0x1.f8340202d69d2p-56 },
        .{ .chi = 0x1.e2fffdb52c8ddp-1, .clo = 0x1.b00c1ca1b0864p-56 },
        .{ .chi = 0x1.e4ffff24216efp-1, .clo = 0x1.2ffa8b094ab51p-56 },
        .{ .chi = 0x1.e6fffe88a5e11p-1, .clo = -0x1.7f673b1efbe59p-58 },
        .{ .chi = 0x1.e9000119eff0dp-1, .clo = -0x1.4808d5e0bc801p-55 },
        .{ .chi = 0x1.eafffdfa51744p-1, .clo = 0x1.80006d54320b5p-56 },
        .{ .chi = 0x1.ed0001a127fa1p-1, .clo = -0x1.002f860565c92p-58 },
        .{ .chi = 0x1.ef00007babcc4p-1, .clo = -0x1.540445d35e611p-55 },
        .{ .chi = 0x1.f0ffff57a8d02p-1, .clo = -0x1.ffb3139ef9105p-59 },
        .{ .chi = 0x1.f30001ee58ac7p-1, .clo = 0x1.a81acf2731155p-55 },
        .{ .chi = 0x1.f4ffff5823494p-1, .clo = 0x1.a3f41d4d7c743p-55 },
        .{ .chi = 0x1.f6ffffca94c6bp-1, .clo = -0x1.202f41c987875p-57 },
        .{ .chi = 0x1.f8fffe1f9c441p-1, .clo = 0x1.77dd1f477e74bp-56 },
        .{ .chi = 0x1.fafffd2e0e37ep-1, .clo = -0x1.f01199a7ca331p-57 },
        .{ .chi = 0x1.fd0001c77e49ep-1, .clo = 0x1.181ee4bceacb1p-56 },
        .{ .chi = 0x1.feffff7e0c331p-1, .clo = -0x1.e05370170875ap-57 },
        .{ .chi = 0x1.00ffff465606ep+0, .clo = -0x1.a7ead491c0adap-55 },
        .{ .chi = 0x1.02ffff3867a58p+0, .clo = -0x1.77f69c3fcb2ep-54 },
        .{ .chi = 0x1.04ffffdfc0d17p+0, .clo = 0x1.7bffe34cb945bp-54 },
        .{ .chi = 0x1.0700003cd4d82p+0, .clo = 0x1.20083c0e456cbp-55 },
        .{ .chi = 0x1.08ffff9f2cbe8p+0, .clo = -0x1.dffdfbe37751ap-57 },
        .{ .chi = 0x1.0b000010cda65p+0, .clo = -0x1.13f7faee626ebp-54 },
        .{ .chi = 0x1.0d00001a4d338p+0, .clo = 0x1.07dfa79489ff7p-55 },
        .{ .chi = 0x1.0effffadafdfdp+0, .clo = -0x1.7040570d66bcp-56 },
        .{ .chi = 0x1.110000bbafd96p+0, .clo = 0x1.e80d4846d0b62p-55 },
        .{ .chi = 0x1.12ffffae5f45dp+0, .clo = 0x1.dbffa64fd36efp-54 },
        .{ .chi = 0x1.150000dd59ad9p+0, .clo = 0x1.a0077701250aep-54 },
        .{ .chi = 0x1.170000f21559ap+0, .clo = 0x1.dfdf9e2e3deeep-55 },
        .{ .chi = 0x1.18ffffc275426p+0, .clo = 0x1.10030dc3b7273p-54 },
        .{ .chi = 0x1.1b000123d3c59p+0, .clo = 0x1.97f7980030188p-54 },
        .{ .chi = 0x1.1cffff8299eb7p+0, .clo = -0x1.5f932ab9f8c67p-57 },
        .{ .chi = 0x1.1effff48ad4p+0, .clo = 0x1.37fbf9da75bebp-54 },
        .{ .chi = 0x1.210000c8b86a4p+0, .clo = 0x1.f806b91fd5b22p-54 },
        .{ .chi = 0x1.2300003854303p+0, .clo = 0x1.3ffc2eb9fbf33p-54 },
        .{ .chi = 0x1.24fffffbcf684p+0, .clo = 0x1.601e77e2e2e72p-56 },
        .{ .chi = 0x1.26ffff52921d9p+0, .clo = 0x1.ffcbb767f0c61p-56 },
        .{ .chi = 0x1.2900014933a3cp+0, .clo = -0x1.202ca3c02412bp-56 },
        .{ .chi = 0x1.2b00014556313p+0, .clo = -0x1.2808233f21f02p-54 },
        .{ .chi = 0x1.2cfffebfe523bp+0, .clo = -0x1.8ff7e384fdcf2p-55 },
        .{ .chi = 0x1.2f0000bb8ad96p+0, .clo = -0x1.5ff51503041c5p-55 },
        .{ .chi = 0x1.30ffffb7ae2afp+0, .clo = -0x1.10071885e289dp-55 },
        .{ .chi = 0x1.32ffffeac5f7fp+0, .clo = -0x1.1ff5d3fb7b715p-54 },
        .{ .chi = 0x1.350000ca66756p+0, .clo = 0x1.57f82228b82bdp-54 },
        .{ .chi = 0x1.3700011fbf721p+0, .clo = 0x1.000bac40dd5ccp-55 },
        .{ .chi = 0x1.38ffff9592fb9p+0, .clo = -0x1.43f9d2db2a751p-54 },
        .{ .chi = 0x1.3b00004ddd242p+0, .clo = 0x1.57f6b707638e1p-55 },
        .{ .chi = 0x1.3cffff5b2c957p+0, .clo = 0x1.a023a10bf1231p-56 },
        .{ .chi = 0x1.3efffeab0b418p+0, .clo = 0x1.87f6d66b152bp-54 },
        .{ .chi = 0x1.410001532aff4p+0, .clo = 0x1.7f8375f198524p-57 },
        .{ .chi = 0x1.4300017478b29p+0, .clo = 0x1.301e672dc5143p-55 },
        .{ .chi = 0x1.44fffe795b463p+0, .clo = 0x1.9ff69b8b2895ap-55 },
        .{ .chi = 0x1.46fffe80475ep+0, .clo = -0x1.5c0b19bc2f254p-54 },
        .{ .chi = 0x1.48fffef6fc1e7p+0, .clo = 0x1.b4009f23a2a72p-54 },
        .{ .chi = 0x1.4afffe5bea704p+0, .clo = -0x1.4ffb7bf0d7d45p-54 },
        .{ .chi = 0x1.4d000171027dep+0, .clo = -0x1.9c06471dc6a3dp-54 },
        .{ .chi = 0x1.4f0000ff03ee2p+0, .clo = 0x1.77f890b85531cp-54 },
        .{ .chi = 0x1.5100012dc4bd1p+0, .clo = 0x1.004657166a436p-57 },
        .{ .chi = 0x1.530001605277ap+0, .clo = -0x1.6bfcece233209p-54 },
        .{ .chi = 0x1.54fffecdb704cp+0, .clo = -0x1.902720505a1d7p-55 },
        .{ .chi = 0x1.56fffef5f54a9p+0, .clo = 0x1.bbfe60ec96412p-54 },
        .{ .chi = 0x1.5900017e61012p+0, .clo = 0x1.87ec581afef9p-55 },
        .{ .chi = 0x1.5b00003c93e92p+0, .clo = -0x1.f41080abf0ccp-54 },
        .{ .chi = 0x1.5d0001d4919bcp+0, .clo = -0x1.8812afb254729p-54 },
        .{ .chi = 0x1.5efffe7b87a89p+0, .clo = -0x1.47eb780ed6904p-54 },
    };

    var ix: i64 = @bitCast(x);

    const LO: i64 = @bitCast(@as(f64, 1.0 - 0x1p-4));
    const HI: i64 = @bitCast(@as(f64, 1.0 + 0x1.09p-4));
    if (LO <= ix and ix < HI) {
        @branchHint(.unlikely);
        if (ix == @as(i64, @bitCast(@as(f64, 1)))) {
            @branchHint(.unlikely);
            return 0;
        }

        const r = x - 1;
        const r2 = r * r;
        const r3 = r * r2;

        const y = r3 * (poly1[1] + r * poly1[2] + r2 * poly1[3] +
            r3 * (poly1[4] + r * poly1[5] + r2 * poly1[6] +
                r3 * (poly1[7] + r * poly1[8] + r2 * poly1[9] + r3 * poly1[10])));

        var w = r * 0x1p27;
        const rhi = r + w - w;
        const rlo = r - rhi;
        w = rhi * rhi * poly1[0];
        const hi = r + w;
        const lo = r - hi + w + poly1[0] * rlo * (rhi + r);
        return y + lo + hi;
    }
    const top = @as(u64, @bitCast(ix)) >> 48;
    if (top < 0x0010 or 0x7ff0 <= top) {
        @branchHint(.unlikely);

        if (ix << 1 == 0)
            return -std.math.inf(f64);

        if (ix == @as(i64, @bitCast(std.math.inf(f64))))
            return x;

        if (top & 0x8000 != 0 or top & 0x7ff0 == 0x7ff0)
            return std.math.nan(f64);

        ix = @as(i64, @bitCast(x * 0x1p52)) - (52 << 52);
    }

    const tmp: packed struct(i64) { unused: u45, i: u7, k: i12 } = @bitCast(ix - 0x3fe6000000000000);
    const i = tmp.i;
    const k = tmp.k;
    const iz = ix - (@as(i64, tmp.k) << 52);
    const invc = tab[i].invc;
    const logc = tab[i].logc;
    const z: f64 = @bitCast(iz);

    // maybe use fma as musl does
    const r = (z - tab2[i].chi - tab2[i].clo) * invc;

    const kd: f64 = k;
    const w = kd * 0x1.62e42fefa3800p-1 + logc;
    const hi = w + r;
    const lo = w - hi + r + kd * 0x1.ef35793c76730p-45;

    const r2 = r * r;

    const y = lo + r2 * poly[0] + r * r2 * (poly[1] + r * poly[2] + r2 * (poly[3] + r * poly[4])) + hi;
    return @bitCast(y);
}

/// ln Γ(x), for x ≥ 0: the standard library's `lgamma`, without the reflection for
/// negative x, which nothing here needs.
pub fn lgamma(x: f64) f64 {
    std.debug.assert(!(x < 0));
    // common integer case first
    if (x == @trunc(x)) {
        // lgamma(+-0.0) = +inf
        if (x <= 0) {
            return std.math.inf(f64);
        }
        // lgamma(1) = +0.0
        // lgamma(2) = +0.0
        if (x < integer_result_table.len) {
            const i = @as(u8, @intFromFloat(x));
            return log(integer_result_table[i]);
        }
        // lgamma(+inf) = +inf
        if (std.math.isPositiveInf(x)) {
            return x;
        }
    }

    const abs = @abs(x);
    // perfect precision here
    if (abs < 0x1p-54) {
        return -log(abs);
    }
    // obvious approach when overflow is not a problem
    const upper_bound = 128;
    if (abs < upper_bound) {
        return log(@abs(gamma(x)));
    }

    const log_base = log(abs + lanczos_minus_half) - 1;
    const exponent = abs - 0.5;
    const log_series = log(series(abs));
    return exponent * log_base + log_series - lanczos;
}

/// Γ(x) for x ≥ 0, as the standard library computes it.
fn gamma(x: f64) f64 {
    // common integer case first
    if (x == @trunc(x)) {
        // gamma(+0.0) = +inf
        if (x == 0) {
            return 1 / x;
        }
        if (x < integer_result_table.len) {
            const i = @as(u8, @intFromFloat(x));
            return integer_result_table[i];
        }
    }
    // above this, result overflows
    // gamma(+inf) = +inf
    const upper_bound = 172;
    if (x > upper_bound) {
        return std.math.inf(f64);
    }

    const abs = @abs(x);
    // perfect precision here
    if (abs < 0x1p-54) {
        return 1 / x;
    }

    const base = abs + lanczos_minus_half;
    const exponent = abs - 0.5;
    // error of y for correction, see
    // https://github.com/python/cpython/blob/5dc79e3d7f26a6a871a89ce3efc9f1bcee7bb447/Modules/mathmodule.c#L286-L324
    const e = if (abs > lanczos_minus_half)
        base - abs - lanczos_minus_half
    else
        base - lanczos_minus_half - abs;
    const correction = lanczos * e / base;
    const initial = series(abs) * exp(-base);

    const corrected = initial + initial * correction;
    const half_pow = pow(base, 0.5 * exponent);
    return corrected * half_pow * half_pow;
}

// table of factorials for integer early return
// stops at 22 because 23 isn't representable with full precision on f64
const integer_result_table = [_]f64{
    std.math.inf(f64), // gamma(+0.0)
    1, // gamma(1)
    1, // ...
    2,
    6,
    24,
    120,
    720,
    5040,
    40320,
    362880,
    3628800,
    39916800,
    479001600,
    6227020800,
    87178291200,
    1307674368000,
    20922789888000,
    355687428096000,
    6402373705728000,
    121645100408832000,
    2432902008176640000,
    51090942171709440000, // gamma(22)
};

// "g" constant, arbitrary
const lanczos = 6.024680040776729583740234375;
const lanczos_minus_half = lanczos - 0.5;

fn series(abs: f64) f64 {
    const numerator = [_]f64{
        23531376880.410759688572007674451636754734846804940,
        42919803642.649098768957899047001988850926355848959,
        35711959237.355668049440185451547166705960488635843,
        17921034426.037209699919755754458931112671403265390,
        6039542586.3520280050642916443072979210699388420708,
        1439720407.3117216736632230727949123939715485786772,
        248874557.86205415651146038641322942321632125127801,
        31426415.585400194380614231628318205362874684987640,
        2876370.6289353724412254090516208496135991145378768,
        186056.26539522349504029498971604569928220784236328,
        8071.6720023658162106380029022722506138218516325024,
        210.82427775157934587250973392071336271166969580291,
        2.5066282746310002701649081771338373386264310793408,
    };
    const denominator = [_]f64{
        0.0,
        39916800.0,
        120543840.0,
        150917976.0,
        105258076.0,
        45995730.0,
        13339535.0,
        2637558.0,
        357423.0,
        32670.0,
        1925.0,
        66.0,
        1.0,
    };
    var num: f64 = 0;
    var den: f64 = 0;
    // split to avoid overflow
    if (abs < 8) {
        // big abs would overflow here
        for (0..numerator.len) |i| {
            num = num * abs + numerator[numerator.len - 1 - i];
            den = den * abs + denominator[numerator.len - 1 - i];
        }
    } else {
        // small abs would overflow here
        for (0..numerator.len) |i| {
            num = num / abs + numerator[i];
            den = den / abs + denominator[i];
        }
    }
    return num / den;
}

/// x^y for finite x > 0 and finite y, as the standard library's `pow` computes it.
fn pow(x: f64, y: f64) f64 {
    std.debug.assert(x > 0 and std.math.isFinite(x) and std.math.isFinite(y));
    // pow(x, +-0) = 1      for any x
    // pow(1, y) = 1        for any y
    if (y == 0 or x == 1) {
        return 1;
    }
    // pow(x, 1) = x        for any x
    if (y == 1) {
        return x;
    }
    // special case sqrt
    if (y == 0.5) {
        return @sqrt(x);
    }
    if (y == -0.5) {
        return 1 / @sqrt(x);
    }

    const r1 = std.math.modf(@abs(y));
    var yi = r1.ipart;
    var yf = r1.fpart;

    if (yi >= 1 << 63) {
        // yi is a large even int, so the result is always positive
        // and the sign of x doesn't matter
        return exp(y * log(@abs(x)));
    }

    // a = a1 * 2^ae
    var a1: f64 = 1.0;
    var ae: i32 = 0;

    // a *= x^yf
    if (yf != 0) {
        if (yf > 0.5) {
            yf -= 1;
            yi += 1;
        }
        a1 = exp(yf * log(x));
    }

    // a *= x^yi
    const r2 = std.math.frexp(x);
    var xe = r2.exponent;
    var x1 = r2.significand;

    var i = @as(i64, @intFromFloat(yi));
    while (i != 0) : (i >>= 1) {
        const overflow_shift = std.math.floatExponentBits(f64) + 1;
        if (xe < -(1 << overflow_shift) or (1 << overflow_shift) < xe) {
            // catch xe before it overflows the left shift below
            // Since i != 0 it has at least one bit still set, so ae will accumulate xe
            // on at least one more iteration, ae += xe is a lower bound on ae
            // the lower bound on ae exceeds the size of a float exp
            // so the final call to Ldexp will produce under/overflow (0/Inf)
            ae += xe;
            break;
        }
        if (i & 1 == 1) {
            a1 *= x1;
            ae += xe;
        }
        x1 *= x1;
        xe <<= 1;
        if (x1 < 0.5) {
            x1 += x1;
            xe -= 1;
        }
    }

    // a *= a1 * 2^ae
    if (y < 0) {
        a1 = 1 / a1;
        ae = -ae;
    }

    return std.math.scalbn(a1, ae);
}

// -- tests -------------------------------------------------------------------

const testing = std.testing;

/// Every input the pinned-bits test hashes: the solver's range for each function.
fn hashOutputs() [64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var prng = std.Random.DefaultPrng.init(0x11B);
    const random = prng.random();
    for (0..20_000) |_| {
        // exp: the solver's arguments are logs of densities, mostly large and negative.
        const a = -800 + 1000 * random.float(f64);
        // log: across magnitudes, and near 1, where it takes its other path.
        const b = std.math.scalbn(random.float(f64) + 0.5, random.intRangeAtMost(i32, -1000, 1000));
        const c = 1 + (random.float(f64) - 0.5) * 0.2;
        const lg = 300 * random.float(f64);
        for ([_]f64{ exp(a), log(b), log(c), lgamma(lg) }) |v| {
            h.update(std.mem.asBytes(&std.mem.nativeToLittle(u64, @bitCast(v))));
        }
    }
    // lgamma at d/2 for every dimension the library accepts, which covers (d−1)/2 too.
    var d: u32 = 1;
    while (d <= 1 << 16) : (d += 1) {
        const v = lgamma(@as(f64, @floatFromInt(d)) * 0.5);
        h.update(std.mem.asBytes(&std.mem.nativeToLittle(u64, @bitCast(v))));
    }
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

// Computed with these functions as they were taken from compiler-rt and std, so equal to
// what builds without libc computed before. Every platform CI tests on must agree.
test "exp, log and lgamma give the same bits everywhere" {
    try testing.expectEqualStrings("69980318cb1bef553b2e9709cd744f2a79c88bd555274a27051c51e298f9e3ea", &hashOutputs());
}

// And they are the functions they claim to be: within an ulp or two of whichever libm
// this test links.
test "exp, log and lgamma agree with the platform's to an ulp or two" {
    var prng = std.Random.DefaultPrng.init(0x11C);
    const random = prng.random();
    const ulps = struct {
        fn apart(x: f64, y: f64) u64 {
            const a: i64 = @bitCast(x);
            const b: i64 = @bitCast(y);
            return @abs(a - b);
        }
    }.apart;
    for (0..5_000) |_| {
        const a = -700 + 1400 * random.float(f64);
        try testing.expect(ulps(@exp(a), exp(a)) <= 1);
        const b = std.math.scalbn(random.float(f64) + 0.5, random.intRangeAtMost(i32, -1000, 1000));
        try testing.expect(ulps(@log(b), log(b)) <= 1);
        // lgamma crosses zero near 1 and 2, so the bound is absolute there.
        const x = 0.5 + 300 * random.float(f64);
        const want = std.math.lgamma(f64, x);
        try testing.expectApproxEqAbs(want, lgamma(x), 1e-13 * @max(1, @abs(want)));
    }
}

test "special values" {
    try testing.expectEqual(@as(f64, 1), exp(0));
    try testing.expectEqual(@as(f64, 0), exp(-std.math.inf(f64)));
    try testing.expect(std.math.isPositiveInf(exp(1000)));
    try testing.expect(std.math.isNan(exp(std.math.nan(f64))));
    try testing.expectEqual(@as(f64, 0), log(1));
    try testing.expect(std.math.isNegativeInf(log(0)));
    try testing.expect(std.math.isNan(log(-1)));
    try testing.expect(std.math.isPositiveInf(log(std.math.inf(f64))));
    try testing.expectEqual(@as(f64, 0), lgamma(1));
    try testing.expectEqual(@as(f64, 0), lgamma(2));
    try testing.expect(std.math.isPositiveInf(lgamma(0)));
    try testing.expectApproxEqRel(@as(f64, 3.1780538303479458), lgamma(5), 1e-15);
}
