//! Phase 2 Task 2: bit-exact golden tests for `nn::activations`
//! (ported from `ActivationFunctions.h`: `Logistic`, `GatesFunction`, `Maxmin2`).
//!
//! `act_sweep.bin` is dumped by the REAL compiled legacy `ActivationFunctions.h`
//! structs (header-only, called directly in `tools/oracle_harness/main.cpp`).
//! exp() is a libm transcendental, so the sweep golden uses the canary-gated
//! hybrid comparator (bit-exact on the oracle env, `<=4` ULP or scaled-absolute
//! elsewhere); the saturation-boundary constants (0.0/1.0) are NOT libm-dependent
//! and are asserted bit-exact everywhere via plain `assert_eq!`.

mod common;

use speech::nn::activations::*;

#[test]
fn activation_sweep_matches_oracle() {
    let dump = common::load_bin_phase2("act_sweep.bin");
    for k in 0..dump.ncols() {
        let x = dump[[0, k]];
        common::assert_oracle_eq_f64(gates_fn(x), dump[[1, k]], &format!("gates[{k}]"));
        common::assert_oracle_eq_f64(logistic_fn(x), dump[[2, k]], &format!("logistic[{k}]"));
        common::assert_oracle_eq_f64(maxmin2_fn(x), dump[[3, k]], &format!("asinh[{k}]"));
    }
}

// Saturation-boundary bit-pins. NOTE: `10.0 * exp_limit()` is NOT the same f64 as
// `exp_limit() / 0.1` (1 ULP apart: 0.1 * (10.0*el) rounds to el's next-higher
// double, not el itself) -- see the harness dump columns 7/8 (exp_limit()/0.1,
// the value where 0.1*x == exp_limit() bit-for-bit) vs 19/20 (10.0*exp_limit()).
// The dump's column 7/8 is the exact input GatesFunction::fn saturates on; using
// it here (rather than the algebraic `10.0 * el`) is what makes this an exact
// boundary probe rather than a 1-ULP-adjacent one.
//
// The brief's proposed `logistic_fn(el - 1.0) < 1.0` does not hold: expLimit ==
// f64::MAX.ln() ~= 709.78 guards against exp(-x) OVERFLOWING (x very negative),
// not against 1/(1+exp(-x)) underflowing to 1.0 at the top end. For ANY x more
// than ~36.7 below expLimit, exp(-x) is already below f64 epsilon relative to
// 1.0, so 1.0 + exp(-x) rounds to exactly 1.0 -- the pass-through formula itself
// saturates to 1.0 far inside the guard, well before x reaches expLimit. Verified
// numerically: logistic_fn(el - 40.0) still == 1.0 bit-for-bit. Using x=10.0
// (sweep column 3, logistic ~= 0.9999546, confirmed < 1.0) as the non-saturated
// probe instead.
#[test]
fn saturation_boundaries_exact() {
    let el = exp_limit();
    let dump = common::load_bin_phase2("act_sweep.bin");
    let el_over_p1 = dump[[0, 7]]; // == el / 0.1 bit-for-bit; 0.1*el_over_p1 == el exactly
    assert_eq!(el_over_p1, el / 0.1);

    assert_eq!(gates_fn(el_over_p1), 1.0); // 0.1*x == el exactly -> EXCLUSIVE -> saturated
    assert_eq!(gates_fn(-el_over_p1), 0.0);
    assert_eq!(logistic_fn(el), 1.0); // INCLUSIVE at boundary
    assert_eq!(logistic_fn(-el), 0.0);
    assert!(logistic_fn(10.0) < 1.0);
}
