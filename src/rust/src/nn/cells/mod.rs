//! Cell abstraction for the exact f64 recurrent stacks: the closed `CellLayer` enum.
//!
//! Port-only (Phase 9 Task 1) -- NO legacy source. Spec:
//! `docs/superpowers/specs/2026-07-27-phase-9-new-architectures-design.md`, S1.1
//! (the cut) and S1.4 (the discipline).
//!
//! The cut is at the LAYER, not the network: [`super::network::Network`] is already
//! generic over the 10-method [`Layer`] trait (whole-sequence
//! `Array2<f64>` in/out), so swapping a recurrent cell means swapping `L` and nothing
//! else. `Network` itself, the output `Network<NeuronLayer>`, the four windowed
//! drivers, input normalization, scoring/cost accumulation, the flat-weight seam,
//! `train_modern` and batching never see the cell type.
//!
//! `CellLayer` implements `Layer` by match delegation -- the
//! `engine::bag_of_processors::Processor` precedent: a CLOSED set, static dispatch,
//! the concrete methods stay reachable, no trait-object gymnastics. Variants grow
//! with the phase: `Lstm` here, `Slstm` (Task 2) and `Mamba` (Task 3) next. The
//! existing `impl Layer for LstmLayer` in [`super::network`] STAYS (the phase-2/3
//! unit + golden suites drive `LstmLayer` directly); `CellLayer::Lstm` wraps that
//! same struct, so every f64 operation below the enum is byte-untouched -- the wrap
//! adds a match indirection and nothing else.
//!
//! ## Exact-tree discipline (spec S1.4)
//!
//! The committed golden suite must stay byte-green at EVERY commit. Phase 9 has
//! THREE sanctioned touch classes outside `cells/`, all behavior-free on LSTM
//! configs:
//!
//! - (a) the `CellLayer` enum wrap here plus the `nn::blstm` stack re-typing it
//!   forces (LSTM f64 arithmetic untouched -- the golden suite is the proof),
//! - (b) the [`Direction`](super::blstm::Direction) forward-only branches (the
//!   `nn::blstm` ctor/seam arms + `Network::feed_forward_double` /
//!   `feed_backward_double`) -- dead on every legacy/default config, since a
//!   bidirectional net always hands those methods a non-empty `second` half,
//! - (c) the `toml_config::KEY_TABLE` / config rows (spec S6).
//!
//! Anything else touching `nn/` outside `cells/` must be justified against that list
//! in review. Numeric divergence of the NEW cells is BY DESIGN (port-only code, no
//! legacy source): it is documented in module docs + `RESULTS.md`, NEVER
//! `IMPROVEMENTS.md` (which tracks legacy-quirk debt only -- the phase-7 rule).

pub mod slstm;

use ndarray::Array2;

pub use slstm::SlstmLayer;

use super::layers::LstmLayer;
use super::network::Layer;

/// The recurrent cell a `Network<CellLayer>` stacks. One variant per architecture;
/// the set is closed and statically dispatched (see the module doc).
#[derive(Clone)]
pub enum CellLayer {
    /// The legacy peephole LSTM (`nn::layers::LstmLayer`), byte-untouched.
    Lstm(LstmLayer),
}

impl Layer for CellLayer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool) {
        match self {
            CellLayer::Lstm(l) => LstmLayer::feed_forward(l, input, output, last_layer),
        }
    }

    fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        match self {
            CellLayer::Lstm(l) => LstmLayer::feed_forward_reverse(l, input, output, last_layer),
        }
    }

    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        match self {
            CellLayer::Lstm(l) => LstmLayer::feed_backward(
                l,
                input,
                output,
                deltas,
                inv_sub_sampling_ratio,
                last_layer,
            ),
        }
    }

    fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        match self {
            CellLayer::Lstm(l) => LstmLayer::feed_backward_reverse(
                l,
                input,
                output,
                deltas,
                inv_sub_sampling_ratio,
                last_layer,
            ),
        }
    }

    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        match self {
            CellLayer::Lstm(l) => LstmLayer::get_weights_derivatives(l, out),
        }
    }

    fn reset_weights_derivatives(&mut self) {
        match self {
            CellLayer::Lstm(l) => LstmLayer::reset_weights_derivatives(l),
        }
    }

    fn ponderate_weights_derivatives(&mut self, factor: f64) {
        match self {
            CellLayer::Lstm(l) => LstmLayer::ponderate_weights_derivatives(l, factor),
        }
    }

    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        match self {
            CellLayer::Lstm(l) => LstmLayer::set_weights(l, flat),
        }
    }

    fn get_weights(&self, out: &mut Vec<f64>) {
        match self {
            CellLayer::Lstm(l) => LstmLayer::get_weights(l, out),
        }
    }

    fn nb_of_weights(&self) -> usize {
        match self {
            CellLayer::Lstm(l) => LstmLayer::nb_of_weights(l),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const I: usize = 3;
    const O: usize = 2;
    const T: usize = 5;

    fn bare() -> LstmLayer {
        LstmLayer::new(I, O, true, true, true)
    }

    /// Deterministic, non-degenerate weight fill (no RNG dependency): a mildly
    /// irrational-looking ramp so no two blocks share a value.
    fn weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| 0.37 - 0.011 * (k as f64) + 0.003 * ((k % 7) as f64))
            .collect()
    }

    fn seq(rows: usize, cols: usize, seed: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(r, c)| {
            seed + 0.13 * (r as f64) - 0.29 * (c as f64) + 0.017 * ((r * cols + c) as f64)
        })
    }

    /// The wrap is an indirection ONLY: forward + reverse forward through
    /// `CellLayer::Lstm` are BIT-identical to the bare `LstmLayer` (spec S1.4 (a)).
    #[test]
    fn wrapped_forward_is_bit_identical_to_bare_lstm() {
        let w = weights(bare().nb_of_weights());
        let input = seq(T, I, 0.4);

        for reverse in [false, true] {
            let mut plain = bare();
            let mut wrapped = CellLayer::Lstm(bare());
            plain.set_weights(&w);
            Layer::set_weights(&mut wrapped, &w);

            let mut out_plain = Array2::zeros((T, O));
            let mut out_wrapped = Array2::zeros((T, O));
            if reverse {
                plain.feed_forward_reverse(&input, &mut out_plain, true);
                Layer::feed_forward_reverse(&mut wrapped, &input, &mut out_wrapped, true);
            } else {
                plain.feed_forward(&input, &mut out_plain, true);
                Layer::feed_forward(&mut wrapped, &input, &mut out_wrapped, true);
            }
            assert_eq!(
                out_plain, out_wrapped,
                "forward mismatch (reverse={reverse})"
            );
        }
    }

    /// Backward + the whole derivative seam (accumulate, harvest, ponderate, reset)
    /// delegate bit-identically, in both temporal directions.
    #[test]
    fn wrapped_backward_and_deriv_seam_are_bit_identical() {
        let w = weights(bare().nb_of_weights());
        let input = seq(T, I, 0.4);
        let deltas = seq(T, O, -0.2);

        for reverse in [false, true] {
            let mut plain = bare();
            let mut wrapped = CellLayer::Lstm(bare());
            plain.set_weights(&w);
            Layer::set_weights(&mut wrapped, &w);

            let mut out_plain = Array2::zeros((T, O));
            let mut out_wrapped = Array2::zeros((T, O));
            if reverse {
                plain.feed_forward_reverse(&input, &mut out_plain, true);
                Layer::feed_forward_reverse(&mut wrapped, &input, &mut out_wrapped, true);
            } else {
                plain.feed_forward(&input, &mut out_plain, true);
                Layer::feed_forward(&mut wrapped, &input, &mut out_wrapped, true);
            }

            let (dp, dw) = if reverse {
                (
                    plain.feed_backward_reverse(&input, &out_plain, &deltas, 1, true),
                    Layer::feed_backward_reverse(
                        &mut wrapped,
                        &input,
                        &out_wrapped,
                        &deltas,
                        1,
                        true,
                    ),
                )
            } else {
                (
                    plain.feed_backward(&input, &out_plain, &deltas, 1, true),
                    Layer::feed_backward(&mut wrapped, &input, &out_wrapped, &deltas, 1, true),
                )
            };
            assert_eq!(dp, dw, "propagated deltas mismatch (reverse={reverse})");

            plain.ponderate_weights_derivatives(0.25);
            Layer::ponderate_weights_derivatives(&mut wrapped, 0.25);
            let (mut rp, mut rw) = (Vec::new(), Vec::new());
            plain.get_weights_derivatives(&mut rp);
            Layer::get_weights_derivatives(&wrapped, &mut rw);
            assert_eq!(rp, rw, "derivative rows mismatch (reverse={reverse})");
            assert!(
                rp.iter().any(|r| r[0] != 0.0),
                "derivatives are all zero -- the test is vacuous"
            );

            plain.reset_weights_derivatives();
            Layer::reset_weights_derivatives(&mut wrapped);
            let (mut zp, mut zw) = (Vec::new(), Vec::new());
            plain.get_weights_derivatives(&mut zp);
            Layer::get_weights_derivatives(&wrapped, &mut zw);
            assert_eq!(zp, zw, "post-reset derivative rows mismatch");
        }
    }

    /// The flat weight seam (`nb_of_weights` / `set_weights` head-and-tail /
    /// `get_weights`) is identical through the wrap -- the chaining contract the
    /// `Network` container relies on.
    #[test]
    fn wrapped_weight_seam_is_bit_identical() {
        let mut plain = bare();
        let mut wrapped = CellLayer::Lstm(bare());
        assert_eq!(plain.nb_of_weights(), Layer::nb_of_weights(&wrapped));

        // A deliberately LONGER vector: `set_weights` must hand back the same tail.
        let w = weights(plain.nb_of_weights() + 9);
        let tail_plain = plain.set_weights(&w);
        let tail_wrapped = Layer::set_weights(&mut wrapped, &w);
        assert_eq!(tail_plain, tail_wrapped);
        assert_eq!(tail_plain.len(), 9);

        let (mut gp, mut gw) = (Vec::new(), Vec::new());
        plain.get_weights(&mut gp);
        Layer::get_weights(&wrapped, &mut gw);
        assert_eq!(gp, gw);
        assert_eq!(gp, w[..plain.nb_of_weights()].to_vec());
    }
}
