//! Generic stacked-layer container over a `Layer` trait; sub-sample decimation.
//!
//! Ported from legacy C++: `NeuralNetwork.hpp` (Phase 2 Task 6). The container
//! stacks `Layer` impls, threads a chained flat weight vector through them
//! (layer-0-first head/tail split), sub-samples between layers (frame-contiguous
//! temporal stacking, dropped tail), and drives forward / reverse / double-input
//! passes. `_LayersOutput` intermediate caches are retained as fields (Phase 3's
//! backward pass reads them). `InvSubSample`/`feedBackward*` are Phase 3 -- not
//! ported here.

use ndarray::Array2;

use super::layers::{LstmLayer, NeuronLayer};

/// A network layer: forward (and reverse) over a whole sequence, plus flat weight
/// (de)serialization with the chained head/tail contract.
///
/// `set_weights` consumes `flat[..nb_of_weights()]` and returns the tail slice for
/// the next layer in the stack (legacy `LayerType::setWeights` returns
/// `weights.tail(...)`). `get_weights` appends this layer's weights (same order) to
/// `out`. `feed_forward`/`feed_forward_reverse` write the caller-allocated `output`
/// (never resize the caller's buffer semantically -- they overwrite it in place).
pub trait Layer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool);
    fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    );
    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64];
    fn get_weights(&self, out: &mut Vec<f64>);
    fn nb_of_weights(&self) -> usize;
}

impl Layer for LstmLayer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool) {
        LstmLayer::feed_forward(self, input, output, last_layer);
    }
    fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        LstmLayer::feed_forward_reverse(self, input, output, last_layer);
    }
    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        LstmLayer::set_weights(self, flat)
    }
    fn get_weights(&self, out: &mut Vec<f64>) {
        LstmLayer::get_weights(self, out);
    }
    fn nb_of_weights(&self) -> usize {
        LstmLayer::nb_of_weights(self)
    }
}

impl Layer for NeuronLayer {
    fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, last_layer: bool) {
        NeuronLayer::feed_forward(self, input, output, last_layer);
    }
    /// `NeuronLayer` has no reverse variant -- the legacy `NeuralNetwork<NeuronLayer>`
    /// is only ever run forward (dense output MLPs). Panics if ever driven reversed.
    fn feed_forward_reverse(
        &mut self,
        _input: &Array2<f64>,
        _output: &mut Array2<f64>,
        _last_layer: bool,
    ) {
        panic!("NeuronLayer is never run reversed (legacy never instantiates it)");
    }
    fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        NeuronLayer::set_weights(self, flat)
    }
    fn get_weights(&self, out: &mut Vec<f64>) {
        NeuronLayer::get_weights(self, out);
    }
    fn nb_of_weights(&self) -> usize {
        NeuronLayer::nb_of_weights(self)
    }
}

/// `NeuralNetwork<LayerType>` (`NeuralNetwork.hpp:13-249`): a stack of `_NeuronNb.len()
/// - 1` layers with per-layer sub-sampling, a chained flat weight vector, and forward
/// / reverse / double-input drivers. `neuron_nb[jj]`/`neuron_nb[jj+1]` are the fan-in
/// (times `sub_sampling[jj]`) / fan-out of layer `jj`.
pub struct Network<L: Layer> {
    neuron_nb: Vec<usize>,
    sub_sampling: Vec<usize>,
    /// Cumulative sub-sampling products (`_SubSamplingRatios`): `sub_sampling_ratios
    /// [jj] = prod(sub_sampling[0..=jj])`.
    sub_sampling_ratios: Vec<usize>,
    layers: Vec<L>,
    /// Retained intermediate outputs (`_LayersOutput`), one per non-final layer; the
    /// Phase 3 backward pass reads them. Filled by `feed_forward`/`feed_forward_reverse`.
    layers_output: Vec<Array2<f64>>,
}

impl<L: Layer> Network<L> {
    /// `NeuralNetwork(conf, prep, neuronNb, subSampling, weightsSetExternally)` ctor
    /// (`:25-37`), factoring the config/prep out into a `mk(layer_id, input, output)`
    /// factory: layer `jj` has input `neuron_nb[jj]*sub_sampling[jj]`, output
    /// `neuron_nb[jj+1]`. `sub_sampling_ratios` are the cumulative products (`:31-36`).
    ///
    /// The ILL-FORMED legacy copy ctor (`:39-51`) is deliberately NOT ported -- it
    /// compiles only because it is never instantiated.
    pub fn new(
        neuron_nb: Vec<usize>,
        sub_sampling: Vec<usize>,
        mut mk: impl FnMut(usize, usize, usize) -> L,
    ) -> Self {
        assert!(
            neuron_nb.len() >= 2,
            "network needs >= 2 neuron-count entries"
        );
        assert_eq!(
            sub_sampling.len(),
            neuron_nb.len() - 1,
            "sub_sampling must have one entry per layer"
        );
        let n_layers = neuron_nb.len() - 1;
        let mut layers = Vec::with_capacity(n_layers);
        let mut layers_output = Vec::with_capacity(n_layers);
        for jj in 0..n_layers {
            layers.push(mk(jj, neuron_nb[jj] * sub_sampling[jj], neuron_nb[jj + 1]));
            layers_output.push(Array2::zeros((0, 0)));
        }
        // :31-36 cumulative products: sub_sampling_ratios[jj] = prod(sub_sampling[0..=jj]).
        // Kept as the legacy's index-nested double loop for a line-for-line source match
        // (an iterator rewrite obscures the `ll <= jj` triangular accumulation).
        let mut sub_sampling_ratios = vec![1usize; sub_sampling.len()];
        #[allow(clippy::needless_range_loop)]
        for jj in 0..sub_sampling.len() {
            for ll in 0..=jj {
                sub_sampling_ratios[jj] *= sub_sampling[ll];
            }
        }
        Network {
            neuron_nb,
            sub_sampling,
            sub_sampling_ratios,
            layers,
            layers_output,
        }
    }

    /// `getInputSize` (`:53-55`): `neuron_nb[0]`.
    pub fn input_size(&self) -> usize {
        self.neuron_nb[0]
    }

    /// `getOutputSize` (`:57-59`): `neuron_nb.last()`.
    pub fn output_size(&self) -> usize {
        self.neuron_nb[self.neuron_nb.len() - 1]
    }

    /// `getSubSampling` (`:61-63`): the raw per-layer sub-sampling factors.
    pub fn sub_samplings(&self) -> &[usize] {
        &self.sub_sampling
    }

    /// `getSubSamplingRatio` (`:65-67`): the LAST cumulative product (overall decimation).
    pub fn sub_sampling_ratio(&self) -> usize {
        self.sub_sampling_ratios[self.sub_sampling_ratios.len() - 1]
    }

    /// `getNbOfWeights` (`:79-85`): sum over layers.
    pub fn nb_of_weights(&self) -> usize {
        self.layers.iter().map(|l| l.nb_of_weights()).sum()
    }

    /// `setWeights` (`:69-77`): thread `flat` through the layers layer-0-first, each
    /// consuming its `nb_of_weights()` head and passing the tail on. Returns the
    /// leftover tail (the legacy returns the final `all`).
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        let mut rest = flat;
        for layer in &mut self.layers {
            rest = layer.set_weights(rest);
        }
        rest
    }

    /// `getWeights` (`:87-96`): concatenate every layer's weights, layer-0-first,
    /// same block order as `set_weights` consumes.
    pub fn get_weights(&self, out: &mut Vec<f64>) {
        for layer in &self.layers {
            layer.get_weights(out);
        }
    }

    /// `feedForward` (`:158-198`). EMPTY input (0 rows) -> silent no-op (`output`
    /// untouched, `:159`). Single-layer net: optional input sub-sample then layer
    /// forward `last_layer=true`. Multi-layer ascending: layer 0 (optionally
    /// sub-sampled input) -> `layers_output[0]` (`last_layer=false`); middle layers
    /// same over `layers_output[jj-1]`; the FINAL layer writes the caller-allocated
    /// `output` (`last_layer=true`). `sub_sampling[jj] > 1` sub-samples that layer's
    /// input first.
    pub fn feed_forward(&mut self, input: &Array2<f64>, output: &mut Array2<f64>) {
        self.drive(input, output, false);
    }

    /// `feedForwardReverse` (`:200-240`): identical driver to `feed_forward`, but each
    /// layer step dispatches `feed_forward_reverse` (the reversal lives inside the
    /// layer). Ascending layer order (only the per-layer temporal direction flips).
    pub fn feed_forward_reverse(&mut self, input: &Array2<f64>, output: &mut Array2<f64>) {
        self.drive(input, output, true);
    }

    /// Shared driver for `feed_forward`/`feed_forward_reverse` (the legacy copy-pastes
    /// the two methods verbatim except the layer call). `reverse` picks the per-layer
    /// kernel.
    fn drive(&mut self, input: &Array2<f64>, output: &mut Array2<f64>, reverse: bool) {
        if input.nrows() == 0 {
            return; // :159 / :201 -- empty input is a no-op
        }
        let n_layers = self.neuron_nb.len() - 1;
        let step = |layer: &mut L, inp: &Array2<f64>, outp: &mut Array2<f64>, last: bool| {
            if reverse {
                layer.feed_forward_reverse(inp, outp, last);
            } else {
                layer.feed_forward(inp, outp, last);
            }
        };

        if self.neuron_nb.len() == 2 {
            // Single-layer net (:160-166 / :202-208).
            if self.sub_sampling[0] > 1 {
                let sub = sub_sample(self.sub_sampling[0], input);
                step(&mut self.layers[0], &sub, output, true);
            } else {
                step(&mut self.layers[0], input, output, true);
            }
            return;
        }

        // Multi-layer ascending (:167-196 / :209-238). `neuron_nb[jj+1]` sizes the
        // intermediate output buffer; the layer forward overwrites it in place.
        for jj in 0..n_layers {
            let out_cols = self.neuron_nb[jj + 1];
            if jj == 0 {
                if self.sub_sampling[0] > 1 {
                    let sub = sub_sample(self.sub_sampling[0], input);
                    let mut buf = Array2::zeros((sub.nrows(), out_cols));
                    step(&mut self.layers[0], &sub, &mut buf, false);
                    self.layers_output[0] = buf;
                } else {
                    let mut buf = Array2::zeros((input.nrows(), out_cols));
                    step(&mut self.layers[0], input, &mut buf, false);
                    self.layers_output[0] = buf;
                }
            } else if jj == n_layers - 1 {
                let prev =
                    std::mem::replace(&mut self.layers_output[jj - 1], Array2::zeros((0, 0)));
                if self.sub_sampling[jj] > 1 {
                    let sub = sub_sample(self.sub_sampling[jj], &prev);
                    step(&mut self.layers[jj], &sub, output, true);
                } else {
                    step(&mut self.layers[jj], &prev, output, true);
                }
                self.layers_output[jj - 1] = prev;
            } else {
                let prev =
                    std::mem::replace(&mut self.layers_output[jj - 1], Array2::zeros((0, 0)));
                let (buf, restored) = if self.sub_sampling[jj] > 1 {
                    let sub = sub_sample(self.sub_sampling[jj], &prev);
                    let mut buf = Array2::zeros((sub.nrows(), out_cols));
                    step(&mut self.layers[jj], &sub, &mut buf, false);
                    (buf, prev)
                } else {
                    let mut buf = Array2::zeros((prev.nrows(), out_cols));
                    step(&mut self.layers[jj], &prev, &mut buf, false);
                    (buf, prev)
                };
                self.layers_output[jj - 1] = restored;
                self.layers_output[jj] = buf;
            }
        }
    }

    /// `feedForwardDouble` (`:242-249`): hcat `first | second` into an
    /// `(rows x neuron_nb[0])` matrix (forward half on the LEFT), then normal forward.
    /// Empty `first` (0 rows) -> no-op. Requires `first.ncols() + second.ncols() ==
    /// neuron_nb[0]` and equal row counts.
    pub fn feed_forward_double(
        &mut self,
        first: &Array2<f64>,
        second: &Array2<f64>,
        output: &mut Array2<f64>,
    ) {
        if first.nrows() == 0 {
            return; // :243
        }
        assert_eq!(first.nrows(), second.nrows(), "double: row-count mismatch");
        assert_eq!(
            first.ncols() + second.ncols(),
            self.neuron_nb[0],
            "double: first|second cols must sum to neuron_nb[0]"
        );
        let rows = first.nrows();
        let mut hcat = Array2::zeros((rows, self.neuron_nb[0]));
        let fc = first.ncols();
        for t in 0..rows {
            for j in 0..fc {
                hcat[[t, j]] = first[[t, j]];
            }
            for j in 0..second.ncols() {
                hcat[[t, fc + j]] = second[[t, j]];
            }
        }
        self.feed_forward(&hcat, output);
    }
}

/// `NeuralNetwork<L>::SubSample` (`NeuralNetwork.hpp:125-134`). `T x C -> floor(T/R) x
/// C*R`: the trailing `T mod R` rows are DROPPED (integer floor); source row `jj*R+kk`
/// lands in output columns `[kk*C, (kk+1)*C)` of row `jj` (frame-contiguous temporal
/// stacking). `R == 1` is a plain copy (`floor(T/1) == T`, single frame block).
pub fn sub_sample(ratio: usize, input: &Array2<f64>) -> Array2<f64> {
    let (t, c) = input.dim();
    let sub_len = t / ratio; // floor: trailing t mod ratio rows dropped
    let mut out = Array2::zeros((sub_len, c * ratio));
    for jj in 0..sub_len {
        for kk in 0..ratio {
            let src = jj * ratio + kk;
            for j in 0..c {
                out[[jj, kk * c + j]] = input[[src, j]];
            }
        }
    }
    out
}

/// `NeuralNetwork<L>::Repeat` (`NeuralNetwork.hpp:147-156`): each input row duplicated
/// `n` times consecutively (`T x C -> T*n x C`, row `jj` at output rows `[jj*n,
/// (jj+1)*n)`). Not used by the forward drivers; ported for the Phase 2b callers.
pub fn repeat_rows(input: &Array2<f64>, n: usize) -> Array2<f64> {
    let (t, c) = input.dim();
    let mut out = Array2::zeros((t * n, c));
    for jj in 0..t {
        for kk in 0..n {
            for j in 0..c {
                out[[jj * n + kk, j]] = input[[jj, j]];
            }
        }
    }
    out
}
