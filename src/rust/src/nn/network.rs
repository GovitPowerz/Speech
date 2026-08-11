//! Generic stacked-layer container over a `Layer` trait; sub-sample decimation.
//!
//! Ported from legacy C++: `NeuralNetwork.hpp` (Phase 2 Task 6). The container
//! stacks `Layer` impls, threads a chained flat weight vector through them
//! (layer-0-first head/tail split), sub-samples between layers (frame-contiguous
//! temporal stacking, dropped tail), and drives forward / reverse / double-input
//! passes. `_LayersOutput` intermediate caches are retained as fields; the Phase 3
//! backward pass (`feed_backward`/`feed_backward_reverse`/`feed_backward_double`,
//! `NeuralNetwork.hpp:251-362`) reads them and threads the `inv_sub_sample`
//! (`:136-145`) delta inflation + running `invSubSamplingRatio` through the
//! reverse-order layer loop.
//!
//! PHASE 10 (spec S7) added ONE port-only field, [`Network::retain_layers_output`], and
//! it is this file's whole footprint in that task. It is the phase's declared
//! SANCTIONED behaviour-free touch class -- the one diff inside `nn/` but outside
//! `nn/cells/` -- adjudicated in the spec up front rather than argued after the fact:
//! `_LayersOutput` is a BACKWARD-ONLY consumer (inside `drive` it is the inter-layer
//! working buffer of a single call; the only reader that outlives the call is
//! `drive_backward`, and there is no accessor), so a net whose backward can never run
//! may drop it with no observable forward difference. R2 names the committed golden
//! suite byte-green as the arbiter of that claim. See the field's own doc for the
//! mechanism and `RESULTS.md`'s Task-9 row for the memory it buys back.

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
    /// `LayerType::feedBackward(input, output, deltas, invSubSamplingRatio, lastLayer)`:
    /// accumulate this layer's weight derivatives from the reverse pass and return the
    /// deltas propagated to the previous layer (T x input_size). The layer scales its
    /// deriv blocks by `inv_sub_sampling_ratio` when > 1 (Tasks 3/4 already implement
    /// the layer-side scaling).
    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64>;
    /// `LayerType::feedBackwardReverse` (reverse-time variant). `NeuronLayer` never runs
    /// reversed and panics, mirroring `feed_forward_reverse`.
    fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64>;
    /// `LayerType::getWeightsDerivatives`: append this layer's Nx2 `[deriv | count]`
    /// rows (flat weight-packer block order, col1 = frame count replicated) to `out`.
    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>);
    /// `LayerType::resetWeightsDerivatives`: zero the deriv accumulators + frame count.
    fn reset_weights_derivatives(&mut self);
    /// `LayerType::ponderateWeightsDerivatives`: scale the deriv accumulators (col0) by
    /// `factor` in place; the frame-count column (col1) is untouched.
    fn ponderate_weights_derivatives(&mut self, factor: f64);
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
    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        LstmLayer::feed_backward(
            self,
            input,
            output,
            deltas,
            inv_sub_sampling_ratio,
            last_layer,
        )
    }
    fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        LstmLayer::feed_backward_reverse(
            self,
            input,
            output,
            deltas,
            inv_sub_sampling_ratio,
            last_layer,
        )
    }
    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        LstmLayer::get_weights_derivatives(self, out);
    }
    fn reset_weights_derivatives(&mut self) {
        LstmLayer::reset_weights_derivatives(self);
    }
    fn ponderate_weights_derivatives(&mut self, factor: f64) {
        LstmLayer::ponderate_weights_derivatives(self, factor);
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
    /// `NeuronLayer::feedBackward` takes NO `output` argument (the dense backward reads
    /// the RAW `input` for the hidden asinh-deriv fold, `NeuronLayer.cpp:204`); the
    /// container passes the layer's forward output uniformly, so the trait method drops
    /// it here.
    fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        _output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        NeuronLayer::feed_backward(self, input, deltas, inv_sub_sampling_ratio, last_layer)
    }
    /// `NeuronLayer` has no reverse variant -- the dense output MLP is never driven
    /// reversed. Panics, mirroring `feed_forward_reverse`.
    fn feed_backward_reverse(
        &mut self,
        _input: &Array2<f64>,
        _output: &Array2<f64>,
        _deltas: &Array2<f64>,
        _inv_sub_sampling_ratio: usize,
        _last_layer: bool,
    ) -> Array2<f64> {
        panic!("NeuronLayer is never run reversed (legacy never instantiates it)");
    }
    fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        NeuronLayer::get_weights_derivatives(self, out);
    }
    fn reset_weights_derivatives(&mut self) {
        NeuronLayer::reset_weights_derivatives(self);
    }
    fn ponderate_weights_derivatives(&mut self, factor: f64) {
        NeuronLayer::ponderate_weights_derivatives(self, factor);
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
///
/// `Clone` requires `L: Clone` (the derive bound); both `LstmLayer` and `NeuronLayer`
/// derive it. Enables the whole-net clone the Task 4/7 driver bags depend on.
#[derive(Clone)]
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
    /// PORT-ONLY, phase 10 spec S7 -- the phase's ONE adjudicated behaviour-free touch
    /// class inside `nn/` but outside `nn/cells/` (spec S7 declares it up front; R2 names
    /// the committed golden suite byte-green as its arbiter).
    ///
    /// `true` (the default, and every pre-phase-10 behaviour) keeps `layers_output` alive
    /// after `drive` returns; `false` clears it on the way out. Behaviour-free because
    /// `layers_output` has NO forward-path reader that outlives the call: inside `drive` it
    /// is the inter-layer working buffer (layer `jj` writes what layer `jj+1` reads in the
    /// SAME call, which is why the clear happens at the END, never in place of the writes),
    /// and the only reader that survives the call is `drive_backward`. There is no
    /// accessor, so no caller outside this module can observe the difference.
    ///
    /// Flipped at CONSTRUCTION only (`BlstmNetwork::set_inference_only`, keyed off
    /// `BackPropagationActivated` -- NEVER off `Epochs`, the F11 lesson), so a net either
    /// retains for its whole life or never does. A backward after a non-retaining forward
    /// bails LOUDLY in `drive_backward` rather than folding a stale or empty buffer.
    retain_layers_output: bool,
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
            retain_layers_output: true,
        }
    }

    /// Phase 10 spec S7: turn the `layers_output` retention off (inference-only) or back on.
    ///
    /// Turning it OFF also drops whatever the last forward left behind, so a subsequent
    /// backward hits the loud bail instead of folding a stale buffer from an earlier
    /// sequence.
    pub fn set_retain_layers_output(&mut self, retain: bool) {
        self.retain_layers_output = retain;
        if !retain {
            self.clear_layers_output();
        }
    }

    /// Whether this network retains its intermediate layer outputs (spec S7; `true` by
    /// default).
    pub fn retains_layers_output(&self) -> bool {
        self.retain_layers_output
    }

    /// Mutable access to the layer stack, so a concrete instantiation (`Network<CellLayer>`)
    /// can reach its cells' own inherent methods -- phase 10 spec S7 threads
    /// `set_retain_cache` this way, WITHOUT widening the 10-method `Layer` trait.
    pub fn layers_mut(&mut self) -> &mut [L] {
        &mut self.layers
    }

    fn clear_layers_output(&mut self) {
        for lo in &mut self.layers_output {
            *lo = Array2::zeros((0, 0));
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

        // Spec S7: the retention, and the ONE place it is skipped. Deliberately AFTER the
        // loop -- the writes above are the inter-layer working buffer this same call reads
        // back, so only their survival past the call is optional.
        if !self.retain_layers_output {
            self.clear_layers_output();
        }
    }

    /// `feedForwardDouble` (`:242-249`): hcat `first | second` into an
    /// `(rows x neuron_nb[0])` matrix (forward half on the LEFT), then normal forward.
    /// Empty `first` (0 rows) -> no-op. Requires `first.ncols() + second.ncols() ==
    /// neuron_nb[0]` and equal row counts.
    ///
    /// FORWARD-ONLY branch (port-only, phase-9 spec S1.2, sanctioned touch class
    /// (b)): a zero-COLUMN `second` means the caller has no reverse half at all (a
    /// `Direction::Forward` net), so the hcat is skipped and `first` drives the
    /// network directly -- its width already IS `neuron_nb[0]`, as the surviving
    /// assert states. Behavior-identical to hcat'ing an empty right half, minus the
    /// copy. DEAD on every legacy config: a bidirectional net's reverse stack always
    /// contributes at least one column.
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
        if second.ncols() == 0 {
            self.feed_forward(first, output);
            return;
        }
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

    /// `feedBackward` (`NeuralNetwork.hpp:251-300`): the reverse-order layer loop, each
    /// layer fed its RETAINED forward input (`input` for layer 0, `layers_output[jj-1]`
    /// otherwise) + output (`layers_output[jj]` hidden, `output_seq` final) and the
    /// running `inv_sub_sampling_ratio` (starts at 1). A sub-sampled layer (`sub_sampling
    /// [jj] > 1`) re-`sub_sample`s its stored input + output, truncated to the incoming
    /// delta row count via `.topRows(...)`, then `inv_sub_sample`s the returned deltas
    /// back to the pre-decimation resolution and multiplies the running ratio by
    /// `sub_sampling[jj]` (fed to the NEXT-earlier layer). Layer 0's `deltas_out` (already
    /// `inv_sub_sample`d) is returned. Empty input -> empty. Transcribed from the legacy;
    /// the harness `netBackwardLoop` dumps are the arbiter.
    pub fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output_seq: &Array2<f64>,
        deltas: &Array2<f64>,
    ) -> Array2<f64> {
        self.drive_backward(input, output_seq, deltas, false)
    }

    /// `feedBackwardReverse` (`NeuralNetwork.hpp:302-351`): byte-identical driver to
    /// `feed_backward`, only each per-layer step dispatches the REVERSE layer backward
    /// (the reversal lives inside the layer). Same ascending SubSample bookkeeping.
    pub fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output_seq: &Array2<f64>,
        deltas: &Array2<f64>,
    ) -> Array2<f64> {
        self.drive_backward(input, output_seq, deltas, true)
    }

    /// Shared driver for `feed_backward`/`feed_backward_reverse` (the legacy copy-pastes
    /// the two methods verbatim except the per-layer backward call). `reverse` picks the
    /// per-layer kernel.
    fn drive_backward(
        &mut self,
        input: &Array2<f64>,
        output_seq: &Array2<f64>,
        deltas: &Array2<f64>,
        reverse: bool,
    ) -> Array2<f64> {
        // Spec S7: an inference-only network dropped its intermediate outputs, so every
        // hidden layer's retained input/output is gone. Bail LOUDLY rather than fold
        // deltas against empty buffers -- a silently wrong gradient is the one failure
        // mode this class of change must never produce. (A single-layer net never touches
        // `layers_output` at all, but the bail stays unconditional: the flag says the
        // CALLER promised no backward, and honouring that promise uniformly is what makes
        // the promise checkable.)
        assert!(
            self.retain_layers_output,
            "Network::feed_backward/_reverse/_double on a network constructed inference-only \
             (retain_layers_output = false, phase-10 spec S7): the forward dropped its \
             intermediate layer outputs, so no gradient can be folded here. This net was \
             built with BackPropagationActivated off; turn backprop on for it (or call \
             set_retain_layers_output(true) before the forward) if a backward is really \
             wanted."
        );
        // :252-253 deltas_out empty, invSubSamplingRatio = 1.
        let mut deltas_out: Array2<f64> = Array2::zeros((0, 0));
        let mut inv_sub_sampling_ratio = 1usize;
        if input.nrows() == 0 {
            return deltas_out; // :254 -- empty input, empty result
        }
        let n_layers = self.neuron_nb.len() - 1;
        let step = |layer: &mut L,
                    inp: &Array2<f64>,
                    outp: &Array2<f64>,
                    din: &Array2<f64>,
                    ratio: usize,
                    last: bool| {
            if reverse {
                layer.feed_backward_reverse(inp, outp, din, ratio, last)
            } else {
                layer.feed_backward(inp, outp, din, ratio, last)
            }
        };

        if self.neuron_nb.len() == 2 {
            // Single-layer net (:255-263).
            if self.sub_sampling[0] > 1 {
                let sub = sub_sample(self.sub_sampling[0], input);
                let dr = deltas.nrows();
                let sub_top = sub.slice(ndarray::s![..dr, ..]).to_owned();
                let out_top = output_seq.slice(ndarray::s![..dr, ..]).to_owned();
                let dpl = step(
                    &mut self.layers[0],
                    &sub_top,
                    &out_top,
                    deltas,
                    inv_sub_sampling_ratio,
                    true,
                );
                deltas_out = inv_sub_sample(self.sub_sampling[0], &dpl);
                inv_sub_sampling_ratio *= self.sub_sampling[0];
            } else {
                deltas_out = step(
                    &mut self.layers[0],
                    input,
                    output_seq,
                    deltas,
                    inv_sub_sampling_ratio,
                    true,
                );
            }
            let _ = inv_sub_sampling_ratio; // consumed above; kept for source parity
            return deltas_out;
        }

        // Multi-layer reverse order (:264-296). `kk` counts up; `jj = L-2-kk` counts down.
        // The current deltas are the SEED (`deltas`) on the first iteration (last layer,
        // kk==0) and the previous `deltas_out` thereafter -- see :271/:280/:289 which all
        // read `deltas_out` for jj != L-2, `deltas` at jj == L-2.
        for kk in 0..n_layers {
            let jj = n_layers - 1 - kk; // n_layers == _NeuronNb.size()-1, so jj = L-2-kk
            if jj == 0 {
                // :268-276 first layer: input is Input/SubInput, output is layers_output[0].
                let cur = std::mem::replace(&mut deltas_out, Array2::zeros((0, 0)));
                if self.sub_sampling[0] > 1 {
                    let sub = sub_sample(self.sub_sampling[0], input);
                    let dr = cur.nrows();
                    let sub_top = sub.slice(ndarray::s![..dr, ..]).to_owned();
                    let lo = std::mem::replace(&mut self.layers_output[jj], Array2::zeros((0, 0)));
                    let lo_top = lo.slice(ndarray::s![..dr, ..]).to_owned();
                    let dpl = step(
                        &mut self.layers[jj],
                        &sub_top,
                        &lo_top,
                        &cur,
                        inv_sub_sampling_ratio,
                        true,
                    );
                    self.layers_output[jj] = lo;
                    deltas_out = inv_sub_sample(self.sub_sampling[0], &dpl);
                    inv_sub_sampling_ratio *= self.sub_sampling[0];
                } else {
                    let lo = std::mem::replace(&mut self.layers_output[jj], Array2::zeros((0, 0)));
                    deltas_out = step(
                        &mut self.layers[jj],
                        input,
                        &lo,
                        &cur,
                        inv_sub_sampling_ratio,
                        true,
                    );
                    self.layers_output[jj] = lo;
                }
            } else if jj == n_layers - 1 {
                // :277-285 last layer: input is layers_output[jj-1], output is output_seq,
                // deltas is the SEED (`deltas`), lastLayer = false.
                let prev =
                    std::mem::replace(&mut self.layers_output[jj - 1], Array2::zeros((0, 0)));
                if self.sub_sampling[jj] > 1 {
                    let sub = sub_sample(self.sub_sampling[jj], &prev);
                    let dr = deltas.nrows();
                    let sub_top = sub.slice(ndarray::s![..dr, ..]).to_owned();
                    let out_top = output_seq.slice(ndarray::s![..dr, ..]).to_owned();
                    let dpl = step(
                        &mut self.layers[jj],
                        &sub_top,
                        &out_top,
                        deltas,
                        inv_sub_sampling_ratio,
                        false,
                    );
                    deltas_out = inv_sub_sample(self.sub_sampling[jj], &dpl);
                    inv_sub_sampling_ratio *= self.sub_sampling[jj];
                } else {
                    deltas_out = step(
                        &mut self.layers[jj],
                        &prev,
                        output_seq,
                        deltas,
                        inv_sub_sampling_ratio,
                        false,
                    );
                }
                self.layers_output[jj - 1] = prev;
            } else {
                // :286-295 interior layer: input is layers_output[jj-1], output is
                // layers_output[jj], deltas is the running `deltas_out`, lastLayer = false.
                let cur = std::mem::replace(&mut deltas_out, Array2::zeros((0, 0)));
                let prev =
                    std::mem::replace(&mut self.layers_output[jj - 1], Array2::zeros((0, 0)));
                if self.sub_sampling[jj] > 1 {
                    let sub = sub_sample(self.sub_sampling[jj], &prev);
                    let dr = cur.nrows();
                    let sub_top = sub.slice(ndarray::s![..dr, ..]).to_owned();
                    let lo = std::mem::replace(&mut self.layers_output[jj], Array2::zeros((0, 0)));
                    let lo_top = lo.slice(ndarray::s![..dr, ..]).to_owned();
                    let dpl = step(
                        &mut self.layers[jj],
                        &sub_top,
                        &lo_top,
                        &cur,
                        inv_sub_sampling_ratio,
                        false,
                    );
                    self.layers_output[jj] = lo;
                    deltas_out = inv_sub_sample(self.sub_sampling[jj], &dpl);
                    inv_sub_sampling_ratio *= self.sub_sampling[jj];
                } else {
                    let lo = std::mem::replace(&mut self.layers_output[jj], Array2::zeros((0, 0)));
                    deltas_out = step(
                        &mut self.layers[jj],
                        &prev,
                        &lo,
                        &cur,
                        inv_sub_sampling_ratio,
                        false,
                    );
                    self.layers_output[jj] = lo;
                }
                self.layers_output[jj - 1] = prev;
            }
        }
        deltas_out
    }

    /// `feedBackwardDouble` (`NeuralNetwork.hpp:353-362`): hcat `first | second` (forward
    /// half on the LEFT) into an `(rows x neuron_nb[0])` matrix, run `feed_backward`, and
    /// return the split-ready `deltas_out`. Empty `first` (0 rows) -> empty result.
    ///
    /// FORWARD-ONLY branch (port-only, phase-9 spec S1.2, sanctioned touch class
    /// (b)): the `feed_forward_double` twin -- a zero-COLUMN `second` skips the hcat,
    /// and the returned `deltas_out` is `hidden`- rather than `2*hidden`-wide, so the
    /// caller routes ALL of it to the forward stack (no half-split). DEAD on every
    /// legacy config.
    pub fn feed_backward_double(
        &mut self,
        first: &Array2<f64>,
        second: &Array2<f64>,
        output_seq: &Array2<f64>,
        deltas: &Array2<f64>,
    ) -> Array2<f64> {
        if first.nrows() == 0 {
            return Array2::zeros((0, 0)); // :355
        }
        assert_eq!(first.nrows(), second.nrows(), "double: row-count mismatch");
        assert_eq!(
            first.ncols() + second.ncols(),
            self.neuron_nb[0],
            "double: first|second cols must sum to neuron_nb[0]"
        );
        if second.ncols() == 0 {
            return self.feed_backward(first, output_seq, deltas);
        }
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
        self.feed_backward(&hcat, output_seq, deltas)
    }

    /// `getWeightsDerivatives` (`NeuralNetwork.hpp:98-111`): concatenate each layer's Nx2
    /// `[deriv | count]` rows into `out`, layer-0-first (the `jj == 0` seed then vertical
    /// hcat). Same block order `set_weights`/`get_weights` walk, so col0 matches the
    /// Phase 0a flat weight packer element-for-element.
    pub fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        for layer in &self.layers {
            layer.get_weights_derivatives(out);
        }
    }

    /// `resetWeightsDerivatives` (`NeuralNetwork.hpp:113-117`): zero every layer's deriv
    /// accumulators + frame count.
    pub fn reset_weights_derivatives(&mut self) {
        for layer in &mut self.layers {
            layer.reset_weights_derivatives();
        }
    }

    /// `ponderateWeightsDerivatives` (`NeuralNetwork.hpp:119-123`): scale every layer's
    /// deriv accumulators (col0) by `factor`; frame counts (col1) are untouched.
    pub fn ponderate_weights_derivatives(&mut self, factor: f64) {
        for layer in &mut self.layers {
            layer.ponderate_weights_derivatives(factor);
        }
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

/// `NeuralNetwork<L>::InvSubSample` (`NeuralNetwork.hpp:136-145`), the inverse of
/// `sub_sample`'s column-block stacking: `T x (C*R) -> (T*R) x C`. Source row `jj`'s
/// block `[kk*C, (kk+1)*C)` un-stacks into output row `jj*R+kk`. The backward container
/// loop uses it to inflate a sub-sampled layer's returned deltas back to the
/// pre-decimation row count before feeding the layer below.
///
/// Quirk (Task 5): `sub_sample` FLOORS the input to `floor(T/R)` rows (dropping the
/// trailing `T mod R`), so a round-trip `inv_sub_sample(R, sub_sample(R, x))` yields
/// `floor(T/R)*R` rows, NOT `T` -- the dropped tail frames are never restored (this is
/// how the container's returned `deltas_out` ends up shorter than the original input
/// for odd `T`). See IMPROVEMENTS.md.
pub fn inv_sub_sample(ratio: usize, input: &Array2<f64>) -> Array2<f64> {
    let (t, c_wide) = input.dim();
    let c = c_wide / ratio;
    let mut out = Array2::zeros((t * ratio, c));
    for jj in 0..t {
        for kk in 0..ratio {
            for j in 0..c {
                out[[jj * ratio + kk, j]] = input[[jj, kk * c + j]];
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

/// Spec S1.2 / touch class (b): the `Direction::Forward` zero-width-`second` path
/// through `feed_forward_double` / `feed_backward_double`.
///
/// These assertions pin the CONTRACT (double-with-an-empty-right-half == plain), not
/// the branch: the short-circuit is pure copy elision, so deleting it leaves every
/// number here unchanged. What would fail is a branch that stopped agreeing with the
/// general path -- e.g. a half-split applied to a zero-width `second`.
#[cfg(test)]
mod forward_only_double_tests {
    use super::*;
    use crate::nn::cells::CellLayer;

    const IN: usize = 3;
    const HIDDEN: usize = 2;
    const OUT: usize = 2;
    const T: usize = 4;

    fn recurrent_net() -> Network<CellLayer> {
        let mut net = Network::<CellLayer>::new(vec![IN, HIDDEN], vec![1], |_id, i, o| {
            CellLayer::Lstm(LstmLayer::new(i, o, true, true, true))
        });
        let nb = net.nb_of_weights();
        let w: Vec<f64> = (0..nb)
            .map(|k| 0.19 - 0.006 * (k as f64) + 0.003 * ((k % 5) as f64))
            .collect();
        net.set_weights(&w);
        net
    }

    /// The output MLP a `Direction::Forward` net owns: `hidden` in, `OUT` out (the
    /// bidirectional twin would declare `2*hidden` in).
    fn narrow_output_net() -> Network<NeuronLayer> {
        let mut net = Network::<NeuronLayer>::new(vec![HIDDEN, OUT], vec![1], |_id, i, o| {
            NeuronLayer::new(i, o)
        });
        let nb = net.nb_of_weights();
        let w: Vec<f64> = (0..nb).map(|k| 0.4 - 0.05 * (k as f64)).collect();
        net.set_weights(&w);
        net
    }

    fn hidden_rows() -> Array2<f64> {
        Array2::from_shape_fn((T, HIDDEN), |(r, c)| {
            0.2 + 0.31 * (r as f64) - 0.17 * (c as f64)
        })
    }

    /// Spec S1.2 / touch class (b): a zero-COLUMN `second` short-circuits the hcat.
    /// The branch is behavior-PRESERVING -- it must agree bit-for-bit with the
    /// general path fed a zero-width right half, and with a plain `feed_forward`.
    #[test]
    fn forward_only_double_matches_the_plain_forward_bit_for_bit() {
        let rows = hidden_rows();
        let empty = Array2::<f64>::zeros((T, 0));

        let mut via_double = Array2::<f64>::zeros((T, OUT));
        narrow_output_net().feed_forward_double(&rows, &empty, &mut via_double);

        let mut via_plain = Array2::<f64>::zeros((T, OUT));
        narrow_output_net().feed_forward(&rows, &mut via_plain);

        assert_eq!(via_double, via_plain);
        // Non-vacuity: the dense layer actually did something (softmax rows).
        assert!(via_plain.iter().all(|v| v.is_finite()));
        assert!((via_plain.row(0).sum() - 1.0).abs() < 1e-12);
    }

    /// The backward twin: `feed_backward_double` with a zero-width `second` returns
    /// the FULL `hidden`-wide delta block (no half-split for the caller to make),
    /// bit-identical to `feed_backward` on the same input.
    #[test]
    fn forward_only_backward_double_returns_full_width_deltas() {
        let rows = hidden_rows();
        let empty = Array2::<f64>::zeros((T, 0));
        let deltas =
            Array2::from_shape_fn((T, OUT), |(r, c)| 0.05 * (r as f64) - 0.09 * (c as f64));

        let mut net_a = narrow_output_net();
        let mut out_a = Array2::<f64>::zeros((T, OUT));
        net_a.feed_forward_double(&rows, &empty, &mut out_a);
        let via_double = net_a.feed_backward_double(&rows, &empty, &out_a, &deltas);

        let mut net_b = narrow_output_net();
        let mut out_b = Array2::<f64>::zeros((T, OUT));
        net_b.feed_forward(&rows, &mut out_b);
        let via_plain = net_b.feed_backward(&rows, &out_b, &deltas);

        assert_eq!(via_double.dim(), (T, HIDDEN));
        assert_eq!(via_double, via_plain);
        assert!(
            via_double.iter().any(|v| *v != 0.0),
            "no deltas propagated -- the test is vacuous"
        );
    }

    /// The forward-only path is genuinely wired to a recurrent stack: `hidden`-wide
    /// rows out of a `Network<CellLayer>` feed the narrow output net directly.
    #[test]
    fn recurrent_hidden_rows_feed_the_narrow_output_net() {
        let mut stack = recurrent_net();
        let input =
            Array2::from_shape_fn((T, IN), |(r, c)| 0.3 + 0.11 * (r as f64) - 0.2 * (c as f64));
        let mut hidden = Array2::<f64>::zeros((T, HIDDEN));
        stack.feed_forward(&input, &mut hidden);
        assert_eq!(hidden.ncols(), HIDDEN);

        let mut output = Array2::<f64>::zeros((T, OUT));
        narrow_output_net().feed_forward_double(&hidden, &Array2::zeros((T, 0)), &mut output);
        assert!(output.iter().all(|v| v.is_finite()));
        // Non-vacuity: the dense head really produced a softmax row, not zeros.
        assert!((output.row(0).sum() - 1.0).abs() < 1e-12);
    }
}

// ==== Phase 10 Task 9 / spec S7: inference-only `layers_output` gating ====

#[cfg(test)]
mod retention_tests {
    use super::*;
    use crate::nn::cells::CellLayer;

    const IN: usize = 3;
    const HID: usize = 4;
    const OUT: usize = 2;
    const T: usize = 6;

    /// A MULTI-layer net: the single-layer branch of `drive` never touches
    /// `layers_output` at all, so it could not tell the two flag states apart.
    fn net() -> Network<CellLayer> {
        let mut n = Network::<CellLayer>::new(vec![IN, HID, OUT], vec![1, 1], |_id, i, o| {
            CellLayer::Lstm(LstmLayer::new(i, o, true, true, true))
        });
        let nb = n.nb_of_weights();
        let w: Vec<f64> = (0..nb)
            .map(|k| 0.21 - 0.0043 * ((k % 29) as f64) + 0.0017 * ((k % 7) as f64))
            .collect();
        n.set_weights(&w);
        n
    }

    fn input() -> Array2<f64> {
        Array2::from_shape_fn((T, IN), |(r, c)| {
            0.3 + 0.11 * (r as f64) - 0.19 * (c as f64)
        })
    }

    fn run(n: &mut Network<CellLayer>, x: &Array2<f64>) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((x.nrows(), OUT));
        n.feed_forward(x, &mut out);
        out
    }

    #[test]
    fn a_fresh_network_retains_its_layer_outputs() {
        assert!(net().retains_layers_output());
    }

    /// THE BEHAVIOUR-FREE CLAIM: `layers_output` is the inter-layer working buffer of a
    /// SINGLE `drive` call, so clearing it on the way out cannot move the output. Both
    /// drivers (forward and reverse) are covered -- they share `drive`, but the reverse
    /// leg is what proves the clear was not accidentally put inside the layer loop.
    #[test]
    fn forward_is_bit_identical_without_retention() {
        let x = input();
        let retained = run(&mut net(), &x);
        let mut lean = net();
        lean.set_retain_layers_output(false);
        assert_eq!(retained, run(&mut lean, &x));

        let mut a = net();
        let mut b = net();
        b.set_retain_layers_output(false);
        let (mut oa, mut ob) = (
            Array2::<f64>::zeros((T, OUT)),
            Array2::<f64>::zeros((T, OUT)),
        );
        a.feed_forward_reverse(&x, &mut oa);
        b.feed_forward_reverse(&x, &mut ob);
        assert_eq!(oa, ob);
    }

    /// ... and the buffers really are released (the memory claim). The retaining twin is
    /// the non-vacuity contrast: it must hold a `T x HID` block.
    #[test]
    fn an_inference_only_forward_releases_the_layer_outputs() {
        let x = input();
        let mut lean = net();
        lean.set_retain_layers_output(false);
        run(&mut lean, &x);
        assert!(lean.layers_output.iter().all(|lo| lo.dim() == (0, 0)));

        let mut keeper = net();
        run(&mut keeper, &x);
        assert_eq!(keeper.layers_output[0].dim(), (T, HID));
    }

    /// THE R6 CLONE NOTE (spec S7): the per-lane bag clone deep-copies these buffers, so
    /// under inference-only the clone cost dies with them.
    #[test]
    fn a_clone_after_an_inference_only_forward_carries_no_layer_outputs() {
        let mut lean = net();
        lean.set_retain_layers_output(false);
        run(&mut lean, &input());
        let twin = lean.clone();
        assert!(!twin.retains_layers_output());
        assert!(twin.layers_output.iter().all(|lo| lo.dim() == (0, 0)));
    }

    /// Turning retention off drops what an earlier forward left behind (no stale buffer
    /// can survive into a later backward).
    #[test]
    fn turning_retention_off_drops_the_existing_buffers() {
        let mut n = net();
        run(&mut n, &input());
        assert_eq!(n.layers_output[0].dim(), (T, HID));
        n.set_retain_layers_output(false);
        assert!(n.layers_output.iter().all(|lo| lo.dim() == (0, 0)));
    }

    #[test]
    #[should_panic(expected = "constructed inference-only")]
    fn backward_after_an_inference_only_forward_panics() {
        let x = input();
        let mut n = net();
        n.set_retain_layers_output(false);
        let out = run(&mut n, &x);
        let seed = Array2::from_shape_fn((T, OUT), |(r, c)| 0.05 * (r as f64) - 0.02 * (c as f64));
        let _ = n.feed_backward(&x, &out, &seed);
    }

    /// The reverse driver shares `drive_backward`, so it must bail the same way -- pinned
    /// because the two public entry points are what callers actually reach for.
    #[test]
    #[should_panic(expected = "constructed inference-only")]
    fn reverse_backward_after_an_inference_only_forward_panics() {
        let x = input();
        let mut n = net();
        n.set_retain_layers_output(false);
        let mut out = Array2::<f64>::zeros((T, OUT));
        n.feed_forward_reverse(&x, &mut out);
        let seed = Array2::from_shape_fn((T, OUT), |(r, c)| 0.05 * (r as f64) - 0.02 * (c as f64));
        let _ = n.feed_backward_reverse(&x, &out, &seed);
    }

    /// The contrast that makes the two panics a GATE: retention on, the backward runs and
    /// returns full-width deltas.
    #[test]
    fn backward_still_works_when_retention_is_on() {
        let x = input();
        let mut n = net();
        let out = run(&mut n, &x);
        let seed = Array2::from_shape_fn((T, OUT), |(r, c)| 0.05 * (r as f64) - 0.02 * (c as f64));
        let d = n.feed_backward(&x, &out, &seed);
        assert_eq!(d.dim(), (T, IN));
        assert!(d.iter().all(|v| v.is_finite()));
    }
}
