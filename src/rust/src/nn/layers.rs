//! Layer implementations: peephole LSTM, SRN, Clockwork-RNN, dense, conv.
//!
//! Ported from legacy C++: LSTMLayer.*, SRNLayer.*, CWRNNLayer.*, NeuronLayer.*,
//! Convolutional*.*. Phase 2.

use ndarray::Array2;

use super::activations::{
    gates_deriv, gates_fn, identity_fn, logistic_fn, maxmin2_deriv, maxmin2_fn,
};

/// Ascending-loop matrix product `a (m x k) * b (k x n)`, accumulating the `k`
/// index in strictly ascending order (`i`/`j` outer, `k` inner). This is the
/// measured product-order contract for every NN product site (Task 1 probes: Eigen
/// DIVERGES from ascending loops on the LSTM input-projection shape, and is bit-
/// exact on the recurrence row-product; the harness reimpl uses `matSeq` for both,
/// so the Rust port mirrors it uniformly). Matches `main.cpp`'s `matSeq`.
pub(crate) fn matmul_seq(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = a.dim();
    let (kb, n) = b.dim();
    debug_assert_eq!(k, kb, "matmul_seq inner dim mismatch");
    let mut out = Array2::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for kk in 0..k {
                acc += a[[i, kk]] * b[[kk, j]];
            }
            out[[i, j]] = acc;
        }
    }
    out
}

/// Peephole LSTM layer (legacy `LSTMLayer`). Task 3 ports the weight layout +
/// flat (de)serialization; forward/backward (Task 4+) are not yet implemented.
///
/// Member sizes are the CTOR RESIZES (`LSTMLayer.cpp:19-22`), NOT the stale size
/// comments in `LSTMLayer.h:35-36` (which claim `_PeepWeight` is `3*_OutputSize`
/// and predate a peephole-bundle rework -- the real ctor resizes it to
/// `12 x _OutputSize`):
/// - `input_weights`: `_InputSize x 4*_OutputSize` (gate column blocks `[i|f|o|g]`).
/// - `feedback_weights`: `_OutputSize x 4*_OutputSize` (same gate blocks, recurrent).
/// - `peep_weight`: `12 x _OutputSize`.
/// - `biases`: `1 x 4*_OutputSize` (row vector; stored as `Array2<f64>` for a
///   uniform (de)serialization loop with the other three blocks).
#[derive(Clone)]
pub struct LstmLayer {
    input_size: usize,
    output_size: usize,
    cells_peep: bool,
    gates_peep: bool,
    gates_rec_peep: bool,

    input_weights: Array2<f64>,
    feedback_weights: Array2<f64>,
    peep_weight: Array2<f64>,
    biases: Array2<f64>,

    // Forward-pass caches (`_Gates`/`_CellsIn`/`_CellStates`), filled by
    // `feed_forward`; the backward pass reads them. `_Gates` holds the
    // POST-activation gate values (T x 4O).
    gates: Array2<f64>,
    cells_in: Array2<f64>,
    cell_states: Array2<f64>,

    // Backward-pass deriv accumulators (`_InputWeightsDerivatives` I x 4O,
    // `_FeedbackWeightsDerivatives` O x 4O, `_PeepWeightDerivatives` 12 x O,
    // `_BiaisesDerivatives` 1 x 4O; `LSTMLayer.cpp:518-522`) + frame count
    // (`_NbOfSeqFedBackward`). Reset via `reset_weights_derivatives`, accumulated
    // across `feed_backward` calls, harvested via `get_weights_derivatives`.
    input_weights_derivatives: Array2<f64>,
    feedback_weights_derivatives: Array2<f64>,
    peep_weight_derivatives: Array2<f64>,
    biases_derivatives: Array2<f64>,
    nb_of_seq_fed_backward: i64,
}

impl LstmLayer {
    /// New layer with zeroed weights (legacy ctor with `weightsSetExternally =
    /// true` skips the config-key/random-table fill; callers set weights via
    /// `set_weights`). `cells_peep`/`gates_peep`/`gates_rec_peep` mirror
    /// `_isCellsPeepholesActive`/`_isGatesPeepholesActive`/
    /// `_isGatesReccurentPeepholesActive` (`LSTMLayer.cpp:11-13`), consumed by
    /// the Task 4 forward pass.
    pub fn new(
        input_size: usize,
        output_size: usize,
        cells_peep: bool,
        gates_peep: bool,
        gates_rec_peep: bool,
    ) -> Self {
        LstmLayer {
            input_size,
            output_size,
            cells_peep,
            gates_peep,
            gates_rec_peep,
            input_weights: Array2::zeros((input_size, 4 * output_size)),
            feedback_weights: Array2::zeros((output_size, 4 * output_size)),
            peep_weight: Array2::zeros((12, output_size)),
            biases: Array2::zeros((1, 4 * output_size)),
            gates: Array2::zeros((0, 0)),
            cells_in: Array2::zeros((0, 0)),
            cell_states: Array2::zeros((0, 0)),
            input_weights_derivatives: Array2::zeros((input_size, 4 * output_size)),
            feedback_weights_derivatives: Array2::zeros((output_size, 4 * output_size)),
            peep_weight_derivatives: Array2::zeros((12, output_size)),
            biases_derivatives: Array2::zeros((1, 4 * output_size)),
            nb_of_seq_fed_backward: 0,
        }
    }

    /// `getNbOfWeights` (`LSTMLayer.cpp:205-207`):
    /// `4*_InputSize*_OutputSize + _OutputSize*4*_OutputSize + 12*_OutputSize + 4*_OutputSize`.
    pub fn nb_of_weights(&self) -> usize {
        4 * self.input_size * self.output_size
            + 4 * self.output_size * self.output_size
            + 12 * self.output_size
            + 4 * self.output_size
    }

    /// `setWeights` (`LSTMLayer.cpp:162-203`): consumes `flat[..nb_of_weights()]`
    /// into the four blocks in order (InputWeights, FeedbackWeights, PeepWeight,
    /// Biaises), each read in COLUMN-major element order (`needed(previousSize +
    /// jj*currentRows + ii)`, `jj` outer/cols, `ii` inner/rows), and returns the
    /// tail slice (`weights.tail(weights.size()-length)`) for the next layer to
    /// consume in a chain.
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        let nb = self.nb_of_weights();
        let (needed, left) = flat.split_at(nb);

        let mut pos = 0usize;
        fill_col_major(&mut self.input_weights, needed, &mut pos);
        fill_col_major(&mut self.feedback_weights, needed, &mut pos);
        fill_col_major(&mut self.peep_weight, needed, &mut pos);
        fill_col_major(&mut self.biases, needed, &mut pos);

        left
    }

    /// `getWeights` (`LSTMLayer.cpp:209-249`): the mirror of `set_weights`, same
    /// block order and column-major element order, appended into `out`.
    pub fn get_weights(&self, out: &mut Vec<f64>) {
        drain_col_major(&self.input_weights, out);
        drain_col_major(&self.feedback_weights, out);
        drain_col_major(&self.peep_weight, out);
        drain_col_major(&self.biases, out);
    }

    /// `feedForward` (`LSTMLayer.cpp:312-413`): whole-sequence peephole-LSTM forward
    /// pass. Fills `output` (T x O) and the `self.gates`/`self.cell_states`/
    /// `self.cells_in` caches (T x 4O / T x O / T x O). `last_layer` is UNUSED in the
    /// LSTM forward (kept for signature parity with the legacy). Op order + FP
    /// grouping follow spec S4.3 verbatim -- do NOT reorder or split the combined
    /// gates-peep expressions (`:362-363`), which are single fused elementwise
    /// passes, not two adds.
    ///
    /// - Input projection with width tolerance (`:313-319`): `cols > I` uses the
    ///   left `I` input columns; `cols < I` uses the top `cols` weight rows; else the
    ///   full product. All via `matmul_seq` (the measured ascending-loop contract).
    /// - Peephole families gated by the flags: rows 0,1,2 (`cells_peep`),
    ///   4,5,6,8,9,10 (`gates_peep`), 3,7,11 (`gates_rec_peep`).
    /// - `t=0` has NO forget contribution and NO row-11 term.
    /// - All `t-1` gate reads are POST-activation (the gates cache is activated in
    ///   place before the next step reads it).
    pub fn feed_forward(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        let _ = last_layer;
        let o = self.output_size;
        let i = self.input_size;
        let cols = input.dim().1;

        // :313-319 input projection with width tolerance.
        let proj = if cols > i {
            matmul_seq(
                &input.slice(ndarray::s![.., ..i]).to_owned(),
                &self.input_weights,
            )
        } else if cols < i {
            matmul_seq(
                input,
                &self.input_weights.slice(ndarray::s![..cols, ..]).to_owned(),
            )
        } else {
            matmul_seq(input, &self.input_weights)
        };
        let t_len = proj.dim().0;

        // .rowwise() + _Biaises into the gates cache.
        let mut gates = proj;
        for t in 0..t_len {
            for j in 0..4 * o {
                gates[[t, j]] += self.biases[[0, j]];
            }
        }
        let mut cell_states = Array2::<f64>::zeros((t_len, o));
        let mut cells_in = Array2::<f64>::zeros((t_len, o));
        output.fill(0.0);

        let p = &self.peep_weight;

        // --- t = 0 (:325-348) ------------------------------------------------
        if t_len > 0 {
            // :333 activate i,f (GatesFunction); :334 activate g (Maxmin2).
            for j in 0..2 * o {
                gates[[0, j]] = gates_fn(gates[[0, j]]);
            }
            for j in 0..o {
                gates[[0, 3 * o + j]] = maxmin2_fn(gates[[0, 3 * o + j]]);
            }
            // :336 c_0 = i_0 .* g_0 (NO forget term).
            for j in 0..o {
                cell_states[[0, j]] = gates[[0, j]] * gates[[0, 3 * o + j]];
            }
            // :338-342 o extras IN ORDER: c_0.*P[2]; i_0.*P[9]; f_0.*P[10].
            if self.cells_peep {
                for j in 0..o {
                    gates[[0, 2 * o + j]] += cell_states[[0, j]] * p[[2, j]];
                }
            }
            if self.gates_peep {
                for j in 0..o {
                    gates[[0, 2 * o + j]] += gates[[0, j]] * p[[9, j]];
                }
                for j in 0..o {
                    gates[[0, 2 * o + j]] += gates[[0, o + j]] * p[[10, j]];
                }
            }
            // :346 activate o; :347 cells_in = asinh(c_0); :348 y_0 = o_0 .* cells_in_0.
            for j in 0..o {
                gates[[0, 2 * o + j]] = gates_fn(gates[[0, 2 * o + j]]);
            }
            for j in 0..o {
                cells_in[[0, j]] = identity_fn(cell_states[[0, j]]);
            }
            for j in 0..o {
                output[[0, j]] = gates[[0, 2 * o + j]] * cells_in[[0, j]];
            }
        }

        // --- t >= 1 (:350-412) -----------------------------------------------
        for t in 1..t_len {
            // :351 gates.row(t) += y_{t-1} * FeedbackWeights (all four blocks).
            let prev_out = output.slice(ndarray::s![t - 1..t, ..]).to_owned();
            let rec = matmul_seq(&prev_out, &self.feedback_weights); // 1 x 4O
            for j in 0..4 * o {
                gates[[t, j]] += rec[[0, j]];
            }
            // :353-356 cells peep into i,f: i += c_{t-1}.*P[0]; f += c_{t-1}.*P[1].
            if self.cells_peep {
                for j in 0..o {
                    gates[[t, j]] += cell_states[[t - 1, j]] * p[[0, j]];
                }
                for j in 0..o {
                    gates[[t, o + j]] += cell_states[[t - 1, j]] * p[[1, j]];
                }
            }
            // :357-360 gates-rec into i,f: i += i_{t-1}.*P[3]; f += f_{t-1}.*P[7].
            if self.gates_rec_peep {
                for j in 0..o {
                    gates[[t, j]] += gates[[t - 1, j]] * p[[3, j]];
                }
                for j in 0..o {
                    gates[[t, o + j]] += gates[[t - 1, o + j]] * p[[7, j]];
                }
            }
            // :362-363 gates peep, each ONE COMBINED expression (single fused pass):
            //   i += (f_{t-1}.*P[4] + o_{t-1}.*P[5]); f += (i_{t-1}.*P[6] + o_{t-1}.*P[8]).
            if self.gates_peep {
                for j in 0..o {
                    gates[[t, j]] +=
                        gates[[t - 1, o + j]] * p[[4, j]] + gates[[t - 1, 2 * o + j]] * p[[5, j]];
                }
                for j in 0..o {
                    gates[[t, o + j]] +=
                        gates[[t - 1, j]] * p[[6, j]] + gates[[t - 1, 2 * o + j]] * p[[8, j]];
                }
            }
            // :370 activate i,f; :385 activate g.
            for j in 0..2 * o {
                gates[[t, j]] = gates_fn(gates[[t, j]]);
            }
            for j in 0..o {
                gates[[t, 3 * o + j]] = maxmin2_fn(gates[[t, 3 * o + j]]);
            }
            // :387 c_t = i_t.*g_t + c_{t-1}.*f_t (single combined expression).
            for j in 0..o {
                cell_states[[t, j]] = gates[[t, j]] * gates[[t, 3 * o + j]]
                    + cell_states[[t - 1, j]] * gates[[t, o + j]];
            }
            // :389-394 o extras IN ORDER: c_t.*P[2]; i_t.*P[9]; f_t.*P[10]; o_{t-1}.*P[11].
            if self.cells_peep {
                for j in 0..o {
                    gates[[t, 2 * o + j]] += cell_states[[t, j]] * p[[2, j]];
                }
            }
            if self.gates_peep {
                for j in 0..o {
                    gates[[t, 2 * o + j]] += gates[[t, j]] * p[[9, j]];
                }
                for j in 0..o {
                    gates[[t, 2 * o + j]] += gates[[t, o + j]] * p[[10, j]];
                }
            }
            if self.gates_rec_peep {
                for j in 0..o {
                    gates[[t, 2 * o + j]] += gates[[t - 1, 2 * o + j]] * p[[11, j]];
                }
            }
            // :398 activate o; :399 cells_in = asinh(c_t); :400 y_t = o_t .* cells_in_t.
            for j in 0..o {
                gates[[t, 2 * o + j]] = gates_fn(gates[[t, 2 * o + j]]);
            }
            for j in 0..o {
                cells_in[[t, j]] = identity_fn(cell_states[[t, j]]);
            }
            for j in 0..o {
                output[[t, j]] = gates[[t, 2 * o + j]] * cells_in[[t, j]];
            }
        }

        self.gates = gates;
        self.cell_states = cell_states;
        self.cells_in = cells_in;
    }

    /// Post-activation gates cache (`_Gates`, T x 4O) from the last forward pass.
    /// Exposed for golden tests (the harness dumps this) and the Task 5+ backward
    /// pass. Column blocks are `[i|f|o|g]`; after a reverse pass the rows are in
    /// reversed-input order (see `feed_forward_reverse`).
    pub fn gates(&self) -> &Array2<f64> {
        &self.gates
    }

    /// Cell states cache (`_CellStates`, T x O) from the last forward pass.
    pub fn cell_states(&self) -> &Array2<f64> {
        &self.cell_states
    }

    /// Cell-input (`asinh`) cache (`_CellsIn`, T x O) from the last forward pass.
    pub fn cells_in(&self) -> &Array2<f64> {
        &self.cells_in
    }

    /// `feedForwardReverse` (`LSTMLayer.cpp:415-421`): reverse input rows, run
    /// `feed_forward`, reverse output rows. The `self.gates`/`self.cell_states`/
    /// `self.cells_in` caches are left in REVERSED-input order (exactly as the legacy
    /// leaves `_Gates` after `feedForwardReverse` -- it never un-reverses them);
    /// only `output` is un-reversed to match `outputSeq`.
    pub fn feed_forward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        let input_rev = input.slice(ndarray::s![..;-1, ..]).to_owned();
        let mut output_rev = Array2::<f64>::zeros(output.dim());
        self.feed_forward(&input_rev, &mut output_rev, last_layer);
        output.assign(&output_rev.slice(ndarray::s![..;-1, ..]));
    }

    /// `feedBackward` (`LSTMLayer.cpp:518-723`): whole-sequence peephole-LSTM BPTT.
    /// Reverse-time loop (`row = T-1 .. 0`); per row the gate-derivative order is
    /// O -> state -> C -> F -> I (`:582/613/616/624/659`). Derivatives are recomputed
    /// FROM the post-activation caches (`self.gates`/`cell_states`/`cells_in`) via the
    /// activation derivatives (`gates_deriv`/`maxmin2_deriv`, each on the CACHED
    /// activated value). Returns `deltas_previous_layer` (T x I). Accumulates into the
    /// four deriv members and `nb_of_seq_fed_backward += deltas.nrows()` (`:720`).
    ///
    /// The 1xO gate-delta vectors (`deltas_{input,forget,output}_gate`) carry state
    /// ACROSS reverse-time iterations: at each row they still hold the PRIOR (later
    /// time) step's values when the peephole cross-terms read them, and are only then
    /// overwritten with the current row's. The `deltas_forget_gate_tmp` save (`:624`)
    /// captures the prior forget-gate delta BEFORE the forget block overwrites it, for
    /// the input-gate block to read at `:660`.
    ///
    /// Products: the two NN-wide-k GEMMs (`(_FeedbackWeights*test)^T` `:703`,
    /// `(_InputWeights*testM)^T` `:707`) go through `matmul_seq` (the measured
    /// ascending-loop contract); the rank-1 outer-product deriv accumulations
    /// (`input.col(row)*deltasX`, `output.col(row-1)*deltasX`) are k=1, done as
    /// explicit scalar loops (bit-identical to the legacy Eigen product at k=1).
    pub fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        let _ = last_layer; // lastLayer is UNUSED in the LSTM backward (signature parity).
        let o = self.output_size;
        let i = self.input_size;
        let cols = input.dim().1;
        let lines_nb = deltas.dim().0; // :578 linesNb = deltas.rows().

        // :534-543 width tolerance. Legacy transposes to I x T; the Rust port keeps
        // the sequence T x I and reads `recon_input.row(row)` where the legacy reads
        // `input.col(row)`. cols > I -> leftCols(I); cols < I -> zero-PAD columns up
        // to I (the input matrix itself is padded, NOT truncated -- differs from the
        // dense width-recon only in that both there and here the accumulation half
        // reads the reconstructed tensor); else as-is.
        let recon_input: Array2<f64> = if cols > i {
            input.slice(ndarray::s![.., ..i]).to_owned()
        } else if cols < i {
            let mut padded = Array2::<f64>::zeros((lines_nb, i));
            padded.slice_mut(ndarray::s![.., ..cols]).assign(input);
            padded
        } else {
            input.to_owned()
        };

        let p = &self.peep_weight;
        let g = &self.gates;
        let cs = &self.cell_states;
        let ci = &self.cells_in;

        // Per-call accumulators (`:559-562`), added into the members at the end.
        let mut iw_d = Array2::<f64>::zeros((i, 4 * o));
        let mut fw_d = Array2::<f64>::zeros((o, 4 * o));
        let mut pp_d = Array2::<f64>::zeros((12, o));
        let mut bs_d = Array2::<f64>::zeros((1, 4 * o));

        // 1xO state vectors persisting across the reverse-time loop (`:547-557`).
        let mut deltas_feedback = vec![0.0f64; o];
        let mut deltas_input_gate = vec![0.0f64; o];
        let mut deltas_forget_gate = vec![0.0f64; o];
        let mut deltas_output_gate = vec![0.0f64; o];
        let mut deltas_cells = vec![0.0f64; o];
        // epsilonState from the NEXT (later) step, read by epsilonForget at :608.
        let mut epsilon_state = vec![0.0f64; o];

        // testM (4O x T) stacks per-row [i|f|o|c] gate deltas for the :707 GEMM.
        let mut test_m = Array2::<f64>::zeros((4 * o, lines_nb));

        for row in (0..lines_nb).rev() {
            let last_row = row == lines_nb - 1;

            // :580 epsilonCells = deltas.row(row) + deltasFeedback.
            let mut epsilon_cells = vec![0.0f64; o];
            for j in 0..o {
                epsilon_cells[j] = deltas[[row, j]] + deltas_feedback[j];
            }

            // :582-585 deltasOutputGate. tmp = _CellsIn.row(row) .* epsilonCells,
            // += gatesPeep (rows 5/8 from PRIOR i/f deltas), += gatesRecPeep (row 11
            // from PRIOR o delta), then * GatesFunction'(_Gates[o-block]).
            let mut tmp = vec![0.0f64; o];
            for j in 0..o {
                tmp[j] = ci[[row, j]] * epsilon_cells[j];
            }
            if self.gates_peep {
                for j in 0..o {
                    tmp[j] += deltas_input_gate[j] * p[[5, j]] + deltas_forget_gate[j] * p[[8, j]];
                }
            }
            if self.gates_rec_peep {
                for j in 0..o {
                    tmp[j] += deltas_output_gate[j] * p[[11, j]];
                }
            }
            for j in 0..o {
                deltas_output_gate[j] = gates_deriv(g[[row, 2 * o + j]]) * tmp[j];
            }

            // :595-605 output-gate deriv accumulation.
            for ii in 0..i {
                for j in 0..o {
                    iw_d[[ii, 2 * o + j]] += recon_input[[row, ii]] * deltas_output_gate[j];
                }
            }
            if row > 0 {
                for ii in 0..o {
                    for j in 0..o {
                        fw_d[[ii, 2 * o + j]] += output[[row - 1, ii]] * deltas_output_gate[j];
                    }
                }
                if self.gates_rec_peep {
                    for j in 0..o {
                        pp_d[[11, j]] += deltas_output_gate[j] * g[[row - 1, 2 * o + j]];
                    }
                }
            }
            if self.cells_peep {
                for j in 0..o {
                    pp_d[[2, j]] += deltas_output_gate[j] * cs[[row, j]];
                }
            }
            if self.gates_peep {
                for j in 0..o {
                    pp_d[[9, j]] += deltas_output_gate[j] * g[[row, j]];
                }
                for j in 0..o {
                    pp_d[[10, j]] += deltas_output_gate[j] * g[[row, o + j]];
                }
            }
            for j in 0..o {
                bs_d[[0, 2 * o + j]] += deltas_output_gate[j];
            }

            // :607-611 epsilonForget from the NEXT step's forget gate .* NEXT epsilonState.
            let mut epsilon_forget = vec![0.0f64; o];
            if !last_row {
                for j in 0..o {
                    epsilon_forget[j] = g[[row + 1, o + j]] * epsilon_state[j];
                }
            }

            // :613-614 epsilonState = _Gates[o].*Identity'(_CellsIn).*epsilonCells
            //   + epsilonForget + cellsPeep(rows 0/1/2 from PRIOR i/f/o deltas).
            let mut tmp_epsilon_state = vec![0.0f64; o];
            if self.cells_peep {
                for j in 0..o {
                    tmp_epsilon_state[j] = deltas_input_gate[j] * p[[0, j]]
                        + deltas_forget_gate[j] * p[[1, j]]
                        + deltas_output_gate[j] * p[[2, j]];
                }
            }
            for j in 0..o {
                epsilon_state[j] =
                    g[[row, 2 * o + j]] * maxmin2_deriv(ci[[row, j]]) * epsilon_cells[j]
                        + epsilon_forget[j]
                        + tmp_epsilon_state[j];
            }

            // :616 deltasCells = _Gates[i] .* Maxmin2'(_Gates[g-block]) .* epsilonState.
            for j in 0..o {
                deltas_cells[j] =
                    g[[row, j]] * maxmin2_deriv(g[[row, 3 * o + j]]) * epsilon_state[j];
            }
            for ii in 0..i {
                for j in 0..o {
                    iw_d[[ii, 3 * o + j]] += recon_input[[row, ii]] * deltas_cells[j];
                }
            }
            if row > 0 {
                for ii in 0..o {
                    for j in 0..o {
                        fw_d[[ii, 3 * o + j]] += output[[row - 1, ii]] * deltas_cells[j];
                    }
                }
            }
            for j in 0..o {
                bs_d[[0, 3 * o + j]] += deltas_cells[j];
            }

            // :624 save the PRIOR forget-gate delta (read at :660 by the input gate).
            let deltas_forget_gate_tmp = deltas_forget_gate.clone();

            // :624-657 deltasForgetGate: row>0 has the _CellStates.row(row-1) term;
            // row==0 (`:648-657`) DROPS it (no `_CellStates.row(-1)` read).
            if row > 0 {
                let mut tmp_f = vec![0.0f64; o];
                for j in 0..o {
                    tmp_f[j] = cs[[row - 1, j]] * epsilon_state[j];
                }
                if self.gates_peep {
                    for j in 0..o {
                        tmp_f[j] +=
                            deltas_input_gate[j] * p[[4, j]] + deltas_output_gate[j] * p[[10, j]];
                    }
                }
                if self.gates_rec_peep {
                    for j in 0..o {
                        tmp_f[j] += deltas_forget_gate[j] * p[[7, j]];
                    }
                }
                for j in 0..o {
                    deltas_forget_gate[j] = gates_deriv(g[[row, o + j]]) * tmp_f[j];
                }
                for ii in 0..i {
                    for j in 0..o {
                        iw_d[[ii, o + j]] += recon_input[[row, ii]] * deltas_forget_gate[j];
                    }
                }
                for ii in 0..o {
                    for j in 0..o {
                        fw_d[[ii, o + j]] += output[[row - 1, ii]] * deltas_forget_gate[j];
                    }
                }
                if self.cells_peep {
                    for j in 0..o {
                        pp_d[[1, j]] += deltas_forget_gate[j] * cs[[row - 1, j]];
                    }
                }
                if self.gates_peep {
                    for j in 0..o {
                        pp_d[[6, j]] += deltas_forget_gate[j] * g[[row - 1, j]];
                    }
                    for j in 0..o {
                        pp_d[[8, j]] += deltas_forget_gate[j] * g[[row - 1, 2 * o + j]];
                    }
                }
                if self.gates_rec_peep {
                    for j in 0..o {
                        pp_d[[7, j]] += deltas_forget_gate[j] * g[[row - 1, o + j]];
                    }
                }
                for j in 0..o {
                    bs_d[[0, o + j]] += deltas_forget_gate[j];
                }
            } else {
                // :648-657 row==0: no cellstate term; still folds the gates-peep
                // (rows 4/10) + gates-rec (row 7) cross-terms and the bias.
                let mut tmp_f = vec![0.0f64; o];
                if self.gates_peep {
                    for j in 0..o {
                        tmp_f[j] +=
                            deltas_input_gate[j] * p[[4, j]] + deltas_output_gate[j] * p[[10, j]];
                    }
                }
                if self.gates_rec_peep {
                    for j in 0..o {
                        tmp_f[j] += deltas_forget_gate[j] * p[[7, j]];
                    }
                }
                for j in 0..o {
                    deltas_forget_gate[j] = gates_deriv(g[[row, o + j]]) * tmp_f[j];
                }
                for ii in 0..i {
                    for j in 0..o {
                        iw_d[[ii, o + j]] += recon_input[[row, ii]] * deltas_forget_gate[j];
                    }
                }
                for j in 0..o {
                    bs_d[[0, o + j]] += deltas_forget_gate[j];
                }
            }

            // :659-682 deltasInputGate. tmp = _Gates[g-block].*epsilonState, +=
            // gatesPeep (row 6 from the SAVED prior forget delta, row 9 from the
            // current o delta), += gatesRecPeep (row 3 from the PRIOR i delta), then
            // * GatesFunction'(_Gates[i-block]).
            let mut tmp_i = vec![0.0f64; o];
            for j in 0..o {
                tmp_i[j] = g[[row, 3 * o + j]] * epsilon_state[j];
            }
            if self.gates_peep {
                for j in 0..o {
                    tmp_i[j] +=
                        deltas_forget_gate_tmp[j] * p[[6, j]] + deltas_output_gate[j] * p[[9, j]];
                }
            }
            if self.gates_rec_peep {
                for j in 0..o {
                    tmp_i[j] += deltas_input_gate[j] * p[[3, j]];
                }
            }
            for j in 0..o {
                deltas_input_gate[j] = gates_deriv(g[[row, j]]) * tmp_i[j];
            }
            for ii in 0..i {
                for j in 0..o {
                    iw_d[[ii, j]] += recon_input[[row, ii]] * deltas_input_gate[j];
                }
            }
            if row > 0 {
                for ii in 0..o {
                    for j in 0..o {
                        fw_d[[ii, j]] += output[[row - 1, ii]] * deltas_input_gate[j];
                    }
                }
                if self.cells_peep {
                    for j in 0..o {
                        pp_d[[0, j]] += deltas_input_gate[j] * cs[[row - 1, j]];
                    }
                }
                if self.gates_peep {
                    for j in 0..o {
                        pp_d[[4, j]] += deltas_input_gate[j] * g[[row - 1, o + j]];
                    }
                    for j in 0..o {
                        pp_d[[5, j]] += deltas_input_gate[j] * g[[row - 1, 2 * o + j]];
                    }
                }
                if self.gates_rec_peep {
                    for j in 0..o {
                        pp_d[[3, j]] += deltas_input_gate[j] * g[[row - 1, j]];
                    }
                }
            }
            for j in 0..o {
                bs_d[[0, j]] += deltas_input_gate[j];
            }

            // :688-690 stack [i|f|o|c] into testM.col(row) (4O x 1).
            for j in 0..o {
                test_m[[j, row]] = deltas_input_gate[j];
                test_m[[o + j, row]] = deltas_forget_gate[j];
                test_m[[2 * o + j, row]] = deltas_output_gate[j];
                test_m[[3 * o + j, row]] = deltas_cells[j];
            }

            // :703 deltasFeedback = (_FeedbackWeights * test)^T, for the next (earlier)
            // iteration's :580. test is testM.col(row) (4O x 1).
            let test_col = test_m.slice(ndarray::s![.., row..row + 1]).to_owned();
            let fb = matmul_seq(&self.feedback_weights, &test_col); // O x 1
            for j in 0..o {
                deltas_feedback[j] = fb[[j, 0]];
            }
        }

        // :707 deltasPreviousLayer = (_InputWeights * testM)^T -> T x I.
        let dpl = matmul_seq(&self.input_weights, &test_m); // I x T
        let deltas_previous_layer = dpl.t().to_owned();

        // :710-715 invSubSamplingRatio > 1 scales all four deriv blocks.
        if inv_sub_sampling_ratio > 1 {
            let r = inv_sub_sampling_ratio as f64;
            iw_d.mapv_inplace(|v| v * r);
            fw_d.mapv_inplace(|v| v * r);
            pp_d.mapv_inplace(|v| v * r);
            bs_d.mapv_inplace(|v| v * r);
        }

        // :716-720 accumulate into the members; count += linesNb (deltas.rows()).
        self.input_weights_derivatives += &iw_d;
        self.feedback_weights_derivatives += &fw_d;
        self.peep_weight_derivatives += &pp_d;
        self.biases_derivatives += &bs_d;
        self.nb_of_seq_fed_backward += lines_nb as i64;

        deltas_previous_layer
    }

    /// `feedBackwardReverse` (`LSTMLayer.cpp:725-732`): reverse input/output/deltas
    /// rows, run the forward-order `feed_backward`, reverse the returned deltas. The
    /// caches are STORED time-reversed after `feed_forward_reverse` and consumed
    /// AS-IS here (do NOT un-reverse them -- see `feed_forward_reverse`, risk R9).
    pub fn feed_backward_reverse(
        &mut self,
        input: &Array2<f64>,
        output: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        let input_rev = input.slice(ndarray::s![..;-1, ..]).to_owned();
        let output_rev = output.slice(ndarray::s![..;-1, ..]).to_owned();
        let deltas_rev = deltas.slice(ndarray::s![..;-1, ..]).to_owned();
        let dpl = self.feed_backward(
            &input_rev,
            &output_rev,
            &deltas_rev,
            inv_sub_sampling_ratio,
            last_layer,
        );
        dpl.slice(ndarray::s![..;-1, ..]).to_owned()
    }

    /// `getWeightsDerivatives` (`LSTMLayer.cpp:251-295`): Nx2 harvest, col0 = the
    /// summed derivative in the flat weight-packer block order (InputWeights ->
    /// FeedbackWeights -> PeepWeight -> Biaises, each COLUMN-major), col1 =
    /// `_NbOfSeqFedBackward` REPLICATED per element. Appended into `out`.
    pub fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        let count = self.nb_of_seq_fed_backward as f64;
        for m in [
            &self.input_weights_derivatives,
            &self.feedback_weights_derivatives,
            &self.peep_weight_derivatives,
            &self.biases_derivatives,
        ] {
            let (rows, cols) = m.dim();
            for jj in 0..cols {
                for ii in 0..rows {
                    out.push([m[[ii, jj]], count]);
                }
            }
        }
    }

    /// `resetWeightsDerivatives` (`LSTMLayer.cpp:297-303`): zero all four deriv
    /// blocks and the frame count.
    pub fn reset_weights_derivatives(&mut self) {
        self.nb_of_seq_fed_backward = 0;
        self.input_weights_derivatives = Array2::zeros((self.input_size, 4 * self.output_size));
        self.feedback_weights_derivatives = Array2::zeros((self.output_size, 4 * self.output_size));
        self.peep_weight_derivatives = Array2::zeros((12, self.output_size));
        self.biases_derivatives = Array2::zeros((1, 4 * self.output_size));
    }

    /// `ponderateWeightsDerivatives` (`LSTMLayer.cpp:305-310`): scale the four deriv
    /// accumulators (the harvested Nx2 col0) IN PLACE by `factor`. The frame count
    /// `_NbOfSeqFedBackward` (col1) is deliberately NOT touched.
    pub fn ponderate_weights_derivatives(&mut self, factor: f64) {
        self.input_weights_derivatives *= factor;
        self.feedback_weights_derivatives *= factor;
        self.peep_weight_derivatives *= factor;
        self.biases_derivatives *= factor;
    }
}

/// Read `rows*cols` elements from `needed[*pos..]` in column-major order (`jj`
/// outer over cols, `ii` inner over rows) into `m`, then advance `*pos`.
fn fill_col_major(m: &mut Array2<f64>, needed: &[f64], pos: &mut usize) {
    let (rows, cols) = m.dim();
    for jj in 0..cols {
        for ii in 0..rows {
            m[[ii, jj]] = needed[*pos + jj * rows + ii];
        }
    }
    *pos += rows * cols;
}

/// Append `m`'s elements to `out` in column-major order (`jj` outer over cols,
/// `ii` inner over rows) -- the mirror of `fill_col_major`.
fn drain_col_major(m: &Array2<f64>, out: &mut Vec<f64>) {
    let (rows, cols) = m.dim();
    for jj in 0..cols {
        for ii in 0..rows {
            out.push(m[[ii, jj]]);
        }
    }
}

/// Dense output layer (legacy `NeuronLayer`). Ported: weight (de)serialization
/// (`NeuronLayer.cpp:69-99`) + forward (`:127-149`).
///
/// - `weights`: `_InputSize x _OutputSize` (unlike the LSTM gate matrices, no
///   peepholes -- one plain projection matrix).
/// - `biases`: `1 x _OutputSize` (row vector; stored as `Array2<f64>` for a
///   uniform (de)serialization loop with `weights`).
#[derive(Clone)]
pub struct NeuronLayer {
    input_size: usize,
    output_size: usize,

    weights: Array2<f64>,
    biases: Array2<f64>,

    // Backward-pass deriv accumulators (`_WeightsDerivatives`/`_BiaisesDerivatives`,
    // NeuronLayer.cpp:53-54) + frame count (`_NbOfSeqFedBackward`). Reset via
    // `reset_weights_derivatives`, accumulated across `feed_backward` calls,
    // harvested via `get_weights_derivatives`.
    weights_derivatives: Array2<f64>,
    biases_derivatives: Array2<f64>,
    nb_of_seq_fed_backward: i64,
}

impl NeuronLayer {
    /// New layer with zeroed weights (legacy ctor with `weightsSetExternally =
    /// true` skips the config-key/random-table fill; callers set weights via
    /// `set_weights`).
    pub fn new(input_size: usize, output_size: usize) -> Self {
        NeuronLayer {
            input_size,
            output_size,
            weights: Array2::zeros((input_size, output_size)),
            biases: Array2::zeros((1, output_size)),
            weights_derivatives: Array2::zeros((input_size, output_size)),
            biases_derivatives: Array2::zeros((1, output_size)),
            nb_of_seq_fed_backward: 0,
        }
    }

    /// `getNbOfWeights` (`NeuronLayer.cpp:84-86`): `_OutputSize*(_InputSize+1)`.
    pub fn nb_of_weights(&self) -> usize {
        self.output_size * (self.input_size + 1)
    }

    /// `setWeights` (`NeuronLayer.cpp:69-82`): consumes `flat[..nb_of_weights()]`
    /// into the weights matrix (COLUMN-major element order,
    /// `needed(jj*_InputSize+ii)`, `jj` outer/cols, `ii` inner/rows) then the bias
    /// row, and returns the tail slice for the next layer to consume in a chain.
    pub fn set_weights<'a>(&mut self, flat: &'a [f64]) -> &'a [f64] {
        let nb = self.nb_of_weights();
        let (needed, left) = flat.split_at(nb);

        let mut pos = 0usize;
        fill_col_major(&mut self.weights, needed, &mut pos);
        fill_col_major(&mut self.biases, needed, &mut pos);

        left
    }

    /// `getWeights` (`NeuronLayer.cpp:88-99`): the mirror of `set_weights`, same
    /// column-major element order (weights then bias), appended into `out`.
    pub fn get_weights(&self, out: &mut Vec<f64>) {
        drain_col_major(&self.weights, out);
        drain_col_major(&self.biases, out);
    }

    /// `feedForward` (`NeuronLayer.cpp:127-149`): whole-sequence dense forward
    /// pass. Fills `output` (T x O).
    ///
    /// - Width-tolerant projection (`:129-135`): `cols > I` uses the left `I`
    ///   input columns; `cols < I` uses the top `cols` weight rows; else the full
    ///   product. Via `matmul_seq` (the measured ascending-loop contract).
    /// - Bias added `.rowwise()` BEFORE activation (`:136-137`), shared by all
    ///   three activation paths below.
    /// - `last_layer && output_size > 1`: softmax (`:138-142`). The legacy is
    ///   UNSTABILIZED (`exp(a+b)` computed directly, no max-subtraction); F8
    ///   (phase5) adds an OVERFLOW-GUARD-ONLY shift -- the per-row max is
    ///   subtracted ONLY when it exceeds the exp-overflow threshold (700; `exp`
    ///   is finite iff its arg `<= ln(f64::MAX) ~ 709.78), so below the threshold
    ///   the raw `exp(a+b)` path is BIT-IDENTICAL to the legacy (every committed
    ///   golden is unchanged) and above it the shift makes an otherwise
    ///   inf/inf=NaN row finite + correct (softmax is shift-invariant). Per-row
    ///   sum via a SEQUENTIAL ascending-column loop (not a reduction), then
    ///   per-COLUMN quotient by that sum. See IMPROVEMENTS.md [phase2]/[phase5].
    /// - `last_layer && output_size == 1`: `Logistic` (`:144`).
    /// - `!last_layer`: `Maxmin2`/asinh (`:147`).
    pub fn feed_forward(
        &mut self,
        input: &Array2<f64>,
        output: &mut Array2<f64>,
        last_layer: bool,
    ) {
        let i = self.input_size;
        let o = self.output_size;
        let cols = input.dim().1;

        // :129-135 input projection with width tolerance.
        let activations = if cols > i {
            matmul_seq(&input.slice(ndarray::s![.., ..i]).to_owned(), &self.weights)
        } else if cols < i {
            matmul_seq(
                input,
                &self.weights.slice(ndarray::s![..cols, ..]).to_owned(),
            )
        } else {
            matmul_seq(input, &self.weights)
        };
        let t_len = activations.dim().0;

        // .rowwise() + _Biaises, BEFORE activation (:136-137).
        let mut pre_act = activations;
        for t in 0..t_len {
            for j in 0..o {
                pre_act[[t, j]] += self.biases[[0, j]];
            }
        }

        output.fill(0.0);
        if last_layer && o > 1 {
            // :138 exp(a+b); :139 SEQUENTIAL per-row sum (ascending columns, not a
            // reduction); :140-142 per-COLUMN cwiseQuotient.
            //
            // F8 (phase5): OVERFLOW-GUARDED stabilization. The legacy softmax is
            // unstabilized (no max-subtraction), so a large pre-activation overflows
            // `exp` to inf and the inf/inf row-quotient is NaN. `exp(x)` is finite iff
            // `x <= ln(f64::MAX) ~ 709.78`; the guard subtracts the per-row max ONLY
            // when it exceeds EXP_OVERFLOW_GUARD (700, leaving headroom for the O-term
            // sum). Below the threshold the else-branch is LITERALLY `pre_act.exp()`,
            // byte-identical to the legacy path -- so every committed golden (none
            // overflow) stays bit-exact; above it the shift makes the row finite and
            // correct (softmax is shift-invariant). Pinned by `dense_softmax_overflow_guarded`.
            const EXP_OVERFLOW_GUARD: f64 = 700.0;
            let mut exp_out = Array2::<f64>::zeros((t_len, o));
            for t in 0..t_len {
                let mut row_max = f64::NEG_INFINITY;
                for j in 0..o {
                    if pre_act[[t, j]] > row_max {
                        row_max = pre_act[[t, j]];
                    }
                }
                if row_max > EXP_OVERFLOW_GUARD {
                    for j in 0..o {
                        exp_out[[t, j]] = (pre_act[[t, j]] - row_max).exp();
                    }
                } else {
                    for j in 0..o {
                        exp_out[[t, j]] = pre_act[[t, j]].exp();
                    }
                }
            }
            let mut row_sum = vec![0.0f64; t_len];
            for t in 0..t_len {
                let mut acc = 0.0;
                for j in 0..o {
                    acc += exp_out[[t, j]];
                }
                row_sum[t] = acc;
            }
            for j in 0..o {
                for t in 0..t_len {
                    output[[t, j]] = exp_out[[t, j]] / row_sum[t];
                }
            }
        } else if last_layer {
            // O == 1: Logistic.
            for t in 0..t_len {
                for j in 0..o {
                    output[[t, j]] = logistic_fn(pre_act[[t, j]]);
                }
            }
        } else {
            // Maxmin2 (asinh).
            for t in 0..t_len {
                for j in 0..o {
                    output[[t, j]] = maxmin2_fn(pre_act[[t, j]]);
                }
            }
        }
    }

    /// `feedBackward` (`NeuronLayer.cpp:151-207`). Returns `deltas_out` (T x I).
    ///
    /// - Width-tolerant input reconstruction (`:157-166`, mirrors the forward's
    ///   `:129-135`): `cols > I` -> take the left `I` input columns; `cols < I`
    ///   -> zero-pad up to `I` columns; else the input as-is. (The legacy stores
    ///   this transposed, `I x T`; the Rust port keeps `input` T x I throughout
    ///   and reads columns/rows accordingly -- an equivalent, not identical,
    ///   layout, since ndarray column ops are just as cheap either way here.)
    /// - `weightsDerivatives += input.col(jj)*deltas.row(jj)` accumulated PER
    ///   ROW (`:178-182`, ascending-loop outer product per `jj`, NOT a single
    ///   GEMM); bias derivs `+= deltas.colwise().sum()` (`:183`).
    /// - `invSubSamplingRatio > 1` scales BOTH deriv blocks (`:192-195`).
    /// - `_NbOfSeqFedBackward += InputSeq.rows()` (`:199` -- the ORIGINAL input's
    ///   row count, NOT `deltas.rows()`; they coincide here since this layer
    ///   never decimates internally, but the count source is InputSeq by the
    ///   legacy's own choice).
    /// - `lastLayer`: `deltas_out = deltas * weights^T`, NO activation
    ///   derivative -- the softmax+CE fusion (and the scalar Logistic fold) is
    ///   pre-seeded into `deltas` by `CostLaw::computeDeltas` upstream; adding a
    ///   Jacobian here would double-count it (spec S5, risk R3).
    /// - hidden: `deltas_out = (deltas * weights^T) .* Maxmin2'(InputSeq)`,
    ///   the asinh-deriv taken on the RAW `InputSeq` parameter (`:204`), not the
    ///   output AND not the width-reconstructed `input` local used above for the
    ///   deriv-accumulation half -- the two halves read different tensors in the
    ///   legacy. See the fold site below for the divergence-reachability note.
    /// - No retained cache: `input`/`deltas` are passed in fresh each call.
    pub fn feed_backward(
        &mut self,
        input: &Array2<f64>,
        deltas: &Array2<f64>,
        inv_sub_sampling_ratio: usize,
        last_layer: bool,
    ) -> Array2<f64> {
        let i = self.input_size;
        let o = self.output_size;
        let cols = input.dim().1;
        let t = input.dim().0;

        // :157-166 width-tolerant input reconstruction (T x I, zero-padded /
        // truncated on the COLUMN axis to match _InputSize).
        let recon_input: Array2<f64> = if cols > i {
            input.slice(ndarray::s![.., ..i]).to_owned()
        } else if cols < i {
            let mut padded = Array2::<f64>::zeros((t, i));
            padded.slice_mut(ndarray::s![.., ..cols]).assign(input);
            padded
        } else {
            input.to_owned()
        };

        // :178-182 weightsDerivatives += input.col(jj)*deltas.row(jj) per row.
        let mut weights_derivatives = Array2::<f64>::zeros((i, o));
        for jj in 0..deltas.dim().0 {
            let in_row = recon_input
                .slice(ndarray::s![jj..jj + 1, ..])
                .t()
                .to_owned(); // I x 1
            let delta_row = deltas.slice(ndarray::s![jj..jj + 1, ..]).to_owned(); // 1 x O
            let outer = matmul_seq(&in_row, &delta_row); // I x O
            weights_derivatives += &outer;
        }
        // :183 biaisesDerivatives += deltas.colwise().sum().
        let mut biases_derivatives = Array2::<f64>::zeros((1, o));
        for jj in 0..deltas.dim().0 {
            for j in 0..o {
                biases_derivatives[[0, j]] += deltas[[jj, j]];
            }
        }

        if inv_sub_sampling_ratio > 1 {
            let ratio = inv_sub_sampling_ratio as f64;
            weights_derivatives.mapv_inplace(|v| v * ratio);
            biases_derivatives.mapv_inplace(|v| v * ratio);
        }

        self.weights_derivatives += &weights_derivatives;
        self.biases_derivatives += &biases_derivatives;
        self.nb_of_seq_fed_backward += input.dim().0 as i64; // :199 InputSeq.rows(), NOT deltas.rows()

        // deltas (T x O) * weights^T (O x I) -> T x I, via matmul_seq (the
        // measured ascending-loop contract; Eigen diverges at NN-wide k here).
        let w_t = self.weights.t().to_owned();
        let mut deltas_out = matmul_seq(deltas, &w_t);
        if !last_layer {
            // :204 hidden -> .* Maxmin2'(InputSeq), literally the UN-RECONSTRUCTED
            // `InputSeq` parameter (not the width-reconstructed/transposed `input`
            // local used for the :157-166 deriv-accumulation half above). The two
            // halves read different tensors in the legacy: accumulation uses the
            // reconstructed `input`, this fold uses the raw `input` arg as-is. The
            // divergence between raw and reconstructed is only observable when
            // `cols != I`, which this layer's sole legacy caller never produces
            // (BLSTMNeuralNetwork.h:29, .cpp:90 always feeds cols == I).
            for r in 0..deltas_out.dim().0 {
                for c in 0..deltas_out.dim().1 {
                    deltas_out[[r, c]] *= maxmin2_deriv(input[[r, c]]);
                }
            }
        }
        deltas_out
    }

    /// `getWeightsDerivatives` (`NeuronLayer.cpp:101-114`): Nx2 harvest, col0 =
    /// the summed derivative (weights column-major, i.e. weight-block ordering
    /// matching the flat packer, then bias), col1 = `_NbOfSeqFedBackward`
    /// REPLICATED per element. Appended into `out` (threaded through
    /// `Network`/`BlstmNetwork` in later tasks; final `Array2` assembly is
    /// Task 6).
    pub fn get_weights_derivatives(&self, out: &mut Vec<[f64; 2]>) {
        let count = self.nb_of_seq_fed_backward as f64;
        for jj in 0..self.output_size {
            for ii in 0..self.input_size {
                out.push([self.weights_derivatives[[ii, jj]], count]);
            }
        }
        for jj in 0..self.output_size {
            out.push([self.biases_derivatives[[0, jj]], count]);
        }
    }

    /// `resetWeightsDerivatives` (`NeuronLayer.cpp:116-120`): zero both deriv
    /// matrices and the frame count.
    pub fn reset_weights_derivatives(&mut self) {
        self.nb_of_seq_fed_backward = 0;
        self.weights_derivatives = Array2::zeros((self.input_size, self.output_size));
        self.biases_derivatives = Array2::zeros((1, self.output_size));
    }

    /// `ponderateWeightsDerivatives` (`NeuronLayer.cpp:122-125`): scale the weight +
    /// bias deriv accumulators (the harvested Nx2 col0) IN PLACE by `factor`. The frame
    /// count `_NbOfSeqFedBackward` (col1) is deliberately NOT touched.
    pub fn ponderate_weights_derivatives(&mut self, factor: f64) {
        self.weights_derivatives *= factor;
        self.biases_derivatives *= factor;
    }
}
