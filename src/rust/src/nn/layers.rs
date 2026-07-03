//! Layer implementations: peephole LSTM, SRN, Clockwork-RNN, dense, conv.
//!
//! Ported from legacy C++: LSTMLayer.*, SRNLayer.*, CWRNNLayer.*, NeuronLayer.*,
//! Convolutional*.*. Phase 2.

use ndarray::Array2;

/// Peephole LSTM layer (legacy `LSTMLayer`). Task 3 ports the weight layout +
/// flat (de)serialization; forward/backward (Task 4+) are not yet implemented.
///
/// Member sizes are the CTOR RESIZES (`LSTMLayer.cpp:19-22`), NOT the stale size
/// comments in `LSTMLayer.h:33-36` (which claim `_PeepWeight` is `3*_OutputSize`
/// and predate a peephole-bundle rework -- the real ctor resizes it to
/// `12 x _OutputSize`):
/// - `input_weights`: `_InputSize x 4*_OutputSize` (gate column blocks `[i|f|o|g]`).
/// - `feedback_weights`: `_OutputSize x 4*_OutputSize` (same gate blocks, recurrent).
/// - `peep_weight`: `12 x _OutputSize`.
/// - `biases`: `1 x 4*_OutputSize` (row vector; stored as `Array2<f64>` for a
///   uniform (de)serialization loop with the other three blocks).
pub struct LstmLayer {
    input_size: usize,
    output_size: usize,
    // Stored for Task 4's forward pass; unused until then.
    #[allow(dead_code)]
    cells_peep: bool,
    #[allow(dead_code)]
    gates_peep: bool,
    #[allow(dead_code)]
    gates_rec_peep: bool,

    input_weights: Array2<f64>,
    feedback_weights: Array2<f64>,
    peep_weight: Array2<f64>,
    biases: Array2<f64>,

    // Forward-pass caches (Task 4). Left empty until then.
    #[allow(dead_code)]
    gates: Array2<f64>,
    #[allow(dead_code)]
    cells_in: Array2<f64>,
    #[allow(dead_code)]
    cell_states: Array2<f64>,
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
