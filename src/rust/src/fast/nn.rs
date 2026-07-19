//! `fast::nn` -- the f32 BLSTM forward core (Phase 7 Task 2). FORWARD ONLY.
//!
//! An `f32`, `faer`-backed transcription of the exact bidirectional forward pass.
//! The exact sources this mirrors line-for-line (cite them when auditing drift, per
//! the plan's R2):
//!
//! - `nn/blstm.rs::BlstmNetwork::feed_forward` (`:810-840`, legacy
//!   `BLSTMNeuralNetwork.cpp:419-437`): output-length division, the
//!   `LSTMRatios[0] > 1 && netInput < inputCols` crop gate, forward
//!   `feed_forward` + backward `feed_forward_reverse`, then the output MLP's
//!   `feed_forward_double` (HCAT forward-LEFT).
//! - `nn/blstm.rs::BlstmNetwork::set_weights` (`:459-480`): the flat-pack block
//!   order Forward -> Backward -> Output -> mean -> std, adim ALREADY APPLIED by the
//!   config-reader path (`config.rs::apply_adim`); `from_flat` consumes exactly that
//!   layout and narrows once.
//! - `nn/layers.rs::LstmLayer::feed_forward` (`:174-337`, `LSTMLayer.cpp:312-413`):
//!   the `[i|f|o|g]` gate blocks, the two-matmul summed pre-activation (input
//!   projection + recurrence), the 12-row peephole bundle with its gate ordering,
//!   the CellWeight-narrow layout (cell rows carry NO peepholes -- the parity
//!   hazard), the `t=0` no-forget/no-P11 special case, and the width-tolerance crop
//!   (`:186-198`).
//! - `nn/layers.rs::NeuronLayer::feed_forward` (`:914-1006`): asinh hidden, the
//!   700-guarded softmax (`:946-990`, F8), Logistic for `output_size == 1`.
//! - `nn/network.rs::sub_sample` (`:695-708`) integer-floor decimation +
//!   `Network::drive` (`:328-394`) per-layer subsampling chaining.
//! - `nn/activations.rs`: `gates_fn` (sigmoid(0.1z) + expLimit guards), `maxmin2_fn`/
//!   `identity_fn` (asinh), `logistic_fn`.
//!
//! DELIBERATE f32 DIVERGENCES (spec S4; not bugs, not IMPROVEMENTS.md material):
//! 1. `faer::linalg::matmul` (blocked SIMD) replaces `matmul_seq` (ascending f64) for
//!    the batched input/output projections -- different reduction order + f32.
//! 2. The recurrence matvec is a hand-written contiguous dot over the column-major
//!    feedback blocks, NOT a per-timestep faer matvec: the batched projection uses
//!    faer (the brief's mandate), but a per-frame faer matvec's view-setup + dispatch
//!    overhead exceeds a tight autovectorized `O`-length dot for the sequential
//!    recurrence, which the brief explicitly blesses as the recurrence fallback.
//! 3. The exp saturation/overflow guards use the f32 bound `ln(f32::MAX) ~ 88.72`
//!    (gates/logistic) and an `80.0` softmax guard, not the exact path's f64 `700.0`
//!    -- the correct f32 analogues (the guards only fire in saturation, unreachable on
//!    real posteriors, so this never moves a non-degenerate value).
//!
//! Workspace note: `from_flat(spec, flat)` carries no max-sequence hint, so the
//! workspace is created empty and grown geometrically (`ensure_len`, `max(n, 2*len)`)
//! on first/subsequent `feed_forward` -- never per-frame. Lazy growth to exactly the
//! largest sequence seen also serves the phase's low-memory goal (no eager
//! over-allocation).

use anyhow::{Result, bail};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};

use crate::config::{self, NnetSpec};

// ---------------------------------------------------------------------------
// f32 activation transcriptions (nn/activations.rs, f32).
// ---------------------------------------------------------------------------

/// f32 analogue of `Log<double>::expLimit` (`ln(f64::MAX)`): `ln(f32::MAX)`
/// (~88.72284). On the fast path the exp-overflow bound is f32's, so the gate/
/// logistic saturation guards test against this, not the exact f64 limit ~709.78.
///
/// Const (Phase 7 Task 4, mirroring `EXP_OVERFLOW_GUARD_F32` below): the literal
/// `ln(f32::MAX)` value, promoted from the previous `f32::MAX.ln()` call. The gate/
/// logistic guards only fire in saturation (`|scaled| >= ~88.72`), unreachable on
/// real posteriors, so the sub-ULP difference between this literal and
/// `f32::MAX.ln()` never moves a non-degenerate value (the committed activation
/// grids sample far from the edge).
const EXP_LIMIT_F32: f32 = 88.72284;

/// `Maxmin2::fn`/`Identity::fn` (ActivationFunctions.h:158-160,206-208): `asinh`, f32.
#[inline]
pub fn asinh_f32(x: f32) -> f32 {
    x.asinh()
}

/// `GatesFunction::fn` (ActivationFunctions.h:230-237, `nn/activations.rs::gates_fn`):
/// sigmoid on `0.1*x`, saturation guards on the SCALED value, f32.
#[inline]
pub fn gates_fn_f32(x: f32) -> f32 {
    let scaled = 0.1_f32 * x;
    if scaled < EXP_LIMIT_F32 {
        if scaled > -EXP_LIMIT_F32 {
            1.0 / (1.0 + (-scaled).exp())
        } else {
            0.0
        }
    } else {
        1.0
    }
}

/// `Logistic::fn` (ActivationFunctions.h:41-48, `nn/activations.rs::logistic_fn`):
/// plain sigmoid, saturation guards, f32.
#[inline]
pub fn logistic_f32(x: f32) -> f32 {
    if x < EXP_LIMIT_F32 {
        if x > -EXP_LIMIT_F32 {
            1.0 / (1.0 + (-x).exp())
        } else {
            0.0
        }
    } else {
        1.0
    }
}

/// f32 whole-sequence self-normalization (`BLSTMNeuralNetwork::self_normalize`,
/// `nn/blstm.rs:769-801`, the input-normalization type -1 branch): all per-column
/// means, then center all; all per-column stds `sqrt((sum(centered^2) + 1e-32)/rows)`,
/// then `asinh(x/std)` all. In-place, mirroring the exact 4-pass structure.
///
/// The algo-3 fast SAD driver applies this to the whole input sequence ONCE per
/// channel BEFORE the overlap windowing, exactly as the exact scoring path applies
/// it at the top of `feed_forward_backward` (`nn/blstm.rs:1025-1028`) before it
/// dispatches to `feed_forward_backward_overlap`. f32 divergence from the exact f64
/// normalization is by design (spec S4).
pub fn self_normalize_f32(m: &mut FastMatrix) {
    let r = m.rows;
    let c = m.cols;
    if r == 0 {
        return;
    }
    let rf = r as f32;
    let mut mean = vec![0.0_f32; c];
    for (col, mn) in mean.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for row in 0..r {
            acc += m.data[row * c + col];
        }
        *mn = acc / rf;
    }
    for row in 0..r {
        for (col, &mn) in mean.iter().enumerate() {
            m.data[row * c + col] -= mn;
        }
    }
    let mut stdv = vec![0.0_f32; c];
    for (col, sd) in stdv.iter_mut().enumerate() {
        let mut acc = 0.0_f32;
        for row in 0..r {
            let v = m.data[row * c + col];
            acc += v * v;
        }
        *sd = ((acc + 1e-32) / rf).sqrt();
    }
    for row in 0..r {
        for (col, &sd) in stdv.iter().enumerate() {
            m.data[row * c + col] = (m.data[row * c + col] / sd).asinh();
        }
    }
}

/// f32 external (type 1) input normalization (`BLSTMNeuralNetwork::
/// feedForwardBackward` `:720-723`, exact port `nn/blstm.rs:1008-1021`, the
/// `input_normalization_type == 1` branch): per column `jj < min(cols,
/// mean.len())`, `(x - mean_jj) / max(1e-12, std_jj)`; columns beyond the
/// mean/std tail are UNTOUCHED. `mean`/`std` are the pack-carried normalize
/// tail ([`FastBlstm::normalize_mean`]/[`FastBlstm::normalize_std`], narrowed
/// f64 -> f32 once at `from_flat`). No centering/asinh structure here -- unlike
/// the type -1 self-normalization above, type 1 is a plain affine transform.
///
/// Phase 8 Task 1 (spec S1.1): the frozen-stats input normalization the
/// streaming reference mode requires -- the fast SAD driver applies this ONCE
/// per channel before the overlap windowing, at the SAME pipeline position
/// where the -1 self-norm runs (the exact path's pre-dispatch application at
/// `nn/blstm.rs:1008`). The exact type-1 branch also feeds the normalized
/// snapshot to `analyse_input_seq` (`:1022-1023`, InputStatistics accumulation
/// -- a training-side bookkeeping read); the fast path is FORWARD-ONLY and
/// carries no InputStatistics (the phase-7 T6b audit: `get_input_statistics`
/// is an inert-empty read on fast), so that call is deliberately absent here.
pub fn external_normalize_f32(m: &mut FastMatrix, mean: &[f32], std: &[f32]) {
    let r = m.rows;
    let c = m.cols;
    let max_col = c.min(mean.len());
    for jj in 0..max_col {
        let denom = 1e-12_f32.max(std[jj]);
        let mn = mean[jj];
        for row in 0..r {
            let v = &mut m.data[row * c + jj];
            *v = (*v - mn) / denom;
        }
    }
}

/// f32 softmax overflow guard (exact path `nn/layers.rs:959` uses f64 `700.0`).
/// f32's `exp` is finite iff arg `<= ln(f32::MAX) ~ 88.72`; `80.0` leaves headroom
/// for the per-row class-count sum. Below this the path is literally
/// `exp(x)/sum(exp(x))` -- the unguarded legacy softmax in f32.
const EXP_OVERFLOW_GUARD_F32: f32 = 80.0;

/// One row of the 700-guarded softmax (`nn/layers.rs:946-990`, f32). Sequential
/// ascending-column exp+sum (fused, same order as the exact two-pass), then a
/// per-column quotient. `out.len() == pre.len()`.
pub fn softmax_row_f32(pre: &[f32], out: &mut [f32]) {
    let mut row_max = f32::NEG_INFINITY;
    for &v in pre {
        if v > row_max {
            row_max = v;
        }
    }
    let mut sum = 0.0_f32;
    if row_max > EXP_OVERFLOW_GUARD_F32 {
        for (o, &v) in out.iter_mut().zip(pre) {
            *o = (v - row_max).exp();
            sum += *o;
        }
    } else {
        for (o, &v) in out.iter_mut().zip(pre) {
            *o = v.exp();
            sum += *o;
        }
    }
    for o in out.iter_mut() {
        *o /= sum;
    }
}

// ---------------------------------------------------------------------------
// FastMatrix -- the tree's row-major f32 buffer type.
// ---------------------------------------------------------------------------

/// Row-major dense f32 matrix (`data[r*cols + c]`), the fast path's public matrix
/// type. Posteriors come out as `frames x classes`.
#[derive(Debug, Clone, PartialEq)]
pub struct FastMatrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl FastMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        FastMatrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    /// Narrow an f64 row-major buffer (`rows*cols` elements) into a FastMatrix.
    pub fn from_f64_rows(rows: usize, cols: usize, src: &[f64]) -> Self {
        assert_eq!(src.len(), rows * cols, "from_f64_rows: shape mismatch");
        FastMatrix {
            data: src.iter().map(|&x| x as f32).collect(),
            rows,
            cols,
        }
    }

    #[inline]
    pub fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..r * self.cols + self.cols]
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
}

// ---------------------------------------------------------------------------
// Per-layer f32 weight blocks (SoA, narrowed once at construction).
// ---------------------------------------------------------------------------

/// One peephole LSTM layer's f32 weights. `input_weights`/`feedback_weights` are
/// stored COLUMN-major exactly as the flat pack delivers them (`LSTMLayer::set_weights`
/// reads them column-major), so faer views them zero-copy; `peep` is transposed to
/// ROW-major (`peep[row*O + j]`) for contiguous per-unit peephole reads. Gate column
/// blocks are `[i|f|o|g]` (0..O input, O..2O forget, 2O..3O output, 3O..4O cell/g).
#[derive(Clone)]
struct FastLstmLayer {
    input_size: usize,
    output_size: usize,
    cells_peep: bool,
    gates_peep: bool,
    gates_rec_peep: bool,
    input_weights: Vec<f32>,    // col-major (I x 4O)
    feedback_weights: Vec<f32>, // col-major (O x 4O)
    peep: Vec<f32>,             // row-major (12 x O)
    biases: Vec<f32>,           // 4O
}

/// One dense output layer's f32 weights (col-major `I x O`, plus the `O` bias row).
#[derive(Clone)]
struct FastDenseLayer {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>, // col-major (I x O)
    biases: Vec<f32>,  // O
}

// ---------------------------------------------------------------------------
// Preallocated, geometrically-grown workspace.
// ---------------------------------------------------------------------------

/// Reusable scratch buffers for one network's forward pass. Every buffer is grown
/// geometrically by `ensure_len` (never per-frame) and sliced to the exact live
/// length at each use, so over-sized tails are inert and repeated calls are
/// bit-identical (the workspace-hygiene contract).
#[derive(Default, Clone)]
struct Scratch {
    sub: Vec<f32>,        // sub_sample output (T x C*R)
    proj: Vec<f32>,       // gate/dense projection (T x 4O or T x O)
    ping: Vec<f32>,       // layer-output ping (also the materialized net input)
    pong: Vec<f32>,       // layer-output pong
    prev_gates: Vec<f32>, // rolling previous-step activated gates (4O)
    prev_cell: Vec<f32>,  // rolling previous-step cell state (O)
    prev_out: Vec<f32>,   // rolling previous-step output (O)
    cur_cell: Vec<f32>,   // current-step cell state (O)
}

/// The full workspace: the per-net `Scratch` plus the buffers that must persist
/// simultaneously (the two LSTM-net outputs feeding the HCAT).
#[derive(Default, Clone)]
struct Workspace {
    scratch: Scratch,
    lstm_fwd: Vec<f32>, // forward LSTM-net output (T x lstm_out)
    lstm_bwd: Vec<f32>, // backward LSTM-net output
    hcat: Vec<f32>,     // [fwd | bwd] (T x 2*lstm_out)
}

/// Grow `v` to hold at least `n` elements, geometrically (`max(n, 2*len)`), never
/// shrinking. New tail is zeroed but every live slice is fully overwritten before
/// read, so the zero-fill is inert.
#[inline]
fn ensure_len(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        let new_len = n.max(v.len().saturating_mul(2));
        v.resize(new_len, 0.0);
    }
}

// ---------------------------------------------------------------------------
// FastBlstm.
// ---------------------------------------------------------------------------

/// f32 bidirectional forward net: forward + backward LSTM stacks feeding a dense
/// output MLP, mirroring `nn::blstm::BlstmNetwork` (non-MLP mode only -- every
/// committed config has `LSTMNeuronNb[0] != 0`). FORWARD ONLY: no backward, no
/// trainer, no input normalization (the driver, Task 4, normalizes upstream exactly
/// as the exact scoring path does before calling the core forward).
///
/// `Clone` deep-copies the narrowed f32 weights + the (reusable) workspace/output
/// buffers, so the enclosing `Processor::FastSpectral` satisfies the bag's
/// `#[derive(Clone)]`; the fast path is inference-only, so a clone is off any hot path.
#[derive(Clone)]
pub struct FastBlstm {
    lstm_neuron_nb: Vec<usize>,
    lstm_subsampling: Vec<usize>,
    output_subsampling: Vec<usize>,
    lstm_out: usize,    // lstm_neuron_nb.last()
    output_size: usize, // output_neuron_nb.last()

    forward_layers: Vec<FastLstmLayer>,
    backward_layers: Vec<FastLstmLayer>,
    output_layers: Vec<FastDenseLayer>,

    normalize_mean: Vec<f32>,
    normalize_std: Vec<f32>,

    ws: Workspace,
    output: FastMatrix,
}

impl FastBlstm {
    /// Build from a `NnetSpec` + the flat f64 pack (adim ALREADY applied by the
    /// config-reader path). Narrows f64 -> f32 ONCE, per block, in the exact
    /// `BlstmNetwork::set_weights` order. Errors on MLP mode (unsupported: no
    /// committed config uses it) or a too-short pack (mirroring the exact `exit(1)`
    /// path; an over-long pack consumes only the head, like the legacy).
    pub fn from_flat(spec: &NnetSpec, flat: &[f64]) -> Result<FastBlstm> {
        if spec.lstm_neuron_nb.is_empty() || spec.lstm_neuron_nb[0] == 0 {
            bail!(
                "fast::nn::FastBlstm is BLSTM-only; MLP mode (LSTMNeuronNb[0]==0) is unsupported (no committed config uses it)"
            );
        }
        let needed = config::element_count(spec);
        if flat.len() < needed {
            bail!(
                "flat weight vector too short for fast net: {} < {needed}",
                flat.len()
            );
        }

        let lstm = &spec.lstm_neuron_nb;
        let lsub = &spec.lstm_subsampling;
        let outn = &spec.output_neuron_nb;
        let osub = &spec.output_subsampling;
        let n_lstm = lstm.len() - 1;
        let n_out = outn.len() - 1;

        let mut pos = 0usize;
        let mut take = |k: usize| -> &[f64] {
            let seg = &flat[pos..pos + k];
            pos += k;
            seg
        };
        let narrow = |s: &[f64]| -> Vec<f32> { s.iter().map(|&x| x as f32).collect() };

        // One LSTM layer's four blocks, in flat order: InputWeights (I x 4O col-
        // major), FeedbackWeights (O x 4O col-major), PeepWeight (12 x O col-major,
        // stored ROW-major), Biaises (4O). `cells`/`gates`/`gates_rec` are the
        // per-direction peephole flags.
        let mut build_lstm_layer =
            |i: usize, o: usize, cells: bool, gates: bool, gates_rec: bool| -> FastLstmLayer {
                let input_weights = narrow(take(i * 4 * o));
                let feedback_weights = narrow(take(4 * o * o));
                // PeepWeight arrives column-major (12 x O): seg[jj*12 + ii]. Store
                // row-major peep[ii*O + jj] for contiguous per-unit reads.
                let peep_seg = take(12 * o);
                let mut peep = vec![0.0_f32; 12 * o];
                for ii in 0..12 {
                    for jj in 0..o {
                        peep[ii * o + jj] = peep_seg[jj * 12 + ii] as f32;
                    }
                }
                let biases = narrow(take(4 * o));
                FastLstmLayer {
                    input_size: i,
                    output_size: o,
                    cells_peep: cells,
                    gates_peep: gates,
                    gates_rec_peep: gates_rec,
                    input_weights,
                    feedback_weights,
                    peep,
                    biases,
                }
            };

        // Per-direction peephole flags (NnetSpec.peepholes layout, config.rs:46-53).
        let (fc, fg, fr) = (spec.peepholes[0], spec.peepholes[2], spec.peepholes[4]);
        let (bc, bg, br) = (spec.peepholes[1], spec.peepholes[3], spec.peepholes[5]);

        let mut forward_layers = Vec::with_capacity(n_lstm);
        for jj in 0..n_lstm {
            let i = lstm[jj] * lsub[jj];
            let o = lstm[jj + 1];
            forward_layers.push(build_lstm_layer(i, o, fc, fg, fr));
        }
        let mut backward_layers = Vec::with_capacity(n_lstm);
        for jj in 0..n_lstm {
            let i = lstm[jj] * lsub[jj];
            let o = lstm[jj + 1];
            backward_layers.push(build_lstm_layer(i, o, bc, bg, br));
        }

        let mut output_layers = Vec::with_capacity(n_out);
        for jj in 0..n_out {
            let i = outn[jj] * osub[jj];
            let o = outn[jj + 1];
            let weights = narrow(take(i * o)); // col-major (I x O)
            let biases = narrow(take(o));
            output_layers.push(FastDenseLayer {
                input_size: i,
                output_size: o,
                weights,
                biases,
            });
        }

        let input_size = lstm[0];
        let normalize_mean = narrow(take(input_size));
        let normalize_std = narrow(take(input_size));
        debug_assert_eq!(pos, needed, "from_flat consumed != element_count");

        Ok(FastBlstm {
            lstm_neuron_nb: lstm.clone(),
            lstm_subsampling: lsub.clone(),
            output_subsampling: osub.clone(),
            lstm_out: *lstm.last().unwrap(),
            output_size: *outn.last().unwrap(),
            forward_layers,
            backward_layers,
            output_layers,
            normalize_mean,
            normalize_std,
            ws: Workspace::default(),
            output: FastMatrix::zeros(0, 0),
        })
    }

    /// The mean/std input-normalization tail (`BlstmNetwork::normalize_input_mean/std`).
    /// The core forward does NOT apply it (the driver normalizes upstream); exposed
    /// for Task 4.
    pub fn normalize_mean(&self) -> &[f32] {
        &self.normalize_mean
    }
    pub fn normalize_std(&self) -> &[f32] {
        &self.normalize_std
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Test hook: the forward layer-0 input-weight block (column-major), which is the
    /// FIRST block `from_flat` consumes (a direct narrow copy of `flat[0..I*4O]`), for
    /// the narrowing construction pin (`flat[i] as f32`).
    #[cfg(feature = "test-support")]
    pub fn debug_forward_l0_input_weights(&self) -> &[f32] {
        &self.forward_layers[0].input_weights
    }

    /// Core bidirectional forward (`BlstmNetwork::feed_forward`, `:810-840`). `input`
    /// is `frames x cols` (cols may exceed the net input via the crop gate). Returns
    /// the posteriors `output_length x output_size`, held in the reused `output`
    /// buffer (valid until the next `feed_forward`).
    pub fn feed_forward(&mut self, input: &FastMatrix) -> &FastMatrix {
        let frames = input.rows;
        // Output length: frames divided SEQUENTIALLY by each forward LSTM ratio.
        let mut out_len = frames;
        for &r in &self.lstm_subsampling {
            out_len /= r;
        }
        if frames == 0 || out_len == 0 {
            self.output = FastMatrix::zeros(0, self.output_size);
            return &self.output;
        }

        // Crop gate (`:827`): LSTMRatios[0] > 1 && netInput < inputCols -> use the
        // left `fwd_in` columns. Otherwise the full width (the LSTM layer's own
        // width tolerance then handles a mismatch).
        let fwd_in = self.lstm_neuron_nb[0];
        let in_cols = if self.lstm_subsampling[0] > 1 && fwd_in < input.cols {
            fwd_in
        } else {
            input.cols
        };
        let in_stride = input.cols;

        // Forward + backward LSTM stacks.
        let (lf_rows, _) = lstm_net_forward(
            &self.forward_layers,
            &self.lstm_subsampling,
            &input.data,
            frames,
            in_cols,
            in_stride,
            false,
            &mut self.ws.scratch,
            &mut self.ws.lstm_fwd,
        );
        let (lb_rows, _) = lstm_net_forward(
            &self.backward_layers,
            &self.lstm_subsampling,
            &input.data,
            frames,
            in_cols,
            in_stride,
            true,
            &mut self.ws.scratch,
            &mut self.ws.lstm_bwd,
        );
        debug_assert_eq!(lf_rows, lb_rows, "fwd/bwd row-count mismatch");
        let rows = lf_rows;
        let lstm_out = self.lstm_out;

        // HCAT [fwd | bwd] (forward LEFT), `Network::feed_forward_double`.
        let hcat_cols = 2 * lstm_out;
        ensure_len(&mut self.ws.hcat, rows * hcat_cols);
        {
            let hcat = &mut self.ws.hcat;
            let lf = &self.ws.lstm_fwd;
            let lb = &self.ws.lstm_bwd;
            for t in 0..rows {
                let hbase = t * hcat_cols;
                hcat[hbase..hbase + lstm_out]
                    .copy_from_slice(&lf[t * lstm_out..t * lstm_out + lstm_out]);
                hcat[hbase + lstm_out..hbase + hcat_cols]
                    .copy_from_slice(&lb[t * lstm_out..t * lstm_out + lstm_out]);
            }
        }

        // Dense output MLP over the HCAT -> posteriors in self.output.data.
        let (o_rows, o_cols) = dense_net_forward(
            &self.output_layers,
            &self.output_subsampling,
            &self.ws.hcat,
            rows,
            hcat_cols,
            &mut self.ws.scratch,
            &mut self.output.data,
        );
        self.output.rows = o_rows;
        self.output.cols = o_cols;
        &self.output
    }

    /// `getSubSamplingRatio` (`nn/blstm.rs:403-410`): the whole-BLSTM decimation
    /// ratio -- product of the forward LSTM sub-samplings times the output-net
    /// sub-samplings (non-MLP; every committed SAD config is non-MLP).
    pub fn sub_sampling_ratio(&self) -> usize {
        let mut r = 1usize;
        for &s in &self.lstm_subsampling {
            r *= s;
        }
        for &s in &self.output_subsampling {
            r *= s;
        }
        r
    }

    /// Overlapping-window forward accumulation (`BLSTMNeuralNetwork::
    /// feedForwardBackwardOverLap`, `nn/blstm.rs:1594-1726`; forward + OUTPUT-only).
    /// Mirrors the windowed driver the algo-3 SAD path dispatches to when
    /// `window_size > 0 && overlaps` (tier2 / the phase-6 SAD config): each window
    /// is a `feed_forward` over a contiguous row-slice of `input`, its posteriors
    /// accumulated into `output` at `begin/ssr` with a per-row count, then the whole
    /// `output` divided by the counts (rows covered by NO window -> `0/0 = NaN`,
    /// reproduced with no guard, per `:1710`).
    ///
    /// `output` is the CALLER'S buffer and is NOT zeroed here: the exact driver
    /// reuses ONE `result_vec` across channels and the `+=` accumulation seeds
    /// channel N from channel N-1's post-division contents (the load-bearing
    /// cross-channel reuse quirk, `tasks/sad.rs:1433-1436` + `:1594-1726`) -- the
    /// caller reproduces it by passing the same `output` for every channel and NOT
    /// re-zeroing between channels. The LSTM hidden-state accumulators
    /// (`_OutputForward`/`_OutputBackward`, `:1612-1613`) are NOT tracked: the SAD
    /// result path never reads them (only the LID Twin's hidden-state concat does,
    /// out of scope for the algo-3 driver).
    ///
    /// f32 + faer divergence from the exact f64 overlap is by design (spec S4); the
    /// windowing STRUCTURE (grid-snapped begin/end, partial-window recompute,
    /// count-quotient) is transcribed bit-for-bit in integer arithmetic.
    pub fn feed_forward_overlap(
        &mut self,
        input: &FastMatrix,
        window_size: usize,
        window_shift: usize,
        output: &mut FastMatrix,
    ) {
        let lstm_ratios = self.lstm_subsampling.clone();
        let out_ratios = self.output_subsampling.clone();
        let ssr = self.sub_sampling_ratio();
        let is_sub = ssr > 1; // :1604 isSubSamplingUsed

        // :1620-1630 nominal window output length: (2*window_size+1) divided
        // sequentially by each LSTM then Output ratio (only when subsampling is on).
        let mut length = 2 * window_size + 1;
        if is_sub {
            for &r in &lstm_ratios {
                length /= r;
            }
            for &r in &out_ratios {
                length /= r;
            }
        }
        let nominal_len = length; // :1630

        let cols = output.cols;
        let in_cols = input.cols;
        let input_rows = input.rows;
        let mut output_count = vec![0.0_f32; output.rows]; // :1614

        let mut jj = 0usize;
        while jj < input_rows {
            // :1638-1645 begin snapped DOWN to the ssr grid, end snapped UP.
            let mut begin = jj.saturating_sub(window_size);
            begin = (begin / ssr) * ssr;
            let mut end = begin + 2 * window_size;
            if end >= input_rows {
                end = input_rows - 1;
            }
            end = ((end + 1) / ssr) * ssr - 1;
            let length_seq = end - begin + 1; // :1646

            // :1649-1664 partial-window length recompute (sequential floors).
            let length_short = if length_seq != 2 * window_size + 1 {
                let mut ls = length_seq;
                if is_sub {
                    for &r in &lstm_ratios {
                        ls /= r;
                    }
                    for &r in &out_ratios {
                        ls /= r;
                    }
                }
                ls
            } else {
                nominal_len
            };

            // :1667-1705 process the window; accumulate sums + counts.
            if length_short > 0 {
                // block = input rows [begin, begin+length_seq), all cols. Rows are
                // contiguous in row-major, so the slice is one copy (mirrors the
                // exact `input.slice(begin..).to_owned()`, `:1668-1670`).
                let block = FastMatrix {
                    data: input.data[begin * in_cols..(begin + length_seq) * in_cols].to_vec(),
                    rows: length_seq,
                    cols: in_cols,
                };
                let out_short = self.feed_forward(&block);
                debug_assert_eq!(
                    out_short.rows, length_short,
                    "overlap window output-row count"
                );
                // :1687-1693 output sum + count at begin/ssr; `+=` seeds from the
                // caller's prior contents (cross-channel quirk).
                let obeg = begin / ssr;
                for r in 0..length_short {
                    for c in 0..cols {
                        output.data[(obeg + r) * cols + c] += out_short.get(r, c);
                    }
                    output_count[obeg + r] += 1.0;
                }
            }
            jj += window_shift;
        }

        // :1712-1716 divide by the per-row counts (0/0 -> NaN on uncovered rows).
        // `r` indexes both `output.data` (2D `r*cols+c`) and `output_count`, so a plain
        // range loop is clearest here.
        #[allow(clippy::needless_range_loop)]
        for r in 0..output.rows {
            for c in 0..cols {
                output.data[r * cols + c] /= output_count[r];
            }
        }
    }

    /// Scoring windowed forward for the Mode-7 LID Twin (Task 5), the f32 counterpart
    /// of the INFERENCE slice of `BlstmNetwork::feed_forward_scoring`
    /// (`nn/blstm.rs:1091-1186`). Mode 7 always passes `target_index >= 0`, so this
    /// mirrors: the `rows < ssr -> Zero(1, O)` guard (`:1101-1103`), the sequential
    /// output-length division (`:1106-1114`), the windowed forward, and the BINARY
    /// expansion into `[1-p, p]` when `output_size == 1` (`:1175-1185`). The exact's
    /// target construction + cost/backward are SKIPPED: the fast path is forward-only
    /// (spec S1), the forward is target-independent, and Mode 7's langID/confusion
    /// derive from the posteriors alone (the NN-cost columns are a documented
    /// divergence, like the SAD driver's). `two_sweeps` selects the TwoSweeps truncate;
    /// the driver guarantees the truncate dispatch (bails overlap/plain), so only the
    /// truncate windowing is implemented here.
    pub fn feed_forward_scoring(
        &mut self,
        input: &FastMatrix,
        window_size: usize,
        two_sweeps: bool,
    ) -> FastMatrix {
        let output_size = self.output_size;
        let rows = input.rows;
        if rows < self.sub_sampling_ratio() {
            return FastMatrix::zeros(1, output_size); // :1101-1103
        }
        // :1106-1114 output length (only re-divided when the whole-BLSTM ratio > 1).
        let mut out_len = rows;
        if self.sub_sampling_ratio() > 1 {
            for &r in &self.lstm_subsampling {
                out_len /= r;
            }
            for &r in &self.output_subsampling {
                out_len /= r;
            }
        }
        let output = self.feed_forward_truncate(input, window_size, out_len, two_sweeps);
        // :1175-1185 binary expansion into [1-p, p] (Mode 7 always has targets, so the
        // `target_index >= 0` half of the legacy gate is always true here).
        if output_size == 1 {
            let mut expanded = FastMatrix::zeros(out_len, 2);
            for ii in 0..out_len {
                let p = output.data[ii];
                expanded.data[ii * 2] = 1.0 - p;
                expanded.data[ii * 2 + 1] = p;
            }
            expanded
        } else {
            output
        }
    }

    /// f32 TwoSweeps/single-sweep truncate forward (`feed_forward_backward_truncate`,
    /// `nn/blstm.rs:1450-1577`), OUTPUT-ONLY. The exact stitches the fwd/bwd hidden
    /// states (`_OutputForward`/`_OutputBackward`) across windows + sweeps; Mode 7 reads
    /// them ONLY for `DumpLIDInternals` (off on the gate configs, the fast driver bails
    /// if it is on), so the fast path tracks only `output`. `out_len` is the caller's
    /// posterior row count (from `feed_forward_scoring`).
    fn feed_forward_truncate(
        &mut self,
        input: &FastMatrix,
        window_size: usize,
        out_len: usize,
        two_sweeps: bool,
    ) -> FastMatrix {
        let output_size = self.output_size;
        if !two_sweeps {
            // :1457-1459 single TruncateSweep.
            let mut output = FastMatrix::zeros(out_len, output_size);
            self.feed_forward_truncate_sweep(input, window_size, &mut output);
            return output;
        }

        // :1462-1467 shift/window sizing (INTEGER-division ORDER: /2 FIRST, then /ssr).
        let ssr = self.sub_sampling_ratio();
        let shift_short = (window_size / 2) / ssr;
        let shift = shift_short * ssr;
        let window_size_short = window_size / ssr;

        // :1471 input padding (first/last row replicated window_size times).
        let input_padded = pad_replicate_ends_f32(input, window_size, window_size);

        // :1490-1506 sweep 1: input drops the FRONT window_size padding, keeps the back;
        // sweep1_out is the tail of the zero-padded output (out_len + window_size_short).
        let sweep1_in = slice_rows_f32(&input_padded, window_size, input_padded.rows);
        let mut sweep1_out = FastMatrix::zeros(out_len + window_size_short, output_size);
        self.feed_forward_truncate_sweep(&sweep1_in, window_size, &mut sweep1_out);

        // :1518-1546 sweep 2: input from `shift`; output written into a fresh zero buffer
        // (output_padded2) offset by shift_short (the exact re-stitches the shifted block
        // back in place after the sweep, so we mirror that write).
        let sweep2_in = slice_rows_f32(&input_padded, shift, input_padded.rows);
        let mut sweep2_buf = FastMatrix::zeros(out_len + 2 * window_size_short, output_size);
        let mut sweep2_out = FastMatrix::zeros(sweep2_buf.rows - shift_short, output_size);
        self.feed_forward_truncate_sweep(&sweep2_in, window_size, &mut sweep2_out);
        for r in 0..sweep2_out.rows {
            let dst = (shift_short + r) * output_size;
            let src = r * output_size;
            sweep2_buf.data[dst..dst + output_size]
                .copy_from_slice(&sweep2_out.data[src..src + output_size]);
        }

        // :1548-1559 output[r] = (sweep1[r] + sweep2[r]) / 2 over the first out_len rows.
        let mut output = FastMatrix::zeros(out_len, output_size);
        for r in 0..out_len {
            for c in 0..output_size {
                let s1 = sweep1_out.data[r * output_size + c];
                let s2 = sweep2_buf.data[(window_size_short + r) * output_size + c];
                output.data[r * output_size + c] = (s1 + s2) / 2.0;
            }
        }
        output
    }

    /// One truncate sweep (`feed_forward_backward_truncate_sweep`, `:1332-1442`),
    /// OUTPUT-ONLY: non-overlapping windows of `window_size`, each a `feed_forward` over
    /// a contiguous row-slice, written at `begin/ssr`. A `length_short == 0` window is
    /// silently dropped (`:1399-1400`); a partial last window recomputes its length via
    /// the sequential sub-sampling floors (`:1382-1397`).
    fn feed_forward_truncate_sweep(
        &mut self,
        input: &FastMatrix,
        window_size: usize,
        output: &mut FastMatrix,
    ) {
        let ssr = self.sub_sampling_ratio();
        let lstm_ratios = self.lstm_subsampling.clone();
        let out_ratios = self.output_subsampling.clone();
        let is_sub = ssr > 1;
        let cols = output.cols;

        // :1354-1368 nominal length (window /= each LSTM then Output ratio when sub on).
        let mut length = window_size;
        if is_sub {
            for &r in &lstm_ratios {
                length /= r;
            }
            for &r in &out_ratios {
                length /= r;
            }
        }
        let nominal_len = length;

        let input_rows = input.rows;
        let mut jj = 0;
        while jj < input_rows {
            let begin = jj;
            let mut end = jj + window_size - 1;
            if end >= input_rows {
                end = input_rows - 1;
            }
            let length_seq = end - begin + 1;
            let length_short = if length_seq != window_size {
                let mut ls = length_seq;
                if is_sub {
                    for &r in &lstm_ratios {
                        ls /= r;
                    }
                    for &r in &out_ratios {
                        ls /= r;
                    }
                }
                ls
            } else {
                nominal_len
            };
            if length_short > 0 {
                let block = slice_rows_f32(input, begin, begin + length_seq);
                let out_short = self.feed_forward(&block);
                debug_assert_eq!(out_short.rows, length_short, "truncate sweep row count");
                let obeg = begin / ssr;
                for r in 0..length_short {
                    let dst = (obeg + r) * cols;
                    let src = r * out_short.cols;
                    output.data[dst..dst + cols].copy_from_slice(&out_short.data[src..src + cols]);
                }
            }
            jj += window_size;
        }
    }

    /// Test hook: the effective per-direction peephole flags in the
    /// `NnetSpec.peepholes` order `[fwd.cells, bwd.cells, fwd.gates, bwd.gates,
    /// fwd.gates_rec, bwd.gates_rec]`, read back from layer 0 of each direction.
    /// Pins the RIDER-1 peephole-default alignment (the fast driver builds its spec
    /// with `BlstmConfig`-default-TRUE peepholes, not `NnetSpec`'s default-FALSE).
    #[cfg(feature = "test-support")]
    pub fn debug_peepholes(&self) -> [bool; 6] {
        let f = &self.forward_layers[0];
        let b = &self.backward_layers[0];
        [
            f.cells_peep,
            b.cells_peep,
            f.gates_peep,
            b.gates_peep,
            f.gates_rec_peep,
            b.gates_rec_peep,
        ]
    }
}

// ---------------------------------------------------------------------------
// Forward kernels.
// ---------------------------------------------------------------------------

/// `sub_sample` (`network.rs:695-708`): `T x C -> floor(T/R) x C*R`, row `jj*R+kk`
/// into output columns `[kk*C, (kk+1)*C)` of row `jj`; trailing `T mod R` dropped.
/// Reads `in_data[r*in_stride + c]` for `c in 0..in_cols`.
fn sub_sample_into(
    ratio: usize,
    in_data: &[f32],
    in_rows: usize,
    in_cols: usize,
    in_stride: usize,
    out: &mut Vec<f32>,
) -> (usize, usize) {
    let out_rows = in_rows / ratio;
    let out_cols = in_cols * ratio;
    ensure_len(out, out_rows * out_cols);
    for jj in 0..out_rows {
        for kk in 0..ratio {
            let src_base = (jj * ratio + kk) * in_stride;
            let dst_base = jj * out_cols + kk * in_cols;
            out[dst_base..dst_base + in_cols]
                .copy_from_slice(&in_data[src_base..src_base + in_cols]);
        }
    }
    (out_rows, out_cols)
}

/// Extract rows `[start, end)` of a row-major FastMatrix as an owned copy (the
/// Task-5 truncate windows + sweep input slices).
fn slice_rows_f32(m: &FastMatrix, start: usize, end: usize) -> FastMatrix {
    let c = m.cols;
    FastMatrix {
        data: m.data[start * c..end * c].to_vec(),
        rows: end - start,
        cols: c,
    }
}

/// f32 `pad_replicate_ends` (`nn/blstm.rs:1791-1810`): row 0 replicated `front` times,
/// the matrix, row `r-1` replicated `back` times. Requires `m.rows >= 1` (the Mode-7
/// per-block guard ensures every scored block has at least `ssr >= 1` rows).
fn pad_replicate_ends_f32(m: &FastMatrix, front: usize, back: usize) -> FastMatrix {
    let c = m.cols;
    let r = m.rows;
    let mut out = FastMatrix::zeros(front + r + back, c);
    for k in 0..front {
        out.data[k * c..k * c + c].copy_from_slice(&m.data[0..c]);
    }
    for i in 0..r {
        out.data[(front + i) * c..(front + i) * c + c].copy_from_slice(&m.data[i * c..i * c + c]);
    }
    for k in 0..back {
        out.data[(front + r + k) * c..(front + r + k) * c + c]
            .copy_from_slice(&m.data[(r - 1) * c..(r - 1) * c + c]);
    }
    out
}

/// Copy a possibly-strided view `(rows x cols, row stride)` into a contiguous
/// `rows x cols` buffer (materializes the crop-gate view before the layer loop).
fn copy_view_into(in_data: &[f32], rows: usize, cols: usize, stride: usize, out: &mut Vec<f32>) {
    ensure_len(out, rows * cols);
    if stride == cols {
        out[..rows * cols].copy_from_slice(&in_data[..rows * cols]);
    } else {
        for r in 0..rows {
            out[r * cols..r * cols + cols].copy_from_slice(&in_data[r * stride..r * stride + cols]);
        }
    }
}

/// Batched projection via faer: `dst (t x w_cols) = X (t x k) * W (k x w_cols)` where
/// `k = min(in_cols, w_rows)` implements the layer width tolerance (`cols > I` -> left
/// I input cols; `cols < I` -> top `cols` weight rows). `in_data` is row-major with
/// `in_stride`; `w` is column-major `w_rows x w_cols`; `dst` is a contiguous row-major
/// `t x w_cols` slice. This is the mandated faer matmul site.
///
/// `pub` (Phase 7 Task 7): `benches/kernels.rs`'s `faer_project_92x96` calls this
/// EXACT function (not a bench-local reimplementation) at the `matmul_seq_92x96`
/// shape, so the criterion twin measures the real kernel every LSTM gate/dense
/// projection in this module actually dispatches through, mirroring how
/// `nn/layers.rs::matmul_seq` was widened `pub` in Task 1 for the same reason.
#[allow(clippy::too_many_arguments)]
pub fn faer_project(
    in_data: &[f32],
    t: usize,
    in_cols: usize,
    in_stride: usize,
    w: &[f32],
    w_rows: usize,
    w_cols: usize,
    dst: &mut [f32],
) {
    let k = in_cols.min(w_rows);
    let lhs = MatRef::from_row_major_slice_with_stride(&in_data[..t * in_stride], t, k, in_stride);
    let rhs = MatRef::from_column_major_slice_with_stride(w, k, w_cols, w_rows);
    let dmut = MatMut::from_row_major_slice_mut(dst, t, w_cols);
    matmul(dmut, Accum::Replace, lhs, rhs, 1.0_f32, Par::Seq);
}

/// One peephole-LSTM layer forward (`LstmLayer::feed_forward`, f32). `in_data` is the
/// contiguous `t x in_cols` layer input (post-subsample). `reverse` runs reverse-time
/// (frame `T-1` first) by indexing `src = T-1-step`, equivalent to reverse-input ->
/// forward -> reverse-output but with a single natural-order batched projection. Fills
/// `out` (`t x O`). The rolling `prev_*`/`cur_cell` scratch carry the previous step's
/// activated gates / cell / output (distinct buffers so `proj`/`out` never self-alias).
#[allow(clippy::too_many_arguments)]
fn lstm_layer_forward(
    layer: &FastLstmLayer,
    in_data: &[f32],
    t: usize,
    in_cols: usize,
    in_stride: usize,
    reverse: bool,
    proj: &mut Vec<f32>,
    prev_gates: &mut [f32],
    prev_cell: &mut [f32],
    prev_out: &mut [f32],
    cur_cell: &mut [f32],
    out: &mut Vec<f32>,
) {
    let o = layer.output_size;
    let four_o = 4 * o;
    ensure_len(proj, t * four_o);
    ensure_len(out, t * o);

    // Batched gate projection (all four blocks at once): proj[t, 0..4O].
    faer_project(
        in_data,
        t,
        in_cols,
        in_stride,
        &layer.input_weights,
        layer.input_size,
        four_o,
        &mut proj[..t * four_o],
    );

    let p = &layer.peep; // row-major 12 x O
    let fb = &layer.feedback_weights; // col-major O x 4O
    let bias = &layer.biases;

    for step in 0..t {
        let src = if reverse { t - 1 - step } else { step };
        let g0 = src * four_o;

        // + bias (all four gate blocks).
        for j in 0..four_o {
            proj[g0 + j] += bias[j];
        }

        if step > 0 {
            // Recurrence: += y_{t-1} * FeedbackWeights (all 4O columns). Feedback is
            // column-major, so gate column j's O weights are contiguous.
            for j in 0..four_o {
                let col = &fb[j * o..j * o + o];
                let mut acc = 0.0_f32;
                for k in 0..o {
                    acc += prev_out[k] * col[k];
                }
                proj[g0 + j] += acc;
            }
            // Cells peep into i,f (rows 0,1): += c_{t-1} .* P.
            if layer.cells_peep {
                for j in 0..o {
                    proj[g0 + j] += prev_cell[j] * p[j];
                }
                for j in 0..o {
                    proj[g0 + o + j] += prev_cell[j] * p[o + j];
                }
            }
            // Gates-recurrent peep into i,f (rows 3,7): += i_{t-1}/f_{t-1} .* P.
            if layer.gates_rec_peep {
                for j in 0..o {
                    proj[g0 + j] += prev_gates[j] * p[3 * o + j];
                }
                for j in 0..o {
                    proj[g0 + o + j] += prev_gates[o + j] * p[7 * o + j];
                }
            }
            // Gates peep into i,f (rows 4,5 / 6,8), each one combined expression.
            if layer.gates_peep {
                for j in 0..o {
                    proj[g0 + j] +=
                        prev_gates[o + j] * p[4 * o + j] + prev_gates[2 * o + j] * p[5 * o + j];
                }
                for j in 0..o {
                    proj[g0 + o + j] +=
                        prev_gates[j] * p[6 * o + j] + prev_gates[2 * o + j] * p[8 * o + j];
                }
            }
        }

        // Activate i,f (GatesFunction), g (Maxmin2/asinh).
        for j in 0..2 * o {
            proj[g0 + j] = gates_fn_f32(proj[g0 + j]);
        }
        for j in 0..o {
            proj[g0 + 3 * o + j] = asinh_f32(proj[g0 + 3 * o + j]);
        }

        // Cell state: c_t = i_t .* g_t (+ c_{t-1} .* f_t for step > 0).
        if step == 0 {
            for j in 0..o {
                cur_cell[j] = proj[g0 + j] * proj[g0 + 3 * o + j];
            }
        } else {
            for j in 0..o {
                cur_cell[j] = proj[g0 + j] * proj[g0 + 3 * o + j] + prev_cell[j] * proj[g0 + o + j];
            }
        }

        // o-gate extras IN ORDER: c_t.*P2 (cells); i_t.*P9, f_t.*P10 (gates);
        // o_{t-1}.*P11 (gates-rec, step > 0 only).
        if layer.cells_peep {
            for j in 0..o {
                proj[g0 + 2 * o + j] += cur_cell[j] * p[2 * o + j];
            }
        }
        if layer.gates_peep {
            for j in 0..o {
                proj[g0 + 2 * o + j] += proj[g0 + j] * p[9 * o + j];
            }
            for j in 0..o {
                proj[g0 + 2 * o + j] += proj[g0 + o + j] * p[10 * o + j];
            }
        }
        if step > 0 && layer.gates_rec_peep {
            for j in 0..o {
                proj[g0 + 2 * o + j] += prev_gates[2 * o + j] * p[11 * o + j];
            }
        }

        // Activate o; cells_in = asinh(c_t); y_t = o_t .* cells_in.
        for j in 0..o {
            proj[g0 + 2 * o + j] = gates_fn_f32(proj[g0 + 2 * o + j]);
        }
        for j in 0..o {
            out[src * o + j] = proj[g0 + 2 * o + j] * asinh_f32(cur_cell[j]);
        }

        // Roll the previous-step state forward.
        prev_gates[..four_o].copy_from_slice(&proj[g0..g0 + four_o]);
        prev_cell[..o].copy_from_slice(&cur_cell[..o]);
        prev_out[..o].copy_from_slice(&out[src * o..src * o + o]);
    }
}

/// One LSTM stack forward (`Network::drive` with LSTM layers). Materializes the
/// (crop-gate) net input into `scr.ping`, then chains layers with per-layer
/// subsampling; `reverse` runs every layer reverse-time (the backward direction).
/// The final layer output is copied into `dest`.
#[allow(clippy::too_many_arguments)]
fn lstm_net_forward(
    layers: &[FastLstmLayer],
    subs: &[usize],
    in_data: &[f32],
    in_rows: usize,
    in_cols: usize,
    in_stride: usize,
    reverse: bool,
    scr: &mut Scratch,
    dest: &mut Vec<f32>,
) -> (usize, usize) {
    // Materialize the net input contiguously (resolves the crop stride).
    copy_view_into(in_data, in_rows, in_cols, in_stride, &mut scr.ping);
    let mut cur_rows = in_rows;
    let mut cur_cols = in_cols;

    for (jj, layer) in layers.iter().enumerate() {
        // Ensure the rolling buffers fit this layer's O before the borrows split.
        let o = layer.output_size;
        ensure_len(&mut scr.prev_gates, 4 * o);
        ensure_len(&mut scr.prev_cell, o);
        ensure_len(&mut scr.prev_out, o);
        ensure_len(&mut scr.cur_cell, o);

        if subs[jj] > 1 {
            let (sr, sc) = sub_sample_into(
                subs[jj],
                &scr.ping[..cur_rows * cur_cols],
                cur_rows,
                cur_cols,
                cur_cols,
                &mut scr.sub,
            );
            lstm_layer_forward(
                layer,
                &scr.sub[..sr * sc],
                sr,
                sc,
                sc,
                reverse,
                &mut scr.proj,
                &mut scr.prev_gates,
                &mut scr.prev_cell,
                &mut scr.prev_out,
                &mut scr.cur_cell,
                &mut scr.pong,
            );
            cur_rows = sr;
        } else {
            lstm_layer_forward(
                layer,
                &scr.ping[..cur_rows * cur_cols],
                cur_rows,
                cur_cols,
                cur_cols,
                reverse,
                &mut scr.proj,
                &mut scr.prev_gates,
                &mut scr.prev_cell,
                &mut scr.prev_out,
                &mut scr.cur_cell,
                &mut scr.pong,
            );
        }
        cur_cols = o;
        std::mem::swap(&mut scr.ping, &mut scr.pong);
    }
    dest.clear();
    dest.extend_from_slice(&scr.ping[..cur_rows * cur_cols]);
    (cur_rows, cur_cols)
}

/// One dense layer forward (`NeuronLayer::feed_forward`, f32): faer projection +
/// bias, then softmax (`last && O>1`), Logistic (`last && O==1`), or asinh (hidden).
#[allow(clippy::too_many_arguments)]
fn dense_layer_forward(
    layer: &FastDenseLayer,
    in_data: &[f32],
    t: usize,
    in_cols: usize,
    in_stride: usize,
    last: bool,
    proj: &mut Vec<f32>,
    out: &mut Vec<f32>,
) {
    let o = layer.output_size;
    ensure_len(proj, t * o);
    ensure_len(out, t * o);
    faer_project(
        in_data,
        t,
        in_cols,
        in_stride,
        &layer.weights,
        layer.input_size,
        o,
        &mut proj[..t * o],
    );
    for step in 0..t {
        let base = step * o;
        for j in 0..o {
            proj[base + j] += layer.biases[j];
        }
        let pre = &proj[base..base + o];
        let orow = &mut out[base..base + o];
        if last && o > 1 {
            softmax_row_f32(pre, orow);
        } else if last {
            for j in 0..o {
                orow[j] = logistic_f32(pre[j]);
            }
        } else {
            for j in 0..o {
                orow[j] = asinh_f32(pre[j]);
            }
        }
    }
}

/// The dense output MLP stack (`Network::drive` with NeuronLayers). `in_data` is the
/// HCAT (`in_rows x in_cols`); it is materialized into `scr.ping`, then layers chain
/// with per-layer subsampling. The final (last) layer applies softmax/Logistic and
/// writes `dest`; hidden layers apply asinh into the ping/pong scratch.
fn dense_net_forward(
    layers: &[FastDenseLayer],
    subs: &[usize],
    in_data: &[f32],
    in_rows: usize,
    in_cols: usize,
    scr: &mut Scratch,
    dest: &mut Vec<f32>,
) -> (usize, usize) {
    copy_view_into(in_data, in_rows, in_cols, in_cols, &mut scr.ping);
    let mut cur_rows = in_rows;
    let mut cur_cols = in_cols;
    let n = layers.len();
    for (jj, layer) in layers.iter().enumerate() {
        let last = jj == n - 1;
        let (src_rows, src_cols) = if subs[jj] > 1 {
            sub_sample_into(
                subs[jj],
                &scr.ping[..cur_rows * cur_cols],
                cur_rows,
                cur_cols,
                cur_cols,
                &mut scr.sub,
            )
        } else {
            (cur_rows, cur_cols)
        };
        let use_sub = subs[jj] > 1;
        let dst: &mut Vec<f32> = if last { dest } else { &mut scr.pong };
        if use_sub {
            dense_layer_forward(
                layer,
                &scr.sub[..src_rows * src_cols],
                src_rows,
                src_cols,
                src_cols,
                last,
                &mut scr.proj,
                dst,
            );
        } else {
            dense_layer_forward(
                layer,
                &scr.ping[..cur_rows * cur_cols],
                cur_rows,
                cur_cols,
                cur_cols,
                last,
                &mut scr.proj,
                dst,
            );
        }
        if last {
            return (src_rows, layer.output_size);
        }
        cur_rows = src_rows;
        cur_cols = layer.output_size;
        std::mem::swap(&mut scr.ping, &mut scr.pong);
    }
    (cur_rows, cur_cols)
}
