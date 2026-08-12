//! `fast::bicell` -- the BIDIRECTIONAL f32 twins of the new recurrent cells
//! (Phase 10 Task 7, spec S5): `Direction bidirectional` x `Cell_Type
//! {slstm, mamba, cfc}`, the combination the phase-9 fast tree typed-bailed.
//!
//! [`FastBiCell`] is the f32 counterpart of `nn::blstm::BlstmNetwork` under
//! `Direction::Bidirectional` with a `CellLayer` other than `Lstm`. Structurally it is
//! `FastCausalNet` with a SECOND stack: forward stack + REVERSED stack -> hcat ->
//! the shared per-row `DenseRowChain` output MLP. The exact sources it mirrors (cite
//! when auditing drift):
//!
//! - `nn/blstm.rs::feed_forward` (`:1123-1159`): the sequential output-length division,
//!   the `LSTMRatios[0] > 1 && netInput < inputCols` crop gate feeding BOTH stacks the
//!   SAME (cropped) input, `forward.feed_forward` + `backward.feed_forward_reverse`,
//!   then `output_network.feed_forward_double(out_forward, out_backward, output)`.
//! - `nn/network.rs::feed_forward_double` (`:408-439`): the hcat, FORWARD HALF ON THE
//!   LEFT (`hcat[t][0..fc] = first`, `hcat[t][fc..] = second`, and `feed_forward`
//!   passes `first = out_forward`). [`FastBiCell::feed_forward`] writes exactly that
//!   column order into its per-row buffer, and
//!   `output_mlp_reads_the_forward_half_from_the_left_columns` pins it directly.
//! - `nn/network.rs::drive` (`:328-394`) + `nn/cells/mod.rs`'s
//!   `Layer::feed_forward_reverse`: the reverse stack is the SAME ascending layer chain
//!   with each layer's TIME direction flipped, and the per-layer sub-sampling still runs
//!   on natural-order rows. Reproduced by `super::cells::cell_stack_forward(.., reverse
//!   = true)`, which is the causal driver with one flag -- not a second copy.
//! - `nn/blstm.rs::feed_forward_backward_overlap` (`:1928-2083`, and see the CITATION
//!   OFFSET note on [`FastBiCell::feed_forward_overlap`]'s table): the windowed OVERLAP
//!   driver, reproduced FRESH in [`FastBiCell::feed_forward_overlap`] (see its docs for
//!   the term-by-term correspondence and why it is a fresh loop rather than a call into
//!   `FastBlstm::overlap_window_step`).
//!
//! **`FastBlstm` AND `overlap_window_step` ARE BYTE-UNTOUCHED** (spec S5, approach A
//! ratified). They are the phase-8 streaming bit-identity kernels: `fast::stream::
//! StreamOverlap` fires the SAME `overlap_window_step` the offline
//! `FastBlstm::feed_forward_overlap` does, and that shared-kernel identity is what makes
//! streamed-vs-offline bit-equality hold BY CONSTRUCTION. Threading a second net shape
//! through that kernel would have put a branch inside it. Instead this module writes its
//! own accumulate/average loop over the ALREADY-SHARED span helpers
//! (`super::nn::window_begin` / `window_end`), so the window GRID is one fact in one
//! place while the accumulation stays local. The phase-7/8 suites passing WITHOUT EDITS
//! is the behavior-preservation proof.
//!
//! **NO STREAMING TWIN, AND NOT AS AN OMISSION.** A bidirectional net is UNSTREAMABLE BY
//! CONSTRUCTION: the reverse stack's state at time `t` is a function of the samples
//! AFTER `t`, so its output at the first frame depends on the last one -- there is no
//! bounded lookahead that makes it causal, unlike the phase-8 windowed BLSTM (bounded by
//! the window) or the phase-9 causal cells (no lookahead at all). `fast::stream::
//! StreamingSession::new` therefore typed-bails this shape, pinned by `phase8_gate.rs` +
//! `phase9_stream_causal.rs`, BOTH UNMODIFIED. Precisely why they stay green: the refusal
//! moved from `classify_fast_shape` to the streaming session with its LEADING CLAUSE
//! preserved (`cell type '<x>' is not supported on the fast inference path ... in the
//! BIDIRECTIONAL direction`), which is exactly what those legs assert -- a PREFIX
//! SUBSTRING, not the whole body. The TAIL was rewritten, deliberately: the classifier's
//! old advice ("run this config on the exact path") is now WRONG, because the offline
//! fast path implements this shape.
//!
//! SCOPE, same posture as the rest of `fast/`: INFERENCE-ONLY (forward, no backward, no
//! trainer), no MLP mode, and the two windowing regimes the exact tree's algo-3 SAD
//! driver actually dispatches -- PLAIN (`window_size == 0`) and OVERLAP. The TRUNCATE
//! (non-overlap) variant typed-bails in `super::driver`, exactly as it does for
//! `FastBlstm`. f32 numeric divergence from the exact f64 tree is BY DESIGN (spec S4/S5),
//! documented here + in `RESULTS.md`, NEVER `IMPROVEMENTS.md`.

use anyhow::{Result, bail};

use crate::config::NnetSpec;
use crate::nn::blstm::{CellType, CfcParams, MambaParams, TransformerParams};

use super::cells::{
    FastCell, backward_peep, build_cell, build_dense_tail, cell_stack_forward, check_net_spec,
    drive_output_rows, forward_peep, stack_element_count, stack_input_cols,
};
use super::nn::{DenseRowChain, FastDenseLayer, FastMatrix, Scratch, window_begin, window_end};

/// The f32 bidirectional cell net: TWO cell stacks (forward + reversed) feeding the
/// shared per-row dense output MLP.
///
/// Flat pack layout, `BlstmNetwork::set_weights` (`nn/blstm.rs:766-791`) element for
/// element: `[forward stack | backward stack | output MLP | mean | std]`. Both stacks
/// walk their layers layer-0-first (`Network::set_weights`, `nn/network.rs:291-297`),
/// and the `2 * lstm_neuron_nb[0]` mean/std tail is CARRIED, not applied -- the driver
/// normalizes upstream, exactly as it does for `FastBlstm` and `FastCausalNet`.
///
/// STACKS AND SUB-SAMPLING are supported for the same reason the causal net supports
/// them (the real SAD arms are `23,24,24` / `4,1`), and by the same mechanism:
/// `super::cells::cell_stack_forward`, i.e. `Network::drive` semantics reused rather
/// than new logic.
#[derive(Clone)]
pub struct FastBiCell {
    lstm_neuron_nb: Vec<usize>,
    lstm_subsampling: Vec<usize>,
    output_subsampling: Vec<usize>,
    output_size: usize,

    cells_fwd: Vec<FastCell>,
    cells_rev: Vec<FastCell>,
    output_layers: Vec<FastDenseLayer>,

    normalize_mean: Vec<f32>,
    normalize_std: Vec<f32>,

    /// Per-stack workspaces. TWO of them, not one reused twice: the stacks run
    /// back-to-back over the same input and each keeps its hidden output alive for the
    /// hcat, so sharing a `Scratch` would mean copying one stack's result out before the
    /// other could start. The preallocation contract (never allocate per frame) is
    /// unchanged.
    scratch_fwd: Scratch,
    scratch_rev: Scratch,
    hidden_fwd: Vec<f32>,
    hidden_rev: Vec<f32>,
    /// The hcat row handed to the dense chain (`fwd | rev`), reused across rows.
    row_buf: Vec<f32>,
    output: FastMatrix,
    /// The dense output MLP's PER-ROW driver -- the same [`DenseRowChain`] the causal net
    /// runs. Reused across calls and `reset` at the top of every [`Self::feed_forward`],
    /// so a reused net starts where a fresh one would (the run-twice bit-identity
    /// contract, which the overlap loop depends on: it calls `feed_forward` once per
    /// window).
    dense_chain: DenseRowChain,
}

impl FastBiCell {
    /// The whole-net element count: TWO cell stacks + the output MLP + the
    /// `2 * input_size` normalize tail. Mirrors `BlstmNetwork::nb_of_weights` with both
    /// recurrent stacks present.
    pub fn element_count(
        spec: &NnetSpec,
        cell_type: CellType,
        mamba: &MambaParams,
        cfc: &CfcParams,
        transformer: &TransformerParams,
    ) -> Result<usize> {
        check_net_spec(spec, "fast::bicell::FastBiCell")?;
        // THE LSTM REFUSAL, now EXPLICIT (phase-10 Task 8). It used to fall out of
        // `cell_weight_count`'s `None` sentinel, which that task retired when it gave the
        // LSTM a real f32 cell kernel ([`super::cells::FastLstm`]). The refusal itself is
        // UNCHANGED and still load-bearing: a bidirectional LSTM's fast twin is
        // `super::nn::FastBlstm` (batched faer projection, phase-7-pinned), and routing
        // one through a cell stack would silently swap it for the per-step kernel. It is
        // also unreachable through the dispatch -- `classify_fast_shape` maps
        // `(lstm, bidirectional)` to `FastNetShape::Blstm` -- so this is the belt to that
        // braces, pinned by `bicell_bails_on_the_lstm_cell`. It stays HERE, after the
        // shape guard and before the count, which is the order the pre-dedupe copy had.
        if cell_type == CellType::Lstm {
            bail!(
                "fast::bicell::FastBiCell is for the phase-9/10 cells only; the legacy \
                 peephole LSTM's bidirectional fast twin is `super::nn::FastBlstm` (spec S5)"
            );
        }
        // TWO stacks: the forward and backward blocks are separate weight blocks of
        // identical shape (`BlstmNetwork::from_config` builds both from the same
        // `lstm_neuron_nb`), which is the whole difference from the causal net's count.
        Ok(stack_element_count(
            spec,
            cell_type,
            mamba,
            cfc,
            transformer,
            2,
        ))
    }

    /// Build from a `NnetSpec` + the cell type/geometry + the flat f64 pack, narrowing
    /// f64 -> f32 once per block (the `FastBlstm::from_flat` / `FastCausalNet::from_flat`
    /// contract: the narrowing is the ONLY lossy step, and it happens AFTER adim -- new
    /// cells carry no adim by construction, spec S1.3).
    ///
    /// Errors on MLP mode, on the LSTM cell (whose bidirectional fast twin is
    /// `FastBlstm`), and on a pack shorter than [`Self::element_count`] -- the length
    /// check that stops a FORWARD-sized pack (short by exactly one stack) or another
    /// architecture's pack from being consumed head-first. An OVER-long pack consumes
    /// only the head, as the legacy does.
    pub fn from_flat(
        spec: &NnetSpec,
        cell_type: CellType,
        mamba: &MambaParams,
        cfc: &CfcParams,
        transformer: &TransformerParams,
        flat: &[f64],
    ) -> Result<FastBiCell> {
        if spec.lstm_neuron_nb.is_empty() || spec.lstm_neuron_nb[0] == 0 {
            bail!(
                "fast::bicell::FastBiCell is recurrent-only; MLP mode (LSTMNeuronNb[0]==0) is \
                 unsupported"
            );
        }
        let needed = Self::element_count(spec, cell_type, mamba, cfc, transformer)?;
        if flat.len() < needed {
            bail!(
                "flat weight vector too short for the fast bidirectional net: {} < {needed}",
                flat.len()
            );
        }

        let lstm = &spec.lstm_neuron_nb;
        let lsub = &spec.lstm_subsampling;
        let outn = &spec.output_neuron_nb;
        let osub = &spec.output_subsampling;

        let mut pos = 0usize;
        // Pack ORDER: the WHOLE forward stack, then the WHOLE backward stack -- not
        // interleaved per layer (`BlstmNetwork::set_weights` hands the tail from
        // `forward_network.set_weights` to `backward_network.set_weights`, and each of
        // those walks all of its own layers).
        let stack = |pos: &mut usize, peep: [bool; 3]| -> Vec<FastCell> {
            let mut cells = Vec::with_capacity(lstm.len() - 1);
            for jj in 0..lstm.len() - 1 {
                let i = lstm[jj] * lsub[jj];
                let o = lstm[jj + 1];
                let (cell, used) = build_cell(
                    cell_type,
                    &flat[*pos..],
                    i,
                    o,
                    mamba,
                    cfc,
                    transformer,
                    peep,
                );
                cells.push(cell);
                *pos += used;
            }
            cells
        };
        // Each stack takes ITS OWN peephole triple, mirroring `BlstmNetwork::from_config`
        // (`nn/blstm.rs:588-602`: `make(cfg.forward_peep)` then `make(cfg.backward_peep)`).
        // Read by the LSTM cell alone, which `element_count` above refuses here -- so this
        // threading is correctness-by-construction for a cell this net cannot currently
        // build, not live behaviour. Passing one triple to both stacks would be the
        // subtle wrong thing to leave lying around.
        let cells_fwd = stack(&mut pos, forward_peep(spec));
        let cells_rev = stack(&mut pos, backward_peep(spec));

        let (output_layers, normalize_mean, normalize_std) =
            build_dense_tail(flat, &mut pos, outn, osub, lstm[0]);
        // Construction-time, once per net: a plain assert (not `debug_assert`), since a
        // layout/count disagreement here silently decodes the WHOLE pack wrong and the
        // release build is exactly where that must not pass quietly.
        assert_eq!(pos, needed, "from_flat consumed != element_count");

        let dense_chain = DenseRowChain::new(output_layers.len());
        Ok(FastBiCell {
            lstm_neuron_nb: lstm.clone(),
            lstm_subsampling: lsub.clone(),
            output_subsampling: osub.clone(),
            output_size: *outn.last().unwrap(),
            cells_fwd,
            cells_rev,
            output_layers,
            normalize_mean,
            normalize_std,
            scratch_fwd: Scratch::default(),
            scratch_rev: Scratch::default(),
            hidden_fwd: Vec::new(),
            hidden_rev: Vec::new(),
            row_buf: Vec::new(),
            output: FastMatrix::zeros(0, 0),
            dense_chain,
        })
    }

    /// The pack-carried mean/std tail (`BlstmNetwork::normalize_input_mean/std`). Not
    /// applied here -- the driver normalizes upstream, matching `FastBlstm`.
    pub fn normalize_mean(&self) -> &[f32] {
        &self.normalize_mean
    }
    pub fn normalize_std(&self) -> &[f32] {
        &self.normalize_std
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// `getSubSamplingRatio` (`nn/blstm.rs:403-410`): the recurrent sub-samplings times
    /// the output-net ones.
    pub fn sub_sampling_ratio(&self) -> usize {
        self.lstm_subsampling.iter().product::<usize>()
            * self.output_subsampling.iter().product::<usize>()
    }

    /// Whole-sequence bidirectional forward, the f32 twin of `BlstmNetwork::feed_forward`
    /// (`nn/blstm.rs:1123-1159`) under `Direction::Bidirectional`:
    ///
    /// 1. output length = rows divided SEQUENTIALLY by each recurrent sub-sampling ratio;
    /// 2. the `LSTMRatios[0] > 1 && netInput < inputCols` crop gate, applied ONCE and fed
    ///    to BOTH stacks (`:1142-1147` passes the same `cropped` to `forward` and
    ///    `backward`);
    /// 3. the forward stack, then the REVERSED stack (`feed_forward_reverse`);
    /// 4. the hcat `[fwd | rev]` (forward LEFT, `feed_forward_double`) driving the dense
    ///    output MLP one row at a time.
    ///
    /// THE DENSE STAGE IS PER-ROW, through the same [`DenseRowChain`] the causal net
    /// uses. Not for streaming (this shape has none) but for the reason that made the
    /// chain exist: `faer_project`'s result depends on the ROW COUNT once the weight
    /// matrix has more than one column, so a windowed run (whose per-window row counts
    /// differ) and a whole-sequence run would disagree in the last f32 ULP at a wide
    /// hidden dense layer. Per-row makes the plain and overlap regimes read the same
    /// kernel at the same granularity.
    pub fn feed_forward(&mut self, input: &FastMatrix) -> &FastMatrix {
        let frames = input.rows;
        let Some(in_cols) = stack_input_cols(&self.lstm_subsampling, self.lstm_neuron_nb[0], input)
        else {
            self.output = FastMatrix::zeros(0, self.output_size);
            return &self.output;
        };

        let (rows_f, cols_f) = cell_stack_forward(
            &self.cells_fwd,
            &self.lstm_subsampling,
            &input.data,
            frames,
            in_cols,
            input.cols,
            &mut self.scratch_fwd,
            &mut self.hidden_fwd,
            false,
        );
        let (rows_r, cols_r) = cell_stack_forward(
            &self.cells_rev,
            &self.lstm_subsampling,
            &input.data,
            frames,
            in_cols,
            input.cols,
            &mut self.scratch_rev,
            &mut self.hidden_rev,
            true,
        );
        // Both stacks are built from the same `lstm_neuron_nb`/`lstm_subsampling`, so
        // this is a construction invariant, not input-dependent -- hence `debug_assert`.
        debug_assert_eq!(
            (rows_f, cols_f),
            (rows_r, cols_r),
            "bi stack shape mismatch"
        );

        // hcat + dense, ONE row at a time. Column order: FORWARD HALF LEFT
        // (`Network::feed_forward_double`, `nn/network.rs:430-437`) -- the shared drive
        // stages `[fwd | rev]` into `row_buf` in exactly that order.
        drive_output_rows(
            &mut self.dense_chain,
            &self.output_layers,
            &self.output_subsampling,
            rows_f,
            (&self.hidden_fwd, cols_f),
            Some((&self.hidden_rev, cols_r)),
            &mut self.row_buf,
            &mut self.output,
            self.output_size,
        );
        &self.output
    }

    /// Overlapping-window forward accumulation, the f32 twin of the FORWARD half of
    /// `BLSTMNeuralNetwork::feedForwardBackwardOverLap`
    /// (`nn/blstm.rs::feed_forward_backward_overlap`, `:1928-2083`).
    ///
    /// CITATION OFFSET, stated once for every `nn/blstm.rs` line number in this doc and the
    /// table below (including the `:1928-2083` span above and the one in this module's own
    /// header): they are AS OF TASK 7 (`c19bfac`). Task 9's retention work (`471c0be`)
    /// inserted 55 lines above `:1963`, so ADD +55 to read them at HEAD -- e.g. the nominal
    /// length block cited as `:1964-1973` lives at `:2019-2028` today. The numbers were left
    /// as-authored rather than renumbered because they will drift again; the SYMBOL names
    /// (`feed_forward_backward_overlap`, `window_begin`, `window_end`) are the stable handles,
    /// and replacing line citations with symbol/anchor references tree-wide is a named
    /// follow-on in `RESULTS.md`.
    ///
    /// A FRESH loop, deliberately (spec S5, approach A): it does NOT call
    /// `FastBlstm::overlap_window_step`, because that kernel is shared with
    /// `fast::stream::StreamOverlap` and its bit-identity contract is what phase 8's gate
    /// rests on -- putting a second net shape through it would put a branch inside it.
    /// What IS shared is the window GRID: [`window_begin`] / [`window_end`], the same
    /// `pub(crate)` span helpers the offline and streaming overlap engines use, so the
    /// three cannot disagree about WHICH rows a window spans.
    ///
    /// TERM FOR TERM against the exact driver (line numbers are `nn/blstm.rs`):
    ///
    /// | exact | here |
    /// |---|---|
    /// | `:1964-1973` nominal length `(2w+1)` / each LSTM then output ratio, gated on `is_sub` | identical |
    /// | `:1981-1982` begin = `max(0, jj-w)` snapped DOWN to the ssr grid | [`window_begin`] |
    /// | `:1984-1988` end = `begin + 2w`, clamped to `rows-1`, snapped UP | [`window_end`] |
    /// | `:1992-2007` partial-window `length_short` recompute (sequential floors) | identical |
    /// | `:2011-2013` block = input rows `[begin, begin+length_seq)` | identical (one row-major copy) |
    /// | `:2026` `feed_forward_backward_plain` over the block | [`Self::feed_forward`] |
    /// | `:2030-2036` `output[obeg+r] += short[r]`, `count[obeg+r] += 1` at `obeg = begin/ssr` | identical |
    /// | `:2049` `jj += window_shift` | identical |
    /// | `:2055-2059` divide by the counts, `0/0 -> NaN` on uncovered rows, NO guard | identical |
    ///
    /// NOT reproduced, because the SAD result path never reads them: the
    /// `_OutputForward`/`_OutputBackward` hidden-state accumulators and their own count
    /// quotient (`:1955-1958`, `:2038-2047`, `:2073-2082`) -- only the LID Twin's
    /// hidden-state concat consumes those, and it is out of the algo-3 driver's scope
    /// (the same omission `FastBlstm::feed_forward_overlap` documents).
    ///
    /// THE ACCUMULATION ORDER IS `jj`-ASCENDING, and that is the arithmetic contract, not
    /// an implementation detail: f32 `+=` is not associative, so a row covered by several
    /// windows must sum their contributions in the exact order the exact driver does.
    /// `jj` starts at 0 and steps up by `window_shift`, windows fire in that order, and
    /// each fires exactly once -- the same convention `overlap_window_step`'s docs state
    /// for the `FastBlstm` pair.
    ///
    /// `output` is the CALLER'S buffer and is NOT zeroed here: the exact driver reuses
    /// ONE `result_vec` across channels and the `+=` seeds channel N from channel N-1's
    /// post-division contents (the load-bearing cross-channel reuse quirk,
    /// `tasks/sad.rs:1433-1436`).
    pub fn feed_forward_overlap(
        &mut self,
        input: &FastMatrix,
        window_size: usize,
        window_shift: usize,
        output: &mut FastMatrix,
    ) {
        // Cloned out of `self` before the loop: `feed_forward` below takes `&mut self`.
        let lstm_ratios = self.lstm_subsampling.clone();
        let out_ratios = self.output_subsampling.clone();
        let ssr = self.sub_sampling_ratio();
        let is_sub = ssr > 1; // :1938

        // :1964-1973 nominal window output length.
        let mut length = 2 * window_size + 1;
        if is_sub {
            for &r in &lstm_ratios {
                length /= r;
            }
            for &r in &out_ratios {
                length /= r;
            }
        }
        let nominal_len = length; // :1973

        let cols = output.cols;
        let in_cols = input.cols;
        let input_rows = input.rows;
        let mut output_count = vec![0.0_f32; output.rows]; // :1957

        let mut jj = 0usize;
        while jj < input_rows {
            let begin = window_begin(jj, window_size, ssr); // :1981-1982
            let end = window_end(begin, window_size, ssr, input_rows); // :1984-1988
            let length_seq = end - begin + 1; // :1989

            // :1992-2007 partial-window length recompute (sequential floors).
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

            if length_short > 0 {
                // :2011-2013 block = input rows [begin, begin+length_seq), all cols.
                // Rows are contiguous in row-major, so the slice is one copy.
                let block = FastMatrix {
                    data: input.data[begin * in_cols..(begin + length_seq) * in_cols].to_vec(),
                    rows: length_seq,
                    cols: in_cols,
                };
                let obeg = begin / ssr;
                let out = self.feed_forward(&block);
                debug_assert_eq!(out.rows, length_short, "overlap window output-row count");
                // :2030-2036 output sum + count; `+=` seeds from the caller's prior
                // contents (cross-channel quirk).
                for r in 0..length_short {
                    for c in 0..cols {
                        output.data[(obeg + r) * cols + c] += out.get(r, c);
                    }
                    output_count[obeg + r] += 1.0;
                }
            }
            jj += window_shift; // :2049
        }

        // :2055-2059 divide by the per-row counts (0/0 -> NaN on uncovered rows).
        // `r` indexes both `output.data` (2D `r*cols+c`) and `output_count`, so a plain
        // range loop is clearest here.
        #[allow(clippy::needless_range_loop)]
        for r in 0..output.rows {
            for c in 0..cols {
                output.data[r * cols + c] /= output_count[r];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // `cell_weight_count` is a TEST-only import here since the phase-11 Task 1 dedupe:
    // `element_count` now reaches it through `super::cells::stack_element_count`, while
    // these legs still size a single stack directly to cross-check the count.
    use crate::fast::cells::{FastCfc, FastMamba, FastSlstm, cell_weight_count};

    fn spec(lstm: &[usize], lsub: &[usize], outn: &[usize], osub: &[usize]) -> NnetSpec {
        NnetSpec {
            input_size: lstm[0],
            lstm_neuron_nb: lstm.to_vec(),
            lstm_subsampling: lsub.to_vec(),
            output_neuron_nb: outn.to_vec(),
            output_subsampling: osub.to_vec(),
            peepholes: [false; 6],
        }
    }

    /// A BOUNDED deterministic fill (the `fast::cells` convention): a linear ramp over a
    /// net-sized pack saturates the output logistic to a constant column, which makes
    /// every value comparison below vacuous.
    fn weights(n: usize) -> Vec<f64> {
        (0..n)
            .map(|k| {
                0.29 * ((0.7 + 0.013 * (k as f64)).sin()) + 0.05 * ((k % 7) as f64 - 3.0) * 0.1
            })
            .collect()
    }

    fn seq(rows: usize, cols: usize, seed: f64) -> FastMatrix {
        let mut m = FastMatrix::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                m.data[r * cols + c] = (seed + 0.17 * (r as f64) - 0.23 * (c as f64)
                    + 0.011 * ((r * cols + c) as f64))
                    .sin() as f32;
            }
        }
        m
    }

    fn mamba_params() -> MambaParams {
        MambaParams {
            d_state: 4,
            d_conv: 3,
            expand: 2,
            dt_rank: 0,
        }
    }

    fn cfc_params() -> CfcParams {
        CfcParams {
            backbone_units: 4,
            backbone_layers: 2,
        }
    }

    /// The transformer geometry the legs below thread through the shared sizing/build
    /// functions. INERT for every cell in [`CELLS`] -- the transformer row itself is
    /// phase-11 Task 6's, and this exists only so the shared signatures line up.
    /// `heads = 1` keeps it valid at any cell width, which matters because these specs
    /// carry 5-wide layers.
    fn transformer_params() -> TransformerParams {
        TransformerParams {
            window: 3,
            heads: 1,
            d_ff: 5,
        }
    }

    const CELLS: [CellType; 3] = [CellType::Slstm, CellType::Mamba, CellType::Cfc];

    /// The pack is TWO stacks + the head: `element_count` is the causal count plus one
    /// more stack, cross-checked against an independent per-cell arithmetic.
    #[test]
    #[allow(clippy::identity_op)]
    fn element_count_is_two_stacks_plus_the_head() {
        let sp = spec(&[6, 5, 4], &[2, 1], &[8, 3, 1], &[1, 1]);
        let p = mamba_params();
        let c = cfc_params();
        let tf = transformer_params();
        for cell in CELLS {
            let n = FastBiCell::element_count(&sp, cell, &p, &c, &tf).unwrap();
            let stack = match cell {
                CellType::Slstm => {
                    FastSlstm::weight_count(6 * 2, 5) + FastSlstm::weight_count(5 * 1, 4)
                }
                CellType::Mamba => {
                    FastMamba::weight_count(12, 5, p.d_state, p.d_conv, p.expand, p.dt_rank)
                        + FastMamba::weight_count(5, 4, p.d_state, p.d_conv, p.expand, p.dt_rank)
                }
                CellType::Cfc => {
                    FastCfc::weight_count(12, 5, c.backbone_units, c.backbone_layers)
                        + FastCfc::weight_count(5, 4, c.backbone_units, c.backbone_layers)
                }
                // Neither is in the iterated set: the bidirectional LSTM's fast twin is
                // `FastBlstm` (not a cell stack), and the transformer's f32 kernel lands in
                // phase-11 T5/T6.
                CellType::Lstm | CellType::Transformer => unreachable!(),
            };
            assert_eq!(
                n,
                2 * stack + (8 * 3 + 3) + (3 * 1 + 1) + 2 * 6,
                "{cell:?} bidirectional element count"
            );
            // ...and `from_flat` consumes EXACTLY it (the ctor assert), while one
            // element less is refused.
            FastBiCell::from_flat(&sp, cell, &p, &c, &tf, &weights(n)).unwrap();
            let err = match FastBiCell::from_flat(&sp, cell, &p, &c, &tf, &weights(n - 1)) {
                Err(e) => e,
                Ok(_) => panic!("{cell:?}: a one-element-short pack must be refused"),
            };
            assert!(err.to_string().contains("too short"), "{cell:?}: {err}");
        }
    }

    /// The LSTM cell has no bidirectional twin HERE -- that shape is `FastBlstm`, which
    /// this task leaves byte-untouched.
    #[test]
    fn lstm_cell_is_refused() {
        let sp = spec(&[6, 4], &[1], &[8, 1], &[1]);
        let err = FastBiCell::element_count(
            &sp,
            CellType::Lstm,
            &mamba_params(),
            &cfc_params(),
            &transformer_params(),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("FastBlstm"),
            "expected the LSTM refusal to name FastBlstm, got: {err}"
        );
    }

    /// THE REVERSE STACK REALLY RUNS BACKWARD IN TIME. Feeding the SAME weights and the
    /// same input to both stacks (a pack whose two stack blocks are byte-identical), the
    /// two hidden halves must DIFFER -- otherwise `reverse` is being ignored and the net
    /// is two copies of the forward pass.
    #[test]
    fn the_reverse_stack_is_not_the_forward_stack() {
        let sp = spec(&[4, 3], &[1], &[6, 1], &[1]);
        let p = mamba_params();
        let c = cfc_params();
        let tf = transformer_params();
        for cell in CELLS {
            let stack = cell_weight_count(cell, 4, 3, &p, &c, &tf);
            let n = FastBiCell::element_count(&sp, cell, &p, &c, &tf).unwrap();
            // Make the two stacks byte-IDENTICAL, so any difference in their outputs is
            // the time direction and nothing else.
            let mut flat = weights(n);
            let (head, tail) = flat.split_at_mut(stack);
            tail[..stack].copy_from_slice(head);
            let mut net = FastBiCell::from_flat(&sp, cell, &p, &c, &tf, &flat).unwrap();
            let input = seq(9, 4, 0.3);
            net.feed_forward(&input);
            assert_eq!(net.hidden_fwd.len(), net.hidden_rev.len());
            let differing = net
                .hidden_fwd
                .iter()
                .zip(net.hidden_rev.iter())
                .filter(|(a, b)| a != b)
                .count();
            assert!(
                differing > 0,
                "{cell:?}: the reverse stack reproduced the forward stack exactly -- the time \
                 flip is not happening"
            );
            // ...and WHICH row each pass starts on is pinned directly, by stepping the
            // cell kernel ONCE from a fresh state (single layer, sub-sampling 1, no crop
            // -- so a hidden row is exactly one `step`):
            //   - the FORWARD pass's first output row (row 0) is `step(input[0])`;
            //   - the REVERSE pass's first output row is the LAST one (row T-1), and it
            //     is `step(input[T-1])`.
            // Swap the direction flag and both of these fail.
            let (o, t_last, in_cols) = (3usize, input.rows - 1, input.cols);
            let probe = |c: &FastCell, row: usize| -> Vec<f32> {
                let mut st = c.state();
                let mut out = vec![0.0_f32; o];
                c.step(
                    &input.data[row * in_cols..row * in_cols + in_cols],
                    &mut st,
                    &mut out,
                );
                out
            };
            assert_eq!(
                probe(&net.cells_fwd[0], 0),
                net.hidden_fwd[..o].to_vec(),
                "{cell:?}: the forward pass does not start at row 0"
            );
            assert_eq!(
                probe(&net.cells_rev[0], t_last),
                net.hidden_rev[t_last * o..t_last * o + o].to_vec(),
                "{cell:?}: the reverse pass does not start at row T-1"
            );
        }
    }

    /// THE HCAT COLUMN ORDER, pinned directly rather than inferred from the parity
    /// numbers: the FORWARD half occupies the LEFT columns of the dense layer's input
    /// (`Network::feed_forward_double`, forward = `first`).
    ///
    /// Method: zero the dense layer's weight ROWS for one half and perturb the OTHER
    /// stack's weights. With the RIGHT (reverse) rows zeroed, perturbing the REVERSE
    /// stack must change NOTHING; perturbing the FORWARD stack must change the output.
    /// A swapped hcat inverts both statements.
    #[test]
    fn output_mlp_reads_the_forward_half_from_the_left_columns() {
        let hidden = 3usize;
        let sp = spec(&[4, hidden], &[1], &[2 * hidden, 1], &[1]);
        let p = mamba_params();
        let c = cfc_params();
        let tf = transformer_params();
        for cell in CELLS {
            let stack = cell_weight_count(cell, 4, hidden, &p, &c, &tf);
            let n = FastBiCell::element_count(&sp, cell, &p, &c, &tf).unwrap();
            let base = weights(n);
            // The single dense layer sits after both stacks; col-major (I x O) with
            // I = 2*hidden, O = 1, so its weight rows are `base[2*stack .. 2*stack + I]`.
            let dense0 = 2 * stack;
            let input = seq(11, 4, -0.4);

            // Perturb the FORWARD stack (block 0) / the REVERSE stack (block 1).
            let bump = |src: &[f64], at: usize| -> Vec<f64> {
                let mut v = src.to_vec();
                for w in v[at..at + stack].iter_mut() {
                    *w += 0.37;
                }
                v
            };
            let run = |flat: &[f64]| -> Vec<f32> {
                let mut net = FastBiCell::from_flat(&sp, cell, &p, &c, &tf, flat).unwrap();
                net.feed_forward(&input).data.clone()
            };

            // (a) zero the RIGHT half of the dense input rows -> only the forward stack
            //     can reach the output.
            let mut left_only = base.clone();
            left_only[dense0 + hidden..dense0 + 2 * hidden].fill(0.0);
            assert_eq!(
                run(&left_only),
                run(&bump(&left_only, stack)),
                "{cell:?}: with the RIGHT dense rows zeroed, the REVERSE stack still reached \
                 the output -- the hcat halves are swapped"
            );
            assert_ne!(
                run(&left_only),
                run(&bump(&left_only, 0)),
                "{cell:?}: with the RIGHT dense rows zeroed, the FORWARD stack did NOT reach \
                 the output"
            );

            // (b) the mirror image: zero the LEFT half.
            let mut right_only = base.clone();
            right_only[dense0..dense0 + hidden].fill(0.0);
            assert_eq!(
                run(&right_only),
                run(&bump(&right_only, 0)),
                "{cell:?}: with the LEFT dense rows zeroed, the FORWARD stack still reached the \
                 output -- the hcat halves are swapped"
            );
            assert_ne!(
                run(&right_only),
                run(&bump(&right_only, stack)),
                "{cell:?}: with the LEFT dense rows zeroed, the REVERSE stack did NOT reach the \
                 output"
            );
        }
    }

    /// The preallocation contract: a net run TWICE on the same input returns bit-identical
    /// output (no scratch/dense-chain state leaks between calls). Load-bearing for the
    /// overlap loop, which calls [`FastBiCell::feed_forward`] once per window.
    #[test]
    fn run_twice_is_bit_identical() {
        let sp = spec(&[6, 5, 4], &[2, 1], &[8, 3, 1], &[1, 1]);
        let p = mamba_params();
        let c = cfc_params();
        let tf = transformer_params();
        for cell in CELLS {
            let n = FastBiCell::element_count(&sp, cell, &p, &c, &tf).unwrap();
            let mut net = FastBiCell::from_flat(&sp, cell, &p, &c, &tf, &weights(n)).unwrap();
            let input = seq(13, 6, 0.2);
            let a = net.feed_forward(&input).clone();
            let b = net.feed_forward(&input).clone();
            assert_eq!((a.rows, a.cols), (6, 1), "{cell:?}: decimated row count");
            assert!(
                a.data
                    .iter()
                    .all(|v| v.is_finite() && (0.0..=1.0).contains(v)),
                "{cell:?}: posterior out of range"
            );
            assert_eq!(
                a.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{cell:?}: run-twice is not bit-identical"
            );
        }
    }

    /// The overlap loop's UNCOVERED-ROW quirk survives the port: a result buffer longer
    /// than any window reaches keeps `0/0 = NaN` (the exact driver's `:2055-2059`, no
    /// guard), and the covered prefix is finite.
    #[test]
    fn overlap_leaves_uncovered_rows_nan() {
        let sp = spec(&[4, 3], &[1], &[6, 1], &[1]);
        let p = mamba_params();
        let c = cfc_params();
        let tf = transformer_params();
        let n = FastBiCell::element_count(&sp, CellType::Slstm, &p, &c, &tf).unwrap();
        let mut net =
            FastBiCell::from_flat(&sp, CellType::Slstm, &p, &c, &tf, &weights(n)).unwrap();
        let input = seq(20, 4, 0.1);
        // ssr == 1, so output rows == input rows; size the buffer LONGER than that.
        let mut out = FastMatrix::zeros(26, 1);
        net.feed_forward_overlap(&input, 3, 2, &mut out);
        assert!(
            out.data[..20].iter().all(|v| v.is_finite()),
            "covered rows must be finite"
        );
        assert!(
            out.data[20..].iter().all(|v| v.is_nan()),
            "uncovered rows must be the reproduced 0/0 NaN"
        );
    }

    /// A window grid that covers every row exactly ONCE (shift == span) must reproduce
    /// the plain forward over the same rows -- the count quotient is then a division by
    /// 1.0, so any accumulate/average bug shows up as a value difference rather than a
    /// tolerance one. `window_size 3` spans `2*3+1 = 7` rows at `ssr == 1`.
    #[test]
    fn a_unit_coverage_window_grid_reproduces_the_per_block_forward() {
        let sp = spec(&[4, 3], &[1], &[6, 1], &[1]);
        let p = mamba_params();
        let c = cfc_params();
        let tf = transformer_params();
        for cell in CELLS {
            let n = FastBiCell::element_count(&sp, cell, &p, &c, &tf).unwrap();
            let mut net = FastBiCell::from_flat(&sp, cell, &p, &c, &tf, &weights(n)).unwrap();
            let input = seq(21, 4, 0.45);
            // jj = 0, 7, 14 -> begins 0, 4, 11 with `window_begin`'s `jj - w`... so a
            // hand-built expectation would re-derive the whole grid. Instead assert the
            // WEAKER but independent SINGLE-COVERED-PREFIX property: window 0 covers rows
            // [0, 7) alone (begin 0, end 6) and jj = 7's window begins at 4, so rows
            // [0, 4) are covered exactly ONCE -- their count quotient is a division by
            // 1.0, and they must therefore be BIT-identical to a direct `feed_forward` of
            // that block. (Coverage is NOT total here: at window 3 / shift 7 the last
            // window begins at 11 and ends at 17, so rows 18-20 are genuinely uncovered
            // and carry the reproduced `0/0` NaN -- which is what
            // `overlap_leaves_uncovered_rows_nan` pins.)
            let mut out = FastMatrix::zeros(21, 1);
            net.feed_forward_overlap(&input, 3, 7, &mut out);
            let block = FastMatrix {
                data: input.data[0..7 * 4].to_vec(),
                rows: 7,
                cols: 4,
            };
            let direct = net.feed_forward(&block).clone();
            for r in 0..4 {
                assert_eq!(
                    out.data[r].to_bits(),
                    direct.data[r].to_bits(),
                    "{cell:?}: single-covered row {r} is not the plain block forward"
                );
            }
        }
    }
}
