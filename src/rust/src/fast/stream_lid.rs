//! `fast::stream_lid` -- the per-utterance streaming LID session (Phase 8 Task 7).
//!
//! LID is UTTERANCE-GRANULAR by design (unlike the frame-granular SAD stream in
//! [`super::stream`]): one `external_features` entry (one phSeq line / one cep record) is
//! one "utterance". The offline Mode-7 Twin ([`FastTwinLid::get_segmentation`]) folds ALL of
//! a channel's entries into a per-channel accumulator (`langid`/`confusion`/counts) then runs
//! one epilogue; this session does the SAME fold ONE utterance at a time, exposing the
//! running aggregate after each push.
//!
//! BIT-IDENTICAL TO OFFLINE BY CONSTRUCTION (spec S2, the task gate). [`push_utterance`] folds
//! through the SAME shared kernel the offline loop uses ([`super::driver::score_lid_entry`]),
//! and [`finish`](StreamingLidSession::finish) / the per-push running aggregate run the SAME
//! epilogue ([`super::driver::finalize_lid_channel`]). So:
//!  - [`finish`](StreamingLidSession::finish) == the offline `FastTwinLid` members on the same
//!    entries, BIT-IDENTICAL (`lid_classification_errors`/`lid_segments_confusion`/
//!    `is_lid_correct`), and
//!  - the running aggregate after k pushes == the offline run on the FIRST k entries
//!    (prefix-correctness), the strong pin against the REAL offline driver (`phase8_stream_lid.rs`).
//!
//! Nothing is re-implemented; the phase-7 LID parity legs (`phase7_parity_lid.rs`) prove the
//! kernel refactor left the offline path unchanged.
//!
//! EVERY SHAPE STREAMS (issue #57, D3). The session is UTTERANCE-granular: one push scores one
//! whole `external_features` block through the SAME block forward the offline Twin uses, so
//! a bidirectional LID net is as streamable here as a causal one -- ADR-0004's "a bidirectional
//! net is refused at session construction" is a statement about the FRAME stream
//! (`fast::stream`), and its dated amendment names this session as the exception. Whatever
//! (cell x direction) shape the offline `FastTwinLid` builds (`fast::driver::FastLidNet`),
//! this session runs, prefix-correct after every push and bit-equal to the offline run on
//! `finish` (`tests/phase8_stream_lid.rs`'s matrix legs). A frame-granular LID emission model
//! would be its own issue.
//!
//! NOISE (entry-order-independent). Mode-7 Gaussian noise (`_NoiseMagnitude > 0`, off in the
//! gate configs) uses a `randinit` fixed at 0 PER ENTRY (`fast/driver.rs`'s `score_lid_entry`,
//! the exact `:989`), so each utterance's noise indexing restarts at 0 -- a per-utterance push
//! reproduces the offline sequential call's noise exactly, at any push granularity.
//!
//! SIGNATURE NOTE (the same T4/T5-precedent deviation the SAD session took). The Task-7 brief
//! sketches `new(map)`, but the offline path threads `audio.sample_rate` into the LID window
//! derivation and reads `audio.lang_index` for the scoring target, and the parity fixtures pass
//! the LID weight pack explicitly. None of those live in the config, so [`new`](StreamingLidSession::new)
//! takes `rate`, `lang_index`, and `lid_weights` too. That parenthetical used to read "a
//! `speech stream-lid` CLI / PyO3 binding -- DEFERRED to phase 9, this phase is LIB-ONLY";
//! PHASE 10 TASK 10 LANDED BOTH (`stream_lid_cli.rs` + the `speech_rs.StreamingLidSession`
//! pyclass) without touching a byte of this module, and each resolves those three arguments
//! exactly as predicted: the rate off the loaded input (`read_audio`'s `sample_rate`), the
//! target from `--lang` / the caller, and the weights from `BLSTM_LID_weightsFile` (`None`
//! defers to the config, which since issue #65 also reads and length-checks a non-empty
//! `BLSTM_weightsFile` as the offline Twin does, then discards it).

use anyhow::{Result, bail};
use indexmap::IndexMap;
use ndarray::Array2;

pub use super::driver::LidAggregate;
use super::driver::{FastTwinLid, LidChannelAcc, LidScoreParams, finalize_lid_channel};

/// One utterance's streaming LID score: the per-entry observables (the offline `seg_lid` +
/// its argmax) plus the running aggregate AFTER folding this utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct UtteranceScore {
    /// This utterance's per-block LID scores (the offline `seg_lid`, `class_nb`-wide,
    /// post-`_PostProcessMode`). Empty iff the utterance was SKIPPED (`nrows < lid_ssr` or
    /// below `MinNbOfFrames`).
    pub scores: Vec<f64>,
    /// This utterance's argmax over `scores` (the offline per-block `j`); `None` iff skipped.
    pub argmax: Option<usize>,
    /// Whether THIS utterance's argmax hit the target language (`j == ti`); `None` iff skipped.
    /// NOTE: distinct from `running_aggregate.is_lid_correct` -- that is the CUMULATIVE decision
    /// (the offline `is_lid_correct`: `100` iff the argmax of the langID accumulated over ALL
    /// utterances so far is the target), whereas this is the per-utterance hit/miss.
    pub is_correct: Option<bool>,
    /// The running aggregate AFTER folding this utterance -- the finalized snapshot, bit-
    /// identical to the offline `FastTwinLid` run on the first k entries.
    pub running_aggregate: LidAggregate,
}

/// The per-utterance streaming LID session: the online twin of the offline fast Mode-7 LID
/// Twin ([`FastTwinLid`]), restricted to its phSeq/cep external-features arm. Owns the
/// validated LID net + the derived scoring params + the running per-channel accumulator; folds
/// one utterance per [`push_utterance`](Self::push_utterance) and finalizes on demand.
pub struct StreamingLidSession {
    /// The validated Twin (holds the LID net + the config-derived scoring surface). Its
    /// `get_segmentation` is NEVER called -- the session drives the shared kernel directly, so
    /// it reuses all of `from_legacy`'s construction validation without the offline SAD side.
    twin: FastTwinLid,
    params: LidScoreParams,
    /// The single-channel running accumulator (streaming LID is inherently one utterance
    /// stream; the offline's per-channel loop reuses the same external_features for every
    /// channel, so a session models one channel).
    acc: LidChannelAcc,
}

impl StreamingLidSession {
    /// Build the session from the SAME legacy config `map` the offline Twin consumes, the
    /// stream `rate`, the target `lang_index` (the eval target -- the offline `audio.lang_index`),
    /// and the LID weight pack (`Some(flat)` builds the net immediately; `None` defers to the
    /// config's `BLSTM_LID_weightsFile` through [`FastTwinLid::load_weights_file`], so a
    /// non-empty `BLSTM_weightsFile` must also resolve and be long enough, issue #65). Reuses [`FastTwinLid::from_legacy`]'s construction
    /// validation (Mode 7 only, InputNormalizationType 0, no pitch pass / DumpInternals /
    /// negative TargetEnforcementStep / MLP, ...) and additionally bails if the LID window
    /// resolves to a regime the shape refuses (windowed causal, overlap -- the offline Twin's
    /// own rule, issue #57) or if no LID net is loaded.
    pub fn new(
        map: &IndexMap<String, String>,
        rate: f64,
        lang_index: i32,
        lid_weights: Option<&[f64]>,
    ) -> Result<StreamingLidSession> {
        let mut twin = FastTwinLid::from_legacy(map, None, lid_weights)?;
        // `from_legacy` with `lid_weights: None` defers the net to the config's weightsFile.
        if lid_weights.is_none() {
            twin.load_weights_file(map)?;
        }
        if !twin.lid_net_loaded() {
            bail!(
                "streaming LID: no LID net loaded -- pass lid_weights to new(), or set \
                 BLSTM_LID_weightsFile in the config"
            );
        }
        let params = twin.lid_score_params("streaming LID", rate, lang_index)?;
        let acc = LidChannelAcc::new(params.class_nb);
        Ok(StreamingLidSession { twin, params, acc })
    }

    /// Fold ONE utterance (`external_features` entry -- the `(frames x feat_dim)` matrix the
    /// phSeq/cep readers push) into the running accumulator and return its per-entry score plus
    /// the finalized running aggregate. A too-short / below-`MinNbOfFrames` entry is SKIPPED
    /// (empty `scores`, `None` argmax), exactly as the offline loop's `:1450` guard.
    pub fn push_utterance(&mut self, feat: &Array2<f64>) -> UtteranceScore {
        let outcome = self.twin.fold_entry(feat, &self.params, &mut self.acc);
        let running_aggregate = finalize_lid_channel(&self.acc, &self.params);
        match outcome {
            Some(o) => UtteranceScore {
                is_correct: Some(o.argmax == self.params.ti),
                argmax: Some(o.argmax),
                scores: o.seg_lid,
                running_aggregate,
            },
            None => UtteranceScore {
                scores: Vec::new(),
                argmax: None,
                is_correct: None,
                running_aggregate,
            },
        }
    }

    /// The finalized LID result rows on the utterances folded so far -- BIT-IDENTICAL to the
    /// offline `FastTwinLid` members on the same entries. A pure snapshot of the current
    /// accumulator (`== the last push's running aggregate`), so it is repeatable/idempotent and
    /// the session may keep folding after a call.
    pub fn finish(&self) -> LidAggregate {
        finalize_lid_channel(&self.acc, &self.params)
    }

    /// The number of scored (not skipped) utterances folded so far (the offline `segments_count`).
    pub fn scored_count(&self) -> i32 {
        self.acc.segments_count
    }
}
