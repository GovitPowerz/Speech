//! The channel result: the engine's outcome for one (file, config, channel),
//! and its wire form, the result row (issue #23).
//!
//! Ported from legacy C++: `BagOfProcessors::SegmentationFunction`'s scored
//! (`BagOfProcessors.cpp:313-350`) and unscored (`:358-392`) row assembly. The
//! legacy wrote the row positionally and every reader (`saveAndUpdate`,
//! `CorpusProcessor::gradCheck`, `PrintConfusionMatrix`, `ComputeCost.m`)
//! decoded it by index. Here the layout is written down ONCE, in [`ChannelResult::to_row`];
//! the Rust readers take fields, and the seam hands Python names.
//!
//! # The result row
//!
//! 18 columns for a driver without LID (algo 0-4), `18 + N` for a LID driver
//! (algo 5/6, `N` = the class count). Column `k`:
//!
//! | col | field | legacy member |
//! |---|---|---|
//! | 0 | `pfa` (percent) | `100 * _Errors[SPEECH]._Pfa` |
//! | 1 | `pmiss` (percent) | `100 * _Errors[SPEECH]._Pmiss` |
//! | 2 | `error_rate` (percent) | `100 * sum_j _Errors[j]._ErrorRate`, j in OTHER..EXCLUDED |
//! | 3 | `time_per_hour` | the wall-clock timing (`:308-309`) |
//! | 4 | `seg_cost` | `_CumulativeError` |
//! | 5 | `audio_duration` | `_Duration` |
//! | 6 | `speech_duration` | the speech walk (`:324-330`) |
//! | 7-13 | `wer` | `_NbWords, _Corrects, _Subs, _Ins, _Dels, _CoveragePenalty, _DelayPenalty` |
//! | 14 | `lid.cost` (0.0 without LID) | `_LIDCumulativeError` |
//! | 15 | `lid.correct` as `100.0` / `0.0` | `_IsLIDCorrect` |
//! | 16..16+N | `lid.scores`, the target re-encoded | `_LIDClassificationErrors` |
//! | len-2 | `lid.count` (0 without LID) | `_LIDNbOfClassif` |
//! | len-1 | `seg_count` | `_NbOfClassif` |
//!
//! The two trailing counters sit OUTSIDE the LID gate (`:348-349`), so both
//! widths end with them.
//!
//! # The in-band LID signalling
//!
//! `tasks/lid.rs` (and its fast twin) writes `100 * (langID[c] - target[c])`
//! per class with `target[c] = -2` on the true class and `0` elsewhere: the
//! true class carries `100 * langID + 200` in `[200, 300]`, every other class
//! its raw `100 * langID` in `[0, 100]`. The readers decode "the column `> 150`
//! is the target, its score is `v - 200`". That decode happens once, in
//! [`LidResult::from_encoded`]; [`ChannelResult::to_row`] re-encodes with
//! `+ 200`. On `[200, 300]` the subtraction is exact (Sterbenz: `200 <= v <=
//! 400`), so `(v - 200) + 200 == v` bit for bit and the `.mat` goldens are
//! unmoved. The frozen tree keeps producing the encoding (ADR-0002).
//!
//! # The unscored asymmetry (`:358-392`)
//!
//! An unscored row zeroes the three error columns, `seg_cost` and BOTH trailing
//! counters, keeps the timing, the durations and the LID cost / flag / scores
//! real, and writes the legacy `WordErrorRate` constructor default for the WER
//! block (`nb_words == -1`, the Phase 4a tier-1 golden). [`ChannelResult::unscored`]
//! holds that asymmetry; nothing downstream distinguishes "unscored" from
//! "scored as zero", so the struct does not either.

use crate::tasks::segmentation::SegClass;
use crate::tasks::segmentation_io::{ScoreReport, WerStats};

/// The scored/unscored LID block of one channel: `seg._LID*` members, written
/// only by the LID drivers (algo 5/6), decoded once here.
#[derive(Debug, Clone, PartialEq)]
pub struct LidResult {
    /// `_LIDCumulativeError`, column 14.
    pub cost: f64,
    /// `_LIDNbOfClassif`, column `len-2`; a hard 0 in the unscored row (`:390`).
    pub count: i64,
    /// `_IsLIDCorrect`, column 15: `100.0` / `0.0` on the wire.
    pub correct: bool,
    /// The one column `> 150`; `None` is an out-of-set row (no true class in
    /// the closed set), which the confusion matrix credits to nobody (F5).
    pub target: Option<usize>,
    /// One score per class, the target decoded (`v - 200`), the others raw.
    pub scores: Vec<f64>,
}

impl LidResult {
    /// The one in-band decode. Ascending scan, the LAST column `> 150` is the
    /// target (the legacy `PrintConfusionMatrix` loop's overwrite semantics,
    /// `BagOfProcessors.cpp:512-520`); only that column is decoded, so a row
    /// re-encodes byte-identically whatever its other columns hold. A second
    /// column `> 150` cannot occur (a non-target carries `100 * langID <= 100`).
    pub fn from_encoded(cost: f64, count: i64, correct: bool, encoded: &[f64]) -> Self {
        let mut scores = encoded.to_vec();
        let target = encoded.iter().rposition(|&v| v > 150.0);
        if let Some(t) = target {
            scores[t] = encoded[t] - 200.0;
        }
        LidResult {
            cost,
            count,
            correct,
            target,
            scores,
        }
    }

    /// The wire form of `scores`: the target column `+ 200`, exact on `[0, 100]`.
    fn encoded_scores(&self) -> Vec<f64> {
        let mut out = self.scores.clone();
        if let Some(t) = self.target {
            out[t] = self.scores[t] + 200.0;
        }
        out
    }
}

/// The engine's outcome for one (file, config, channel).
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelResult {
    /// Column 0, percent.
    pub pfa: f64,
    /// Column 1, percent.
    pub pmiss: f64,
    /// Column 2, percent: the per-class error rates summed over OTHER..EXCLUDED.
    pub error_rate: f64,
    /// Column 3, the wall-clock timing (masked in every golden, still written).
    pub time_per_hour: f64,
    /// Column 4, `_CumulativeError`: the SAD net's cost numerator.
    pub seg_cost: f64,
    /// Column 5.
    pub audio_duration: f64,
    /// Column 6.
    pub speech_duration: f64,
    /// Columns 7-13. Dead surface (no reader past the `.mat`), kept for the bytes.
    pub wer: WerStats,
    /// Columns 14, 15, 16..16+N and len-2; `None` writes the two 0.0 slots and a
    /// 0 counter, no score columns.
    pub lid: Option<LidResult>,
    /// Column len-1, `_NbOfClassif`: the SAD net's cost denominator.
    pub seg_count: i64,
}

impl ChannelResult {
    /// The scored row (`:313-350`).
    pub fn scored(
        report: &ScoreReport,
        time_per_hour: f64,
        seg_cost: f64,
        audio_duration: f64,
        speech_duration: f64,
        seg_count: i64,
        lid: Option<LidResult>,
    ) -> Self {
        let speech = report.per_class[SegClass::Speech as usize];
        let mut global_error_rate = 0.0;
        // legacy: :317-319 j from OTHER up to (exclusive) EXCLUDED.
        for j in (SegClass::Other as usize)..(SegClass::Excluded as usize) {
            global_error_rate += report.per_class[j].error_rate;
        }
        // No WER Pass 1 (STM/no reference) -> the legacy WordErrorRate CONSTRUCTOR
        // default (`_NbWords = -1`, rest 0), NOT `WerStats::default()` (nb_words 0).
        let wer = report.wer.unwrap_or(WerStats::legacy_default());
        ChannelResult {
            pfa: 100.0 * speech.pfa,
            pmiss: 100.0 * speech.pmiss,
            error_rate: 100.0 * global_error_rate,
            time_per_hour,
            seg_cost,
            audio_duration,
            speech_duration,
            wer,
            lid,
            seg_count,
        }
    }

    /// The unscored row (`:358-392`): zeros for the error columns, `seg_cost`
    /// and BOTH trailing counters; timing, durations and the LID block real;
    /// the WER block the legacy constructor default (`nb_words == -1`).
    pub fn unscored(
        time_per_hour: f64,
        audio_duration: f64,
        speech_duration: f64,
        lid: Option<LidResult>,
    ) -> Self {
        ChannelResult {
            pfa: 0.0,
            pmiss: 0.0,
            error_rate: 0.0,
            time_per_hour,
            seg_cost: 0.0,
            audio_duration,
            speech_duration,
            wer: WerStats::legacy_default(),
            // legacy: :390 -- the LID counter is a hard 0 in the unscored branch.
            lid: lid.map(|l| LidResult { count: 0, ..l }),
            seg_count: 0,
        }
    }

    /// The result row. The ONLY writer of the layout (see the module doc).
    pub fn to_row(&self) -> Vec<f64> {
        let mut row = vec![
            self.pfa,                  // 0
            self.pmiss,                // 1
            self.error_rate,           // 2
            self.time_per_hour,        // 3
            self.seg_cost,             // 4
            self.audio_duration,       // 5
            self.speech_duration,      // 6
            self.wer.nb_words as f64,  // 7
            self.wer.corrects as f64,  // 8
            self.wer.subs as f64,      // 9
            self.wer.ins as f64,       // 10
            self.wer.dels as f64,      // 11
            self.wer.coverage_penalty, // 12
            self.wer.delay_penalty,    // 13
        ];
        match &self.lid {
            Some(l) => {
                row.push(l.cost); // 14
                row.push(if l.correct { 100.0 } else { 0.0 }); // 15
                row.extend(l.encoded_scores()); // 16..16+N
                row.push(l.count as f64); // len-2
            }
            None => {
                row.push(0.0); // 14
                row.push(0.0); // 15
                row.push(0.0); // len-2
            }
        }
        row.push(self.seg_count as f64); // len-1
        row
    }

    /// The inverse of [`Self::to_row`] for the fold tests and the round-trip
    /// proptest: never production-reachable. `lid` is `Some` when the row is
    /// wider than 18 or any of its three LID slots is non-zero, so a canonical
    /// struct round-trips exactly and a crafted 18-wide row with LID numbers
    /// (the Phase 4a aggregation tests) keeps them.
    #[cfg(any(test, feature = "test-support"))]
    pub fn from_row(row: &[f64]) -> Self {
        assert!(row.len() >= 18, "a result row is at least 18 columns");
        let n = row.len() - 18;
        let lid_slots_live = row[14] != 0.0 || row[15] != 0.0 || row[16 + n] != 0.0;
        let lid = (n > 0 || lid_slots_live).then(|| {
            LidResult::from_encoded(
                row[14],
                row[16 + n] as i64,
                row[15] == 100.0,
                &row[16..16 + n],
            )
        });
        ChannelResult {
            pfa: row[0],
            pmiss: row[1],
            error_rate: row[2],
            time_per_hour: row[3],
            seg_cost: row[4],
            audio_duration: row[5],
            speech_duration: row[6],
            wer: WerStats {
                nb_words: row[7] as i64,
                corrects: row[8] as i64,
                subs: row[9] as i64,
                ins: row[10] as i64,
                dels: row[11] as i64,
                coverage_penalty: row[12],
                delay_penalty: row[13],
            },
            lid,
            seg_count: row[row.len() - 1] as i64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn empty_report() -> ScoreReport {
        ScoreReport {
            per_class: [Default::default(); 23],
            label_counts: [0.0; 23],
            wer: None,
        }
    }

    fn lid3() -> LidResult {
        LidResult::from_encoded(7.5, 42, true, &[210.0, 30.0, 60.0])
    }

    /// The width contract (`:338-349` / `:380-391`), carried over from the
    /// `assemble_*_row` tests: 18 columns without LID (two 0.0 slots), `18 + N`
    /// with, and `[len-2] = _LIDNbOfClassif`, `[len-1] = _NbOfClassif` in both.
    #[test]
    fn row_width_and_slots() {
        let report = empty_report();

        let plain = ChannelResult::scored(&report, 1.0, 2.0, 3.0, 4.0, 9, None).to_row();
        assert_eq!(plain.len(), 18);
        assert_eq!(plain[14], 0.0);
        assert_eq!(plain[15], 0.0);
        assert_eq!(plain[16], 0.0); // len-2 lid count
        assert_eq!(plain[17], 9.0); // len-1 seg count

        let wide = ChannelResult::scored(&report, 1.0, 2.0, 3.0, 4.0, 9, Some(lid3())).to_row();
        assert_eq!(wide.len(), 21, "18 + N(3)");
        assert_eq!(wide[14], 7.5);
        assert_eq!(wide[15], 100.0);
        assert_eq!(&wide[16..19], &[210.0, 30.0, 60.0]);
        assert_eq!(wide[19], 42.0); // len-2
        assert_eq!(wide[20], 9.0); // len-1

        // Unscored: same LID block, hard-0 trailing counters (:390-391), WER default.
        let unscored = ChannelResult::unscored(1.0, 3.0, 4.0, Some(lid3())).to_row();
        assert_eq!(unscored.len(), 21);
        assert_eq!(unscored[14], 7.5);
        assert_eq!(&unscored[16..19], &[210.0, 30.0, 60.0]);
        assert_eq!(unscored[19], 0.0);
        assert_eq!(unscored[20], 0.0);
        assert_eq!(unscored[7], -1.0);
    }

    #[test]
    fn from_encoded_decodes_only_the_target() {
        let l = lid3();
        assert_eq!(l.target, Some(0));
        assert_eq!(l.scores, vec![10.0, 30.0, 60.0]);
        let none = LidResult::from_encoded(0.0, 0, false, &[20.0, 30.0]);
        assert_eq!(none.target, None);
        assert_eq!(none.scores, vec![20.0, 30.0]);
    }

    #[test]
    fn from_row_inverts_to_row() {
        let report = empty_report();
        let wide = ChannelResult::scored(&report, 1.0, 2.0, 3.0, 4.0, 9, Some(lid3()));
        assert_eq!(ChannelResult::from_row(&wide.to_row()), wide);
        let plain = ChannelResult::scored(&report, 1.0, 2.0, 3.0, 4.0, 9, None);
        assert_eq!(ChannelResult::from_row(&plain.to_row()), plain);
        // A crafted 18-wide row with live LID slots keeps them (the aggregation tests).
        let mut row = plain.to_row();
        row[14] = 6.0;
        row[15] = 100.0;
        row[16] = 3.0;
        let back = ChannelResult::from_row(&row);
        assert_eq!(back.lid, Some(LidResult::from_encoded(6.0, 3, true, &[])));
        assert_eq!(back.to_row(), row);
    }

    /// A valid wire row: the first 14 columns any finite value with integer
    /// WER tallies, col 15 in {0, 100}, N in 0..=8 score columns with at most
    /// one target in [200, 300] and the rest in [0, 150], integer counters.
    fn arb_row() -> impl Strategy<Value = Vec<f64>> {
        let head = (
            proptest::collection::vec(-1e6f64..1e6, 7),
            proptest::collection::vec(-1000i64..1000, 5),
            proptest::collection::vec(-1e6f64..1e6, 2),
        );
        let lid = (
            -1e6f64..1e6,
            prop::bool::ANY,
            0usize..=8,
            proptest::collection::vec(0.0f64..=150.0, 8),
            proptest::option::of(0usize..8),
            200.0f64..=300.0,
            0i64..1000,
        );
        (head, lid, 0i64..1000).prop_map(
            |(
                (errs, wer_ints, wer_pen),
                (cost, correct, n, raw, target, enc, lid_count),
                seg_count,
            )| {
                let mut row = errs;
                row.extend(wer_ints.iter().map(|&v| v as f64));
                row.extend(wer_pen);
                row.push(cost);
                row.push(if correct { 100.0 } else { 0.0 });
                let mut scores: Vec<f64> = raw[..n].to_vec();
                if let Some(t) = target.filter(|&t| t < n) {
                    scores[t] = enc;
                }
                row.extend(scores);
                row.push(lid_count as f64);
                row.push(seg_count as f64);
                row
            },
        )
    }

    proptest! {
        /// The wire bytes are golden-pinned, the struct is not: a valid row
        /// survives decode + re-encode bit for bit, and the struct it decodes
        /// to is a fixed point.
        #[test]
        fn wire_row_round_trips_bit_exact(row in arb_row()) {
            let decoded = ChannelResult::from_row(&row);
            let back = decoded.to_row();
            prop_assert_eq!(back.len(), row.len());
            for (a, b) in back.iter().zip(&row) {
                prop_assert_eq!(a.to_bits(), b.to_bits());
            }
            prop_assert_eq!(ChannelResult::from_row(&back), decoded);
        }
    }
}
