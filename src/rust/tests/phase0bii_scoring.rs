//! `compute_errors` scoring: Pass 2 (Pfa/Pmiss/ErrorRate) hand-computed golden,
//! the no-reference labelling branch, and a hand-traced Pass 1 WER unit.
//!
//! Ported target: `Segmentation::compute_errors` (`Segmentation.cpp:288-517`).
//! The Pass-2 golden below is the arbiter: the arithmetic is worked out in the
//! comments and asserted to `1e-12`.

use approx::assert_abs_diff_eq;
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmentation_io::compute_errors;

/// Build `[Other@0, <class>@begin, Other@end, End@dur]` on duration `dur`.
fn one_span(dur: f64, begin: f64, end: f64, class: SegClass) -> Segmentation {
    let mut seg = Segmentation::new(dur);
    seg.label_segment(begin, end, class);
    seg.sanitize();
    seg
}

/// Pass 2 hand-computed golden (brief spec 5).
///
/// ref `[Other@0, Speech@2, Other@5, End@10]`, hyp `[Other@0, Speech@3, Other@6,
/// End@10]`. No SUB/INS so modify_type is a no-op.
/// update_count on the (unmodified) ref: Other = (2-0)+(10-5) = 7, Speech = 5-2
/// = 3, dur = 10, count_excluded = 0.
///
/// Pass 2 two-pointer walk (itRef from 0, it from 1):
/// - The single mismatch window is [2,3): ref says Speech, hyp[it-1] says Other.
///   error = 1s accrues to Pmiss[Speech] (missed speech) and to
///   ErrorRate[Other] + Pfa[Other] (hyp called it Other).
/// - The window [5,6): ref says Other, hyp[it-1] says Speech.
///   error = 1s accrues to Pmiss[Other] and ErrorRate[Speech] + Pfa[Speech].
///
/// Raw seconds accumulated:
///   Pmiss[Speech]=1, Pmiss[Other]=1,
///   ErrorRate[Speech]=1, ErrorRate[Other]=1,
///   Pfa[Speech]=1, Pfa[Other]=1.
///
/// Normalization (dur=10, count_excluded=0):
///   Speech: count=3, count_others = 10-3-0 = 7
///     pmiss = 1/3, pfa = 1/7, error_rate = 1/(10-0) = 0.1
///   Other:  count=7, count_others = 10-7-0 = 3
///     pmiss = 1/7, pfa = 1/3, error_rate = 1/(10-0) = 0.1
#[test]
fn pass2_speech_other_golden() {
    let refseg = one_span(10.0, 2.0, 5.0, SegClass::Speech);
    let mut hyp = one_span(10.0, 3.0, 6.0, SegClass::Speech);

    let report = compute_errors(&mut hyp, Some(&refseg), -1);

    let sp = report.per_class[SegClass::Speech as usize];
    let ot = report.per_class[SegClass::Other as usize];

    assert_abs_diff_eq!(sp.pmiss, 1.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(sp.pfa, 1.0 / 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(sp.error_rate, 0.1, epsilon = 1e-12);

    assert_abs_diff_eq!(ot.pmiss, 1.0 / 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ot.pfa, 1.0 / 3.0, epsilon = 1e-12);
    assert_abs_diff_eq!(ot.error_rate, 0.1, epsilon = 1e-12);

    // label_counts reflect the (here unmodified) reference.
    assert_abs_diff_eq!(
        report.label_counts[SegClass::Speech as usize],
        3.0,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        report.label_counts[SegClass::Other as usize],
        7.0,
        epsilon = 1e-12
    );

    // nb_words = -1 -> Pass 1 (WER) is skipped.
    assert!(report.wer.is_none());
}

/// No-reference branch (brief Branch A): sanitize hyp, return its label counts,
/// no per-class errors, no WER.
///
/// hyp `[Other@0, Speech@3, Other@6, End@10]` -> Speech = 3, Other = 7.
#[test]
fn no_reference_labelling() {
    let mut hyp = one_span(10.0, 3.0, 6.0, SegClass::Speech);

    let report = compute_errors(&mut hyp, None, -1);

    assert_abs_diff_eq!(
        report.label_counts[SegClass::Speech as usize],
        3.0,
        epsilon = 1e-12
    );
    assert_abs_diff_eq!(
        report.label_counts[SegClass::Other as usize],
        7.0,
        epsilon = 1e-12
    );
    // The legacy displays 100*count/dur; verify the ratio the caller computes.
    assert_abs_diff_eq!(
        100.0 * report.label_counts[SegClass::Speech as usize] / hyp.audio_duration(),
        30.0,
        epsilon = 1e-12
    );

    for e in &report.per_class {
        assert_eq!(e.pmiss, 0.0);
        assert_eq!(e.pfa, 0.0);
        assert_eq!(e.error_rate, 0.0);
    }
    assert!(report.wer.is_none());
}

/// Pass 1 WER, hand-traced (brief step 3, Pass 1a `Segmentation.cpp:313-369`).
///
/// ref `[Other@0, Speech@2, Other@5, End@10]`, nb_words = 1.
/// hyp `[Other@0, Speech@2, Other@5, End@10]` (exact match).
///
/// Pass 1a trace (dels init = nb_words = 1; itRef=0, it=0):
///   itRef=0 Other       -> else arm -> ++itRef            (itRef=1)
///   itRef=1 Speech, it=0 Other: else@336, (it+1)@2 <= itRef@2 -> ++it (it=1)
///   itRef=1 Speech, it=1 Speech:
///     (it+1)@5 <= itRef@2 ? no
///     (it+1)@5 <  (itRef+1)@5 ? no
///     it@2 > itRef@2 ? no
///     else: --dels (=0); itRef.ty==SPEECH -> ++corrects (=1); ++itRef (itRef=2)
///   loop guard itRef+1==itEndRef -> stop.
/// => corrects=1, subs=0, dels=0, ins=0.
///
/// Pass 1b: ref == hyp everywhere, so every ref-speech instant is covered by a
/// hyp-speech instant (coverage add gated on `it.ty != SPEECH`), there are no
/// INSERTION segments, and no OTHER-ref instant sees hyp SPEECH. Hence
/// coverage_penalty = 0 and delay_penalty = 0 exactly.
#[test]
fn pass1_wer_exact_match() {
    let refseg = one_span(10.0, 2.0, 5.0, SegClass::Speech);
    let mut hyp = one_span(10.0, 2.0, 5.0, SegClass::Speech);

    let report = compute_errors(&mut hyp, Some(&refseg), 1);

    let wer = report.wer.expect("nb_words >= 0 must yield WER stats");
    assert_eq!(wer.nb_words, 1);
    assert_eq!(wer.corrects, 1);
    assert_eq!(wer.subs, 0);
    assert_eq!(wer.dels, 0);
    assert_eq!(wer.ins, 0);
    assert_abs_diff_eq!(wer.coverage_penalty, 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(wer.delay_penalty, 0.0, epsilon = 1e-12);
}
