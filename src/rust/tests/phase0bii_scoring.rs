//! `compute_errors` scoring: Pass 2 (Pfa/Pmiss/ErrorRate) hand-computed golden,
//! the no-reference labelling branch, and a hand-traced Pass 1 WER unit.
//!
//! Ported target: `Segmentation::compute_errors` (`Segmentation.cpp:288-517`).
//! The Pass-2 golden below is the arbiter: the arithmetic is worked out in the
//! comments and asserted to `1e-12`.

use approx::assert_abs_diff_eq;
use serde::Deserialize;
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

/// Pass 1b `length` formula, non-integer duration (review fix pin).
///
/// Legacy `Segmentation.cpp:371`: `long length = ((long)(itEndRef-1)->_BeginTime)/timeStep+1`.
/// The `(long)` cast is unary and binds to `_BeginTime` (tighter than `/`), so
/// the duration is truncated to WHOLE SECONDS first, then divided by
/// `timeStep`. A naive `(begin / time_step) as i64 + 1` divides first and
/// truncates the quotient instead -- a numeric divergence for any non-integer
/// duration.
///
/// ref = `Other@[0,10.5)` (no labelling -> single OTHER span), hyp = SPEECH
/// throughout `[0,10.5)`. Every one of the `length` frames therefore hits the
/// final `else` branch (`ref_ty == Other`, `hyp.ty == Speech`):
/// `delay_penalty += time_step/100.0`; `coverage_penalty` is never touched.
///
/// Fixed length = `(10.5.trunc()/1e-4) as i64 + 1 = (10/1e-4)+1 = 100001`.
///   delay_penalty = 100001 * (1e-4/100) = 100001 * 1e-6 = 0.100001.
/// Buggy length (divide-first) = `(10.5/1e-4) as i64 + 1 = 105000+1 = 105001`.
///   delay_penalty = 105001 * 1e-6 = 0.105001 (what this test catches).
#[test]
fn pass1b_length_truncates_duration_to_whole_seconds() {
    let refseg = Segmentation::new(10.5);
    let mut hyp = Segmentation::new(10.5);
    hyp.label_segment(0.0, 10.5, SegClass::Speech);

    let report = compute_errors(&mut hyp, Some(&refseg), 0);

    let wer = report.wer.expect("nb_words >= 0 must yield WER stats");
    // Epsilon widened to 1e-7: a 100k-term f64 sum accrues float error beyond 1e-12.
    assert_abs_diff_eq!(wer.delay_penalty, 0.100_001, epsilon = 1e-7);
    assert_abs_diff_eq!(wer.coverage_penalty, 0.0, epsilon = 1e-7);
}

/// Map a raw i32 class code (as emitted in the JSON fixtures) to `SegClass`.
/// Only the codes used by the emitter (Speech/Substitution/Insertion/Excluded)
/// plus Other/End need mapping; anything else is a fixture bug.
fn code_to_segclass(code: i32) -> SegClass {
    match code {
        0 => SegClass::Other,
        1 => SegClass::Speech,
        19 => SegClass::Insertion,
        20 => SegClass::Substitution,
        21 => SegClass::Excluded,
        22 => SegClass::End,
        other => panic!("unexpected class code {other} in fixture"),
    }
}

#[derive(Deserialize)]
struct ScoringCase {
    ref_spans: Vec<[f64; 3]>,
    hyp_spans: Vec<[f64; 3]>,
    audio_duration: f64,
    expected: Vec<[f64; 3]>,
}

fn build_seg(dur: f64, spans: &[[f64; 3]]) -> Segmentation {
    let mut seg = Segmentation::new(dur);
    for s in spans {
        seg.label_segment(s[0], s[1], code_to_segclass(s[2] as i32));
    }
    seg.sanitize();
    seg
}

/// Cross-check the Rust `compute_errors` Pass 2 against the numpy oracle
/// bit-for-bit over the deterministic fixture cases (`compute_errors_cases.json`).
/// `nb_words = -1` gates out Pass 1, isolating the Pass-2 two-pointer walk +
/// normalization (the differential oracle that caught the 0b-i softmax bug).
#[test]
fn oracle_cross_check_bit_exact() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0bii/compute_errors_cases.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let cases: Vec<ScoringCase> = serde_json::from_str(&text).unwrap();
    assert!(cases.len() >= 8, "expected >= 8 cases, got {}", cases.len());

    for (idx, case) in cases.iter().enumerate() {
        let refseg = build_seg(case.audio_duration, &case.ref_spans);
        let mut hyp = build_seg(case.audio_duration, &case.hyp_spans);
        let report = compute_errors(&mut hyp, Some(&refseg), -1);

        assert_eq!(
            case.expected.len(),
            23,
            "case {idx}: expected 23 per-class rows"
        );
        for (j, exp) in case.expected.iter().enumerate() {
            let got = report.per_class[j];
            assert_eq!(
                got.pmiss.to_bits(),
                exp[0].to_bits(),
                "case {idx} class {j}: pmiss {} != {}",
                got.pmiss,
                exp[0]
            );
            assert_eq!(
                got.pfa.to_bits(),
                exp[1].to_bits(),
                "case {idx} class {j}: pfa {} != {}",
                got.pfa,
                exp[1]
            );
            assert_eq!(
                got.error_rate.to_bits(),
                exp[2].to_bits(),
                "case {idx} class {j}: error_rate {} != {}",
                got.error_rate,
                exp[2]
            );
        }
    }
}
