//! Segmenter decision tests (hand-computed from synthetic results rows) plus the
//! Rust==Python differential cross-check over the deterministic fixture cases.

use serde::Deserialize;
use speech::tasks::segmentation::{SegClass, Segmentation};
use speech::tasks::segmenter::{
    SegmenterConfig, get_targets, lid_to_segmentation, update_segmentation, update_segmentation_raw,
};

fn cfg_no_smooth() -> SegmenterConfig {
    // area 0 so a single crossing triggers; padding/min all 0 so smoothing is a no-op.
    SegmenterConfig {
        rising: 0.5,
        area_rising: 0.0,
        falling: 0.5,
        area_falling: 0.0,
        padding: [0.0; 4],
        min_speech: [0.0; 3],
        min_silence: [0.0; 2],
    }
}

fn spans(s: &Segmentation) -> Vec<(f64, SegClass)> {
    s.segments().iter().map(|x| (x.begin, x.ty)).collect()
}

#[test]
fn clean_rising_falling_crossing() {
    // dt=1.0, offset=0. results cross up between idx1(0.0)->idx2(1.0) at 0.5*dt,
    // and back down between idx3(1.0)->idx4(0.0) at 0.5*dt past idx3. area_rising/falling=0.
    let mut seg = Segmentation::new(5.0);
    let results = vec![0.0, 0.0, 1.0, 1.0, 0.0];
    update_segmentation(
        &mut seg,
        &results,
        SegClass::Speech,
        0.0,
        1.0,
        &cfg_no_smooth(),
    );
    // rising: r(2)>=0.5 && r(1)<0.5 -> begin = 1*(2 - (1.0-0.5)/(1.0-0.0)) = 1.5
    // falling: r(4)<=0.5 && r(3)>0.5 -> end = 1*(4 - (0.0-0.5)/(0.0-1.0)) = 3.5
    let sp = spans(&seg);
    assert_eq!(sp[0], (0.0, SegClass::Other));
    assert_eq!(sp[1], (1.5, SegClass::Speech));
    assert_eq!(sp[2], (3.5, SegClass::Other));
    assert_eq!(*sp.last().unwrap(), (5.0, SegClass::End));
}

#[test]
fn lid_single_threshold_col_crossing() {
    // COL vector, single thresh_max=0.5, sanitize only (no smoothing). dt=1, off=0.
    // r = [0,0,1,1,0]:
    //   ii=2: r(2)=1 >= 0.5 && r(1)=0 < 0.5 -> begin = 1*(2 - (1-0.5)/(1-0)) = 1.5, hasBegun.
    //   ii=4: r(4)=0 <= 0.5 && r(3)=1 > 0.5 -> end = 1*(4 - (0-0.5)/(0-1)) = 3.5.
    //   begin(1.5) < end(3.5) -> label_segment(1.5, 3.5, Speech).
    let mut seg = Segmentation::new(5.0);
    let results = vec![0.0, 0.0, 1.0, 1.0, 0.0];
    lid_to_segmentation(&mut seg, &results, SegClass::Speech, 0.0, 1.0, 0.5);
    let sp = spans(&seg);
    assert_eq!(sp[0], (0.0, SegClass::Other));
    assert_eq!(sp[1], (1.5, SegClass::Speech));
    assert_eq!(sp[2], (3.5, SegClass::Other));
    assert_eq!(*sp.last().unwrap(), (5.0, SegClass::End));
}

#[test]
fn get_targets_backprop_off_speech_span() {
    // back_prop_wer < 0 plain branch. Reference boundary list:
    //   [Other@0, Speech@2, Excluded@4, Other@6, End@8].
    // time_step=1, time_offset=0, n_rows=8, class=Speech. t = ii.
    //   ii in {0,1}: itRef=Other   -> 0.0
    //   ii in {2,3}: itRef=Speech  -> 1.0
    //   ii in {4,5}: itRef=Excluded-> -0.5
    //   ii in {6,7}: itRef=Other   -> 0.0
    let mut reference = Segmentation::new(8.0);
    reference.label_segment(2.0, 4.0, SegClass::Speech);
    reference.label_segment(4.0, 6.0, SegClass::Excluded);
    let hyp = Segmentation::new(8.0);
    let targets = get_targets(&hyp, &reference, 1.0, 0.0, -1.0, SegClass::Speech, 8);
    assert_eq!(targets, vec![0.0, 0.0, 1.0, 1.0, -0.5, -0.5, 0.0, 0.0]);
}

#[test]
fn get_targets_backprop_on_neighbor_window() {
    // back_prop_wer >= 0 branch. Reference: [Other@0, Speech@2, Other@5, End@8].
    // back_prop_wer=0.5, time_step=1, time_offset=0, n_rows=8. t = ii.
    //   ii in {0,1}: itRef=Other, no window (Speech begins at 2, window opens at
    //     2-0.5=1.5, not yet <= t) -> time_step/100 = 0.01.
    //   ii in {2,3,4}: itRef=Speech, dur_seg = 5-2 = 3 -> 1 - 0.1/3.
    //   ii=5: itRef=Other@5, preceding Speech window still open (5+0.5=5.5 >= 5),
    //     dur_seg = 5-2 = 3 -> 1 - 0.1/3.
    //   ii in {6,7}: itRef=Other@5, window closed (5.5 < 6) -> 0.01.
    let mut reference = Segmentation::new(8.0);
    reference.label_segment(2.0, 5.0, SegClass::Speech);
    let hyp = Segmentation::new(8.0);
    let targets = get_targets(&hyp, &reference, 1.0, 0.0, 0.5, SegClass::Speech, 8);
    let speech = 1.0 - 0.1 / 3.0;
    assert_eq!(
        targets,
        vec![0.01, 0.01, speech, speech, speech, speech, 0.01, 0.01]
    );
}

#[test]
fn config_parses_seg_golden() {
    let text = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/reference_data/phase0bii/seg.config"),
    )
    .unwrap();
    let m = speech::legacy_config::parse_legacy_config(&text);
    let cfg = SegmenterConfig::from_config(&m, "BLSTM").unwrap();
    // 24-Feb-2014_BLSTM_Spect.config, last-duplicate-key wins: rising 0.6/area 0.05, falling 0.3/area 0.03.
    assert_eq!(cfg.rising, 0.6);
    assert_eq!(cfg.area_rising, 0.05);
    assert_eq!(cfg.falling, 0.3);
    assert_eq!(cfg.area_falling, 0.03);
    assert_eq!(cfg.padding, [0.2, 0.0, 0.3, 0.4]);
    assert_eq!(cfg.min_silence, [0.3, 0.5]);
    assert_eq!(cfg.min_speech, [0.0, 0.3, 0.4]);
}

#[derive(Deserialize)]
struct OracleParams {
    rising: f64,
    area_rising: f64,
    falling: f64,
    area_falling: f64,
    dt: f64,
    off: f64,
}

#[derive(Deserialize)]
struct OracleCase {
    row: Vec<f64>,
    params: OracleParams,
    expected: Vec<[f64; 3]>,
}

/// Cross-check the Rust raw decision against the numpy oracle bit-for-bit over
/// the deterministic fixture cases (`update_seg_cases.json`). This pins
/// Rust == Python on random inputs (the differential oracle that caught the
/// 0b-i softmax bug). Isolates the decision math: no smoothing on either side.
#[test]
fn oracle_cross_check_bit_exact() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0bii/update_seg_cases.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let cases: Vec<OracleCase> = serde_json::from_str(&text).unwrap();
    assert!(
        cases.len() >= 20,
        "expected >= 20 cases, got {}",
        cases.len()
    );

    for (idx, case) in cases.iter().enumerate() {
        // Area 0 for smoothing is irrelevant here: update_segmentation_raw does
        // no smoothing. Padding/min unused by the decision.
        let cfg = SegmenterConfig {
            rising: case.params.rising,
            area_rising: case.params.area_rising,
            falling: case.params.falling,
            area_falling: case.params.area_falling,
            padding: [0.0; 4],
            min_speech: [0.0; 3],
            min_silence: [0.0; 2],
        };
        let got = update_segmentation_raw(
            &case.row,
            SegClass::Speech,
            case.params.off,
            case.params.dt,
            &cfg,
        );
        assert_eq!(
            got.len(),
            case.expected.len(),
            "case {idx}: segment count mismatch (rust {} vs python {})",
            got.len(),
            case.expected.len()
        );
        for (si, ((begin, end, code), exp)) in got.iter().zip(case.expected.iter()).enumerate() {
            // Bit-for-bit: raw f64 bit patterns must match the Python oracle.
            assert_eq!(
                begin.to_bits(),
                exp[0].to_bits(),
                "case {idx} seg {si}: begin {begin:?} != {:?}",
                exp[0]
            );
            assert_eq!(
                end.to_bits(),
                exp[1].to_bits(),
                "case {idx} seg {si}: end {end:?} != {:?}",
                exp[1]
            );
            assert_eq!(
                *code, exp[2] as i32,
                "case {idx} seg {si}: code {code} != {}",
                exp[2] as i32
            );
        }
    }
}
