//! Segmentation container tests (hand-verified boundary lists).

use speech::tasks::segmentation::{SegClass, Segmentation};

fn begins(s: &Segmentation) -> Vec<f64> {
    s.segments().iter().map(|x| x.begin).collect()
}
fn types(s: &Segmentation) -> Vec<SegClass> {
    s.segments().iter().map(|x| x.ty).collect()
}

#[test]
fn new_is_other_then_end_sentinel() {
    let s = Segmentation::new(120.0);
    assert_eq!(begins(&s), vec![0.0, 120.0]);
    assert_eq!(types(&s), vec![SegClass::Other, SegClass::End]);
}

#[test]
fn label_inserts_typed_interval() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(2.0, 5.0, SegClass::Speech);
    assert_eq!(begins(&s), vec![0.0, 2.0, 5.0, 10.0]);
    assert_eq!(
        types(&s),
        vec![
            SegClass::Other,
            SegClass::Speech,
            SegClass::Other,
            SegClass::End
        ]
    );
}

#[test]
fn sanitize_rounds_1e4_half_away_and_merges() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(2.00004, 5.0, SegClass::Speech); // 2.00004 -> 2.0000 (round to 1e-4 grid)
    s.label_segment(5.0, 7.0, SegClass::Speech); // adjacent same-type -> merge on sanitize
    s.sanitize();
    assert_eq!(
        types(&s),
        vec![
            SegClass::Other,
            SegClass::Speech,
            SegClass::Other,
            SegClass::End
        ]
    );
    assert_eq!(begins(&s)[1], 2.0000);
    // terminal sentinel unchanged
    assert_eq!(*begins(&s).last().unwrap(), 10.0);
}

// ---- Additional hand-verified paths ----

#[test]
fn segclass_repr_codes_are_load_bearing() {
    assert_eq!(SegClass::Other as i32, 0);
    assert_eq!(SegClass::Speech as i32, 1);
    assert_eq!(SegClass::Ring as i32, 2);
    assert_eq!(SegClass::Dtmf0 as i32, 3);
    assert_eq!(SegClass::Dtmf9 as i32, 12);
    assert_eq!(SegClass::DtmfA as i32, 13);
    assert_eq!(SegClass::DtmfD as i32, 16);
    assert_eq!(SegClass::DtmfStar as i32, 17);
    assert_eq!(SegClass::DtmfSharp as i32, 18);
    assert_eq!(SegClass::Insertion as i32, 19);
    assert_eq!(SegClass::Substitution as i32, 20);
    assert_eq!(SegClass::Excluded as i32, 21);
    assert_eq!(SegClass::End as i32, 22);
}

#[test]
fn label_clamps_begin_to_zero() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(-3.0, 4.0, SegClass::Speech); // begin clamped to 0.0
    assert_eq!(begins(&s), vec![0.0, 4.0, 10.0]);
    assert_eq!(
        types(&s),
        vec![SegClass::Speech, SegClass::Other, SegClass::End]
    );
}

#[test]
fn label_end_past_duration_reclose_at_sentinel() {
    let mut s = Segmentation::new(10.0);
    // endTime 12.0 > sentinel begin 10.0: no closing boundary inserted before sentinel.
    s.label_segment(6.0, 12.0, SegClass::Speech);
    assert_eq!(begins(&s), vec![0.0, 6.0, 10.0]);
    assert_eq!(
        types(&s),
        vec![SegClass::Other, SegClass::Speech, SegClass::End]
    );
}

#[test]
fn label_endtime_at_or_before_start_noop() {
    let mut s = Segmentation::new(10.0);
    // endTime == first begin (0.0): guard `endTime > segs[0].begin` false -> no-op.
    s.label_segment(0.0, 0.0, SegClass::Speech);
    assert_eq!(begins(&s), vec![0.0, 10.0]);
    assert_eq!(types(&s), vec![SegClass::Other, SegClass::End]);
}

#[test]
fn label_begin_at_sentinel_noop() {
    let mut s = Segmentation::new(10.0);
    // beginTime == sentinel begin: guard `beginTime < segs[last].begin` false -> no-op.
    s.label_segment(10.0, 20.0, SegClass::Speech);
    assert_eq!(begins(&s), vec![0.0, 10.0]);
    assert_eq!(types(&s), vec![SegClass::Other, SegClass::End]);
}

#[test]
fn label_mid_list_overwrites_and_recloses_with_previous_type() {
    let mut s = Segmentation::new(20.0);
    s.label_segment(2.0, 6.0, SegClass::Speech); // [0 O, 2 S, 6 O, 20 End]
    s.label_segment(10.0, 14.0, SegClass::Ring); // [0 O, 2 S, 6 O, 10 R, 14 O, 20 End]
    assert_eq!(begins(&s), vec![0.0, 2.0, 6.0, 10.0, 14.0, 20.0]);
    assert_eq!(
        types(&s),
        vec![
            SegClass::Other,
            SegClass::Speech,
            SegClass::Other,
            SegClass::Ring,
            SegClass::Other,
            SegClass::End
        ]
    );
    // Overlap a new interval starting inside Speech, ending inside Ring: the
    // advance loop leaves previousType=Speech, then the erase loop walks O then
    // R, so the reclose at 12.0 takes the last-erased type = Ring.
    s.label_segment(4.0, 12.0, SegClass::Speech); // spans across S/O/R
    assert_eq!(begins(&s), vec![0.0, 2.0, 4.0, 12.0, 14.0, 20.0]);
    assert_eq!(
        types(&s),
        vec![
            SegClass::Other,
            SegClass::Speech,
            SegClass::Speech,
            SegClass::Ring,
            SegClass::Other,
            SegClass::End
        ]
    );
}

#[test]
fn sanitize_round_half_away_at_00005() {
    let mut s = Segmentation::new(10.0);
    // 2.00005 * 1e4 = 20000.5 -> round half away -> 20001 -> 2.0001
    s.label_segment(2.00005, 5.0, SegClass::Speech);
    s.sanitize();
    assert_eq!(begins(&s)[1], 2.0001);
}

#[test]
fn sanitize_leaves_terminal_sentinel_unrounded() {
    // Give a duration that would move under 1e-4 rounding if it were rounded.
    let mut s = Segmentation::new(9.999_95);
    s.sanitize();
    // Only [0.0, End@9.99995]; the loop exits at the sentinel, so it is never rounded.
    assert_eq!(*begins(&s).last().unwrap(), 9.999_95);
}

#[test]
fn audio_duration_accessor() {
    let s = Segmentation::new(42.5);
    assert_eq!(s.audio_duration(), 42.5);
}
