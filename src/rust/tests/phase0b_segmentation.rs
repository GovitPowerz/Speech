//! Segmentation container tests (hand-verified boundary lists).

use proptest::prelude::*;
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

// ---- Task 5: suppress_short / add_padding / modify_type / update_count ----

#[test]
fn suppress_short_drops_and_merges() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(2.0, 2.05, SegClass::Speech); // 0.05s speech, below threshold 0.1
    s.suppress_short(0.1, SegClass::Speech);
    // the short speech is removed; neighbors (both Other) merge -> whole thing is Other
    assert_eq!(types(&s), vec![SegClass::Other, SegClass::End]);
}

#[test]
fn suppress_short_threshold_zero_is_noop() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(2.0, 2.05, SegClass::Speech);
    let before = begins(&s);
    s.suppress_short(0.0, SegClass::Speech);
    assert_eq!(begins(&s), before);
}

#[test]
fn update_count_sums_class_durations() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(2.0, 5.0, SegClass::Speech);
    let counts = s.update_count();
    assert_eq!(counts[SegClass::Speech as usize], 3.0);
    assert_eq!(counts[SegClass::Other as usize], 7.0);
}

// suppress branch: head. `[S@0(0.05), O@0.05, End]` -> pull O.begin to 0.0, erase S.
#[test]
fn suppress_short_head_branch() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(0.0, 0.05, SegClass::Speech); // [S@0, O@0.05, End@10]
    assert_eq!(
        types(&s),
        vec![SegClass::Speech, SegClass::Other, SegClass::End]
    );
    s.suppress_short(0.1, SegClass::Speech);
    assert_eq!(begins(&s), vec![0.0, 10.0]);
    assert_eq!(types(&s), vec![SegClass::Other, SegClass::End]);
}

// suppress branch: pre-sentinel (it+2 == end). `[O@0, S@9.95, End@10]` -> erase S.
#[test]
fn suppress_short_pre_sentinel_branch() {
    let mut s = Segmentation::new(10.0);
    s.label_segment(9.95, 10.0, SegClass::Speech); // [O@0, S@9.95, End@10]
    assert_eq!(
        types(&s),
        vec![SegClass::Other, SegClass::Speech, SegClass::End]
    );
    s.suppress_short(0.1, SegClass::Speech);
    assert_eq!(begins(&s), vec![0.0, 10.0]);
    assert_eq!(types(&s), vec![SegClass::Other, SegClass::End]);
}

// suppress branch: neighbor-match full merge. previousType == (it+1).ty -> erase both.
// [O@0, R@2, S@5(0.05), R@5.05, O@10, End@20] -> [O@0, R@2, O@10, End@20].
#[test]
fn suppress_short_neighbor_match_full_merge() {
    let mut s = Segmentation::new(20.0);
    s.label_segment(2.0, 10.0, SegClass::Ring); // [O@0, R@2, O@10, End@20]
    s.label_segment(5.0, 5.05, SegClass::Speech); // carve S inside R
    assert_eq!(
        types(&s),
        vec![
            SegClass::Other,
            SegClass::Ring,
            SegClass::Speech,
            SegClass::Ring,
            SegClass::Other,
            SegClass::End
        ]
    );
    s.suppress_short(0.1, SegClass::Speech);
    assert_eq!(begins(&s), vec![0.0, 2.0, 10.0, 20.0]);
    assert_eq!(
        types(&s),
        vec![
            SegClass::Other,
            SegClass::Ring,
            SegClass::Other,
            SegClass::End
        ]
    );
}

// suppress branch: neighbor-differ half-split. (it+1).begin -= dur/2, erase short.
// [O@0, R@2, S@5(0.06), O@5.06, R@10, O@12, End@20]
//   -> (it+1)=O@5.06.begin -= 0.03 = 5.03 -> [O@0, R@2, O@5.03, R@10, O@12, End@20].
#[test]
fn suppress_short_neighbor_differ_half_split() {
    let mut s = Segmentation::new(20.0);
    s.label_segment(2.0, 5.0, SegClass::Ring); // [O@0, R@2, O@5, End@20]
    s.label_segment(5.0, 5.06, SegClass::Speech); // [O@0, R@2, S@5, O@5.06, End@20]
    s.label_segment(10.0, 12.0, SegClass::Ring); // add trailing seg so S is not pre-sentinel
    assert_eq!(
        types(&s),
        vec![
            SegClass::Other,
            SegClass::Ring,
            SegClass::Speech,
            SegClass::Other,
            SegClass::Ring,
            SegClass::Other,
            SegClass::End
        ]
    );
    s.suppress_short(0.1, SegClass::Speech);
    assert_eq!(begins(&s), vec![0.0, 2.0, 5.03, 10.0, 12.0, 20.0]);
    assert_eq!(
        types(&s),
        vec![
            SegClass::Other,
            SegClass::Ring,
            SegClass::Other,
            SegClass::Ring,
            SegClass::Other,
            SegClass::End
        ]
    );
}

// no-advance invariant: two consecutive short Speech separated by Ring. After the
// first full-merge (double erase), `it` must stay put so the shifted-in second short
// Speech is re-tested. A naive `++it` after erase would drop the second short.
// [O@0, R@2, S@5(0.03), R@5.03, S@5.05(0.03), R@5.08, O@10, End@20]
//   -> [O@0, R@2, O@10, End@20].
#[test]
fn suppress_short_no_advance_after_erase() {
    let mut s = Segmentation::new(20.0);
    s.label_segment(2.0, 10.0, SegClass::Ring); // [O@0, R@2, O@10, End@20]
    s.label_segment(5.0, 5.03, SegClass::Speech); // first short inside R
    s.label_segment(5.05, 5.08, SegClass::Speech); // second short inside R
    assert_eq!(
        types(&s),
        vec![
            SegClass::Other,
            SegClass::Ring,
            SegClass::Speech,
            SegClass::Ring,
            SegClass::Speech,
            SegClass::Ring,
            SegClass::Other,
            SegClass::End
        ]
    );
    s.suppress_short(0.1, SegClass::Speech);
    // both shorts gone, all Ring merged
    assert_eq!(begins(&s), vec![0.0, 2.0, 10.0, 20.0]);
    assert_eq!(
        types(&s),
        vec![
            SegClass::Other,
            SegClass::Ring,
            SegClass::Other,
            SegClass::End
        ]
    );
}

#[test]
fn add_padding_grows_both_sides() {
    let mut s = Segmentation::new(20.0);
    s.label_segment(5.0, 8.0, SegClass::Speech); // [O@0, S@5, O@8, End@20]
    s.add_padding(1.0, 2.0, SegClass::Speech); // grow left by 1, right by 2
    // Speech now spans [4.0, 10.0].
    assert_eq!(begins(&s), vec![0.0, 4.0, 10.0, 20.0]);
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
fn add_padding_zero_before_skips_left() {
    let mut s = Segmentation::new(20.0);
    s.label_segment(5.0, 8.0, SegClass::Speech);
    s.add_padding(0.0, 2.0, SegClass::Speech); // only right side grows
    assert_eq!(begins(&s), vec![0.0, 5.0, 10.0, 20.0]);
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
fn add_padding_zero_after_skips_right() {
    let mut s = Segmentation::new(20.0);
    s.label_segment(5.0, 8.0, SegClass::Speech);
    s.add_padding(1.0, 0.0, SegClass::Speech); // only left side grows
    assert_eq!(begins(&s), vec![0.0, 4.0, 8.0, 20.0]);
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
fn modify_type_retypes_and_merges() {
    let mut s = Segmentation::new(20.0);
    s.label_segment(2.0, 5.0, SegClass::Speech); // [O@0, S@2, O@5, End@20]
    s.label_segment(5.0, 8.0, SegClass::Ring); // [O@0, S@2, R@5, O@8, End@20]
    s.modify_type(SegClass::Ring, SegClass::Speech); // R -> S, adjacent S@2/S@5 merge
    assert_eq!(begins(&s), vec![0.0, 2.0, 8.0, 20.0]);
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
fn update_count_multi_class() {
    let mut s = Segmentation::new(20.0);
    s.label_segment(2.0, 5.0, SegClass::Speech); // 3s Speech
    s.label_segment(10.0, 13.0, SegClass::Ring); // 3s Ring
    let c = s.update_count();
    assert_eq!(c[SegClass::Speech as usize], 3.0);
    assert_eq!(c[SegClass::Ring as usize], 3.0);
    assert_eq!(c[SegClass::Other as usize], 14.0); // 2 + 5 + 7
    assert_eq!(c[SegClass::End as usize], 0.0); // sentinel never counted
}

// spec 4.3: sanitize is idempotent and never reorders boundaries (non-decreasing).
proptest! {
    #[test]
    fn sanitize_idempotent_and_non_decreasing(
        dur in 1.0f64..100.0,
        ops in prop::collection::vec(
            (0.0f64..100.0, 0.0f64..100.0, 0usize..3),
            0..12,
        ),
    ) {
        let mut s = Segmentation::new(dur);
        let classes = [SegClass::Other, SegClass::Speech, SegClass::Ring];
        for (a, b, k) in ops {
            let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
            s.label_segment(lo, hi, classes[k]);
        }
        s.sanitize();
        let a = begins(&s);
        s.sanitize();
        let b = begins(&s);
        prop_assert_eq!(&a, &b); // idempotent
        // boundaries never reorder (non-decreasing)
        for w in b.windows(2) {
            prop_assert!(w[0] <= w[1]);
        }
    }
}
