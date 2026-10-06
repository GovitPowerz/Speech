//! Multiclass softmax cross-entropy tests.
//!
//! No `.mat` golden exists for this path; expectations are computed from the
//! exact legacy form in CostLaw.cpp (computeCost/computeDeltas, targetSeq.cols()>1):
//!   - cost += -ln(clamp(output,1e-24,inf)) * ponderation  over target>0.5 entries
//!   - delta = output - onehot(target), row target<0 => 0 (ignore)
//!   - class-ponderation and WER-ponderation applied as the legacy does.

use indexmap::IndexMap;

fn cfg(pairs: &[(&str, &str)]) -> IndexMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn delta_is_output_minus_onehot() {
    // 2 frames x 3 classes; frame 0 target class 1, frame 1 ignored (all target < 0).
    let law = speech::cost::CostLaw::from_config(&Default::default(), "BLSTM").unwrap();
    let outputs = vec![0.2, 0.5, 0.3, 0.1, 0.6, 0.3];
    let target = vec![0.0, 1.0, 0.0, -1.0, -1.0, -1.0];
    let mut deltas = vec![0.0; 6];
    law.compute_deltas(&outputs, &target, 3, &mut deltas);
    assert_eq!(deltas[0], 0.2 - 0.0);
    assert_eq!(deltas[1], 0.5 - 1.0);
    assert_eq!(deltas[2], 0.3 - 0.0);
    assert_eq!(&deltas[3..6], &[0.0, 0.0, 0.0]); // ignored frame
}

#[test]
fn cost_is_cross_entropy_over_on_class() {
    // Only the target>0.5 entry contributes: -ln(output) at that class.
    let law = speech::cost::CostLaw::from_config(&Default::default(), "BLSTM").unwrap();
    let outputs = vec![0.2, 0.5, 0.3, 0.1, 0.6, 0.3];
    let target = vec![0.0, 1.0, 0.0, -1.0, -1.0, -1.0];
    let cost = law.compute_cost(&outputs, &target, 3);
    assert_eq!(cost, -(0.5_f64).ln());
}

#[test]
fn cost_accumulates_column_major_bit_exact() {
    // 3 frames x 2 classes; on-class entries at (frame0,class0), (frame1,class1),
    // (frame2,class0). f64 addition is not associative, so the accumulation order
    // matters: legacy CostLaw.cpp:219-231 sums column-by-column (outer kk, inner jj),
    // not row-by-row. These specific values were found to diverge in the last bit(s)
    // between row-major and column-major summation.
    let law = speech::cost::CostLaw::from_config(&Default::default(), "BLSTM").unwrap();
    let outputs = vec![
        0.276841,
        0.759708,
        0.8187500000000001,
        0.805755,
        0.495479,
        0.8323560000000001,
    ];
    let target = vec![1.0, -1.0, -1.0, 1.0, 1.0, -1.0];
    let cost = law.compute_cost(&outputs, &target, 2);

    // Column-major reference: sum class0's on-class terms first (frame0, frame2),
    // then class1's (frame1) -- same order as the legacy double loop.
    let mut expected = 0.0f64;
    expected += -(outputs[0].max(1e-24)).ln(); // (frame0, class0)
    expected += -(outputs[4].max(1e-24)).ln(); // (frame2, class0)
    expected += -(outputs[3].max(1e-24)).ln(); // (frame1, class1)

    assert_eq!(cost, expected);

    // Sanity: row-major summation of the same terms gives a different f64 (pins the bug).
    let mut row_major = 0.0f64;
    row_major += -(outputs[0].max(1e-24)).ln(); // (frame0, class0)
    row_major += -(outputs[3].max(1e-24)).ln(); // (frame1, class1)
    row_major += -(outputs[4].max(1e-24)).ln(); // (frame2, class0)
    assert_ne!(
        expected, row_major,
        "test fixture no longer exercises order-dependence"
    );
}

#[test]
fn cost_clamps_tiny_output_to_1e_24() {
    // output below 1e-24 is clamped up before the log (legacy: value<1e-24 => 1e-24).
    let law = speech::cost::CostLaw::from_config(&Default::default(), "BLSTM").unwrap();
    let outputs = vec![0.0, 1.0, 0.0];
    let target = vec![1.0, 0.0, 0.0];
    let cost = law.compute_cost(&outputs, &target, 3);
    assert_eq!(cost, -(1e-24_f64).ln());
}

#[test]
fn class_ponderation_scales_cost_and_deltas() {
    // classes_ponderations = [2,3,4]. Legacy cost multiplies -ln(output) by pond[k]
    // of the on-class column. Legacy non-WER deltas: the whole frame row is scaled
    // by the on-class ponderation.
    let law = speech::cost::CostLaw::from_config(
        &cfg(&[("BLSTM_classes_ponderations", "2,3,4")]),
        "BLSTM",
    )
    .unwrap();
    let outputs = vec![0.2, 0.5, 0.3];
    let target = vec![0.0, 1.0, 0.0]; // on-class = 1 => pond 3.0
    let cost = law.compute_cost(&outputs, &target, 3);
    assert_eq!(cost, -(0.5_f64).ln() * 3.0);
    let mut deltas = vec![0.0; 3];
    law.compute_deltas(&outputs, &target, 3, &mut deltas);
    assert_eq!(deltas[0], (0.2 - 0.0) * 3.0);
    assert_eq!(deltas[1], (0.5 - 1.0) * 3.0);
    assert_eq!(deltas[2], (0.3 - 0.0) * 3.0);
}

#[test]
fn back_prop_wer_scales_cost_and_deltas() {
    // BackPropWER on: cost += -ln(output)*10*(1-target) over target>0.5 entries.
    // deltas: on-class => 10*(1-target)*(output-1); off-class => 10*target*output.
    let law =
        speech::cost::CostLaw::from_config(&cfg(&[("BLSTM_BackPropWER", "1")]), "BLSTM").unwrap();
    let outputs = vec![0.2, 0.5, 0.3];
    let target = vec![0.0, 0.8, 0.0]; // on-class target = 0.8
    let cost = law.compute_cost(&outputs, &target, 3);
    assert_eq!(cost, -(0.5_f64).ln() * 10.0 * (1.0 - 0.8));
    let mut deltas = vec![0.0; 3];
    law.compute_deltas(&outputs, &target, 3, &mut deltas);
    assert_eq!(deltas[0], 10.0 * 0.0 * 0.2); // off-class: 10*target*output, target=0
    assert_eq!(deltas[1], 10.0 * (1.0 - 0.8) * (0.5 - 1.0)); // on-class
    assert_eq!(deltas[2], 10.0 * 0.0 * 0.3);
}

#[test]
fn back_prop_wer_with_class_ponderation() {
    // Both on: cost += -ln(output)*pond[k]*10*(1-target); deltas per-element pond[k].
    let law = speech::cost::CostLaw::from_config(
        &cfg(&[
            ("BLSTM_BackPropWER", "1"),
            ("BLSTM_classes_ponderations", "2,3,4"),
        ]),
        "BLSTM",
    )
    .unwrap();
    let outputs = vec![0.2, 0.5, 0.3];
    let target = vec![0.0, 0.8, 0.0]; // on-class = 1 => pond 3.0
    let cost = law.compute_cost(&outputs, &target, 3);
    assert_eq!(cost, -(0.5_f64).ln() * 3.0 * 10.0 * (1.0 - 0.8));
    let mut deltas = vec![0.0; 3];
    law.compute_deltas(&outputs, &target, 3, &mut deltas);
    assert_eq!(deltas[0], 2.0 * 10.0 * 0.0 * 0.2); // off-class k=0 pond 2
    assert_eq!(deltas[1], 3.0 * 10.0 * (1.0 - 0.8) * (0.5 - 1.0)); // on-class k=1 pond 3
    assert_eq!(deltas[2], 4.0 * 10.0 * 0.0 * 0.3); // off-class k=2 pond 4
}
