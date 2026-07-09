//! Phase 4d Task 1: packer/adim byte golden closed against the real 2015 weight packs.
//!
//! Closes the Phase-0a deferral (CLAUDE.md "Phase 0a status"): the flat weight packer +
//! `adim_coeff` seam was never validated against an INDEPENDENT legacy golden pack, because
//! none was known to exist at Phase 0a time. Two real production tuples (real `.config` +
//! real `NNweights_config1.bin`) surfaced in the full legacy copy (see
//! `tests/reference_data/phase4d/phase4d_sources.json` for provenance + SHA-256):
//!   - Tuple A: Optimizer_V6.2.2/15-Oct-2015_BLSTM_OpenSAD15 (byte-identical to the
//!     already-committed `tests/reference_data/phase0` fixtures - the familiar 33,671 shape).
//!   - Tuple B: Optimizer_V6.2.1/14-Oct-2015_BLSTM_OpenSAD15 (genuinely new: a wider
//!     NNetInputSize=35 front end, MEASURED element count 42,911).
//!
//! Unlike `phase0_packer.rs::packer_bijection_on_real_bin` (vector-level `nnet_to_flat(
//! flat_to_nnet(v)) == v` equality on Tuple A only), this suite closes the round trip at the
//! FILE-BYTE level (`write_matrix` output diffed byte-for-byte against the committed fixture)
//! on BOTH tuples, and exercises `config_to_nnet`'s adim invariant (bias exempt, body scaled by
//! `1/adim`) against a genuinely novel architecture (Tuple B's NNetInputSize=35), which the
//! original Phase 0a `best_config_domain.bin` fixture (NNetInputSize=23) never covered.

use std::path::PathBuf;

use speech::config::{NnetSpec, config_to_nnet, element_count, flat_to_nnet, nnet_to_flat};
use speech::io::binary::{read_matrix, write_matrix};
use speech::legacy_config::parse_legacy_config;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase4d")
}

struct Tuple {
    tag: &'static str,
    config_name: &'static str,
    bin_name: &'static str,
    expected_element_count: usize,
}

const TUPLE_A: Tuple = Tuple {
    tag: "A",
    config_name: "tupleA_1_worker_1.config",
    bin_name: "tupleA_NNweights_config1.bin",
    expected_element_count: 33671,
};

const TUPLE_B: Tuple = Tuple {
    tag: "B",
    config_name: "tupleB_1_worker_1.config",
    bin_name: "tupleB_NNweights_config1.bin",
    expected_element_count: 42911,
};

fn spec_for(t: &Tuple) -> NnetSpec {
    let text = std::fs::read_to_string(ref_dir().join(t.config_name)).unwrap();
    let m = parse_legacy_config(&text);
    NnetSpec::from_legacy(&m, "BLSTM").unwrap()
}

#[test]
fn tuple_a_element_count_is_the_familiar_33671_shape() {
    let spec = spec_for(&TUPLE_A);
    assert_eq!(spec.lstm_neuron_nb, vec![23, 24, 24]);
    assert_eq!(spec.input_size, 23);
    assert_eq!(element_count(&spec), TUPLE_A.expected_element_count);
    let (rows, cols, data) = read_matrix(&ref_dir().join(TUPLE_A.bin_name)).unwrap();
    assert_eq!(cols, 1);
    assert_eq!(rows, TUPLE_A.expected_element_count);
    assert_eq!(data.len(), TUPLE_A.expected_element_count);
}

#[test]
fn tuple_b_element_count_is_measured_at_42911() {
    let spec = spec_for(&TUPLE_B);
    assert_eq!(spec.lstm_neuron_nb, vec![35, 24, 24]);
    assert_eq!(spec.input_size, 35);
    assert_eq!(element_count(&spec), TUPLE_B.expected_element_count);
    let (rows, cols, data) = read_matrix(&ref_dir().join(TUPLE_B.bin_name)).unwrap();
    assert_eq!(cols, 1);
    assert_eq!(rows, TUPLE_B.expected_element_count);
    assert_eq!(data.len(), TUPLE_B.expected_element_count);

    // Cross-check independent of NnetSpec/element_count entirely: the raw file size alone
    // pins the element count (16-byte header + 8 bytes/element), so a spec-derivation bug
    // cannot silently agree with a matching-but-wrong element_count implementation.
    let size = std::fs::metadata(ref_dir().join(TUPLE_B.bin_name))
        .unwrap()
        .len();
    assert_eq!((size - 16) / 8, TUPLE_B.expected_element_count as u64);
}

fn assert_repack_byte_identical(t: &Tuple) {
    let spec = spec_for(t);
    let bin_path = ref_dir().join(t.bin_name);
    let (rows, cols, flat) = read_matrix(&bin_path).unwrap();
    assert_eq!(flat.len(), element_count(&spec), "tuple {}", t.tag);

    let nnet = flat_to_nnet(&flat, &spec);
    let repacked = nnet_to_flat(&nnet, &spec);
    assert_eq!(repacked, flat, "tuple {}: vector-level bijection", t.tag);

    let dir = tempfile::tempdir().unwrap();
    let out_path = dir.path().join(t.bin_name);
    write_matrix(&out_path, rows, cols, &repacked).unwrap();

    let original_bytes = std::fs::read(&bin_path).unwrap();
    let repacked_bytes = std::fs::read(&out_path).unwrap();
    assert_eq!(
        repacked_bytes, original_bytes,
        "tuple {}: repacked file bytes must be byte-identical to the committed fixture",
        t.tag
    );
}

#[test]
fn tuple_a_repack_is_byte_identical_to_the_committed_file() {
    assert_repack_byte_identical(&TUPLE_A);
}

#[test]
fn tuple_b_repack_is_byte_identical_to_the_committed_file() {
    assert_repack_byte_identical(&TUPLE_B);
}

/// The `adim_coeff` invariant (CLAUDE.md locked decision: `sqrt(fan_in*sub + fan_out)`,
/// bias-excepted) holds structurally for both tuples' REAL architectures: every non-bias
/// column of every gate/output row is scaled by exactly `1/adim`, and every bias column
/// (the row's last element) passes through `config_to_nnet` untouched. `flat_to_nnet`'s
/// output is fed in as the `Structured` (config-domain) input purely as a shape-compatible
/// probe -- `Structured` and `Nnet` are the same container type (`config.rs:112`), and this
/// checks the SCALING BEHAVIOR of `config_to_nnet`, not a value-level domain claim.
fn assert_adim_invariants(t: &Tuple) {
    let spec = spec_for(t);
    let flat = read_matrix(&ref_dir().join(t.bin_name)).unwrap().2;
    let probe = flat_to_nnet(&flat, &spec);
    let scaled = config_to_nnet(&probe, &spec);

    let lstm = &spec.lstm_neuron_nb;
    for (dir_probe, dir_scaled) in [
        (&probe.forward, &scaled.forward),
        (&probe.backward, &scaled.backward),
    ] {
        for (i, (layer_probe, layer_scaled)) in dir_probe.iter().zip(dir_scaled.iter()).enumerate()
        {
            let out = lstm[i + 1];
            let fin = lstm[i] * spec.lstm_subsampling[i];
            let adim = ((lstm[i] * spec.lstm_subsampling[i] + lstm[i + 1]) as f64).sqrt();
            assert!(
                adim.is_finite() && adim > 0.0,
                "tuple {}: adim must be finite positive",
                t.tag
            );

            for (gate_probe, gate_scaled, ncols) in [
                (&layer_probe.input, &layer_scaled.input, fin + out + 5),
                (&layer_probe.forget, &layer_scaled.forget, fin + out + 5),
                (&layer_probe.output, &layer_scaled.output, fin + out + 5),
                (&layer_probe.cell, &layer_scaled.cell, fin + out + 1),
            ] {
                for r in 0..out {
                    let row_probe = &gate_probe[r * ncols..r * ncols + ncols];
                    let row_scaled = &gate_scaled[r * ncols..r * ncols + ncols];
                    // Bias (last column) is exempt: passes through unscaled, bit-exact.
                    assert_eq!(
                        row_scaled[ncols - 1],
                        row_probe[ncols - 1],
                        "tuple {} layer {}: bias must pass through",
                        t.tag,
                        i
                    );
                    // Every other column is scaled by exactly 1/adim.
                    for c in 0..ncols - 1 {
                        assert_eq!(
                            row_scaled[c],
                            row_probe[c] / adim,
                            "tuple {} layer {} row {} col {}",
                            t.tag,
                            i,
                            r,
                            c
                        );
                    }
                }
            }
        }
    }

    let outn = &spec.output_neuron_nb;
    for (i, (mat_probe, mat_scaled)) in probe.output.iter().zip(scaled.output.iter()).enumerate() {
        let out = outn[i + 1];
        let ncols = outn[i] + 1;
        let adim = ((outn[i] * spec.output_subsampling[i]) as f64).sqrt();
        assert!(
            adim.is_finite() && adim > 0.0,
            "tuple {}: output adim must be finite positive",
            t.tag
        );
        for r in 0..out {
            let row_probe = &mat_probe[r * ncols..r * ncols + ncols];
            let row_scaled = &mat_scaled[r * ncols..r * ncols + ncols];
            assert_eq!(
                row_scaled[ncols - 1],
                row_probe[ncols - 1],
                "tuple {} output layer {}: bias must pass through",
                t.tag,
                i
            );
            for c in 0..ncols - 1 {
                assert_eq!(
                    row_scaled[c],
                    row_probe[c] / adim,
                    "tuple {} output layer {} row {} col {}",
                    t.tag,
                    i,
                    r,
                    c
                );
            }
        }
    }
}

#[test]
fn tuple_a_adim_invariants_hold() {
    assert_adim_invariants(&TUPLE_A);
}

#[test]
fn tuple_b_adim_invariants_hold() {
    assert_adim_invariants(&TUPLE_B);
}
