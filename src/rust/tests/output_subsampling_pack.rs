//! Issue #61: the output MLP's pack length at `OutputSubSampling > 1`.
//!
//! Output layer `i` reads `OutputNeuronNb[i] * OutputSubSampling[i]` inputs: the exact tree
//! builds it so (`Network::new`), the legacy packs it so (`vec2struct.m:908`, `nbNeed =
//! Output_net_size(ii)*OutputSubSampling(ii)+1`), and every fast reader consumes it so.
//! `config::element_count`, the count `FastBlstm::from_flat` checks a pack against, read
//! `OutputNeuronNb[i]` alone. At osub > 1 an exact-length pack then tripped the
//! consumed-vs-count assert, and a pack up to the deficit short passed the length check and
//! panicked on a slice instead of being refused. The packer (`config_to_nnet`, `nnet_to_flat`,
//! `flat_to_nnet`) framed the output rows the same short way. Every committed fixture has
//! osub = 1.
//!
//! One row per net of `twin_mode7.config`, with that net's `OutputSubSampling` set to 2.

use indexmap::IndexMap;
use speech::config::{config_to_nnet, element_count, flat_to_nnet, nnet_to_flat};
use speech::fast::driver::{FastSpectralSegmenter, FastTwinLid, build_aligned_spec};
use speech::nn::blstm::{BlstmConfig, BlstmNetwork};
use speech::tasks::lid::TwinBlstmSpectralLid;
use speech::tasks::sad::BlstmSpectralSegmenter;

/// `(prefix, pack length at osub 2)`. SAD `11,12` (sub 4) / MLP `24,1`: 5760 + (24*2 + 1) +
/// 22 = 5831. LID `36,24` / MLP `48,1`: 12288 + (48*2 + 1) + 72 = 12457. At osub 1 they are
/// 5807 and 12409 (the committed LID pack).
const ROWS: [(&str, usize); 2] = [("BLSTM", 5831), ("BLSTM_LID", 12457)];

fn osub2_map(prefix: &str) -> IndexMap<String, String> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase4b/twin_mode7.config");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    let mut map = speech::legacy_config::parse_legacy_config(&text);
    map.insert(format!("{prefix}_OutputSubSampling"), "2".into());
    // The fast SAD driver is an algo-3 plan since issue #24 (`FastSadPlan::from_map`
    // refuses any other `Algo_choice` and, rate-free, the truncate regime the Twin's
    // `BLSTM_shift 0` resolves to -- refusals the pre-#24 driver raised only at the first
    // `get_segmentation`, which this test never reaches). The exact SAD driver reads
    // neither at construction, so the BLSTM row runs both trees on one map under algo 3
    // and an overlap shift; the pack length depends on neither key.
    if prefix == "BLSTM" {
        map.insert("Algo_choice".into(), "3".into());
        map.insert("BLSTM_shift".into(), "0.8".into());
    }
    map
}

fn ramp(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.5 + (i as f64) * 1e-3).collect()
}

/// Build the driver that owns the `prefix` net on each tree from `pack`: `(exact, fast)`.
fn build(prefix: &str, pack: &[f64]) -> (Result<(), String>, Result<(), String>) {
    let map = osub2_map(prefix);
    let msg = |e: anyhow::Error| e.to_string();
    if prefix == "BLSTM" {
        (
            BlstmSpectralSegmenter::from_legacy(&map, Some(pack))
                .map(drop)
                .map_err(msg),
            FastSpectralSegmenter::from_legacy(&map, Some(pack))
                .map(drop)
                .map_err(msg),
        )
    } else {
        (
            TwinBlstmSpectralLid::from_legacy(&map, None, Some(pack))
                .map(drop)
                .map_err(msg),
            FastTwinLid::from_legacy(&map, None, Some(pack))
                .map(drop)
                .map_err(msg),
        )
    }
}

#[test]
fn element_count_is_the_exact_nets_pack_length() {
    for (prefix, n) in ROWS {
        let map = osub2_map(prefix);
        let net =
            BlstmNetwork::from_config(BlstmConfig::from_legacy(&map, prefix).unwrap()).unwrap();
        assert_eq!(net.nb_of_weights(), n, "{prefix}: the exact net");
        // The spec both fast drivers check their pack against.
        let spec = build_aligned_spec(&map, prefix).unwrap();
        assert_eq!(element_count(&spec), n, "{prefix}: config::element_count");
    }
}

#[test]
fn an_exact_length_pack_builds_on_both_trees_and_a_short_one_is_refused() {
    for (prefix, n) in ROWS {
        let (exact, fast) = build(prefix, &ramp(n));
        for (tree, res) in [("exact", exact), ("fast", fast)] {
            res.unwrap_or_else(|e| panic!("{prefix}/{tree}: the exact length must build: {e}"));
        }

        let (exact, fast) = build(prefix, &ramp(n - 1));
        for (tree, res) in [("exact", exact), ("fast", fast)] {
            let err = res
                .err()
                .unwrap_or_else(|| panic!("{prefix}/{tree}: a short pack must be refused"));
            assert!(
                err.contains(&(n - 1).to_string()) && err.contains(&n.to_string()),
                "{prefix}/{tree}: the refusal must name both lengths, got: {err}"
            );
        }
    }
}

#[test]
fn the_packer_frames_the_output_rows_at_the_subsampled_width() {
    for (prefix, n) in ROWS {
        let spec = build_aligned_spec(&osub2_map(prefix), prefix).unwrap();
        let flat = ramp(n);
        let nnet = flat_to_nnet(&flat, &spec);
        assert_eq!(nnet_to_flat(&nnet, &spec), flat, "{prefix}: round trip");

        let outn = &spec.output_neuron_nb;
        let ncols = outn[0] * 2 + 1;
        assert_eq!(
            nnet.output[0].len(),
            outn[1] * ncols,
            "{prefix}: output matrix"
        );

        // `config_to_nnet` exempts exactly each row's last column, the bias.
        let adim = ((outn[0] * 2) as f64).sqrt();
        let scaled = config_to_nnet(&nnet, &spec);
        for (row, row_scaled) in nnet.output[0]
            .chunks(ncols)
            .zip(scaled.output[0].chunks(ncols))
        {
            assert_eq!(row_scaled[ncols - 1], row[ncols - 1], "{prefix}: bias");
            for c in 0..ncols - 1 {
                assert_eq!(row_scaled[c], row[c] / adim, "{prefix}: col {c}");
            }
        }
    }
}
