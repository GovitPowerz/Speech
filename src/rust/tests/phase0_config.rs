//! Legacy .config parsing tests (Rust side).

use std::path::PathBuf;

fn cfg_text() -> String {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/reference_data/phase0/1_worker_1.config");
    std::fs::read_to_string(p).unwrap()
}

#[test]
fn parse_last_wins() {
    let m = speech::legacy_config::parse_legacy_config("a 1\n# c 2\na 3\n");
    assert_eq!(m.get("a").map(String::as_str), Some("3"));
    assert!(!m.contains_key("c"));
}

#[test]
fn nnet_spec_from_real_config() {
    let m = speech::legacy_config::parse_legacy_config(&cfg_text());
    let spec = speech::config::NnetSpec::from_legacy(&m, "BLSTM").unwrap();
    assert_eq!(spec.lstm_neuron_nb, vec![23, 24, 24]);
    assert_eq!(spec.lstm_subsampling, vec![4, 1]);
    assert_eq!(spec.output_neuron_nb, vec![48, 12, 1]);
    assert_eq!(spec.output_subsampling, vec![1, 1]);
    assert_eq!(spec.input_size, 23);
    assert_eq!(spec.peepholes, [true; 6]);
}

/// Peephole flags follow `PeepholeFlags::from_legacy` (issue #32): absent -> true, exactly
/// `true`/`false` (trimmed) as written, anything else errors naming the key.
#[test]
fn nnet_spec_peephole_flags_read_strictly() {
    let real = speech::legacy_config::parse_legacy_config(&cfg_text());
    let key = "BLSTM_Backward_IsGatesPeepholesActive";
    let spec_with = |v: Option<&str>| {
        let mut m = real.clone();
        match v {
            Some(v) => m.insert(key.into(), v.into()),
            None => m.shift_remove(key),
        };
        speech::config::NnetSpec::from_legacy(&m, "BLSTM")
    };
    assert_eq!(spec_with(None).unwrap().peepholes, [true; 6]);
    assert!(!spec_with(Some(" false ")).unwrap().peepholes[3]);
    for bad in ["True", "1", "ture", ""] {
        let err = format!("{:#}", spec_with(Some(bad)).unwrap_err());
        assert!(
            err.contains(key) && err.contains(&format!("'{bad}'")),
            "{bad}: {err}"
        );
    }
}
