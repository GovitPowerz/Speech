//! Skeleton smoke tests: the crate boots and the real plumbing works.

use speech::cli::Mode;
use speech::config::Config;
use speech::constants::{MAX_RAND_SIZE, random_uniform};

#[test]
fn version_is_set() {
    assert!(!speech::version().is_empty());
}

#[test]
fn mode_parses_flags() {
    assert_eq!(Mode::from_flag("-m"), Some(Mode::Multi));
    assert_eq!(Mode::from_flag("-s"), Some(Mode::Solo));
    assert_eq!(Mode::from_flag("--nope"), None);
}

#[test]
fn config_parses_minimal_toml() {
    let cfg = Config::from_toml_str("[engine]\nalgo = \"twin_blstm_lid\"\n").unwrap();
    assert_eq!(cfg.engine.algo, "twin_blstm_lid");
    assert_eq!(cfg.engine.num_outer_threads, 1);
}

#[test]
fn random_tables_load_and_index_modularly() {
    let u = random_uniform(0);
    assert!((0.0..1.0).contains(&u));
    // Modular indexing: index N wraps to index 0.
    assert_eq!(random_uniform(MAX_RAND_SIZE), random_uniform(0));
}
