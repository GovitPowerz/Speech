//! Skeleton smoke tests: the crate boots and the real plumbing works.

use speech::cli::{Mode, ModeKind};
use speech::constants::{MAX_RAND_SIZE, random_uniform};

#[test]
fn version_is_set() {
    assert!(!speech::version().is_empty());
}

#[test]
fn mode_parses_flags() {
    assert_eq!(
        Mode::from_flag("-m"),
        Some(Mode {
            kind: ModeKind::Multi,
            verbose: false
        })
    );
    assert_eq!(
        Mode::from_flag("-s"),
        Some(Mode {
            kind: ModeKind::Solo,
            verbose: false
        })
    );
    assert_eq!(Mode::from_flag("--nope"), None);
}

#[test]
fn random_tables_load_and_index_modularly() {
    let u = random_uniform(0);
    assert!((0.0..1.0).contains(&u));
    // Modular indexing: index N wraps to index 0.
    assert_eq!(random_uniform(MAX_RAND_SIZE), random_uniform(0));
}
