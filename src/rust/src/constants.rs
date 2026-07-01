//! Fixed random tables ported from the legacy Constants.h.
//!
//! Two 100000-entry f64 tables provide the legacy engine's determinism:
//! `_RandomGaussVector` (N(0,1), weight init) and `_RandomVector` (U[0,1),
//! dither/augmentation). Embedded from data/random_tables.bin (gauss then
//! uniform) and indexed modulo [`MAX_RAND_SIZE`], exactly like the legacy.

/// Length of each random table (legacy `_MaxRandSize`).
pub const MAX_RAND_SIZE: usize = 100_000;

static RANDOM_TABLES: &[u8] = include_bytes!("../../../data/random_tables.bin");

fn table(offset_entries: usize, i: usize) -> f64 {
    let idx = offset_entries + (i % MAX_RAND_SIZE);
    let start = idx * 8;
    let bytes: [u8; 8] = RANDOM_TABLES[start..start + 8].try_into().unwrap();
    f64::from_le_bytes(bytes)
}

/// `_RandomGaussVector[i % MAX_RAND_SIZE]` - standard-normal samples.
pub fn random_gauss(i: usize) -> f64 {
    table(0, i)
}

/// `_RandomVector[i % MAX_RAND_SIZE]` - uniform [0,1) samples.
pub fn random_uniform(i: usize) -> f64 {
    table(MAX_RAND_SIZE, i)
}
