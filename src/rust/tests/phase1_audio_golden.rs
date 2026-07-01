mod common;
use common::{assert_bits_eq, fixture, load_bin};
use ndarray::Array2;
use speech::audio::{convolution_horiz, convolution_vert, read_audio, windowing_coefficients};

#[test]
fn decode_normalize_matches_oracle() {
    let audio = read_audio(&fixture("excerpt_2ch_8k.wav"), 0.35, 2.0).unwrap();
    assert_eq!(audio.sample_rate, 8000);
    assert_bits_eq(&audio.data_raw, &load_bin("sig_norm.bin"), "sig_norm");
}

#[test]
fn preemph_noise_match_oracle() {
    let mut audio = read_audio(&fixture("excerpt_2ch_8k.wav"), 0.35, 2.0).unwrap();
    audio.apply_preemph(0.97);
    assert_bits_eq(&audio.data, &load_bin("sig_preemph.bin"), "sig_preemph");
    audio.apply_noise(0.001);
    assert_bits_eq(&audio.data, &load_bin("sig_noise.bin"), "sig_noise");
    audio.reset();
    assert_bits_eq(&audio.data, &load_bin("sig_norm.bin"), "reset");
}

fn coeffs_row(v: Vec<f64>) -> Array2<f64> {
    let n = v.len();
    Array2::from_shape_vec((1, n), v).unwrap()
}

#[test]
fn windowing_hamming_matches_oracle() {
    let w = windowing_coefficients("hamming", false, 257, 0.83333).unwrap();
    assert_bits_eq(
        &coeffs_row(w),
        &load_bin("win_hamming_257.bin"),
        "win_hamming_257",
    );
}

#[test]
fn windowing_hann_matches_oracle() {
    let w = windowing_coefficients("hann", false, 257, 0.83333).unwrap();
    assert_bits_eq(
        &coeffs_row(w),
        &load_bin("win_hann_257.bin"),
        "win_hann_257",
    );
}

#[test]
fn windowing_hhcw_matches_oracle() {
    let w = windowing_coefficients("hHCw", false, 257, 0.83333).unwrap();
    assert_bits_eq(
        &coeffs_row(w),
        &load_bin("win_hhcw_257.bin"),
        "win_hhcw_257",
    );
}

#[test]
fn windowing_hann_normalized_matches_oracle() {
    let w = windowing_coefficients("hann", true, 7, 0.83333).unwrap();
    assert_bits_eq(
        &coeffs_row(w),
        &load_bin("win_conv_norm_7.bin"),
        "win_conv_norm_7",
    );
}

#[test]
fn uniform_unnormalized_is_rectangular() {
    assert!(windowing_coefficients("uniform", false, 7, 0.83333).is_none()); // Helpers.hpp:260 quirk
    let u = windowing_coefficients("uniform", true, 4, 0.83333).unwrap();
    assert_eq!(u, vec![0.25; 4]);
}

#[test]
fn none_unknown_or_size1_empty() {
    assert!(windowing_coefficients("none", false, 257, 0.8).is_none());
    assert!(windowing_coefficients("blackman", false, 257, 0.8).is_none());
    assert!(windowing_coefficients("hamming", false, 1, 0.8).is_none());
}

#[test]
fn hamming_formula() {
    let w = windowing_coefficients("hamming", false, 5, 0.83333).unwrap();
    for (ii, v) in w.iter().enumerate() {
        assert_eq!(
            *v,
            0.54 - 0.46 * (2.0 * std::f64::consts::PI / 4.0 * ii as f64).cos()
        );
    }
}

#[test]
fn convolution_edges_truncate_without_renormalization() {
    // kernel [0.25,0.5,0.25]; row [1,1,1,1]: interior=1.0, edges=0.75 (missing coeff NOT redistributed)
    let mut r = ndarray::arr2(&[[1.0, 1.0, 1.0, 1.0]]);
    convolution_horiz(&mut r, &[0.25, 0.5, 0.25]);
    assert_eq!(r.row(0).to_vec(), vec![0.75, 1.0, 1.0, 0.75]);
}

#[test]
fn convolution_vert_edges_truncate_without_renormalization() {
    // 4x2 matrix, columns constant at 1.0 and 2.0; kernel [0.25,0.5,0.25] down each column.
    // Per column: interior rows = value, edge rows = 0.75*value (missing tap NOT redistributed).
    // col0 (1.0): [0.75, 1.0, 1.0, 0.75]; col1 (2.0): [1.5, 2.0, 2.0, 1.5]
    let mut m = ndarray::arr2(&[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]);
    convolution_vert(&mut m, &[0.25, 0.5, 0.25]);
    assert_eq!(
        m,
        ndarray::arr2(&[[0.75, 1.5], [1.0, 2.0], [1.0, 2.0], [0.75, 1.5]])
    );
}
