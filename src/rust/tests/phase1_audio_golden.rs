mod common;
use common::{assert_bits_eq, fixture, load_bin};
use speech::audio::read_audio;

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
