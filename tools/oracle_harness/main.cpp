// Oracle harness for Speech Phase 1 feature-parity fixtures.
//
// Compiles the vendored legacy feature sources (git-ignored legacy/src/) WITHOUT
// -ffast-math, with -ffp-contract=off and -DEIGEN_DONT_VECTORIZE, and dumps
// pipeline-stage matrices as little-endian .bin fixtures via the legacy's own
// Matrix2BinaryFile (Helpers.hpp: i64 rows, i64 cols, f64 column-major). A
// straightforward sequential Rust port must reproduce these dumps bit-for-bit.
//
// Usage: ./oracle_harness <output_dir>
// Later phase-1 tasks append stages to this single file.

#include <iostream>
#include <string>

#include "AudioStruct.h"
#include "CorpusItem.h"
#include "Helpers.hpp"
#include "fft.hpp"

// Constants ALL later tasks reuse (kept in sync with the harness manifest).
static const double OFFSET_SEC = 0.35;   // offset skip: round(rate*0.35) frames
static const double MAX_DUR_SEC = 2.0;   // duration cap: round(rate*2.0+1) samples
static const double PREEMPH = 0.97;
static const double NOISE_RATIO = 0.001;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <output_dir>\n";
        return 1;
    }
    std::string out = argv[1];
    if (!out.empty() && out.back() != '/') out += '/';
    std::string wav = out + "excerpt_2ch_8k.wav";

    int dumps = 0;

    // --- Audio stages ---------------------------------------------------------
    // legacy: AudioStruct.cpp:36-137 (file_type==0 ctor: libsndfile read + offset
    // skip round(rate*offset) + duration cap round(rate*dur+1) + per-channel
    // (2*RMS + max)/2 normalization).
    CorpusItem item(wav, "", "RUS", "RU", 0, 0, 1.0);
    AudioStruct audio(OFFSET_SEC, MAX_DUR_SEC, 0, item);
    Matrix2BinaryFile(out + "sig_norm.bin", audio._DataRaw);
    ++dumps;

    // legacy: AudioStruct.cpp:446-454 (applyPreemph: _Data[k] -= ratio*_Data[k-1],
    // reverse in-place per channel, no-op at k==0).
    audio.applyPreemph(PREEMPH);
    Matrix2BinaryFile(out + "sig_preemph.bin", audio._Data);
    ++dumps;

    // legacy: AudioStruct.cpp:437-444 (applyNoise: _Data += 2*ratio*_RandomVector
    // [k % _MaxRandSize] - ratio, using the Constants.h 100000-entry table).
    audio.applyNoise(NOISE_RATIO);
    Matrix2BinaryFile(out + "sig_noise.bin", audio._Data);
    ++dumps;

    // --- Windowing coefficients ----------------------------------------------
    // legacy: Helpers.hpp:222-272 (getWindowingCoefficients).
    Matrix2BinaryFile(out + "win_hamming_257.bin",
                      getWindowingCoefficients("hamming", false, 257, 0.83333));
    ++dumps;
    Matrix2BinaryFile(out + "win_hann_257.bin",
                      getWindowingCoefficients("hann", false, 257, 0.83333));
    ++dumps;
    Matrix2BinaryFile(out + "win_hhcw_257.bin",
                      getWindowingCoefficients("hHCw", false, 257, 0.83333));
    ++dumps;
    // normalized hann of length 7 (coeffs divided by their sum).
    Matrix2BinaryFile(out + "win_conv_norm_7.bin",
                      getWindowingCoefficients("hann", true, 7));
    ++dumps;

    // Pin the quirk at the source: "uniform" with normalized==false falls through
    // to the else branch and returns an EMPTY matrix (legacy Helpers.hpp:260-265).
    {
        Eigen::MatrixXd u = getWindowingCoefficients("uniform", false, 257, 0.83333);
        if (u.size() != 0) {
            std::cerr << "FATAL: uniform(normalized=false) expected empty, got "
                      << u.rows() << "x" << u.cols() << "\n";
            return 1;
        }
    }

    // --- Synthetic matrix -----------------------------------------------------
    // Closed-form matrix later tasks reuse verbatim; dumping it pins codec
    // agreement between the harness writer and the Rust/Python readers.
    Eigen::MatrixXd synth(20, 50);
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 50; ++j) {
            synth(i, j) = ((i * 7 + j * 13) % 100) / 100.0;
        }
    }
    Matrix2BinaryFile(out + "synth_20x50.bin", synth);
    ++dumps;

    // --- GFFT (two-reals-in-one-complex, unnormalized, negative exponent) -----
    // legacy: fft.hpp (V. Myrnyy DDJ 2007): radix-2 DIT with Taylor-seed twiddles
    // + Numerical-Recipes running recurrence. Factory built exactly as
    // BLSTMSpectralSegmenter.cpp:598-603 (Min=1, Max=20). The spectrum-order clamp
    // at BLSTMSpectralSegmenter.cpp:199-203 is Max-1 == 19 (VERIFIED; recorded in
    // the manifest as spectrum_order_clamp). Input rows: data[2i]=S(0,i),
    // data[2i+1]=S(1,i) from synth (i mod 50 for N>50); dumped 1 x 2N interleaved.
    {
        const unsigned Min = 1;
        const unsigned Max = 20;
        Loki::Factory<AbstractFFT<double>, unsigned int> gfft_factory;
        FactoryInit<GFFTList<GFFT, Min, Max>::Result>::apply(gfft_factory);

        const unsigned powers[] = {2u, 3u, 8u};
        for (unsigned p : powers) {
            const int n = 1 << p;
            Eigen::MatrixXd in(1, 2 * n);
            for (int i = 0; i < n; ++i) {
                in(0, 2 * i) = synth(0, i % 50);
                in(0, 2 * i + 1) = synth(1, i % 50);
            }
            Matrix2BinaryFile(out + "fft_in_" + std::to_string(n) + ".bin", in);
            ++dumps;

            double* data = new double[2 * n];
            for (int k = 0; k < 2 * n; ++k) data[k] = in(0, k);
            AbstractFFT<double>* gfft = gfft_factory.CreateObject(p);
            gfft->fft(data);
            Eigen::MatrixXd outm(1, 2 * n);
            for (int k = 0; k < 2 * n; ++k) outm(0, k) = data[k];
            Matrix2BinaryFile(out + "fft_out_" + std::to_string(n) + ".bin", outm);
            ++dumps;
            delete[] data;
            delete gfft;
        }
    }

    std::cout << "OK: " << dumps << " dumps\n";
    return 0;
}
