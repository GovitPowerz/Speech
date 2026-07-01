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
#include "MelFilterBank.h"
#include "fft.hpp"

// Constants ALL later tasks reuse (kept in sync with the harness manifest).
static const double OFFSET_SEC = 0.35;   // offset skip: round(rate*0.35) frames
static const double MAX_DUR_SEC = 2.0;   // duration cap: round(rate*2.0+1) samples
static const double PREEMPH = 0.97;
static const double NOISE_RATIO = 0.001;

// Periodogram framing parameters (kept in sync with the manifest + Rust golden).
static const long PERIO_P = 8;           // window_size = 1<<p = 256; half_window = 128
static const long PERIO_SHIFT = 80;      // window_shift
// Odd-frame-count variant: end chosen so frameNb = ceil((end-begin+1)/shift) is odd.
// begin=0, end=399 -> (399-0+1)/80 = 5 exactly -> frameNb=5 (odd). Asserted below.
static const long long PERIO_ODD_END = 399;

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

        // P=1 (N=2) folded in from Task 4 review: exercises the DanielsonLanczos<2>
        // hard-coded base block directly (no recursion). Same synthetic-row scheme.
        const unsigned powers[] = {1u, 2u, 3u, 8u};
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

    // --- Two-real periodogram + framing driver -------------------------------
    // legacy: AudioStruct.cpp:456-584 (getSequence / computeTwoRealPeriodogram /
    // computeSegmentPeriodogramEstimates). Run on the SAME preemph+noise audio as
    // above, per channel, p=8 shift=80 DC-offset TRUE, hamming-257 window, EMPTY
    // MelFilterBank (default ctor -> notEmpty()==false), begin=0 end=frames-1.
    // Two temporal-convolution variants: empty (no conv) and the normalized hann-7
    // kernel (ConvolutionVert gated on cols()>1). Plus one odd-frame-count variant.
    {
        const unsigned Min = 1;
        const unsigned Max = 20;
        Loki::Factory<AbstractFFT<double>, unsigned int> gfft_factory;
        FactoryInit<GFFTList<GFFT, Min, Max>::Result>::apply(gfft_factory);

        MelFilterBank emptyMel;  // default ctor: _IsBuilt == false -> notEmpty() false
        Eigen::MatrixXd win = getWindowingCoefficients("hamming", false, 257, 0.83333);
        Eigen::MatrixXd noConv;  // empty 0x0 -> cols()==0, ConvolutionVert skipped
        Eigen::MatrixXd hann7 = getWindowingCoefficients("hann", true, 7);  // 1x7 kernel

        const long long frames = audio.getFrameCount();
        const long long endFull = frames - 1;

        // No temporal convolution, both channels.
        audio.computeSegmentPeriodogramEstimates(PERIO_P, PERIO_SHIFT, 0, true, win,
                                                 emptyMel, gfft_factory, noConv, 0, endFull);
        Matrix2BinaryFile(out + "perio_p8_s80_chan1.bin", audio._Periodogram);
        ++dumps;
        audio.computeSegmentPeriodogramEstimates(PERIO_P, PERIO_SHIFT, 1, true, win,
                                                 emptyMel, gfft_factory, noConv, 0, endFull);
        Matrix2BinaryFile(out + "perio_p8_s80_chan2.bin", audio._Periodogram);
        ++dumps;

        // With the normalized hann-7 temporal convolution (kernel cols()==7 > 1).
        audio.computeSegmentPeriodogramEstimates(PERIO_P, PERIO_SHIFT, 0, true, win,
                                                 emptyMel, gfft_factory, hann7, 0, endFull);
        Matrix2BinaryFile(out + "perio_conv_p8_s80_chan1.bin", audio._Periodogram);
        ++dumps;
        audio.computeSegmentPeriodogramEstimates(PERIO_P, PERIO_SHIFT, 1, true, win,
                                                 emptyMel, gfft_factory, hann7, 0, endFull);
        Matrix2BinaryFile(out + "perio_conv_p8_s80_chan2.bin", audio._Periodogram);
        ++dumps;

        // Odd-frame-count variant (chan 0, no conv). Assert oddness in-harness.
        {
            long long begin = 0;
            long long end = PERIO_ODD_END;
            long long span = end - begin + 1;
            long long frameNb = (span / PERIO_SHIFT) * PERIO_SHIFT == span
                                    ? span / PERIO_SHIFT
                                    : span / PERIO_SHIFT + 1;
            if ((frameNb / 2) * 2 == frameNb) {
                std::cerr << "FATAL: perio_odd expected odd frameNb, got " << frameNb << "\n";
                return 1;
            }
            audio.computeSegmentPeriodogramEstimates(PERIO_P, PERIO_SHIFT, 0, true, win,
                                                     emptyMel, gfft_factory, noConv, begin, end);
            Matrix2BinaryFile(out + "perio_odd_chan1.bin", audio._Periodogram);
            ++dumps;
        }
    }

    // --- Mel filterbank + log-mel + regression deltas ------------------------
    // legacy: MelFilterBank.cpp:17-216 (ctor grid walks + triangle coeffs +
    // whole-bank fallback; applyFilterBank log/mel branches + the deltas-no-DCT
    // [delta|delta|dd] overwrite quirk at :192-193). Ctor args mirror a real SAD
    // spectral config: (minMel 64, maxMel 3800, nb_bins 26, minFreq 64, maxFreq
    // 3800, rate 8000, spectrum_size 128, ...). spectrum_size 128 -> 129 bins;
    // the caller snaps _BegFreq/_EndFreq via floor/ceil and the triangles yield
    // 29 mel filters (NOT 26). applyFilterBank writes into a caller-sized,
    // zero-initialized output (Eigen setZero + apply semantics).
    {
        // Recompute the full chan-1 periodogram (the odd variant above left a
        // 5-row buffer in audio._Periodogram); snapshot it into a local matrix.
        const unsigned Min = 1;
        const unsigned Max = 20;
        Loki::Factory<AbstractFFT<double>, unsigned int> gfft_factory;
        FactoryInit<GFFTList<GFFT, Min, Max>::Result>::apply(gfft_factory);
        MelFilterBank emptyMel;
        Eigen::MatrixXd win = getWindowingCoefficients("hamming", false, 257, 0.83333);
        Eigen::MatrixXd noConv;
        const long long endFull = audio.getFrameCount() - 1;
        audio.computeSegmentPeriodogramEstimates(PERIO_P, PERIO_SHIFT, 0, true, win,
                                                 emptyMel, gfft_factory, noConv, 0, endFull);
        Eigen::MatrixXd perio = audio._Periodogram;  // 201 x 129

        // log + mel, no DCT, no deltas -> 201 x 29.
        {
            MelFilterBank mel(64.0, 3800.0, 26, 64.0, 3800.0, 8000.0, 128, true, 0, false, 0, 0);
            Eigen::MatrixXd outm = Eigen::MatrixXd::Zero(perio.rows(), mel.getNbFilters());
            mel.applyFilterBank(perio, outm);
            Matrix2BinaryFile(out + "logmel_26_chan1.bin", outm);
            ++dumps;
        }

        // log + mel + deltas(3) + delta-deltas(3), no DCT -> 201 x 87. Pins the
        // [delta|delta|delta-delta] overwrite quirk (the static log-mel block is
        // clobbered by the delta block) from MelFilterBank.cpp:192-193.
        {
            MelFilterBank mel(64.0, 3800.0, 26, 64.0, 3800.0, 8000.0, 128, true, 0, false, 3, 3);
            Eigen::MatrixXd outm = Eigen::MatrixXd::Zero(perio.rows(), mel.getNbFilters());
            mel.applyFilterBank(perio, outm);
            Matrix2BinaryFile(out + "logmel_deltas_chan1.bin", outm);
            ++dumps;
        }

        // non-log + mel, no DCT, no deltas -> 201 x 29 (plain triangle dots).
        {
            MelFilterBank mel(64.0, 3800.0, 26, 64.0, 3800.0, 8000.0, 128, false, 0, false, 0, 0);
            Eigen::MatrixXd outm = Eigen::MatrixXd::Zero(perio.rows(), mel.getNbFilters());
            mel.applyFilterBank(perio, outm);
            Matrix2BinaryFile(out + "mel_26_chan1.bin", outm);
            ++dumps;
        }

        // The synthetic 20x50 matrix reinterpreted as a 20-frame periodogram with
        // rate 8000, spectrum_size 49 (-> freqStep 81.632...). log + mel, no DCT,
        // no deltas -> 20 x 29. Pins the grid walk on a second (rational) freqStep.
        {
            Eigen::MatrixXd synth(20, 50);
            for (int i = 0; i < 20; ++i) {
                for (int j = 0; j < 50; ++j) {
                    synth(i, j) = ((i * 7 + j * 13) % 100) / 100.0;
                }
            }
            MelFilterBank mel(64.0, 3800.0, 26, 64.0, 3800.0, 8000.0, 49, true, 0, false, 0, 0);
            Eigen::MatrixXd outm = Eigen::MatrixXd::Zero(synth.rows(), mel.getNbFilters());
            mel.applyFilterBank(synth, outm);
            Matrix2BinaryFile(out + "logmel_synth.bin", outm);
            ++dumps;
        }
    }

    std::cout << "OK: " << dumps << " dumps\n";
    return 0;
}
