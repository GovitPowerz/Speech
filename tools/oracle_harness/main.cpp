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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "AudioStruct.h"
#include "CorpusItem.h"
#include "Helpers.hpp"
#include "MelFilterBank.h"
#include "fft.hpp"
#include "fmath.hpp"

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

// Explicit ascending triple-loop replacement for melPeriodogram*_CoeffsDCT.
// legacy semantics: MelFilterBank.cpp:223 under EIGEN_DONT_VECTORIZE, order verified
// (the Eigen GEMM diverges from ascending accumulation; the pre-check aborted, so
// this deterministic order is the parity target the Rust port reproduces).
static Eigen::MatrixXd dctProduct(const Eigen::MatrixXd& mel, const Eigen::MatrixXd& coeffs) {
    const int T = static_cast<int>(mel.rows());
    const int N = static_cast<int>(coeffs.rows());
    const int K = static_cast<int>(coeffs.cols());
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(T, K);
    for (int t = 0; t < T; ++t) {
        for (int kk = 0; kk < K; ++kk) {
            double acc = 0.0;
            for (int n = 0; n < N; ++n) {
                acc += mel(t, n) * coeffs(n, kk);
            }
            out(t, kk) = acc;
        }
    }
    return out;
}

// Shared regression-deltas kernel over a T x nbDCT block (MelFilterBank.cpp:227-244
// / 269-286 / 316-333). Faithful block-op port; used by applyDCTLoop for both the
// SDC delta and the delta/delta-delta bands.
static Eigen::MatrixXd regressionDeltas(const Eigen::MatrixXd& base, int n) {
    const int T = static_cast<int>(base.rows());
    const int F = static_cast<int>(base.cols());
    Eigen::MatrixXd acc = Eigen::MatrixXd::Zero(T, F);
    double adim = 0.0;
    for (int jj = 1; jj <= n; ++jj) {
        Eigen::MatrixXd deltas = Eigen::MatrixXd::Zero(T, F);
        int length = jj;
        if (length > T) {
            length = T;
        } else {
            for (int r = 0; r < T - length; ++r)
                for (int c = 0; c < F; ++c) deltas(r, c) = base(r + length, c);
            for (int r = length; r < T; ++r)
                for (int c = 0; c < F; ++c) deltas(r, c) -= base(r - length, c);
        }
        for (int kk = 0; kk < length; ++kk) {
            for (int c = 0; c < F; ++c) {
                deltas(kk, c) -= base(0, c);
                deltas(T - 1 - kk, c) += base(T - 1, c);
            }
        }
        adim += static_cast<double>(jj * jj);
        acc.noalias() += static_cast<double>(jj) * deltas;
    }
    adim *= 2.0;
    acc /= adim;
    return acc;
}

// Faithful branch-for-branch port of MelFilterBank::applyDCT (MelFilterBank.cpp:
// 218-360) with the one Eigen product replaced by dctProduct (ascending triple
// loop). Returns the T x getNbDCT() MFCC matrix (caller-sized + zero-initialized
// in the legacy; here allocated to the same width and zeroed).
static Eigen::MatrixXd applyDCTLoop(const Eigen::MatrixXd& mel, const Eigen::MatrixXd& coeffs,
                                    int nbDCT, bool ignoreFirst, int deltasNb, int ddNb) {
    const int T = static_cast<int>(mel.rows());
    // Output width = getNbDCT() (MelFilterBank.h:45-89).
    int width;
    if (deltasNb > 0) {
        int mult = (ddNb > 0) ? 3 : 2;
        width = mult * nbDCT - (ignoreFirst ? 1 : 0);
    } else if (deltasNb < 0) {
        width = (ignoreFirst ? nbDCT - 1 : nbDCT) + 7 * nbDCT;
    } else {
        width = ignoreFirst ? nbDCT - 1 : nbDCT;
    }
    Eigen::MatrixXd MFCC = Eigen::MatrixXd::Zero(T, width);
    Eigen::MatrixXd product = dctProduct(mel, coeffs);  // T x nbDCT

    if (deltasNb < 0) {
        // Branch A - SDC. d=3, P=3, k=7; delta kernel n=3 (adim 28) REGARDLESS.
        const int d = 3, P = 3, k = 7;
        MFCC.leftCols(nbDCT) = product;  // statics written first
        Eigen::MatrixXd MFCCtmp = regressionDeltas(MFCC.leftCols(nbDCT), d);
        Eigen::MatrixXd SDC = Eigen::MatrixXd::Zero(T + k * P, k * nbDCT);
        for (int kk = 0; kk < k; ++kk) {
            SDC.block(kk * P, kk * nbDCT, MFCCtmp.rows(), MFCCtmp.cols()) = MFCCtmp;
        }
        // extract at row (k*P-1)/2 = 10; rightCols(SDC.cols()) overwrites col
        // nbDCT-1 (the LAST static) when ignoreFirst (width = 12 + 7*13).
        MFCC.rightCols(SDC.cols()) = SDC.block((k * P - 1) / 2, 0, MFCCtmp.rows(), SDC.cols());
    } else if (ignoreFirst) {
        // Branch B - ignoreFirst && deltasNb >= 0. Compute into T x (width+1) temp,
        // then drop the FIRST column.
        Eigen::MatrixXd MFCCtmp = Eigen::MatrixXd::Zero(T, width + 1);
        if (deltasNb > 0) {
            MFCCtmp.leftCols(nbDCT) = product;
            MFCCtmp.leftCols(2 * nbDCT).rightCols(nbDCT) =
                regressionDeltas(MFCCtmp.leftCols(nbDCT), deltasNb);
            if (ddNb > 0) {
                MFCCtmp.rightCols(nbDCT) =
                    regressionDeltas(MFCCtmp.rightCols(2 * nbDCT).leftCols(nbDCT), ddNb);
            }
        } else {
            MFCCtmp = product;  // plain assignment resizes to T x nbDCT
        }
        MFCC = MFCCtmp.rightCols(width);
    } else {
        // Branch C - !ignoreFirst, deltasNb >= 0. Statics kept, [c|dc|(ddc)].
        if (deltasNb > 0) {
            MFCC.leftCols(nbDCT) = product;
            MFCC.leftCols(2 * nbDCT).rightCols(nbDCT) =
                regressionDeltas(MFCC.leftCols(nbDCT), deltasNb);
            if (ddNb > 0) {
                MFCC.rightCols(nbDCT) =
                    regressionDeltas(MFCC.rightCols(2 * nbDCT).leftCols(nbDCT), ddNb);
            }
        } else {
            MFCC = product;
        }
    }
    return MFCC;
}

// legacy: LongTermSpectralVariation.cpp:82-128 (classifySequence), transcribed
// verbatim as a free function (member -> parameter renames only: `this->` state
// dropped, all six args passed explicitly). `periodogram_cols` is the TIME axis
// (rows of our Array2 convention: `column`/`ii` index it), `periodogram_rows` is
// the FREQ axis (`row` indexes it) -- i.e. `periodogram(ii, row)` is (time, freq).
static double classifySequence(std::vector<double>::size_type column,
                               std::vector<double>::size_type freq_beg,
                               std::vector<double>::size_type freq_end,
                               std::vector<double>::size_type window_size,
                               std::vector<double>::size_type periodogram_rows,
                               std::vector<double>::size_type periodogram_cols,
                               const Eigen::MatrixXd& periodogram) {
    std::vector<double>::size_type beg_conv;
    if (column < window_size) {
        beg_conv = 0;
    } else {
        beg_conv = column-window_size;
    }
    std::vector<double>::size_type end_conv = column+window_size;
    if (end_conv >= periodogram_cols) end_conv = periodogram_cols-1;
    double length = (double) (end_conv-beg_conv+1);
    double nbRows = (double) (freq_end-freq_beg+1);

    std::vector<double> mean_value(periodogram_rows,0.0);
    for(std::vector<double>::size_type row = freq_beg ; row <= freq_end; ++row) {
        mean_value.at(row) = 0.0;
        for(std::vector<double>::size_type ii = beg_conv ; ii <= end_conv ; ++ii) {
            mean_value[row] += periodogram(ii,row);
        }
        mean_value[row] /= length;
        if (mean_value[row] < 1e-12) mean_value[row] = 1e-12;
    }

    std::vector<double> dzeta(periodogram_rows,0.0);
    double mean_dzeta = 0.0;
    double tmp_log;
    for(std::vector<double>::size_type row = freq_beg ; row <= freq_end; ++row) {
        dzeta[row] = 0.0;
        for(std::vector<double>::size_type ii = beg_conv ; ii <= end_conv ; ++ii) {
            tmp_log = periodogram(ii,row)/mean_value[row];
            dzeta[row] -= tmp_log*(tmp_log-1);
        }
        dzeta[row] /= length;
        mean_dzeta += dzeta[row];
    }
    mean_dzeta /= nbRows;

    // Compute standard deviation
    double tmp;
    double std_dzeta = 0.0;
    for(std::vector<double>::size_type row = freq_beg ; row <= freq_end; ++row) {
        tmp = dzeta[row]-mean_dzeta;
        std_dzeta += tmp*tmp;
    }
    std_dzeta /= nbRows;

    return std_dzeta;
}

// legacy: BLSTMSpectralSegmenter.cpp:316-339 (getLTSV), transcribed verbatim
// (member -> parameter renames: `this->LongTermSpectralVariation::classifySequence`
// -> the free function above; `audio._Periodogram` -> the `periodogram` param;
// Timer/logStream removed as harness noise). `LTSV_window_size` is the `window_size`
// (a.k.a. half_window/R) and `LTSV_window_shift` is the `shift`.
static Eigen::MatrixXd getLTSV(const Eigen::MatrixXd& periodogram,
                              std::vector<double>::size_type LTSV_window_size,
                              long LTSV_window_shift,
                              std::vector<double>::size_type freq_beg,
                              std::vector<double>::size_type freq_end) {
    std::vector<double>::size_type vec_size = periodogram.rows();
    Eigen::MatrixXd LTSV = Eigen::MatrixXd::Zero(vec_size,1);
    std::vector<double>::size_type periodogramSize = periodogram.cols();
    for (std::vector<double>::size_type jj = 0 ; jj < vec_size ; jj += LTSV_window_shift) {
        // Compute spectral variation
        LTSV(jj,0) = classifySequence(jj, freq_beg, freq_end, LTSV_window_size, periodogramSize, vec_size, periodogram);
        if (jj != 0) {
            for (std::vector<double>::size_type mm = 1 ; mm < (std::vector<double>::size_type) LTSV_window_shift ; ++mm) {
                double coeffInterp = ((double) mm)/LTSV_window_shift;
                LTSV(jj-LTSV_window_shift+mm,0) = (1.0-coeffInterp)*LTSV(jj-LTSV_window_shift,0)+coeffInterp*LTSV(jj,0);
            }
        }
    }
    return LTSV;
}

// legacy: TimeDomainCorrel.cpp:36-91 (classifySequence), transcribed VERBATIM
// (member -> free function: `this->` state dropped, `_Balance` passed as the
// `balance` param). `windowed_signal` is a 1 x N Eigen row vector; the body is
// byte-for-byte the legacy except the class member `_Balance` -> `balance` and
// `fmath::log` kept as-is (the scalar bit-trick log we double-pin against Rust).
static double tdcClassifySequence(long min_lag, long max_lag, double balance,
                                  const Eigen::MatrixXd& windowed_signal) {
    // Max correlation peak
    Eigen::DenseIndex length_signal = windowed_signal.cols();
    Eigen::VectorXd R = Eigen::VectorXd::Zero(max_lag-min_lag+1);
    double MaxPeak = -1e20;
    double adim = windowed_signal.squaredNorm();
    if (adim < 1e-12) adim = 1e-12;
    Eigen::DenseIndex indice;
    for (Eigen::DenseIndex kk = min_lag; kk <= max_lag; ++kk) {
        indice = kk-min_lag;
        Eigen::DenseIndex length = length_signal-kk;
        R[indice] = (windowed_signal.block(0,0,1,length).array()*windowed_signal.block(0,kk,1,length).array()).sum()/adim;
        if (R[indice] > MaxPeak) {
            MaxPeak = R[indice];
        }
    }

    // Cross-correlation
    double CrossCorr = 0.0;
    Eigen::DenseIndex period1_start = 0;
    Eigen::DenseIndex period2_start = 0;
    Eigen::DenseIndex period3_start = 0;
    double count = 0;
    for (Eigen::DenseIndex ll = 0 ; ll < max_lag-min_lag ; ++ll) {
        if (((R[0] >= 0) && (R[ll] > 0) && (R[ll+1] <= 0)) || ((R[0] < 0) && (R[ll] < 0) && (R[ll+1] >= 0))) {
            if (period1_start == 0) {
                period1_start = ll+1;
            } else {
                if (period2_start == 0) {
                    period2_start = ll+1;
                } else {
                    period3_start = ll+1;
                    Eigen::DenseIndex min_size = period2_start-period1_start+1;
                    Eigen::DenseIndex size2 = period3_start-period2_start+1;
                    if (min_size > size2) min_size = size2;
                    double tmp_xcorr = -1e12;
                    for (Eigen::DenseIndex mm = 0 ; mm < min_size ; ++mm) {
                        double xcor = 0;
                        for (Eigen::DenseIndex nn = 0 ; nn < min_size-mm ; ++nn) {
                            xcor += R[period1_start+nn]*R[period2_start+nn+mm]+R[period1_start+mm+nn]*R[period2_start+nn];
                        }
                        if (tmp_xcorr < xcor) tmp_xcorr = xcor;
                    }
                    CrossCorr = CrossCorr+tmp_xcorr/2;
                    period1_start = period2_start;
                    period2_start = period3_start;
                    ++count;
                }
            }
        }
    }
    if (count != 0) {
        CrossCorr = (CrossCorr/count);
    }
    return balance*(-fmath::log(1-MaxPeak))+(1-balance)*CrossCorr;
}

// legacy: BLSTMSpectralSegmenter.cpp:172-192 (computePitch), transcribed VERBATIM
// (member -> free function). Returns frameRate/indiceMaxPeak (a raw estimate; the
// accept/reject bounds live in getPitch below).
static double computePitch(long min_lag, long max_lag, long frameRate,
                           const Eigen::Ref<const Eigen::MatrixXd> windowed_signal) {
    // Max correlation peak
    Eigen::DenseIndex length_signal = windowed_signal.cols();
    Eigen::VectorXd R = Eigen::VectorXd::Zero(max_lag-min_lag+1);
    double MaxPeak = -1e20;
    double adim = windowed_signal.squaredNorm();
    if (adim < 1e-12) adim = 1e-12;
    Eigen::DenseIndex indice;
    long indiceMaxPeak = min_lag;
    for (Eigen::DenseIndex kk = min_lag; kk <= max_lag; ++kk) {
        indice = kk-min_lag;
        Eigen::DenseIndex length = length_signal-kk;
        R[indice] = (windowed_signal.block(0,0,1,length).array()*windowed_signal.block(0,kk,1,length).array()).sum()/adim;
        if (R[indice] > MaxPeak) {
            MaxPeak = R[indice];
            indiceMaxPeak = kk;
        }
    }

    return frameRate/indiceMaxPeak;
}

// legacy: BLSTMSpectralSegmenter.cpp:762-775 (the homothety warp loop inside
// getSegmentation), transcribed VERBATIM as a free function over `periodogramMem`
// (the snapshot the legacy takes) writing a fresh output matrix (the legacy writes
// back into audio._Periodogram). `coeff_homo` is the legacy pitch/300. `pos`/`pos+1`
// index COLUMNS; the bounds checks are against periodogramMem.cols() exactly as
// written -- Eigen's cols() returns a SIGNED Eigen::Index, so `pos < cols()` is a
// signed comparison (the `pos >= 0` / `pos+1 >= 0` guards are load-bearing).
static Eigen::MatrixXd applyHomothety(const Eigen::MatrixXd& periodogramMem, double coeff_homo) {
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(periodogramMem.rows(), periodogramMem.cols());
    for (std::vector<double>::size_type ii = 0 ; ii < (std::vector<double>::size_type) periodogramMem.cols() ; ++ii) {
        for (std::vector<double>::size_type jj = 0 ; jj < (std::vector<double>::size_type) periodogramMem.rows() ; ++jj) {
            int pos = (int) (coeff_homo*ii);
            double alpha = coeff_homo*ii-pos;
            double tmp = 0.0;
            if ((pos >= 0)&&(pos < periodogramMem.cols())) {
                tmp += (1-alpha)*periodogramMem(jj,pos);
            }
            if ((pos+1 >= 0)&&(pos+1 < periodogramMem.cols())) {
                tmp += alpha*periodogramMem(jj,pos+1);
            }
            out(jj,ii) = tmp;
        }
    }
    return out;
}

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

        // --- DCT / MFCC / deltas / SDC (Task 7) ------------------------------
        // legacy: MelFilterBank.cpp:218-360 (applyDCT, three branches) + the DCT
        // table at :108-127. All five dumps run applyFilterBank on the SAME chan-1
        // periodogram with a DCT-active bank (nb_dct=13 < 29 filters, no clamp) ->
        // plain log-mel T x 29 (the DCT params do NOT touch applyFilterBank), then
        // the DCT into T x getNbDCT().
        //
        // GEMM PRE-CHECK OUTCOME (mandatory step, MelFilterBank.cpp:223): the
        // legacy applyDCT computes melPeriodogram*_CoeffsDCT as an Eigen GEMM
        // (201x29 * 29x13). Comparing that Eigen GEMM against the explicit ascending
        // triple loop during development ABORTED - they diverge (Eigen's blocked
        // gebp kernel does not accumulate in plain ascending order). Per the brief,
        // the Eigen product is therefore REPLACED by the explicit ascending triple
        // loop (dctProduct), which is the portable/deterministic parity target the
        // Rust apply_dct reproduces; that substitution is LOCKED regardless of what
        // the check below measures. The dumps below are produced by applyDCTLoop (a
        // faithful branch-for-branch port of MelFilterBank.cpp:218-360 with that one
        // product swapped), NOT by the legacy mel.applyDCT, because the legacy would
        // bake in the non-portable Eigen order.
        //
        // The check below is NOT decorative: it recomputes the FULL 201x13 product
        // both ways on the real chan-1 log-mel input on every fixture regeneration,
        // compares every element bit-for-bit (not just (0,0), not by value), and
        // prints a GEMM_CHECK line that extract_phase1_fixtures.py parses verbatim
        // into manifest.json's dct_gemm_substitution object -- so the manifest never
        // carries a stale/hardcoded divergence claim. It does not gate anything
        // (no abort()): the ascending-loop fallback below is applied unconditionally
        // either way, per the locked policy above.
        {
            MelFilterBank melLog(64.0, 3800.0, 26, 64.0, 3800.0, 8000.0, 128, true, 13, false, 0, 0);
            Eigen::MatrixXd logmel = Eigen::MatrixXd::Zero(perio.rows(), melLog.getNbFilters());
            melLog.applyFilterBank(perio, logmel);

            // Reconstruct _CoeffsDCT with the exact ctor formula (cannot reach the
            // private member without editing legacy/): 29 x 13, PI = _PI literal.
            const int NB_FILTERS = 29;
            const int NB_DCT = 13;
            const double PI = 3.14159265358979323846264338327;  // legacy _PI (Constants.h:15)
            Eigen::MatrixXd coeffs = Eigen::MatrixXd::Zero(NB_FILTERS, NB_DCT);
            for (int col = 0; col < NB_FILTERS; ++col) {
                for (int row = 0; row < NB_DCT; ++row) {
                    coeffs(col, row) = std::cos(PI / NB_FILTERS * (col + 0.5) * row);
                }
            }

            // Self-enforcing full-product bit-for-bit divergence measurement: compute
            // the SAME 201x13 product both ways and compare every element by bit
            // pattern (memcpy'd to uint64_t; NOT operator==, which would compare by
            // value and miss the 1-ULP-only differences this check exists to catch).
            {
                Eigen::MatrixXd eig = logmel * coeffs;
                Eigen::MatrixXd loop = dctProduct(logmel, coeffs);
                const int total = static_cast<int>(eig.rows() * eig.cols());
                int mismatches = 0;
                int firstRow = -1, firstCol = -1;
                uint64_t firstEigenBits = 0, firstLoopBits = 0;
                for (int r = 0; r < eig.rows(); ++r) {
                    for (int c = 0; c < eig.cols(); ++c) {
                        uint64_t eBits, lBits;
                        double eVal = eig(r, c);
                        double lVal = loop(r, c);
                        std::memcpy(&eBits, &eVal, sizeof(double));
                        std::memcpy(&lBits, &lVal, sizeof(double));
                        if (eBits != lBits) {
                            if (mismatches == 0) {
                                firstRow = r;
                                firstCol = c;
                                firstEigenBits = eBits;
                                firstLoopBits = lBits;
                            }
                            ++mismatches;
                        }
                    }
                }
                const bool diverged = mismatches > 0;
                std::cout << "GEMM_CHECK diverged=" << (diverged ? 1 : 0)
                          << " mismatches=" << mismatches << " total=" << total
                          << " first=(" << firstRow << "," << firstCol << ")"
                          << " eigen=0x" << std::hex << firstEigenBits
                          << " loop=0x" << firstLoopBits << std::dec << "\n";
            }

            // (a) plain MFCC: deltas 0, dd 0, ignoreFirst false -> T x 13.
            Matrix2BinaryFile(out + "mfcc_chan1.bin",
                              applyDCTLoop(logmel, coeffs, NB_DCT, false, 0, 0));
            ++dumps;
            // (b) MFCC + deltas(3) + dd(3), ignoreFirst false -> T x 39 ([c|dc|ddc]).
            Matrix2BinaryFile(out + "mfcc_deltas_chan1.bin",
                              applyDCTLoop(logmel, coeffs, NB_DCT, false, 3, 3));
            ++dumps;
            // (c) MFCC + deltas(3) + dd(3), ignoreFirst TRUE -> T x 38 (drops c0).
            Matrix2BinaryFile(out + "mfcc_deltas_if_chan1.bin",
                              applyDCTLoop(logmel, coeffs, NB_DCT, true, 3, 3));
            ++dumps;
            // (d) SDC: deltas -1, dd 0, ignoreFirst false -> T x (13 + 7*13) = 104.
            Matrix2BinaryFile(out + "mfcc_sdc_chan1.bin",
                              applyDCTLoop(logmel, coeffs, NB_DCT, false, -1, 0));
            ++dumps;
            // (e) SDC + ignoreFirst -> T x (12 + 7*13) = 103. Pins the c_{N-1}
            // clobber: statics written first, then SDC col 0 overwrites the LAST
            // static column via rightCols(7*nb_dct) starting at col nb_dct-1.
            Matrix2BinaryFile(out + "mfcc_sdc_if_chan1.bin",
                              applyDCTLoop(logmel, coeffs, NB_DCT, true, -1, 0));
            ++dumps;
        }
    }

    // --- LTSV (Task 8) --------------------------------------------------------
    // legacy: LongTermSpectralVariation.cpp:82-128 (classifySequence) +
    // BLSTMSpectralSegmenter.cpp:316-339 (getLTSV loop), transcribed above.
    {
        // Recompute the chan-1 full periodogram fresh (same recipe as the mel
        // block above; that block's `perio` is out of scope here).
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

        // LTSV column over perio_p8_s80_chan1: freq_beg=0, freq_end=128, R=15, shift=4.
        Eigen::MatrixXd ltsvChan1 = getLTSV(perio, 15, 4, 0, 128);
        Matrix2BinaryFile(out + "ltsv_chan1.bin", ltsvChan1);
        ++dumps;

        // Per-frame scores over synth_20x50: R=3, band 0..49, shift=1.
        Eigen::MatrixXd synth(20, 50);
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 50; ++j) {
                synth(i, j) = ((i * 7 + j * 13) % 100) / 100.0;
            }
        }
        Eigen::MatrixXd ltsvSynth = getLTSV(synth, 3, 1, 0, 49);
        Matrix2BinaryFile(out + "ltsv_synth.bin", ltsvSynth);
        ++dumps;
    }

    // --- fmath::log sweep (Task 9) -------------------------------------------
    // legacy: fmath.hpp:186-226 (LogVar table build) + :713-727 (scalar log eval),
    // the 2048-entry f32 table-log. Dump 2 x N: row0 = input (f32 widened to f64),
    // row1 = fmath::log(input) (f32 result widened to f64). Sweep list = the brief's
    // explicit values PLUS the four table-index anchors 1+i/2048 for i in {0,1,
    // 1023,2047} and nextafterf neighbors of the mantissa table boundaries (the
    // b1-mask edges at 1.0, 1.5=idx1024, and just under 2.0=idx2047) so the golden
    // pins the idx-quantization edges, not just the interior.
    {
        std::vector<float> xs;
        // Brief's explicit sweep.
        xs.push_back(1e-24f);
        xs.push_back(1e-12f);
        xs.push_back(0.5f);
        xs.push_back(1.0f - std::ldexp(1.0f, -24));  // 1.0 - 2^-24
        xs.push_back(1.0f);
        xs.push_back(1.5f);
        xs.push_back(2.0f);
        xs.push_back(88.0f);
        xs.push_back(1e10f);
        xs.push_back(0.0f);
        // Table-index anchors 1 + i/2048 for i in {0,1,1023,2047}.
        for (int i : {0, 1, 1023, 2047}) {
            xs.push_back((float)(1.0 + (double)i / 2048.0));
        }
        // nextafterf neighbors of the table boundaries (idx quantization edges).
        for (float boundary : {1.0f, 1.5f, 2.0f}) {
            xs.push_back(std::nextafterf(boundary, 0.0f));  // toward smaller
            xs.push_back(std::nextafterf(boundary, 3.0f));  // toward larger
        }
        Eigen::MatrixXd sweep(2, (Eigen::Index) xs.size());
        for (Eigen::Index k = 0; k < (Eigen::Index) xs.size(); ++k) {
            sweep(0, k) = (double) xs[(size_t) k];
            sweep(1, k) = (double) fmath::log(xs[(size_t) k]);
        }
        Matrix2BinaryFile(out + "fmath_log_sweep.bin", sweep);
        ++dumps;
    }

    // --- TDC scores + pitch + homothety (Task 9) -----------------------------
    // legacy: TimeDomainCorrel.cpp:36-91 (classifySequence) + :100-236 (getSegmentation
    // frame loop) for the TDC score column; BLSTMSpectralSegmenter.cpp:399-437 (getPitch)
    // for the pitch scalar; :762-775 (homothety warp) for the periodogram warp. TdcParams
    // derived inline from the brief config: TDC_window=0.032, TDC_shift=0.01,
    // lags=(0.002,0.016), balance=0.7 at rate 8000 (TimeDomainCorrel.cpp:100-109):
    //   half_window = round(0.032*8000/2) = 128, full_window = 2*128+1 = 257,
    //   window_shift = round(0.01*8000) = 80,
    //   min_lag = round(0.002*8000) = 16, max_lag = round(0.016*8000) = 128 (< 257),
    //   hamming-257 window with param 0.8.
    {
        // Re-derive the same audio the earlier stages produced: the excerpt is
        // decoded fresh, then preemph + noise applied (matching the audio-stage
        // dumps at the top of main). `audio._Data` above already carries preemph +
        // noise (applyPreemph + applyNoise were called on the single AudioStruct),
        // so we reuse it directly here -- no second decode.
        const double RATE = (double) audio.getFrameRate();
        const long half_window = (long) boost::math::round(0.032 * RATE / 2.0);   // 128
        const long full_window = 2 * half_window + 1;                             // 257
        long window_shift = (long) boost::math::round(0.01 * RATE);               // 80
        long min_lag = (long) boost::math::round(0.002 * RATE);                   // 16
        long max_lag = (long) boost::math::round(0.016 * RATE);                   // 128
        if (max_lag >= full_window) max_lag = full_window - 1;
        const double balance = 0.7;
        Eigen::MatrixXd tdcWin = getWindowingCoefficients("hamming", false, full_window, 0.8);

        // (b) TDC score column over the excerpt (chan 0), single-threaded transcription
        // of the getSegmentation frame loop (TimeDomainCorrel.cpp:178-204): vec_size is
        // the inclusive-count ceil-divide of frameCount by window_shift; per jj = 0,
        // shift, 2*shift, ... < frameCount, get the windowed sequence and classify. The
        // OMP pragma is dropped (deterministic sequential order is the parity target).
        {
            const long frameCount = (long) audio.getFrameCount();
            long vec_size;
            if ((frameCount / window_shift) * window_shift == frameCount) {
                vec_size = frameCount / window_shift;
            } else {
                vec_size = frameCount / window_shift + 1;
            }
            Eigen::MatrixXd result_vec = Eigen::MatrixXd::Zero(1, vec_size);
            Eigen::MatrixXd windowed_signal = Eigen::MatrixXd::Zero(1, full_window);
            for (long jj = 0; jj < frameCount; jj += window_shift) {
                audio.getSequence(jj, half_window, 0, false, tdcWin, windowed_signal);
                result_vec(0, jj / window_shift) = tdcClassifySequence(min_lag, max_lag, balance, windowed_signal);
            }
            Matrix2BinaryFile(out + "tdc_chan1.bin", result_vec);
            ++dumps;
        }

        // (c) Pitch scalar over a hand-built segmentation: a SINGLE SPEECH segment
        // covering the middle 1.0s of the 2.0s excerpt. frameCount = 16001 @ 8000 Hz
        // -> 2.000125s duration; the segment spans [0.5s, 1.5s). Boundary list:
        //   Other@0.0, Speech@0.5, Other@1.5, End@duration.
        // getPitch iterates SPEECH segments; per segment frames step window_shift over
        // [begin*rate + half_window, end*rate - half_window], accepting estimates
        // strictly inside (rate/max_lag, rate/min_lag) (BLSTMSpectralSegmenter.cpp:399-437).
        double pitch = 0.0;
        {
            const long frameCount = (long) audio.getFrameCount();
            const double duration = ((double) frameCount) / RATE;
            const double segBegin = 0.5;
            const double segEnd = 1.5;
            // getPitch frame-range arithmetic (transcribed verbatim), specialized to
            // the one SPEECH segment [segBegin, segEnd):
            Eigen::MatrixXd windowed_signal = Eigen::MatrixXd::Zero(1, full_window);
            int numberOfFrames = 0;
            std::vector<double>::size_type rowBegin =
                ((std::vector<double>::size_type)(segBegin * RATE)) + (std::vector<double>::size_type) half_window;
            std::vector<double>::size_type rowEnd =
                ((std::vector<double>::size_type)(segEnd * RATE));
            if (rowEnd > (std::vector<double>::size_type) half_window) {
                rowEnd -= (std::vector<double>::size_type) half_window;
            } else {
                rowEnd = 0;
            }
            if (rowEnd >= rowBegin) {
                for (std::vector<double>::size_type jj = rowBegin; jj <= rowEnd; jj += window_shift) {
                    audio.getSequence(jj, half_window, 0, false, tdcWin, windowed_signal);
                    double pitchEstimate = computePitch(min_lag, max_lag, (long) RATE, windowed_signal);
                    if ((pitchEstimate > RATE / ((double) max_lag)) && (pitchEstimate < RATE / ((double) min_lag))) {
                        pitch += pitchEstimate;
                        ++numberOfFrames;
                    }
                }
            }
            if (numberOfFrames == 0) numberOfFrames = 1;
            pitch = pitch / numberOfFrames;
            (void) duration;
            Eigen::MatrixXd pitchMat(1, 1);
            pitchMat(0, 0) = pitch;
            Matrix2BinaryFile(out + "pitch_chan1.bin", pitchMat);
            ++dumps;
        }

        // (d) Homothety warp of the chan-1 periodogram with coeff = pitch/300.
        // Recompute the chan-1 periodogram (perio_p8_s80_chan1 recipe: p=8, shift=80,
        // dc_offset=TRUE, hamming-257, empty mel, no conv) on the SAME preemph+noise
        // audio, then warp its columns (BLSTMSpectralSegmenter.cpp:762-775).
        {
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
            double coeff_homo = pitch / 300.0;
            Matrix2BinaryFile(out + "perio_homothety_chan1.bin", applyHomothety(perio, coeff_homo));
            ++dumps;
        }
    }

    std::cout << "OK: " << dumps << " dumps\n";
    return 0;
}
