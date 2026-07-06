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
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "ActivationFunctions.h"
#include "AudioStruct.h"
#include "BLSTMNeuralNetwork.h"
#include "BLSTMSignalSegmenter.h"
#include "BLSTMSpectralSegmenter.h"
#include "ConfigFile.h"
#include "CorpusItem.h"
#include "Helpers.hpp"
#include "InputStatistics.h"
#include "iof/io.hpp"
#include "LSTMLayer.h"
#include "LongTermSpectralVariation.h"
#include "MelFilterBank.h"
#include "NeuronLayer.h"
#include "Segmenter.h"
#include "TimeDomainCorrel.h"
#include "fft.hpp"
#include "fmath.hpp"

// Phase 2b Task 2: exposes the protected Segmenter surface (buildFromConf,
// results2segmentation, updateSegmentation) for the harness, and satisfies the
// pure-virtual getSegmentation with a trivial override (never invoked here).
struct SegProbe : Segmenter {
    void getSegmentation(AudioStruct&, Segmentation&) override {}
    using Segmenter::buildFromConf;
    using Segmenter::results2segmentation;
    using Segmenter::updateSegmentation;
    using Segmenter::_ConvolutionCoeff;
};

// Phase 2b Task 4: TdcSegmenter (Algo 1) oracle. `TimeDomainCorrel::getSegmentation`
// (TimeDomainCorrel.cpp:93-259) calls no NN (its only libm is fmath::log/cos via
// classifySequence + the hamming/hann window coefficients), so the REAL compiled
// method is used AS-IS -- no transcription. TdcProbe exposes the protected
// Segmenter fields the harness needs to independently replicate the pre-conv
// result_vec (`_MinMaxLag`/`_Balance` are private in TimeDomainCorrel itself and
// inaccessible even to a derived class; the harness instead re-reads `TDC_lags`
// from the same ConfigFile and reproduces the ctor's >= 0.0 clamp, TDC_balance
// unclamped, which is all `getSegmentation` does with them before calling the
// public `classifySequence`).
struct TdcProbe : TimeDomainCorrel {
    TdcProbe(ConfigFile &conf) : TimeDomainCorrel(conf, false, false) {}
    using Segmenter::_WindowShift;
    using Segmenter::_WindowSize;
    using Segmenter::_FlagDCOffset;
    using Segmenter::_WindowingType;
    using Segmenter::_WindowingParam;
    using Segmenter::results2segmentation;
};

// Phase 2b Task 5: LtsvSegmenter (Algo 2) oracle. `LongTermSpectralVariation::
// getSegmentation` (LongTermSpectralVariation.cpp:130-407) calls no NN (the mel/DCT
// branch's only libm is fmath::log/cos via the windowing coefficients + the DCT
// cosine table; classifySequence itself is cwise+sum), so under the primary
// (nb_DCT=0) config the REAL compiled getSegmentation is used AS-IS -- no
// transcription. LtsvProbe exposes the protected fields the harness needs to
// independently replicate the pre-convolution decimated result_vec (a
// getSegmentation-local, not otherwise observable) via the REAL public
// classifySequence, matching the TdcProbe pattern.
struct LtsvProbe : LongTermSpectralVariation {
    LtsvProbe(ConfigFile &conf) : LongTermSpectralVariation(conf, false, false) {}
    using Segmenter::_WindowShift;
    using Segmenter::_WindowSize;
    using Segmenter::_FlagDCOffset;
    using Segmenter::_WindowingType;
    using Segmenter::_WindowingParam;
    using Segmenter::_PreemphRatio;
    using Segmenter::_NoiseSeed;
    using Segmenter::_NoiseRatio;
    using Segmenter::results2segmentation;
    using LongTermSpectralVariation::_SpectrumOrder;
    using LongTermSpectralVariation::_SpectrumShift;
    using LongTermSpectralVariation::_TemporalConvolutionSize;
    using LongTermSpectralVariation::_TemporalConvolutionType;
    using LongTermSpectralVariation::_MinMelFreq;
    using LongTermSpectralVariation::_MaxMelFreq;
    using LongTermSpectralVariation::_NbBins;
    using LongTermSpectralVariation::_IsLog;
    using LongTermSpectralVariation::_NbDCT;
    using LongTermSpectralVariation::_IgnoreFirstDCT;
    using LongTermSpectralVariation::_ComputeDeltasNb;
    using LongTermSpectralVariation::_ComputeDeltaDeltasNb;
    using LongTermSpectralVariation::_MinFreq;
    using LongTermSpectralVariation::_MaxFreq;
    using LongTermSpectralVariation::classifySequence;
};

// Phase 2b Task 6: BlstmSignalSegmenter (Algo 4) oracle. Unlike TDC/LTSV, the signal
// driver runs the REAL BLSTMNeuralNetwork.feedForwardBackward (BLSTMSignalSegmenter.
// cpp:260), whose Eigen GEMMs diverge from the ascending-loop port. SignalProbe
// exposes the protected Segmenter/segmenter fields + the NN member so the harness can
// (a) transcribe getSegmentation with ONLY that FFB call swapped for the reimpl
// family above (signalReimplFFB), and (b) run the REAL getSegmentation beside it as a
// SECONDARY structural probe (segment count + types must match EXACTLY or abort).
struct SignalProbe : BLSTMSignalSegmenter {
    SignalProbe(ConfigFile &conf, const Eigen::Ref<const Eigen::VectorXd> weights)
        : BLSTMSignalSegmenter(conf, weights, false, false) {}
    using Segmenter::_WindowShift;
    using Segmenter::_WindowSize;
    using Segmenter::_FlagDCOffset;
    using Segmenter::_WindowingType;
    using Segmenter::_WindowingParam;
    using Segmenter::_PreemphRatio;
    using Segmenter::_NoiseSeed;
    using Segmenter::_NoiseRatio;
    using Segmenter::_ConvolutionCoeff;
    using Segmenter::results2segmentation;
    using Segmenter::getTargets;
    using BLSTMSignalSegmenter::_BLSTMNeuralNetwork;
    using BLSTMSignalSegmenter::_BLSTMTwoSweeps;
};

// Phase 2b Task 7: BlstmSpectralSegmenter (Algo 3) oracle -- the REAL 1_worker_1.
// config's algorithm. Like the signal driver, the spectral result_vec is produced by
// _BLSTMNeuralNetwork.feedForwardBackward (BLSTMSpectralSegmenter.cpp:740), whose real
// Eigen GEMMs diverge from the ascending-loop port at the real net's k>=23 shapes. So
// SpectralProbe exposes the protected machinery so the harness can (a) transcribe the
// getSegmentation body (:593-887, MINUS the pitch second pass :757-805 which the
// Task-7 configs short-circuit via TDCwindow 0) with ONLY that FFB call swapped for
// the reimpl family (signalReimplFFB, shared with Task 6 -- the plain/truncate/overlap
// windowed drivers are input-column-agnostic and the leftCols crop stays FALSE at
// D=11<23), and (b) run the REAL getSegmentation beside it as a SECONDARY structural
// probe (segment count + types must match EXACTLY or abort). The spectral param
// derivation (initSpectralAnalysis, getWindowingCoeff, getTemporalConvolution,
// getLTSVParam, getTDCParam, getBLSTMParam, getLTSV, getBLSTMInputSequence) uses the
// REAL compiled protected helpers -- exposed here so the main-side transcription can
// call them (they mutate _SpectrumShiftInFrames/_SpectrumShift/_WindowShift as real
// side effects, exactly as the driver's getBLSTMParam does).
struct SpectralProbe : BLSTMSpectralSegmenter {
    SpectralProbe(ConfigFile &conf) : BLSTMSpectralSegmenter(conf, false, false) {}
    using Segmenter::_WindowShift;
    using Segmenter::_WindowSize;
    using Segmenter::_FlagDCOffset;
    using Segmenter::_ConvolutionCoeff;
    using Segmenter::results2segmentation;
    using Segmenter::getTargets;
    using LongTermSpectralVariation::_SpectrumOrder;
    using LongTermSpectralVariation::_SpectrumShift;
    using BLSTMSpectralSegmenter::_SpectrumShiftInFrames;
    using BLSTMSpectralSegmenter::_BLSTMNeuralNetwork;
    using BLSTMSpectralSegmenter::initSpectralAnalysis;
    using BLSTMSpectralSegmenter::getWindowingCoeff;
    using BLSTMSpectralSegmenter::getTemporalConvolution;
    using BLSTMSpectralSegmenter::getLTSVParam;
    using BLSTMSpectralSegmenter::getTDCParam;
    using BLSTMSpectralSegmenter::getBLSTMParam;
    using BLSTMSpectralSegmenter::getLTSV;
    using BLSTMSpectralSegmenter::getBLSTMInputSequence;
    using BLSTMSpectralSegmenter::getPitch;
};

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

// Phase 2 Task 1: general ascending-loop matrix product A(m x k) * B(k x n),
// accumulating k in strictly ascending order (i / j outer, k inner). This is the
// portable/deterministic accumulation order the Rust NN port will reproduce for
// every NN product site; the NN_PROBE stage below compares it bit-for-bit against
// Eigen's blocked product on the real NN shapes (extends the DCT GEMM_CHECK from
// Phase 1 to the LSTM/dense/softmax sites). Measurement only -- nothing aborts on
// divergence; the reimpl uses this order unconditionally.
static Eigen::MatrixXd matSeq(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += A(i, k) * B(k, j);
            }
            out(i, j) = acc;
        }
    }
    return out;
}

// Phase 2 Task 1: bit-for-bit divergence probe for one product site. Computes the
// Eigen product AND matSeq on the SAME operands, compares every element by bit
// pattern (memcpy to uint64_t, NOT operator== -- 1-ULP-only differences must be
// caught), and prints a parseable NN_PROBE line the phase2 extractor parses into
// manifest.json:nn_product_probes. Mirrors the Phase 1 GEMM_CHECK format exactly.
static void nnProbe(const std::string& site, const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    Eigen::MatrixXd eig = A * B;
    Eigen::MatrixXd loop = matSeq(A, B);
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
    std::cout << "NN_PROBE site=" << site
              << " diverged=" << (diverged ? 1 : 0)
              << " mismatches=" << mismatches << " total=" << total
              << " first=(" << firstRow << "," << firstCol << ")"
              << " eigen=0x" << std::hex << firstEigenBits
              << " loop=0x" << firstLoopBits << std::dec << "\n";
}

// Phase 2 Task 4: unpack a flat weight vector (column-major, block order
// InputWeights -> FeedbackWeights -> PeepWeight -> Biaises) into the four LSTM
// matrices, EXACTLY as LSTMLayer::setWeights (LSTMLayer.cpp:162-203) does, so the
// reimpl and the real layer share one flat vector bit-for-bit. Shapes: InputW
// (I x 4O), FeedbackW (O x 4O), Peep (12 x O), Bias (1 x 4O).
static void unpackLstmWeights(const Eigen::VectorXd& flat, int I, int O,
                              Eigen::MatrixXd& inputW, Eigen::MatrixXd& feedbackW,
                              Eigen::MatrixXd& peep, Eigen::MatrixXd& bias) {
    inputW.resize(I, 4 * O);
    feedbackW.resize(O, 4 * O);
    peep.resize(12, O);
    bias.resize(1, 4 * O);
    long pos = 0;
    for (int jj = 0; jj < 4 * O; ++jj)
        for (int ii = 0; ii < I; ++ii) inputW(ii, jj) = flat(pos + (long)jj * I + ii);
    pos += (long)I * 4 * O;
    for (int jj = 0; jj < 4 * O; ++jj)
        for (int ii = 0; ii < O; ++ii) feedbackW(ii, jj) = flat(pos + (long)jj * O + ii);
    pos += (long)O * 4 * O;
    for (int jj = 0; jj < O; ++jj)
        for (int ii = 0; ii < 12; ++ii) peep(ii, jj) = flat(pos + (long)jj * 12 + ii);
    pos += 12L * O;
    for (int jj = 0; jj < 4 * O; ++jj) bias(0, jj) = flat(pos + jj);
}

// Phase 2 Task 4: faithful reimpl of LSTMLayer::feedForward (LSTMLayer.cpp:312-413),
// the crux the Rust port reproduces. Uses matSeq for the input projection AND the
// recurrence row-product (measured product-order contract, spec S4.3). Combined
// Eigen array expressions (e.g. a.*P[4] + b.*P[5] at :362-363) are ONE evaluation
// pass -- the two elementwise products are computed and summed, THEN added -- so
// each is replicated as a single fused elementwise loop here (NOT two += adds).
// Separate += adds elsewhere (:340-341, :391-392) stay separate. Peephole families:
// rows 0,1,2 = cells_peep; 4,5,6,8,9,10 = gates_peep; 3,7,11 = gates_rec_peep.
// Writes the post-activation _Gates cache (T x 4O) and outputs (T x O). lastLayer
// is unused in the LSTM forward (kept for signature parity). Width tolerance on the
// input projection matches :313-319 (cols>I -> leftCols(I); cols<I -> topRows(cols)).
static void lstmForwardLoop(const Eigen::MatrixXd& inputSeq, const Eigen::MatrixXd& inputW,
                            const Eigen::MatrixXd& feedbackW, const Eigen::MatrixXd& peep,
                            const Eigen::MatrixXd& bias, int O,
                            bool cellsPeep, bool gatesPeep, bool gatesRecPeep,
                            Eigen::MatrixXd& gates, Eigen::MatrixXd& output) {
    const int I = static_cast<int>(inputW.rows());
    const int cols = static_cast<int>(inputSeq.cols());
    // legacy: LSTMLayer.cpp:313-319 -- input projection with width tolerance.
    Eigen::MatrixXd proj;
    if (cols > I) {
        proj = matSeq(inputSeq.leftCols(I), inputW);
    } else if (cols < I) {
        proj = matSeq(inputSeq, inputW.topRows(cols));
    } else {
        proj = matSeq(inputSeq, inputW);
    }
    const int T = static_cast<int>(proj.rows());
    gates.resize(T, 4 * O);
    for (int t = 0; t < T; ++t)                                        // .rowwise() + _Biaises
        for (int j = 0; j < 4 * O; ++j) gates(t, j) = proj(t, j) + bias(0, j);
    Eigen::MatrixXd cellStates = Eigen::MatrixXd::Zero(T, O);
    Eigen::MatrixXd cellsIn = Eigen::MatrixXd::Zero(T, O);
    output.resize(T, O);

    // --- t = 0 (LSTMLayer.cpp:325-348) --------------------------------------
    {
        const int t = 0;
        // :333 activate i,f (GatesFunction); :334 activate g (Maxmin2)
        for (int j = 0; j < 2 * O; ++j) gates(t, j) = GatesFunction::fn(gates(t, j));
        for (int j = 0; j < O; ++j) gates(t, 3 * O + j) = Maxmin2::fn(gates(t, 3 * O + j));
        // :336 c_0 = i_0 .* g_0 (NO forget term)
        for (int j = 0; j < O; ++j) cellStates(t, j) = gates(t, j) * gates(t, 3 * O + j);
        // :338-342 o extras IN ORDER: c_0.*P[2]; i_0.*P[9]; f_0.*P[10] (separate adds)
        if (cellsPeep)
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += cellStates(t, j) * peep(2, j);
        if (gatesPeep) {
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += gates(t, j) * peep(9, j);
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += gates(t, O + j) * peep(10, j);
        }
        // :346 activate o; :347 cells_in = asinh(c_0); :348 y_0 = o_0 .* cells_in_0
        for (int j = 0; j < O; ++j) gates(t, 2 * O + j) = GatesFunction::fn(gates(t, 2 * O + j));
        for (int j = 0; j < O; ++j) cellsIn(t, j) = Identity::fn(cellStates(t, j));
        for (int j = 0; j < O; ++j) output(t, j) = gates(t, 2 * O + j) * cellsIn(t, j);
    }

    // --- t >= 1 (LSTMLayer.cpp:350-412) -------------------------------------
    for (int t = 1; t < T; ++t) {
        // :351 gates.row(t) += y_{t-1} * FeedbackWeights (all four blocks)
        Eigen::MatrixXd rec = matSeq(output.row(t - 1), feedbackW);    // 1 x 4O, ascending
        for (int j = 0; j < 4 * O; ++j) gates(t, j) += rec(0, j);
        // :353-356 cells peep into i,f: i += c_{t-1}.*P[0]; f += c_{t-1}.*P[1]
        if (cellsPeep) {
            for (int j = 0; j < O; ++j) gates(t, j) += cellStates(t - 1, j) * peep(0, j);
            for (int j = 0; j < O; ++j) gates(t, O + j) += cellStates(t - 1, j) * peep(1, j);
        }
        // :357-360 gates-rec into i,f: i += i_{t-1}.*P[3]; f += f_{t-1}.*P[7]
        if (gatesRecPeep) {
            for (int j = 0; j < O; ++j) gates(t, j) += gates(t - 1, j) * peep(3, j);
            for (int j = 0; j < O; ++j) gates(t, O + j) += gates(t - 1, O + j) * peep(7, j);
        }
        // :362-363 gates peep, each ONE COMBINED expression (single fused pass):
        //   i += (f_{t-1}.*P[4] + o_{t-1}.*P[5]); f += (i_{t-1}.*P[6] + o_{t-1}.*P[8])
        if (gatesPeep) {
            for (int j = 0; j < O; ++j)
                gates(t, j) += gates(t - 1, O + j) * peep(4, j) + gates(t - 1, 2 * O + j) * peep(5, j);
            for (int j = 0; j < O; ++j)
                gates(t, O + j) += gates(t - 1, j) * peep(6, j) + gates(t - 1, 2 * O + j) * peep(8, j);
        }
        // :370 activate i,f; :385 activate g
        for (int j = 0; j < 2 * O; ++j) gates(t, j) = GatesFunction::fn(gates(t, j));
        for (int j = 0; j < O; ++j) gates(t, 3 * O + j) = Maxmin2::fn(gates(t, 3 * O + j));
        // :387 c_t = i_t.*g_t + c_{t-1}.*f_t (single combined expression)
        for (int j = 0; j < O; ++j)
            cellStates(t, j) = gates(t, j) * gates(t, 3 * O + j) + cellStates(t - 1, j) * gates(t, O + j);
        // :389-394 o extras IN ORDER: c_t.*P[2]; i_t.*P[9]; f_t.*P[10]; o_{t-1}.*P[11]
        if (cellsPeep)
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += cellStates(t, j) * peep(2, j);
        if (gatesPeep) {
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += gates(t, j) * peep(9, j);
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += gates(t, O + j) * peep(10, j);
        }
        if (gatesRecPeep)
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += gates(t - 1, 2 * O + j) * peep(11, j);
        // :398 activate o; :399 cells_in = asinh(c_t); :400 y_t = o_t .* cells_in_t
        for (int j = 0; j < O; ++j) gates(t, 2 * O + j) = GatesFunction::fn(gates(t, 2 * O + j));
        for (int j = 0; j < O; ++j) cellsIn(t, j) = Identity::fn(cellStates(t, j));
        for (int j = 0; j < O; ++j) output(t, j) = gates(t, 2 * O + j) * cellsIn(t, j);
    }
}

// Phase 2 Task 4: reimpl of LSTMLayer::feedForwardReverse (LSTMLayer.cpp:415-421):
// reverse input rows, run forward, reverse output rows. The returned `gates` cache
// is left in REVERSED-input order (exactly _Gates after feedForwardReverse -- the
// real layer never un-reverses it); `output` IS un-reversed to match outputSeq.
static void lstmForwardReverseLoop(const Eigen::MatrixXd& inputSeq, const Eigen::MatrixXd& inputW,
                                   const Eigen::MatrixXd& feedbackW, const Eigen::MatrixXd& peep,
                                   const Eigen::MatrixXd& bias, int O,
                                   bool cellsPeep, bool gatesPeep, bool gatesRecPeep,
                                   Eigen::MatrixXd& gates, Eigen::MatrixXd& output) {
    Eigen::MatrixXd inputRev = inputSeq.colwise().reverse();
    Eigen::MatrixXd outputRev;
    lstmForwardLoop(inputRev, inputW, feedbackW, peep, bias, O,
                    cellsPeep, gatesPeep, gatesRecPeep, gates, outputRev);
    output = outputRev.colwise().reverse();
}

// Phase 2 Task 5/6: faithful reimpl of NeuronLayer::feedForward (NeuronLayer.cpp:
// 127-149). Width-tolerant projection via matSeq (cols>I -> leftCols(I); cols<I ->
// topRows(cols); else plain product), bias .rowwise() BEFORE activation, then one of
// three activation paths: lastLayer && O>1 -> UNSTABILIZED softmax (exp(a+b) with NO
// max-subtraction guard, SEQUENTIAL per-row sum in ascending column order, then
// per-COLUMN cwiseQuotient); lastLayer && O==1 -> Logistic; else Maxmin2/asinh.
// File-scope so BOTH the Task 5 dense probe AND the Task 6 container driver share
// one reimpl (Task 6's dense net layers forward through this).
static Eigen::MatrixXd denseForwardLoop(const Eigen::MatrixXd& inputSeq,
                                        const Eigen::MatrixXd& weights,
                                        const Eigen::MatrixXd& bias, bool lastLayer) {
    const int I = static_cast<int>(weights.rows());
    const int O = static_cast<int>(weights.cols());
    const int cols = static_cast<int>(inputSeq.cols());
    // :129-135 width-tolerant projection.
    Eigen::MatrixXd activations;
    if (cols > I) {
        activations = matSeq(inputSeq.leftCols(I), weights);
    } else if (cols < I) {
        activations = matSeq(inputSeq, weights.topRows(cols));
    } else {
        activations = matSeq(inputSeq, weights);
    }
    const int T = static_cast<int>(activations.rows());
    // .rowwise() + _Biaises, BEFORE activation (shared by all three paths).
    Eigen::MatrixXd preAct(T, O);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < O; ++j) preAct(t, j) = activations(t, j) + bias(0, j);

    Eigen::MatrixXd output(T, O);
    if (lastLayer && O > 1) {
        // :138 exp(a+b), :139-142 sequential row-sum THEN per-column quotient.
        Eigen::MatrixXd expOut(T, O);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < O; ++j) expOut(t, j) = std::exp(preAct(t, j));
        std::vector<double> rowSum(T, 0.0);
        for (int t = 0; t < T; ++t) {
            double acc = 0.0;
            for (int j = 0; j < O; ++j) acc += expOut(t, j);
            rowSum[t] = acc;
        }
        for (int j = 0; j < O; ++j)
            for (int t = 0; t < T; ++t) output(t, j) = expOut(t, j) / rowSum[t];
    } else if (lastLayer) {
        // O == 1: Logistic.
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < O; ++j) output(t, j) = Logistic::fn(preAct(t, j));
    } else {
        // Maxmin2 (asinh).
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < O; ++j) output(t, j) = Maxmin2::fn(preAct(t, j));
    }
    return output;
}

// Phase 2 Task 6: transcription of NeuralNetwork<L>::SubSample (NeuralNetwork.hpp:
// 125-134). T x C -> floor(T/R) x C*R: the trailing T mod R rows are DROPPED; source
// row jj*R+kk lands in the block [kk*C, (kk+1)*C) of output row jj (frame-contiguous
// temporal stacking). R==1 is a plain copy.
static Eigen::MatrixXd subSampleLoop(long R, const Eigen::MatrixXd& Input) {
    const long subLen = Input.rows() / R;  // floor: trailing T mod R rows dropped
    const long C = Input.cols();
    Eigen::MatrixXd SubInput(subLen, C * R);
    for (long jj = 0; jj < subLen; ++jj)
        for (long kk = 0; kk < R; ++kk)
            SubInput.block(jj, kk * C, 1, C) = Input.row(jj * R + kk);
    return SubInput;
}

// Phase 3 Task 1: transcription of NeuralNetwork<L>::InvSubSample (NeuralNetwork.hpp:
// 136-145). T x (C*R) -> (T*R) x C: the inverse of SubSample, un-stacking each block
// [kk*C, (kk+1)*C) of source row jj into output row jj*R+kk. Used by the backward
// container loop to inflate a sub-sampled layer's returned deltas back to the full
// (pre-decimation) row count before feeding the layer below it.
static Eigen::MatrixXd invSubSampleLoop(long R, const Eigen::MatrixXd& Input) {
    const long sampledLen = Input.rows() * R;
    const long C = Input.cols() / R;
    Eigen::MatrixXd out(sampledLen, C);
    for (long jj = 0; jj < Input.rows(); ++jj)
        for (long kk = 0; kk < R; ++kk)
            out.row(jj * R + kk) = Input.block(jj, kk * C, 1, C);
    return out;
}

// Read a legacy `.bin` (i64 LE rows, i64 LE cols, column-major f64) into a matrix --
// the inverse of Helpers.hpp's Matrix2BinaryFile. Used by the Phase 3 real-net
// backward probe to re-read the committed e2e_input.bin (201 x 11) without needing
// its shape at the call site.
static Eigen::MatrixXd readBinMatrix(const std::string& fileName) {
    std::ifstream in(fileName, std::ios::binary);
    if (!in) {
        std::cerr << "readBinMatrix: cannot open " << fileName << "\n";
        exit(1);
    }
    long long rows = 0, cols = 0;
    in.read(reinterpret_cast<char*>(&rows), 8);
    in.read(reinterpret_cast<char*>(&cols), 8);
    Eigen::MatrixXd m(rows, cols);
    for (long long c = 0; c < cols; ++c)
        for (long long r = 0; r < rows; ++r) {
            double v;
            in.read(reinterpret_cast<char*>(&v), 8);
            m(r, c) = v;
        }
    return m;
}

// Phase 2 Task 6: a per-layer forward step (LSTM or dense), bundling the layer's
// unpacked weights + its forward direction so the container driver stays layer-type
// agnostic. `forward(in, out)` runs the reimpl (lstmForwardLoop / denseForwardLoop)
// and writes `out` (the LSTM step discards its gates cache -- the container never
// reads it). `lastLayer` is threaded to match the legacy driver's per-layer flag.
struct NetLayerStep {
    std::function<void(const Eigen::MatrixXd&, Eigen::MatrixXd&, bool)> forward;
};

// Phase 2 Task 6: transcription of NeuralNetwork<L>::feedForward (NeuralNetwork.hpp:
// 158-198). EMPTY input (0 rows) -> silent no-op (outputSeq untouched). Single-layer
// net (neuronNb.size()==2): optional subsample of the input then layer forward with
// lastLayer=true. Multi-layer ascending: layer 0 (optionally subsampled input) ->
// layersOutput[0] (lastLayer=false); middle layers same over layersOutput[jj-1];
// FINAL layer writes the CALLER-ALLOCATED outputSeq (lastLayer=true). subSampling[jj]
// gt 1 subsamples that layer's input first. layersOutput is sized per :172/:175/:188/
// :191 (rows_after_subsample x neuronNb[jj+1]) but our reimpl steps resize `out`
// themselves, so we just hold the intermediate matrices.
static void netForwardLoop(const std::vector<long>& neuronNb,
                           const std::vector<long>& subSampling,
                           const std::vector<NetLayerStep>& steps,
                           const Eigen::MatrixXd& Input, Eigen::MatrixXd& outputSeq) {
    if (Input.rows() == 0) return;  // :159 -- empty input is a no-op
    const size_t L = neuronNb.size();
    if (L == 2) {
        if (subSampling[0] > 1) {
            Eigen::MatrixXd sub = subSampleLoop(subSampling[0], Input);
            steps[0].forward(sub, outputSeq, true);
        } else {
            steps[0].forward(Input, outputSeq, true);
        }
        return;
    }
    std::vector<Eigen::MatrixXd> layersOutput(L - 1);
    for (size_t jj = 0; jj < L - 1; ++jj) {
        if (jj == 0) {
            if (subSampling[0] > 1) {
                Eigen::MatrixXd sub = subSampleLoop(subSampling[0], Input);
                steps[jj].forward(sub, layersOutput[0], false);
            } else {
                steps[jj].forward(Input, layersOutput[0], false);
            }
        } else if (jj == L - 2) {
            if (subSampling[jj] > 1) {
                Eigen::MatrixXd sub = subSampleLoop(subSampling[jj], layersOutput[jj - 1]);
                steps[jj].forward(sub, outputSeq, true);
            } else {
                steps[jj].forward(layersOutput[jj - 1], outputSeq, true);
            }
        } else {
            if (subSampling[jj] > 1) {
                Eigen::MatrixXd sub = subSampleLoop(subSampling[jj], layersOutput[jj - 1]);
                steps[jj].forward(sub, layersOutput[jj], false);
            } else {
                steps[jj].forward(layersOutput[jj - 1], layersOutput[jj], false);
            }
        }
    }
}

// Phase 2 Task 6: transcription of NeuralNetwork<L>::feedForwardReverse (:200-240).
// IDENTICAL driver to netForwardLoop; the per-step `forward` closures dispatch the
// REVERSE layer kernel (reversal lives inside the layer). Kept as a separate driver
// (not a flag on netForwardLoop) to mirror the legacy's two copy-pasted methods.
static void netForwardReverseLoop(const std::vector<long>& neuronNb,
                                  const std::vector<long>& subSampling,
                                  const std::vector<NetLayerStep>& steps,
                                  const Eigen::MatrixXd& Input, Eigen::MatrixXd& outputSeq) {
    // The driver is byte-identical to netForwardLoop -- only the step closures differ
    // (they call the reverse kernel). Delegate to keep the two in lockstep.
    netForwardLoop(neuronNb, subSampling, steps, Input, outputSeq);
}

// Phase 2 Task 6: transcription of NeuralNetwork<L>::feedForwardDouble (:242-249).
// hcat firstInputSeq | secondInputSeq into an (rows x neuronNb[0]) matrix (forward
// half on the LEFT), then normal forward. Empty first input -> no-op.
static void netForwardDoubleLoop(const std::vector<long>& neuronNb,
                                 const std::vector<long>& subSampling,
                                 const std::vector<NetLayerStep>& steps,
                                 const Eigen::MatrixXd& first, const Eigen::MatrixXd& second,
                                 Eigen::MatrixXd& outputSeq) {
    if (first.rows() == 0) return;  // :243
    Eigen::MatrixXd Input(first.rows(), neuronNb[0]);
    Input << first, second;  // :245 forward half LEFT, second half RIGHT
    netForwardLoop(neuronNb, subSampling, steps, Input, outputSeq);
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

// --- Task 11: FeatureConfig / SpectralParams derivation ---------------------
// Faithful transcription of LongTermSpectralVariation::buildFromConf (key reads +
// sanitization, LongTermSpectralVariation.cpp:44-80), BLSTMSpectralSegmenter::
// buildFromConf (LTSV/TDC key reads, :44-83), and BLSTMSpectralSegmenter::
// initSpectralAnalysis (order clamp + sizes + shift quantization + BLSTM-variant
// freq band + LTSV params, :194-314). The LTSV/TDC classes cannot be compiled
// standalone (they drag the whole Segmenter hierarchy), so the derivation is
// transcribed here over the real ConfigFile (a compiled TU) rather than linked.
//
// DIVERGENCE NOTE (load-bearing, documented in the task report): the legacy guards
// the LTSVshift read behind `if (_LTSVWindowShift != 0.0)` (BLSTMSpectralSegmenter
// .cpp:50) -- but `_LTSVWindowShift` is an uninitialized double member at that
// point (no in-class initializer, empty ctor body), so that branch reads an
// INDETERMINATE value (UB). We drop the UB guard and read LTSVshift unconditionally
// when the key is present; both the harness and the Rust port do the same, so the
// golden stays valid. All four variant configs supply LTSVshift explicitly.
struct FeatureCfg {
    long order;
    double shift_sec;
    long conv_size;
    std::string conv_type;
    double min_mel;
    double max_mel;
    int nb_bins;
    bool is_log;
    int nb_dct;
    bool ignore_first;
    int deltas_nb;
    int dd_nb;
    double min_freq;
    double max_freq;
    std::string win_type;
    double win_param;
    bool flag_dc;
    double preemph;
    int noise_seed;
    double noise_ratio;
    double ltsv_window;
    double ltsv_shift;
    double tdc_window;
};

static FeatureCfg readFeatureCfg(ConfigFile& conf, const std::string& p) {
    FeatureCfg c;
    // LongTermSpectralVariation::buildFromConf key reads (:44-59, no defaults).
    c.order = conf.get<long>(p + "_spectrum_order");
    c.shift_sec = conf.get<double>(p + "_spectrum_shift");
    c.conv_size = conf.get<long>(p + "_spectrum_temporal_convolution_size");
    c.conv_type = conf.get<std::string>(p + "_spectrum_temporal_convolution_type");
    c.min_mel = conf.get<double>(p + "_minMelFreq");
    c.max_mel = conf.get<double>(p + "_maxMelFreq");
    c.nb_bins = conf.get<int>(p + "_nb_bins");
    c.is_log = conf.get<bool>(p + "_is_log_mel");
    c.nb_dct = conf.get<int>(p + "_nb_DCT");
    c.ignore_first = conf.get<bool>(p + "_IgnoreFirstDCT");
    c.deltas_nb = conf.get<int>(p + "_ComputeDeltasNb");
    c.dd_nb = conf.get<int>(p + "_ComputeDeltaDeltasNb");
    // minFreq/maxFreq: clamp negatives to 0 BEFORE the swap (:56-59).
    c.min_freq = conf.get<double>(p + "_minFreq");
    if (c.min_freq < 0.0) c.min_freq = 0.0;
    c.max_freq = conf.get<double>(p + "_maxFreq");
    if (c.max_freq < 0.0) c.max_freq = 0.0;
    // Mel pair swap + span-widen (:60-69).
    if (c.max_mel < c.min_mel) { double t = c.min_mel; c.min_mel = c.max_mel; c.max_mel = t; }
    if (std::abs(c.max_mel - c.min_mel) < 2.0) {
        double mean = (c.min_mel + c.max_mel) / 2.0;
        c.max_mel = mean + 1.0;
        c.min_mel = mean - 1.0;
    }
    // Hz pair swap + span-widen (:70-79).
    if (c.max_freq < c.min_freq) { double t = c.min_freq; c.min_freq = c.max_freq; c.max_freq = t; }
    if (std::abs(c.max_freq - c.min_freq) < 2.0) {
        double mean = (c.min_freq + c.max_freq) / 2.0;
        c.max_freq = mean + 1.0;
        c.min_freq = mean - 1.0;
    }
    // Segmenter::buildFromConf reads used by the derivation (Segmenter.cpp:94-98,
    // 100-102) + BLSTMSpectralSegmenter LTSV/TDC reads (:48-81).
    c.win_type = conf.get<std::string>(p + "_windowing_type");
    c.win_param = conf.get<double>(p + "_windowing_param");
    c.flag_dc = conf.get<bool>(p + "_flag_DCOffset");
    c.preemph = conf.get<double>(p + "_preemph_ratio");
    c.noise_seed = conf.get<int>(p + "_noise_seed");
    c.noise_ratio = conf.get<double>(p + "_noise_ratio");
    c.ltsv_window = conf.get<double>(p + "_LTSVwindow", 0.0);
    if (c.ltsv_window < 0.0) c.ltsv_window = 0.0;
    // LTSVshift read unconditionally (UB guard dropped, see note above).
    c.ltsv_shift = conf.get<double>(p + "_LTSVshift");
    if (c.ltsv_shift < 0.0) c.ltsv_shift = 0.0;
    c.tdc_window = conf.get<double>(p + "_TDCwindow", 0.0);
    if (c.tdc_window < 0.0) c.tdc_window = 0.0;
    return c;
}

// Derived spectral params (the 8-double params dump order + the freq band the raw
// path uses). BLSTMSpectralSegmenter::initSpectralAnalysis (:199-311), Max=20.
struct SpectralP {
    long order;
    long window_size;
    long bins;             // periodogram_length = 2^(order-1)+1
    long shift_frames;
    long freq_beg;         // BLSTM-variant band (:229-239), PRE-mel (raw-path band)
    long freq_end;
    long ltsv_half_window;
    long ltsv_shift;
    double min_freq_snapped;  // freq_beg*freqStep (mel ctor arg)
    double max_freq_snapped;  // freq_end*freqStep
};

static SpectralP deriveSpectral(const FeatureCfg& c, double rate, unsigned Max) {
    SpectralP s;
    s.order = c.order;
    if (s.order > (long)Max - 1) s.order = (long)Max - 1;   // clamp to 19 (:199-203)
    s.window_size = 1 << s.order;
    long periodogram_length = (1 << (s.order - 1)) + 1;
    s.bins = periodogram_length;
    s.shift_frames = (long)boost::math::round(c.shift_sec * rate);   // :208
    // (audio.hasReadWavFile() is true for the excerpt, so the =80 override is skipped.)
    // Freq band, BLSTM variant (:229-239). freqStep divides by (periodogram_length-1),
    // NOT by periodogram_length; the LTSV.cpp variant differs -- this is the BLSTM one.
    std::vector<double>::size_type freq_beg = 0;
    std::vector<double>::size_type freq_end = periodogram_length - 1;
    double freqStep = rate / 2 / freq_end;
    std::vector<double>::size_type tmp = (std::vector<double>::size_type)std::floor(c.min_freq / freqStep);
    if (freq_beg < tmp) freq_beg = tmp;
    if (freq_beg > freq_end) freq_beg = freq_end;
    tmp = (std::vector<double>::size_type)std::ceil(c.max_freq / freqStep);
    if (freq_end > tmp) freq_end = tmp;
    if (freq_end < freq_beg) freq_end = freq_beg;
    s.freq_beg = (long)freq_beg;
    s.freq_end = (long)freq_end;
    s.min_freq_snapped = freq_beg * freqStep;   // :238
    s.max_freq_snapped = freq_end * freqStep;   // :239
    // LTSV params (:302-305). R and shift both keyed off shift_frames.
    long R = (long)boost::math::round(c.ltsv_window * rate / 2.0 / s.shift_frames);
    if (R < 1) R = 0;
    long ls = (long)boost::math::round(c.ltsv_shift * rate / s.shift_frames);
    if (ls < 1) ls = 1;
    s.ltsv_half_window = R;
    s.ltsv_shift = ls;
    return s;
}

// ===========================================================================
// Phase 2b Task 6: BLSTMSignalSegmenter (Algo 4) reimpl-swap family.
//
// The signal driver is the FIRST driver with the NN in the chain: its result_vec
// is produced by _BLSTMNeuralNetwork.feedForwardBackward (BLSTMSignalSegmenter.cpp:
// 260). Eigen's blocked GEMM diverges from ascending accumulation at the real net's
// k>=23 shapes (Phase 2 NN_PROBE), and the layer-0 GEMM seeds the recurrence, so the
// divergence propagates non-locally. The SignalProbe therefore transcribes
// getSegmentation (:93-398, LIVE code only) faithfully but swaps ONLY that one FFB
// call for the ascending-loop reimpl family below (makeLstmSteps/makeDenseSteps +
// blstmFeedForwardT6 + the three windowed drivers), reusing the file-scope kernel
// statics (lstmForwardLoop/netForwardLoop/denseForwardLoop/...). These are
// standalone statics (not the Task 8/9 scope lambdas) so no prior golden path is
// perturbed; they are byte-identical in logic to the Task 8/9 reimpls.
//
// State bundle for one BLSTM sub-network's per-layer weight matrices + steps. The
// unpacked matrices must outlive the steps (the step closures capture references
// into these vectors), so the caller owns a T6Net and passes it around.
struct T6Net {
    std::vector<Eigen::MatrixXd> iw, fw, pp, bs;   // LSTM per-layer (fwd or bwd)
    std::vector<Eigen::MatrixXd> dw, db;           // dense per-layer
    std::vector<NetLayerStep> lstmFwd, lstmRev, dense;
};

// Build fwd/rev LSTM steps from a flat weight slice (all peepholes on). Mirrors the
// Task 8 makeLstmSteps lambda; returns the consumed weight count.
static long t6MakeLstmSteps(const Eigen::VectorXd& flat, const std::vector<int>& ins,
                            const std::vector<int>& outs, T6Net& n) {
    const size_t L = ins.size();
    n.iw.resize(L); n.fw.resize(L); n.pp.resize(L); n.bs.resize(L);
    n.lstmFwd.resize(L); n.lstmRev.resize(L);
    long pos = 0;
    for (size_t jj = 0; jj < L; ++jj) {
        int I = ins[jj], O = outs[jj];
        long nb = 4L * I * O + 4L * O * O + 12L * O + 4L * O;
        Eigen::VectorXd slice = flat.segment(pos, nb);
        pos += nb;
        unpackLstmWeights(slice, I, O, n.iw[jj], n.fw[jj], n.pp[jj], n.bs[jj]);
    }
    for (size_t jj = 0; jj < L; ++jj) {
        int O = outs[jj];
        n.lstmFwd[jj].forward = [&n, jj, O](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool) {
            Eigen::MatrixXd g;
            lstmForwardLoop(in, n.iw[jj], n.fw[jj], n.pp[jj], n.bs[jj], O, true, true, true, g, outm);
        };
        n.lstmRev[jj].forward = [&n, jj, O](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool) {
            Eigen::MatrixXd g;
            lstmForwardReverseLoop(in, n.iw[jj], n.fw[jj], n.pp[jj], n.bs[jj], O, true, true, true, g, outm);
        };
    }
    return pos;
}

// Build dense (NeuronLayer) steps from a flat weight slice. Mirrors makeDenseSteps.
static long t6MakeDenseSteps(const Eigen::VectorXd& flat, const std::vector<int>& ins,
                             const std::vector<int>& outs, T6Net& n) {
    const size_t L = ins.size();
    n.dw.resize(L); n.db.resize(L); n.dense.resize(L);
    long pos = 0;
    for (size_t jj = 0; jj < L; ++jj) {
        int I = ins[jj], O = outs[jj];
        long nb = (long)O * (I + 1);
        Eigen::VectorXd slice = flat.segment(pos, nb);
        pos += nb;
        n.dw[jj].resize(I, O); n.db[jj].resize(1, O);
        for (int c = 0; c < O; ++c)
            for (int r = 0; r < I; ++r) n.dw[jj](r, c) = slice((long)c * I + r);
        for (int c = 0; c < O; ++c) n.db[jj](0, c) = slice((long)O * I + c);
    }
    for (size_t jj = 0; jj < L; ++jj) {
        n.dense[jj].forward = [&n, jj](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool last) {
            outm = denseForwardLoop(in, n.dw[jj], n.db[jj], last);
        };
    }
    return pos;
}

// The whole real-net BLSTM reimpl state, built once from the flat weight vector.
// Real net: LSTM [23,24,24] sub [4,1]; output [48,12,1] sub [1,1]; ssr = 4.
struct T6Blstm {
    std::vector<long> lstmNN{23, 24, 24}, lstmSS{4, 1}, outNN{48, 12, 1}, outSS{1, 1};
    std::vector<int> lstmIns{92, 24}, lstmOuts{24, 24};   // 23*4=92, 24*1=24
    std::vector<int> outIns{48, 12}, outOuts{12, 1};
    int fwdInputSize = 23;
    long subRatio = 4, outNetRatio = 1, lstmSubRatio = 4;
    int fwdOut = 24;    // last LSTM layer output size
    T6Net fwd, bwd, out;

    explicit T6Blstm(const Eigen::VectorXd& flat) {
        long fwdNb = 0;
        for (size_t jj = 0; jj < lstmIns.size(); ++jj) {
            int I = lstmIns[jj], O = lstmOuts[jj];
            fwdNb += 4L * I * O + 4L * O * O + 12L * O + 4L * O;
        }
        long outNbW = 0;
        for (size_t jj = 0; jj < outIns.size(); ++jj) outNbW += (long)outOuts[jj] * (outIns[jj] + 1);
        Eigen::VectorXd fwdSlice = flat.head(fwdNb);
        Eigen::VectorXd bwdSlice = flat.segment(fwdNb, fwdNb);
        Eigen::VectorXd outSlice = flat.segment(2 * fwdNb, outNbW);
        t6MakeLstmSteps(fwdSlice, lstmIns, lstmOuts, fwd);
        t6MakeLstmSteps(bwdSlice, lstmIns, lstmOuts, bwd);
        t6MakeDenseSteps(outSlice, outIns, outOuts, out);
    }
};

// Core plain feedForward reimpl for the real net (BLSTMNeuralNetwork.cpp:419-437):
// forward net feedForward + backward net feedForwardReverse into (outputLength x
// lstmOut), then output net feedForwardDouble (forward half LEFT). The leftCols
// crop gate (LSTMRatios[0] > 1 && fwdInputSize < input.cols()) is FALSE for signal
// mode's 1-col input (23 < 1 is false), so the whole 1-col input flows into the
// LSTM layer, which itself handles cols<inputSize via inputW.topRows(cols).
static void blstmFeedForwardT6(T6Blstm& b, const Eigen::MatrixXd& input,
                               Eigen::MatrixXd& outForward, Eigen::MatrixXd& outBackward,
                               Eigen::MatrixXd& output) {
    Eigen::MatrixXd fwdIn = input, bwdIn = input;
    if (b.lstmSS[0] > 1 && b.fwdInputSize < input.cols()) {
        fwdIn = input.leftCols(b.fwdInputSize);
        bwdIn = input.leftCols(b.fwdInputSize);
    }
    netForwardLoop(b.lstmNN, b.lstmSS, b.fwd.lstmFwd, fwdIn, outForward);
    netForwardReverseLoop(b.lstmNN, b.lstmSS, b.bwd.lstmRev, bwdIn, outBackward);
    netForwardDoubleLoop(b.outNN, b.outSS, b.out.dense, outForward, outBackward, output);
}

// Reimpl of feedForwardBackwardTruncateSweep (BLSTMNeuralNetwork.cpp:488-546), the
// !two_sweeps Truncate path (signal noOverlap always has _BLSTMTwoSweeps parsed but
// the config sets it false, and the TwoSweeps branch is dead in signal anyway).
static void truncateSweepT6(T6Blstm& b, const Eigen::MatrixXd& input, long window_size,
                            Eigen::MatrixXd& outputSeq) {
    long lengthOutputLSTM = input.rows();
    for (long r : b.lstmSS) lengthOutputLSTM /= r;
    Eigen::MatrixXd outputForward = Eigen::MatrixXd::Zero(lengthOutputLSTM, b.fwdOut);
    Eigen::MatrixXd outputBackward = Eigen::MatrixXd::Zero(lengthOutputLSTM, b.fwdOut);

    long length = window_size, lstmLenShort;
    for (long r : b.lstmSS) length /= r;
    lstmLenShort = length;
    for (long r : b.outSS) length /= r;
    long nominalLen = length, nominalLstm = lstmLenShort;

    for (long jj = 0; jj < input.rows(); jj += window_size) {
        long begin = jj;
        long end = jj + window_size - 1;
        if (end >= input.rows()) end = input.rows() - 1;
        long lengthSeq = end - begin + 1;
        long lengthShort, lstmLength;
        if (lengthSeq != window_size) {
            lengthShort = lengthSeq;
            for (long r : b.lstmSS) lengthShort /= r;
            lstmLength = lengthShort;
            for (long r : b.outSS) lengthShort /= r;
        } else {
            lengthShort = nominalLen;
            lstmLength = nominalLstm;
        }
        if (lengthShort > 0) {
            Eigen::MatrixXd block = input.block(begin, 0, lengthSeq, input.cols());
            Eigen::MatrixXd outShort, oF, oB;
            blstmFeedForwardT6(b, block, oF, oB, outShort);
            outputSeq.block(begin / b.subRatio, 0, lengthShort, outputSeq.cols()) = outShort;
            outputForward.block(begin / b.lstmSubRatio, 0, lstmLength, b.fwdOut) = oF;
            outputBackward.block(begin / b.lstmSubRatio, 0, lstmLength, b.fwdOut) = oB;
        }
    }
}

// Reimpl of feedForwardBackwardOverLap (BLSTMNeuralNetwork.cpp:592-681): overlapping
// windows accumulated INTO the caller's outputSeq (in place, :664 noalias() +=) +
// per-row counts, then quotient (0/0 -> NaN on uncovered rows, no guard). Window
// bounds snapped to the subsampling grid (begin down, end up).
static void overlapT6(T6Blstm& b, const Eigen::MatrixXd& input, long windowSize,
                      long windowShift, Eigen::MatrixXd& outputSeq) {
    long lengthOutputLSTM = outputSeq.rows() * b.outNetRatio;
    Eigen::MatrixXd outputForward = Eigen::MatrixXd::Zero(lengthOutputLSTM, b.fwdOut);
    Eigen::MatrixXd outputBackward = Eigen::MatrixXd::Zero(lengthOutputLSTM, b.fwdOut);
    Eigen::MatrixXd outCount = Eigen::MatrixXd::Zero(outputSeq.rows(), 1);
    Eigen::MatrixXd outCountLSTM = Eigen::MatrixXd::Zero(lengthOutputLSTM, 1);

    long length = 2 * windowSize + 1;
    for (long r : b.lstmSS) length /= r;
    for (long r : b.outSS) length /= r;
    long nominalLen = length, nominalLstm = length;

    for (long jj = 0; jj < input.rows(); jj += windowShift) {
        long begin = (jj < windowSize) ? 0 : jj - windowSize;
        begin = (begin / b.subRatio) * b.subRatio;
        long end = begin + 2 * windowSize;
        if (end >= input.rows()) end = input.rows() - 1;
        end = ((end + 1) / b.subRatio) * b.subRatio - 1;
        long lengthSeq = end - begin + 1;
        long lengthShort, lengthShortLSTM;
        if (lengthSeq != 2 * windowSize + 1) {
            lengthShort = lengthSeq;
            lengthShortLSTM = lengthSeq;
            for (long r : b.lstmSS) { lengthShort /= r; lengthShortLSTM /= r; }
            for (long r : b.outSS) lengthShort /= r;
        } else {
            lengthShort = nominalLen;
            lengthShortLSTM = nominalLstm;
        }
        if (lengthShort > 0) {
            Eigen::MatrixXd block = input.block(begin, 0, lengthSeq, input.cols());
            Eigen::MatrixXd outShort, oF, oB;
            blstmFeedForwardT6(b, block, oF, oB, outShort);
            // :664 accumulate INTO the caller's outputSeq buffer (in place).
            outputSeq.block(begin / b.subRatio, 0, lengthShort, outputSeq.cols()) += outShort;
            outCount.block(begin / b.subRatio, 0, lengthShort, 1) += Eigen::MatrixXd::Ones(lengthShort, 1);
            long lbeg = begin * b.outNetRatio / b.subRatio;
            outputForward.block(lbeg, 0, lengthShortLSTM, b.fwdOut) += oF;
            outputBackward.block(lbeg, 0, lengthShortLSTM, b.fwdOut) += oB;
            outCountLSTM.block(lbeg, 0, lengthShortLSTM, 1) += Eigen::MatrixXd::Ones(lengthShortLSTM, 1);
        }
    }
    for (long c = 0; c < outputSeq.cols(); ++c)
        for (long r = 0; r < outputSeq.rows(); ++r)
            outputSeq(r, c) /= outCount(r, 0);
    (void)outputForward; (void)outputBackward; (void)outCountLSTM;
}

// Top-level feedForwardBackward reimpl for signal mode, dispatching exactly as
// BLSTMNeuralNetwork::feedForwardBackward (:711-773) does after setProcessingType
// ((window>0), !noOverlap): type -1 whole-sequence self-normalization in place
// (the real net's _InputNormalizationType == -1), then the driver:
//   window==0  -> plain (truncates_sequence=false)
//   noOverlap  -> truncate (truncates_sequence=true, overlaps=false)
//   overlap    -> overlap (truncates_sequence=true, overlaps=true)
static void signalReimplFFB(T6Blstm& b, Eigen::MatrixXd& input, long window_size,
                            long window_shift, bool noOverlap, Eigen::MatrixXd& output) {
    // Type -1 self-normalization in place (BLSTMNeuralNetwork.cpp:737-744).
    {
        const int R = static_cast<int>(input.rows());
        const int C = static_cast<int>(input.cols());
        if (R > 0) {
            std::vector<double> mean(C, 0.0);
            for (int c = 0; c < C; ++c) {
                double acc = 0.0;
                for (int r = 0; r < R; ++r) acc += input(r, c);
                mean[c] = acc / (double)R;
            }
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c) input(r, c) -= mean[c];
            std::vector<double> stdv(C, 0.0);
            for (int c = 0; c < C; ++c) {
                double acc = 0.0;
                for (int r = 0; r < R; ++r) acc += input(r, c) * input(r, c);
                stdv[c] = std::sqrt((acc + 1e-32) / (double)R);
            }
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c) input(r, c) = Maxmin2::fn(input(r, c) / stdv[c]);
        }
    }
    if (window_size == 0) {
        Eigen::MatrixXd oF, oB;
        blstmFeedForwardT6(b, input, oF, oB, output);   // plain path
    } else if (noOverlap) {
        truncateSweepT6(b, input, window_size, output);  // truncate path
    } else {
        overlapT6(b, input, window_size, window_shift, output);  // overlap path
    }
}

// =========================================================================
// Phase 3 Task 1: ascending-loop BACKWARD reimpl family. Mirrors the Phase 2
// forward reimpl family above (lstmForwardLoop/denseForwardLoop/netForward*),
// built on matSeq + the same file-scope kernels. The backward contains Eigen
// GEMMs at NN-wide k (_FeedbackWeights*test LSTMLayer.cpp:703, _InputWeights*
// testM :707, NeuronLayer deltas*_Weights.transpose() :202), so the goldens are
// dumped from THESE reimpls and the NN_TOL probes record the real-class deltas
// (synthetic small shapes expected 0 ULP; real-net k>=23 recorded as Phase 4
// calibration). Derivatives are recomputed FROM the post-activation caches
// (_Gates/_CellStates/_CellsIn) via the activation deriv structs -- no forward
// change; the forward-with-caches helper below just also RETURNS cellStates/
// cellsIn (the forward stores gates + output only) so the backward can read all
// three caches the real layer keeps as members.

// Forward-with-caches: identical math to lstmForwardLoop but ALSO returns the
// cellStates (T x O) and cellsIn (T x O = asinh(cellStates)) caches the backward
// consumes (_CellStates / _CellsIn members). Byte-identical to lstmForwardLoop by
// construction (same op order); kept separate so no forward golden path shifts.
static void lstmForwardLoopCache(const Eigen::MatrixXd& inputSeq, const Eigen::MatrixXd& inputW,
                                 const Eigen::MatrixXd& feedbackW, const Eigen::MatrixXd& peep,
                                 const Eigen::MatrixXd& bias, int O,
                                 bool cellsPeep, bool gatesPeep, bool gatesRecPeep,
                                 Eigen::MatrixXd& gates, Eigen::MatrixXd& output,
                                 Eigen::MatrixXd& cellStates, Eigen::MatrixXd& cellsIn) {
    const int I = static_cast<int>(inputW.rows());
    const int cols = static_cast<int>(inputSeq.cols());
    Eigen::MatrixXd proj;
    if (cols > I) {
        proj = matSeq(inputSeq.leftCols(I), inputW);
    } else if (cols < I) {
        proj = matSeq(inputSeq, inputW.topRows(cols));
    } else {
        proj = matSeq(inputSeq, inputW);
    }
    const int T = static_cast<int>(proj.rows());
    gates.resize(T, 4 * O);
    for (int t = 0; t < T; ++t)
        for (int j = 0; j < 4 * O; ++j) gates(t, j) = proj(t, j) + bias(0, j);
    cellStates = Eigen::MatrixXd::Zero(T, O);
    cellsIn = Eigen::MatrixXd::Zero(T, O);
    output.resize(T, O);
    {
        const int t = 0;
        for (int j = 0; j < 2 * O; ++j) gates(t, j) = GatesFunction::fn(gates(t, j));
        for (int j = 0; j < O; ++j) gates(t, 3 * O + j) = Maxmin2::fn(gates(t, 3 * O + j));
        for (int j = 0; j < O; ++j) cellStates(t, j) = gates(t, j) * gates(t, 3 * O + j);
        if (cellsPeep)
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += cellStates(t, j) * peep(2, j);
        if (gatesPeep) {
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += gates(t, j) * peep(9, j);
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += gates(t, O + j) * peep(10, j);
        }
        for (int j = 0; j < O; ++j) gates(t, 2 * O + j) = GatesFunction::fn(gates(t, 2 * O + j));
        for (int j = 0; j < O; ++j) cellsIn(t, j) = Identity::fn(cellStates(t, j));
        for (int j = 0; j < O; ++j) output(t, j) = gates(t, 2 * O + j) * cellsIn(t, j);
    }
    for (int t = 1; t < T; ++t) {
        Eigen::MatrixXd rec = matSeq(output.row(t - 1), feedbackW);
        for (int j = 0; j < 4 * O; ++j) gates(t, j) += rec(0, j);
        if (cellsPeep) {
            for (int j = 0; j < O; ++j) gates(t, j) += cellStates(t - 1, j) * peep(0, j);
            for (int j = 0; j < O; ++j) gates(t, O + j) += cellStates(t - 1, j) * peep(1, j);
        }
        if (gatesRecPeep) {
            for (int j = 0; j < O; ++j) gates(t, j) += gates(t - 1, j) * peep(3, j);
            for (int j = 0; j < O; ++j) gates(t, O + j) += gates(t - 1, O + j) * peep(7, j);
        }
        if (gatesPeep) {
            for (int j = 0; j < O; ++j)
                gates(t, j) += gates(t - 1, O + j) * peep(4, j) + gates(t - 1, 2 * O + j) * peep(5, j);
            for (int j = 0; j < O; ++j)
                gates(t, O + j) += gates(t - 1, j) * peep(6, j) + gates(t - 1, 2 * O + j) * peep(8, j);
        }
        for (int j = 0; j < 2 * O; ++j) gates(t, j) = GatesFunction::fn(gates(t, j));
        for (int j = 0; j < O; ++j) gates(t, 3 * O + j) = Maxmin2::fn(gates(t, 3 * O + j));
        for (int j = 0; j < O; ++j)
            cellStates(t, j) = gates(t, j) * gates(t, 3 * O + j) + cellStates(t - 1, j) * gates(t, O + j);
        if (cellsPeep)
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += cellStates(t, j) * peep(2, j);
        if (gatesPeep) {
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += gates(t, j) * peep(9, j);
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += gates(t, O + j) * peep(10, j);
        }
        if (gatesRecPeep)
            for (int j = 0; j < O; ++j) gates(t, 2 * O + j) += gates(t - 1, 2 * O + j) * peep(11, j);
        for (int j = 0; j < O; ++j) gates(t, 2 * O + j) = GatesFunction::fn(gates(t, 2 * O + j));
        for (int j = 0; j < O; ++j) cellsIn(t, j) = Identity::fn(cellStates(t, j));
        for (int j = 0; j < O; ++j) output(t, j) = gates(t, 2 * O + j) * cellsIn(t, j);
    }
}

// The four LSTM derivative blocks + the frame count, in the same shapes the real
// layer's protected members carry (input I x 4O, feedback O x 4O, peep 12 x O,
// bias 1 x 4O). ACCUMULATED across calls, exactly like the real layer's members.
struct LstmDerivs {
    Eigen::MatrixXd inputW, feedbackW, peep, bias;   // *Derivatives blocks
    long nbFedBackward = 0;
    void init(int I, int O) {
        inputW = Eigen::MatrixXd::Zero(I, 4 * O);
        feedbackW = Eigen::MatrixXd::Zero(O, 4 * O);
        peep = Eigen::MatrixXd::Zero(12, O);
        bias = Eigen::MatrixXd::Zero(1, 4 * O);
        nbFedBackward = 0;
    }
};

// legacy: LSTMLayer.cpp:518-723 (feedBackward), transcribed VERBATIM. Reverse-time
// loop; gate-derivative order O -> state -> C -> F -> I; the 12-row peephole index
// map identical to the forward (rows 0,1,2 cells; 3,7,11 gates-rec; 4,5,6,8,9,10
// gates); the row==0 no-cellstate forget branch (:648-657); the deltasForgetGatetmp
// save (:624 read at :660 for the input gate); the width tolerance (:534-543:
// cols>I -> leftCols transpose; cols<I -> ZERO-PAD to I rows); the 4O-stacked
// testM; deltasPreviousLayer = (_InputWeights*testM).transpose() and deltasFeedback
// = (_FeedbackWeights*test).transpose() via matSeq (:703,:707); the invSubSampling
// block scaling (:710-715); ACCUMULATE into the four deriv members; += linesNb.
// I is the LAYER'S input size (inputW.rows()); cols is InputSeq.cols(). Derivatives
// recomputed FROM the post-activation caches (gates/cellStates/cellsIn) via the
// activation deriv structs -- GatesFunction/Maxmin2/Identity::deriv take the
// ACTIVATED value. DEAD (not ported): the _MaxSaturation truncation (:693-705
// commented), the commented alternate bodies (:519-533,:564-574,:684-704).
static Eigen::MatrixXd lstmBackwardLoop(const Eigen::MatrixXd& InputSeq,
                                        const Eigen::MatrixXd& outputSeq,
                                        const Eigen::MatrixXd& deltas,
                                        const Eigen::MatrixXd& inputW,
                                        const Eigen::MatrixXd& feedbackW,
                                        const Eigen::MatrixXd& peep, int I, int O,
                                        bool cellsPeep, bool gatesPeep, bool gatesRecPeep,
                                        const Eigen::MatrixXd& gatesCache,
                                        const Eigen::MatrixXd& cellStates,
                                        const Eigen::MatrixXd& cellsIn,
                                        long invSubSamplingRatio, LstmDerivs& d) {
    // :534-543 width tolerance -> `input` is (I x InputSeq.rows()) (transposed).
    Eigen::MatrixXd input;
    if (InputSeq.cols() > I) {
        input = InputSeq.leftCols(I).transpose();
    } else if (InputSeq.cols() < I) {
        Eigen::MatrixXd filler = Eigen::MatrixXd::Zero(I - InputSeq.cols(), InputSeq.rows());
        input.resize(I, InputSeq.rows());
        input << InputSeq.transpose(), filler;
    } else {
        input = InputSeq.transpose();
    }
    Eigen::MatrixXd output = outputSeq.transpose();   // O x T

    Eigen::MatrixXd deltasPreviousLayer = Eigen::MatrixXd::Zero(input.cols(), I);
    Eigen::MatrixXd deltasFeedback = Eigen::MatrixXd::Zero(1, O);
    Eigen::MatrixXd deltasInputGate = Eigen::MatrixXd::Zero(1, O);
    Eigen::MatrixXd tmpDeltasInputGate = Eigen::MatrixXd::Zero(1, O);
    Eigen::MatrixXd deltasForgetGate = Eigen::MatrixXd::Zero(1, O);
    Eigen::MatrixXd tmpDeltasForgetGate = Eigen::MatrixXd::Zero(1, O);
    Eigen::MatrixXd deltasCells;
    Eigen::MatrixXd deltasOutputGate = Eigen::MatrixXd::Zero(1, O);
    Eigen::MatrixXd tmpDeltasOutputGate = Eigen::MatrixXd::Zero(1, O);
    Eigen::MatrixXd epsilonCells;
    Eigen::MatrixXd epsilonState;
    Eigen::MatrixXd tmpEpsilonState = Eigen::MatrixXd::Zero(1, O);
    Eigen::MatrixXd epsilonForget;
    Eigen::MatrixXd inputWeightsDerivatives = Eigen::MatrixXd::Zero(I, 4 * O);
    Eigen::MatrixXd feedbackWeightsDerivatives = Eigen::MatrixXd::Zero(O, 4 * O);
    Eigen::MatrixXd peepWeightDerivatives = Eigen::MatrixXd::Zero(12, O);
    Eigen::MatrixXd biaisesDerivatives = Eigen::MatrixXd::Zero(1, 4 * O);

    // Cache block accessors (post-activation): _Gates.block(row, k*O, 1, O).
    auto gRow = [&](long row, int block) { return gatesCache.block(row, (long)block * O, 1, O); };

    Eigen::MatrixXd testM(4 * O, input.cols());
    long long linesNb = deltas.rows();
    for (long long row = linesNb - 1; row >= 0; --row) {
        epsilonCells = deltas.row(row) + deltasFeedback;   // :580

        // :582-585 deltasOutputGate.
        tmpDeltasOutputGate = cellsIn.row(row).cwiseProduct(epsilonCells);
        if (gatesPeep)
            tmpDeltasOutputGate.noalias() +=
                (deltasInputGate.array() * peep.row(5).array() + deltasForgetGate.array() * peep.row(8).array()).matrix();
        if (gatesRecPeep)
            tmpDeltasOutputGate.noalias() += (deltasOutputGate.array() * peep.row(11).array()).matrix();
        deltasOutputGate = (gRow(row, 2).unaryExpr(CwiseActFunctionDeriv<GatesFunction>()).array() * tmpDeltasOutputGate.array()).matrix();

        // :595-605 output-gate deriv accumulation.
        inputWeightsDerivatives.block(0, 2 * O, I, O).noalias() += input.col(row) * deltasOutputGate;
        if (row > 0) {
            feedbackWeightsDerivatives.block(0, 2 * O, O, O).noalias() += output.col(row - 1) * deltasOutputGate;
            if (gatesRecPeep)
                peepWeightDerivatives.row(11).noalias() += (deltasOutputGate.array() * gRow(row - 1, 2).array()).matrix();
        }
        if (cellsPeep) peepWeightDerivatives.row(2).noalias() += deltasOutputGate.cwiseProduct(cellStates.row(row));
        if (gatesPeep) {
            peepWeightDerivatives.row(9).noalias() += deltasOutputGate.cwiseProduct(gRow(row, 0));
            peepWeightDerivatives.row(10).noalias() += deltasOutputGate.cwiseProduct(gRow(row, 1));
        }
        biaisesDerivatives.block(0, 2 * O, 1, O) += deltasOutputGate;

        // :607-611 epsilonForget from NEXT-step forget gate.
        if (row != linesNb - 1) {
            epsilonForget = gRow(row + 1, 1).array() * epsilonState.array();
        } else {
            epsilonForget = Eigen::MatrixXd::Zero(1, O);
        }

        // :613-614 epsilonState.
        if (cellsPeep)
            tmpEpsilonState.noalias() =
                (deltasInputGate.array() * peep.row(0).array() + deltasForgetGate.array() * peep.row(1).array() + deltasOutputGate.array() * peep.row(2).array()).matrix();
        epsilonState.noalias() = (gRow(row, 2).array() * cellsIn.row(row).unaryExpr(CwiseActFunctionDeriv<Identity>()).array() * epsilonCells.array()).matrix() + epsilonForget + tmpEpsilonState;

        // :616 deltasCells.
        deltasCells.noalias() = (gRow(row, 0).array() * gRow(row, 3).unaryExpr(CwiseActFunctionDeriv<Maxmin2>()).array() * epsilonState.array()).matrix();
        inputWeightsDerivatives.block(0, 3 * O, I, O).noalias() += input.col(row) * deltasCells;
        if (row > 0) feedbackWeightsDerivatives.block(0, 3 * O, O, O).noalias() += output.col(row - 1) * deltasCells;
        biaisesDerivatives.block(0, 3 * O, 1, O) += deltasCells;

        // :624-657 deltasForgetGate (row>0 with cellstate; row==0 no cellstate).
        Eigen::MatrixXd deltasForgetGatetmp = deltasForgetGate;   // :624 save (read at :660)
        if (row > 0) {
            tmpDeltasForgetGate.noalias() = cellStates.row(row - 1).cwiseProduct(epsilonState);
            if (gatesPeep)
                tmpDeltasForgetGate.noalias() += (deltasInputGate.array() * peep.row(4).array() + deltasOutputGate.array() * peep.row(10).array()).matrix();
            if (gatesRecPeep)
                tmpDeltasForgetGate.noalias() += (deltasForgetGate.array() * peep.row(7).array()).matrix();
            deltasForgetGate = (gRow(row, 1).unaryExpr(CwiseActFunctionDeriv<GatesFunction>()).array() * tmpDeltasForgetGate.array()).matrix();
            inputWeightsDerivatives.block(0, 1 * O, I, O).noalias() += input.col(row) * deltasForgetGate;
            feedbackWeightsDerivatives.block(0, 1 * O, O, O).noalias() += output.col(row - 1) * deltasForgetGate;
            if (cellsPeep) peepWeightDerivatives.row(1).noalias() += deltasForgetGate.cwiseProduct(cellStates.row(row - 1));
            if (gatesPeep) {
                peepWeightDerivatives.row(6).noalias() += deltasForgetGate.cwiseProduct(gRow(row - 1, 0));
                peepWeightDerivatives.row(8).noalias() += deltasForgetGate.cwiseProduct(gRow(row - 1, 2));
            }
            if (gatesRecPeep) peepWeightDerivatives.row(7).noalias() += deltasForgetGate.cwiseProduct(gRow(row - 1, 1));
            biaisesDerivatives.block(0, 1 * O, 1, O) += deltasForgetGate;
        } else {
            tmpDeltasForgetGate = Eigen::MatrixXd::Zero(1, O);
            if (gatesPeep)
                tmpDeltasForgetGate.noalias() += (deltasInputGate.array() * peep.row(4).array() + deltasOutputGate.array() * peep.row(10).array()).matrix();
            if (gatesRecPeep)
                tmpDeltasForgetGate.noalias() += (deltasForgetGate.array() * peep.row(7).array()).matrix();
            deltasForgetGate = gRow(row, 1).unaryExpr(CwiseActFunctionDeriv<GatesFunction>()).cwiseProduct(tmpDeltasForgetGate);
            inputWeightsDerivatives.block(0, 1 * O, I, O).noalias() += input.col(row) * deltasForgetGate;
            biaisesDerivatives.block(0, 1 * O, 1, O) += deltasForgetGate;
        }

        // :659-682 deltasInputGate (reads deltasForgetGatetmp at :660).
        tmpDeltasInputGate.noalias() = gRow(row, 3).cwiseProduct(epsilonState);
        if (gatesPeep)
            tmpDeltasInputGate.noalias() += (deltasForgetGatetmp.array() * peep.row(6).array() + deltasOutputGate.array() * peep.row(9).array()).matrix();
        if (gatesRecPeep)
            tmpDeltasInputGate.noalias() += (deltasInputGate.array() * peep.row(3).array()).matrix();
        deltasInputGate = gRow(row, 0).unaryExpr(CwiseActFunctionDeriv<GatesFunction>()).cwiseProduct(tmpDeltasInputGate);
        inputWeightsDerivatives.block(0, 0, I, O).noalias() += input.col(row) * deltasInputGate;
        if (row > 0) {
            feedbackWeightsDerivatives.block(0, 0, O, O).noalias() += output.col(row - 1) * deltasInputGate;
            if (cellsPeep) peepWeightDerivatives.row(0).noalias() += deltasInputGate.cwiseProduct(cellStates.row(row - 1));
            if (gatesPeep) {
                peepWeightDerivatives.row(4).noalias() += deltasInputGate.cwiseProduct(gRow(row - 1, 1));
                peepWeightDerivatives.row(5).noalias() += deltasInputGate.cwiseProduct(gRow(row - 1, 2));
            }
            if (gatesRecPeep) peepWeightDerivatives.row(3).noalias() += deltasInputGate.cwiseProduct(gRow(row - 1, 0));
        }
        biaisesDerivatives.block(0, 0, 1, O) += deltasInputGate;

        // :688-704 stacked testM column + deltasFeedback via matSeq.
        Eigen::MatrixXd test(4 * O, 1);
        test << deltasInputGate.transpose(), deltasForgetGate.transpose(), deltasOutputGate.transpose(), deltasCells.transpose();
        testM.col(row) = test;
        deltasFeedback = matSeq(feedbackW, test).transpose();   // :703 (_FeedbackWeights*test)^T
    }
    deltasPreviousLayer = matSeq(inputW, testM).transpose();    // :707 (_InputWeights*testM)^T

    if (invSubSamplingRatio > 1) {
        inputWeightsDerivatives *= (double)invSubSamplingRatio;
        feedbackWeightsDerivatives *= (double)invSubSamplingRatio;
        peepWeightDerivatives *= (double)invSubSamplingRatio;
        biaisesDerivatives *= (double)invSubSamplingRatio;
    }
    d.inputW += inputWeightsDerivatives;
    d.feedbackW += feedbackWeightsDerivatives;
    d.peep += peepWeightDerivatives;
    d.bias += biaisesDerivatives;
    d.nbFedBackward += (long)linesNb;
    return deltasPreviousLayer;
}

// legacy: LSTMLayer.cpp:725-732 (feedBackwardReverse): reverse input/output/deltas,
// call the FORWARD-order feedBackward, reverse the returned deltas. The caches are
// stored time-reversed after feedForwardReverse and consumed AS-IS (do not
// un-reverse) -- so the caller passes the reversed-storage caches here unchanged.
static Eigen::MatrixXd lstmBackwardReverseLoop(const Eigen::MatrixXd& InputSeq,
                                               const Eigen::MatrixXd& outputSeq,
                                               const Eigen::MatrixXd& deltas,
                                               const Eigen::MatrixXd& inputW,
                                               const Eigen::MatrixXd& feedbackW,
                                               const Eigen::MatrixXd& peep, int I, int O,
                                               bool cellsPeep, bool gatesPeep, bool gatesRecPeep,
                                               const Eigen::MatrixXd& gatesCache,
                                               const Eigen::MatrixXd& cellStates,
                                               const Eigen::MatrixXd& cellsIn,
                                               long invSubSamplingRatio, LstmDerivs& d) {
    Eigen::MatrixXd InputSeqRev = InputSeq.colwise().reverse();
    Eigen::MatrixXd outputSeqRev = outputSeq.colwise().reverse();
    Eigen::MatrixXd deltasRev = deltas.colwise().reverse();
    Eigen::MatrixXd dpl = lstmBackwardLoop(InputSeqRev, outputSeqRev, deltasRev, inputW, feedbackW,
                                           peep, I, O, cellsPeep, gatesPeep, gatesRecPeep,
                                           gatesCache, cellStates, cellsIn, invSubSamplingRatio, d);
    return dpl.colwise().reverse();
}

// The dense (NeuronLayer) derivative blocks + count. Weights I x O, bias 1 x O.
struct DenseDerivs {
    Eigen::MatrixXd weights, bias;
    long nbFedBackward = 0;
    void init(int I, int O) {
        weights = Eigen::MatrixXd::Zero(I, O);
        bias = Eigen::MatrixXd::Zero(1, O);
        nbFedBackward = 0;
    }
};

// legacy: NeuronLayer.cpp:151-207 (feedBackward), transcribed VERBATIM. Width
// tolerance (:158-166: cols>I -> leftCols transpose; cols<I -> ZERO-PAD to I rows).
// weightsDerivatives += input.col(jj)*deltas.row(jj) per row (:178-183); bias +=
// deltas.colwise().sum(); invSubSampling scaling (:192-195); += InputSeq.rows()
// (:199 -- NOTE InputSeq.rows(), not deltas.rows()). lastLayer -> deltas_out =
// deltas*_Weights.transpose() with NO activation deriv (the softmax+CE fusion lives
// in CostLaw); hidden -> *= Maxmin2'(InputSeq) on the LAYER INPUT (:204). deltas_out
// via matSeq (deltas * W^T -- k=O, the divergent GEMM at NN width).
static Eigen::MatrixXd denseBackwardLoop(const Eigen::MatrixXd& InputSeq,
                                         const Eigen::MatrixXd& deltas,
                                         const Eigen::MatrixXd& weights, int I, int O,
                                         long invSubSamplingRatio, bool lastLayer, DenseDerivs& d) {
    Eigen::MatrixXd input;
    if (InputSeq.cols() > I) {
        input = InputSeq.leftCols(I).transpose();
    } else if (InputSeq.cols() < I) {
        Eigen::MatrixXd filler = Eigen::MatrixXd::Zero(I - InputSeq.cols(), InputSeq.rows());
        input.resize(I, InputSeq.rows());
        input << InputSeq.transpose(), filler;
    } else {
        input = InputSeq.transpose();
    }
    Eigen::MatrixXd weightsDerivatives = Eigen::MatrixXd::Zero(I, O);
    Eigen::MatrixXd biaisesDerivatives = Eigen::MatrixXd::Zero(1, O);
    for (long jj = 0; jj < deltas.rows(); ++jj)
        weightsDerivatives.noalias() += input.col(jj) * deltas.row(jj);
    biaisesDerivatives.noalias() += deltas.colwise().sum();
    if (invSubSamplingRatio > 1) {
        weightsDerivatives *= (double)invSubSamplingRatio;
        biaisesDerivatives *= (double)invSubSamplingRatio;
    }
    d.weights += weightsDerivatives;
    d.bias += biaisesDerivatives;
    d.nbFedBackward += (long)InputSeq.rows();

    // W^T is (O x I); deltas (T x O) * W^T (O x I) -> (T x I). matSeq needs W^T.
    Eigen::MatrixXd wT = weights.transpose();
    Eigen::MatrixXd deltas_out = matSeq(deltas, wT);
    if (!lastLayer) {
        for (long r = 0; r < deltas_out.rows(); ++r)
            for (long c = 0; c < deltas_out.cols(); ++c)
                deltas_out(r, c) *= Maxmin2::deriv(InputSeq(r, c));   // :204 asinh' on the INPUT
    }
    return deltas_out;
}

// One backward layer step (LSTM or dense), mirroring NetLayerStep for the forward.
// `backward(input, output, deltasIn, invSub, lastLayer)` returns deltas_out and
// accumulates the layer's derivs into its bound Derivs struct.
struct NetBackStep {
    std::function<Eigen::MatrixXd(const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                                  const Eigen::MatrixXd&, long, bool)>
        backward;
};

// legacy: NeuralNetwork.hpp:251-300 (feedBackward). Reverse-order layer loop with the
// running invSubSamplingRatio + the SubSample/InvSubSample inversion. Each layer gets
// its retained forward input (Input for layer 0, layersOutput[jj-1] for jj>0) and its
// forward OUTPUT (layersOutput[jj] for hidden, outputSeq for the final layer);
// SubSampling[jj] > 1 subsamples the layer's input + output.topRows(deltaRows), the
// layer runs, its deltas_out is InvSubSampled, and invSubSamplingRatio *= ratio (fed
// into the NEXT layer up). `deltaRows` tracks the current delta row count (deltas for
// the seed layer, deltas_out.rows() thereafter). The steps close over the retained
// forward tensors; the driver only threads deltas + the ratio.
static Eigen::MatrixXd netBackwardLoop(const std::vector<long>& neuronNb,
                                       const std::vector<long>& subSampling,
                                       const std::vector<NetBackStep>& steps,
                                       const Eigen::MatrixXd& Input,
                                       const Eigen::MatrixXd& outputSeq,
                                       const std::vector<Eigen::MatrixXd>& layersOutput,
                                       const Eigen::MatrixXd& seedDeltas) {
    Eigen::MatrixXd deltas_out;
    long invSubSamplingRatio = 1;
    if (Input.rows() == 0) return deltas_out;
    const size_t L = neuronNb.size();
    if (L == 2) {
        if (subSampling[0] > 1) {
            Eigen::MatrixXd sub = subSampleLoop(subSampling[0], Input);
            deltas_out = steps[0].backward(sub.topRows(seedDeltas.rows()), outputSeq.topRows(seedDeltas.rows()),
                                           seedDeltas, invSubSamplingRatio, true);
            deltas_out = invSubSampleLoop(subSampling[0], deltas_out);
            invSubSamplingRatio *= subSampling[0];
        } else {
            deltas_out = steps[0].backward(Input, outputSeq, seedDeltas, invSubSamplingRatio, true);
        }
        return deltas_out;
    }
    for (size_t kk = 0; kk < L - 1; ++kk) {
        size_t jj = L - 2 - kk;
        const Eigen::MatrixXd& curDeltas = (kk == 0) ? seedDeltas : deltas_out;
        if (jj == 0) {
            if (subSampling[0] > 1) {
                Eigen::MatrixXd sub = subSampleLoop(subSampling[0], Input);
                deltas_out = steps[jj].backward(sub.topRows(curDeltas.rows()), layersOutput[jj].topRows(curDeltas.rows()),
                                                curDeltas, invSubSamplingRatio, true);
                deltas_out = invSubSampleLoop(subSampling[0], deltas_out);
                invSubSamplingRatio *= subSampling[0];
            } else {
                deltas_out = steps[jj].backward(Input, layersOutput[jj], curDeltas, invSubSamplingRatio, true);
            }
        } else if (jj == L - 2) {
            if (subSampling[jj] > 1) {
                Eigen::MatrixXd sub = subSampleLoop(subSampling[jj], layersOutput[jj - 1]);
                deltas_out = steps[jj].backward(sub.topRows(seedDeltas.rows()), outputSeq.topRows(seedDeltas.rows()),
                                                seedDeltas, invSubSamplingRatio, false);
                deltas_out = invSubSampleLoop(subSampling[jj], deltas_out);
                invSubSamplingRatio *= subSampling[jj];
            } else {
                deltas_out = steps[jj].backward(layersOutput[jj - 1], outputSeq, seedDeltas, invSubSamplingRatio, false);
            }
        } else {
            if (subSampling[jj] > 1) {
                Eigen::MatrixXd sub = subSampleLoop(subSampling[jj], layersOutput[jj - 1]);
                deltas_out = steps[jj].backward(sub.topRows(curDeltas.rows()), layersOutput[jj].topRows(curDeltas.rows()),
                                                curDeltas, invSubSamplingRatio, false);
                deltas_out = invSubSampleLoop(subSampling[jj], deltas_out);
                invSubSamplingRatio *= subSampling[jj];
            } else {
                deltas_out = steps[jj].backward(layersOutput[jj - 1], layersOutput[jj], curDeltas, invSubSamplingRatio, false);
            }
        }
    }
    return deltas_out;
}

// legacy: NeuralNetwork.hpp:302-351 (feedBackwardReverse). Byte-identical driver to
// netBackwardLoop -- only the per-layer step closures dispatch the REVERSE layer
// kernel (reversal lives inside the layer step). Kept separate to mirror the legacy's
// two copy-pasted methods.
static Eigen::MatrixXd netBackwardReverseLoop(const std::vector<long>& neuronNb,
                                              const std::vector<long>& subSampling,
                                              const std::vector<NetBackStep>& steps,
                                              const Eigen::MatrixXd& Input,
                                              const Eigen::MatrixXd& outputSeq,
                                              const std::vector<Eigen::MatrixXd>& layersOutput,
                                              const Eigen::MatrixXd& seedDeltas) {
    return netBackwardLoop(neuronNb, subSampling, steps, Input, outputSeq, layersOutput, seedDeltas);
}

// One LSTM sub-network's per-layer weights + forward caches + deriv accumulators,
// enough to run netBackwardLoop over it. Forward caches (gates/cellStates/cellsIn per
// layer, layersOutput per non-final layer) are captured during a forward pass that
// mirrors NeuralNetwork::feedForward's SubSample + layersOutput bookkeeping, so the
// backward can feed each layer its exact forward input/output (the retained
// _LayersOutput the real container holds). `reverse` picks fwd vs reverse LSTM
// kernels. All peepholes on (the real net's config).
struct LstmSubNet {
    std::vector<long> neuronNb, subSampling;
    std::vector<int> ins, outs;
    std::vector<Eigen::MatrixXd> iw, fw, pp, bs;   // per-layer unpacked weights
    std::vector<Eigen::MatrixXd> gcache, ccache, cicache;   // per-layer forward caches
    std::vector<Eigen::MatrixXd> layersOutput;     // non-final layer outputs (retained)
    std::vector<LstmDerivs> derivs;
    bool reverse = false;

    void build(const Eigen::VectorXd& flat, const std::vector<long>& nn,
               const std::vector<long>& ss, bool rev) {
        neuronNb = nn; subSampling = ss; reverse = rev;
        const size_t L = nn.size() - 1;
        ins.resize(L); outs.resize(L);
        for (size_t jj = 0; jj < L; ++jj) {
            ins[jj] = (int)(nn[jj] * (jj == 0 ? ss[0] : ss[jj]));
            outs[jj] = (int)nn[jj + 1];
        }
        // ins[0] already folds subSampling[0] (SubSample widens the layer-0 input);
        // ins[jj>0] folds subSampling[jj] the same way.
        iw.resize(L); fw.resize(L); pp.resize(L); bs.resize(L);
        derivs.resize(L);
        long pos = 0;
        for (size_t jj = 0; jj < L; ++jj) {
            int I = ins[jj], O = outs[jj];
            long nb = 4L * I * O + 4L * O * O + 12L * O + 4L * O;
            Eigen::VectorXd slice = flat.segment(pos, nb);
            pos += nb;
            unpackLstmWeights(slice, I, O, iw[jj], fw[jj], pp[jj], bs[jj]);
            derivs[jj].init(I, O);
        }
    }

    // Forward with cache capture (mirrors NeuralNetwork::feedForward + feedForwardReverse
    // layersOutput bookkeeping). Fills gcache/ccache/cicache per layer + layersOutput,
    // returns the final output (== _OutputForward or _OutputBackward).
    Eigen::MatrixXd forwardCapture(const Eigen::MatrixXd& Input) {
        const size_t L = neuronNb.size() - 1;
        gcache.assign(L, Eigen::MatrixXd());
        ccache.assign(L, Eigen::MatrixXd());
        cicache.assign(L, Eigen::MatrixXd());
        layersOutput.assign(L > 0 ? L - 1 : 0, Eigen::MatrixXd());
        Eigen::MatrixXd finalOut;
        auto runLayer = [&](size_t jj, const Eigen::MatrixXd& in, Eigen::MatrixXd& out) {
            int O = outs[jj];
            if (reverse) {
                Eigen::MatrixXd inRev = in.colwise().reverse(), outRev;
                lstmForwardLoopCache(inRev, iw[jj], fw[jj], pp[jj], bs[jj], O, true, true, true,
                                     gcache[jj], outRev, ccache[jj], cicache[jj]);
                out = outRev.colwise().reverse();  // caches STAY reversed (consumed as-is)
            } else {
                lstmForwardLoopCache(in, iw[jj], fw[jj], pp[jj], bs[jj], O, true, true, true,
                                     gcache[jj], out, ccache[jj], cicache[jj]);
            }
        };
        if (L == 1) {
            Eigen::MatrixXd in = (subSampling[0] > 1) ? subSampleLoop(subSampling[0], Input) : Input;
            runLayer(0, in, finalOut);
            return finalOut;
        }
        for (size_t jj = 0; jj < L; ++jj) {
            Eigen::MatrixXd in;
            if (jj == 0) in = (subSampling[0] > 1) ? subSampleLoop(subSampling[0], Input) : Input;
            else in = (subSampling[jj] > 1) ? subSampleLoop(subSampling[jj], layersOutput[jj - 1]) : layersOutput[jj - 1];
            if (jj == L - 1) runLayer(jj, in, finalOut);
            else runLayer(jj, in, layersOutput[jj]);
        }
        return finalOut;
    }

    // Backward via netBackwardLoop, threading each layer's captured caches through a
    // NetBackStep closure. `Input` is the ORIGINAL sub-net input (pre-SubSample);
    // outputSeq is the sub-net final output; seedDeltas is the delta half from the
    // output-net backward. Accumulates into `derivs`; returns deltasPreviousLayer.
    Eigen::MatrixXd backward(const Eigen::MatrixXd& Input, const Eigen::MatrixXd& outputSeq,
                             const Eigen::MatrixXd& seedDeltas) {
        const size_t L = neuronNb.size() - 1;
        std::vector<NetBackStep> steps(L);
        for (size_t jj = 0; jj < L; ++jj) {
            steps[jj].backward = [this, jj](const Eigen::MatrixXd& lin, const Eigen::MatrixXd& lout,
                                            const Eigen::MatrixXd& din, long invSub, bool last) {
                int I = ins[jj], O = outs[jj];
                if (reverse) {
                    return lstmBackwardReverseLoop(lin, lout, din, iw[jj], fw[jj], pp[jj], I, O,
                                                   true, true, true, gcache[jj], ccache[jj], cicache[jj],
                                                   invSub, derivs[jj]);
                }
                return lstmBackwardLoop(lin, lout, din, iw[jj], fw[jj], pp[jj], I, O,
                                        true, true, true, gcache[jj], ccache[jj], cicache[jj],
                                        invSub, derivs[jj]);
            };
        }
        if (reverse)
            return netBackwardReverseLoop(neuronNb, subSampling, steps, Input, outputSeq, layersOutput, seedDeltas);
        return netBackwardLoop(neuronNb, subSampling, steps, Input, outputSeq, layersOutput, seedDeltas);
    }

    // Nx2 flat deriv assembly matching NeuralNetwork::getWeightsDerivatives (vertical
    // hcat of per-layer LSTMLayer::getWeightsDerivatives, LSTMLayer.cpp:251-295): flat
    // block order InputWeights (col-major) -> FeedbackWeights -> PeepWeight -> Biaises,
    // col0 = deriv, col1 = nbFedBackward replicated.
    Eigen::MatrixXd flatDerivs() const {
        std::vector<double> col0, col1;
        for (size_t jj = 0; jj < derivs.size(); ++jj) {
            const LstmDerivs& d = derivs[jj];
            long cnt = d.nbFedBackward;
            const Eigen::MatrixXd& iwd = d.inputW;   // I x 4O col-major
            for (long c = 0; c < iwd.cols(); ++c)
                for (long r = 0; r < iwd.rows(); ++r) { col0.push_back(iwd(r, c)); col1.push_back((double)cnt); }
            const Eigen::MatrixXd& fwd = d.feedbackW;
            for (long c = 0; c < fwd.cols(); ++c)
                for (long r = 0; r < fwd.rows(); ++r) { col0.push_back(fwd(r, c)); col1.push_back((double)cnt); }
            const Eigen::MatrixXd& ppd = d.peep;     // 12 x O col-major
            for (long c = 0; c < ppd.cols(); ++c)
                for (long r = 0; r < ppd.rows(); ++r) { col0.push_back(ppd(r, c)); col1.push_back((double)cnt); }
            const Eigen::MatrixXd& bsd = d.bias;     // 1 x 4O
            for (long c = 0; c < bsd.cols(); ++c) { col0.push_back(bsd(0, c)); col1.push_back((double)cnt); }
        }
        Eigen::MatrixXd out((long)col0.size(), 2);
        for (long k = 0; k < (long)col0.size(); ++k) { out(k, 0) = col0[k]; out(k, 1) = col1[k]; }
        return out;
    }
};

// One dense (output-MLP) sub-network. Same shape as LstmSubNet but with dense layers;
// the output net is fed HCAT(_OutputForward|_OutputBackward) as input (feedBackward-
// Double, NeuralNetwork.hpp:353-362). Captures layersOutput for the hidden asinh deriv.
struct DenseSubNet {
    std::vector<long> neuronNb, subSampling;
    std::vector<int> ins, outs;
    std::vector<Eigen::MatrixXd> dw, db;
    std::vector<Eigen::MatrixXd> layersOutput;
    std::vector<DenseDerivs> derivs;

    void build(const Eigen::VectorXd& flat, const std::vector<long>& nn, const std::vector<long>& ss) {
        neuronNb = nn; subSampling = ss;
        const size_t L = nn.size() - 1;
        ins.resize(L); outs.resize(L); dw.resize(L); db.resize(L); derivs.resize(L);
        long pos = 0;
        for (size_t jj = 0; jj < L; ++jj) {
            int I = (int)(nn[jj] * (jj == 0 ? ss[0] : ss[jj])), O = (int)nn[jj + 1];
            ins[jj] = I; outs[jj] = O;
            long nb = (long)O * (I + 1);
            Eigen::VectorXd slice = flat.segment(pos, nb);
            pos += nb;
            dw[jj].resize(I, O); db[jj].resize(1, O);
            for (int c = 0; c < O; ++c)
                for (int r = 0; r < I; ++r) dw[jj](r, c) = slice((long)c * I + r);
            for (int c = 0; c < O; ++c) db[jj](0, c) = slice((long)O * I + c);
            derivs[jj].init(I, O);
        }
    }

    // Forward over an already-hcat'd input (first|second), capturing layersOutput.
    Eigen::MatrixXd forwardCapture(const Eigen::MatrixXd& Input) {
        const size_t L = neuronNb.size() - 1;
        layersOutput.assign(L > 0 ? L - 1 : 0, Eigen::MatrixXd());
        Eigen::MatrixXd finalOut;
        if (L == 1) {
            Eigen::MatrixXd in = (subSampling[0] > 1) ? subSampleLoop(subSampling[0], Input) : Input;
            finalOut = denseForwardLoop(in, dw[0], db[0], true);
            return finalOut;
        }
        for (size_t jj = 0; jj < L; ++jj) {
            Eigen::MatrixXd in;
            if (jj == 0) in = (subSampling[0] > 1) ? subSampleLoop(subSampling[0], Input) : Input;
            else in = (subSampling[jj] > 1) ? subSampleLoop(subSampling[jj], layersOutput[jj - 1]) : layersOutput[jj - 1];
            bool last = (jj == L - 1);
            Eigen::MatrixXd o = denseForwardLoop(in, dw[jj], db[jj], last);
            if (last) finalOut = o; else layersOutput[jj] = o;
        }
        return finalOut;
    }

    // feedBackwardDouble backward (NeuralNetwork.hpp:353-362 -> feedBackward :251-300):
    // the input is the hcat(first|second); returns the split-ready deltas_out (rows x
    // neuronNb[0]) whose left half feeds the fwd LSTM stack, right half the bwd stack.
    Eigen::MatrixXd backward(const Eigen::MatrixXd& hcatInput, const Eigen::MatrixXd& outputSeq,
                             const Eigen::MatrixXd& seedDeltas) {
        const size_t L = neuronNb.size() - 1;
        std::vector<NetBackStep> steps(L);
        for (size_t jj = 0; jj < L; ++jj) {
            steps[jj].backward = [this, jj](const Eigen::MatrixXd& lin, const Eigen::MatrixXd&,
                                            const Eigen::MatrixXd& din, long invSub, bool last) {
                return denseBackwardLoop(lin, din, dw[jj], ins[jj], outs[jj], invSub, last, derivs[jj]);
            };
        }
        return netBackwardLoop(neuronNb, subSampling, steps, hcatInput, outputSeq, layersOutput, seedDeltas);
    }

    Eigen::MatrixXd flatDerivs() const {
        std::vector<double> col0, col1;
        for (size_t jj = 0; jj < derivs.size(); ++jj) {
            const DenseDerivs& d = derivs[jj];
            long cnt = d.nbFedBackward;
            const Eigen::MatrixXd& wd = d.weights;   // I x O col-major (NeuronLayer.cpp:105)
            for (long c = 0; c < wd.cols(); ++c)
                for (long r = 0; r < wd.rows(); ++r) { col0.push_back(wd(r, c)); col1.push_back((double)cnt); }
            const Eigen::MatrixXd& bd = d.bias;      // 1 x O
            for (long c = 0; c < bd.cols(); ++c) { col0.push_back(bd(0, c)); col1.push_back((double)cnt); }
        }
        Eigen::MatrixXd out((long)col0.size(), 2);
        for (long k = 0; k < (long)col0.size(); ++k) { out(k, 0) = col0[k]; out(k, 1) = col1[k]; }
        return out;
    }
};

// legacy: BLSTMNeuralNetwork.cpp:439-460 (feedBackward) + getWeightsDerivatives
// (:255-276). The whole-net plain (non-windowed) backward reimpl: run the forward net
// (fwd LSTM + bwd LSTM reverse) into _OutputForward/_OutputBackward, output net over
// hcat(fwd|bwd); seed deltas via the REAL CostLaw::computeDeltas (passed in, since
// CostLaw is bit-portable for polynomial laws); feedBackwardDouble; then (unless
// _BackPropOutputNetworkOnly) fwd feedBackward on deltas.leftCols(half) + bwd
// feedBackwardReverse on deltas.rightCols(half), with the leftCols(inputSize) crop
// gate (:452-454). Assembles the Nx2 flat derivs (fwd|bwd|output|stats-tail:
// [0|1|0|1] so the mean/std slots never move).
struct BlstmBack {
    LstmSubNet fwd, bwd;
    DenseSubNet out;
    long statsRows = 0;   // _NormalizeInputMean.rows() (== lstm input size)
    int fwdInputSize = 0;
    long outNetRatio = 1;
    bool outputNetworkOnly = false;

    // Build from the flat weight vector (fwd | bwd | output blocks, as the real net
    // packs). lstmNN/lstmSS the LSTM sub-net dims; outNN/outSS the output net dims.
    void build(const Eigen::VectorXd& flat, const std::vector<long>& lstmNN,
               const std::vector<long>& lstmSS, const std::vector<long>& outNN,
               const std::vector<long>& outSS) {
        long fwdNb = 0;
        for (size_t jj = 0; jj + 1 < lstmNN.size(); ++jj) {
            long I = lstmNN[jj] * (jj == 0 ? lstmSS[0] : lstmSS[jj]), O = lstmNN[jj + 1];
            fwdNb += 4L * I * O + 4L * O * O + 12L * O + 4L * O;
        }
        long outNbW = 0;
        for (size_t jj = 0; jj + 1 < outNN.size(); ++jj) {
            long I = outNN[jj] * (jj == 0 ? outSS[0] : outSS[jj]), O = outNN[jj + 1];
            outNbW += O * (I + 1);
        }
        fwd.build(flat.head(fwdNb), lstmNN, lstmSS, false);
        bwd.build(flat.segment(fwdNb, fwdNb), lstmNN, lstmSS, true);
        out.build(flat.segment(2 * fwdNb, outNbW), outNN, outSS);
        statsRows = lstmNN[0];
        fwdInputSize = (int)lstmNN[0];
        outNetRatio = 1;
        for (size_t jj = 0; jj + 1 < outSS.size() + 0 && jj < outSS.size(); ++jj) outNetRatio *= outSS[jj];
    }

    // Forward (plain) into outForward/outBackward/output, capturing all caches.
    void forward(const Eigen::MatrixXd& input, Eigen::MatrixXd& outForward,
                 Eigen::MatrixXd& outBackward, Eigen::MatrixXd& output, Eigen::MatrixXd& hcat) {
        Eigen::MatrixXd fwdIn = input, bwdIn = input;
        if (fwd.subSampling[0] > 1 && fwdInputSize < input.cols()) {
            fwdIn = input.leftCols(fwdInputSize);
            bwdIn = input.leftCols(fwdInputSize);
        }
        outForward = fwd.forwardCapture(fwdIn);
        outBackward = bwd.forwardCapture(bwdIn);
        hcat.resize(outForward.rows(), out.neuronNb[0]);
        hcat << outForward, outBackward;
        output = out.forwardCapture(hcat);
    }

    // Backward given the forward caches + the seed deltas (from CostLaw). `input` is
    // the ORIGINAL BLSTM input; outForward/outBackward/hcat/output from forward().
    void backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& outForward,
                  const Eigen::MatrixXd& outBackward, const Eigen::MatrixXd& hcat,
                  const Eigen::MatrixXd& output, Eigen::MatrixXd& seedDeltas) {
        Eigen::MatrixXd deltas = out.backward(hcat, output, seedDeltas);   // rows x neuronNb[0]
        if (outputNetworkOnly) return;
        long half = deltas.cols() / 2;
        Eigen::MatrixXd leftD = deltas.leftCols(half), rightD = deltas.rightCols(half);
        Eigen::MatrixXd fwdIn = input, bwdIn = input;
        if (fwd.subSampling[0] > 1 && fwdInputSize < input.cols()) {
            fwdIn = input.leftCols(fwdInputSize);
            bwdIn = input.leftCols(fwdInputSize);
        }
        fwd.backward(fwdIn, outForward, leftD);
        bwd.backward(bwdIn, outBackward, rightD);
    }

    // Nx2 flat derivs matching BLSTMNeuralNetwork::getWeightsDerivatives (:255-276):
    // fwd|bwd|output|[Zero|Ones|Zero|Ones] stats tail. If outNetRatio > 1 the fwd/bwd
    // LSTM col0 is scaled by it (:266-269).
    Eigen::MatrixXd flatDerivs() const {
        Eigen::MatrixXd f = fwd.flatDerivs(), b = bwd.flatDerivs(), o = out.flatDerivs();
        if (outNetRatio > 1) {
            f.col(0) *= (double)outNetRatio;
            b.col(0) *= (double)outNetRatio;
        }
        long total = f.rows() + b.rows() + o.rows() + 2 * statsRows;
        Eigen::MatrixXd all(total, 2);
        long pos = 0;
        all.block(pos, 0, f.rows(), 2) = f; pos += f.rows();
        all.block(pos, 0, b.rows(), 2) = b; pos += b.rows();
        all.block(pos, 0, o.rows(), 2) = o; pos += o.rows();
        for (long k = 0; k < statsRows; ++k) { all(pos, 0) = 0.0; all(pos, 1) = 1.0; ++pos; }  // mean
        for (long k = 0; k < statsRows; ++k) { all(pos, 0) = 0.0; all(pos, 1) = 1.0; ++pos; }  // std
        return all;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <output_dir>\n";
        return 1;
    }
    std::string out = argv[1];
    if (!out.empty() && out.back() != '/') out += '/';
    std::string wav = out + "excerpt_2ch_8k.wav";

    // Fixture-hygiene fix (Phase 2b Task 5 review, Finding 3): committed VRCTS xml
    // goldens embed `wav`'s directory (`out`, the extractor's throwaway
    // tempfile.TemporaryDirectory()) in the `path=` attribute, which churns on every
    // regeneration since `out` is a fresh random path each run. `path=`/`name=` are
    // OPAQUE CALLER INPUTS to the real `Segmentation::toFile_VRCTS` (T4 review), so
    // swapping in a FIXED literal path weakens nothing. `sf_open` (AudioStruct's
    // ctor) still needs a real, readable file at that path, so the excerpt wav is
    // copied there once per harness run (idempotent, deterministic bytes) and a
    // dedicated CorpusItem/AudioStruct pair reads from the stable path instead of
    // `wav`/`item` for VRCTS-dumping blocks only -- every other dump keeps using the
    // real `wav`/`item` untouched.
    std::string vrctsWav = "/tmp/speech_oracle_harness_vrcts_fixture_audio.wav";
    {
        std::ifstream src(wav, std::ios::binary);
        std::ofstream dst(vrctsWav, std::ios::binary | std::ios::trunc);
        dst << src.rdbuf();
    }

    // Phase 2 Task 1: real-net config + weights paths. Supplied as argv[2]/argv[3]
    // (absolute, from the extractor) so they resolve regardless of the runtime cwd;
    // the repo-relative fallbacks assume the harness is run from tools/oracle_harness.
    std::string nnConfigPath = (argc > 2)
        ? std::string(argv[2])
        : std::string("../../tests/reference_data/phase0/1_worker_1.config");
    std::string nnWeightsPath = (argc > 3)
        ? std::string(argv[3])
        : std::string("../../tests/reference_data/phase0/NNweights_config1.bin");

    // Phase 2b Task 4: TDC-prefixed config, same argv-override convention as
    // nnConfigPath/nnWeightsPath above (absolute from the extractor; repo-relative
    // fallback assumes cwd == tools/oracle_harness).
    std::string tdcConfigPath = (argc > 4)
        ? std::string(argv[4])
        : std::string("../../tests/reference_data/phase2b/tdc.config");

    // Phase 2b Task 5: LTSV-prefixed configs, same argv-override convention.
    std::string ltsvConfigPath = (argc > 5)
        ? std::string(argv[5])
        : std::string("../../tests/reference_data/phase2b/ltsv.config");
    std::string ltsvDctConfigPath = (argc > 6)
        ? std::string(argv[6])
        : std::string("../../tests/reference_data/phase2b/ltsv_dct.config");
    std::string ltsvTinyConfigPath = (argc > 7)
        ? std::string(argv[7])
        : std::string("../../tests/reference_data/phase2b/ltsv_tiny.config");
    // Phase 2b Task 5 review, Finding 1: power-scale (is_log_mel=false) LTSV variant
    // -- see the LtsvProbe powermel block below for why this is needed.
    std::string ltsvPowermelConfigPath = (argc > 8)
        ? std::string(argv[8])
        : std::string("../../tests/reference_data/phase2b/ltsv_powermel.config");

    // Phase 2b Task 6: BLSTM-prefixed signal config, same argv-override convention.
    std::string signalConfigPath = (argc > 9)
        ? std::string(argv[9])
        : std::string("../../tests/reference_data/phase2b/signal.config");

    int dumps = 0;

    // --- Audio stages ---------------------------------------------------------
    // legacy: AudioStruct.cpp:36-137 (file_type==0 ctor: libsndfile read + offset
    // skip round(rate*offset) + duration cap round(rate*dur+1) + per-channel
    // (2*RMS + max)/2 normalization).
    CorpusItem item(wav, "", "RUS", "RU", 0, 0, 1.0);
    AudioStruct audio(OFFSET_SEC, MAX_DUR_SEC, 0, item);
    // Stable-path CorpusItem for VRCTS dumps only (Finding 3) -- same wav bytes, a
    // FIXED path so the `path=` attribute in committed VRCTS xml no longer churns.
    CorpusItem itemVrcts(vrctsWav, "", "RUS", "RU", 0, 0, 1.0);
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

    // --- libm canaries (CI portability gate) ---------------------------------
    // Record the exact transcendental results the parity paths depend on, evaluated
    // EXACTLY as the code above does them (cos in the hamming window + DCT table, log
    // in the mel band + fmath table build, exp in Mel2Hz). Dump 2 x N (row0 = input,
    // row1 = output) with a FIXED column layout the Rust test hardcodes: 10 cos, then
    // 5 log, then 3 exp (18 total). The Rust golden loader recomputes each with its
    // own libm; if all bits match, goldens assert bit-for-bit (this oracle env); if
    // not (e.g. glibc), the transcendental-dependent goldens assert <= 4 ULP instead.
    {
        const double PI = 3.14159265358979323846264338327;  // _PI (Constants.h:15)
        std::vector<double> in;
        std::vector<double> outv;
        // cos: hamming-257 window arg 2*PI/256*k (getWindowingCoefficients, Helpers.hpp:230).
        for (int k : {1, 27, 64, 128, 200, 255}) {
            double x = 2.0 * PI / 256.0 * (double) k;
            in.push_back(x);
            outv.push_back(std::cos(x));
        }
        // cos: DCT table arg PI/29*(n+0.5)*m (MelFilterBank ctor, main.cpp:785).
        for (auto nm : {std::pair<int, int>{0, 1}, {7, 3}, {28, 12}, {13, 7}}) {
            double x = PI / 29.0 * ((double) nm.first + 0.5) * (double) nm.second;
            in.push_back(x);
            outv.push_back(std::cos(x));
        }
        // log (f64): mel-band 1+f/700 (Hz2Mel), raw-band floor 1e-24, table anchors.
        for (double x : {1.0 + 64.0 / 700.0, 1.0 + 3800.0 / 700.0, 1e-24, 0.5,
                         1.0 + 1023.0 / 2048.0}) {
            in.push_back(x);
            outv.push_back(std::log(x));
        }
        // exp (f64): Mel2Hz arg mel/1125 (MelFilterBank.cpp:367), mel from Hz2Mel.
        const double minMel = 1125.0 * std::log(1.0 + 64.0 / 700.0);
        const double maxMel = 1125.0 * std::log(1.0 + 3800.0 / 700.0);
        for (double mel : {minMel, maxMel, (minMel + maxMel) / 2.0}) {
            double x = mel / 1125.0;
            in.push_back(x);
            outv.push_back(std::exp(x));
        }
        const Eigen::Index n = (Eigen::Index) in.size();
        Eigen::MatrixXd canaries(2, n);
        for (Eigen::Index k = 0; k < n; ++k) {
            canaries(0, k) = in[(size_t) k];
            canaries(1, k) = outv[(size_t) k];
        }
        Matrix2BinaryFile(out + "libm_canaries.bin", canaries);
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

    // --- InputStatistics (Task 10) --------------------------------------------
    // legacy: InputStatistics.cpp:6-51 (batch ctor + update/merge), REAL compiled
    // translation unit (no transcription). Batch stats over the full synth_20x50
    // matrix, then a merge of two batches built over the row splits [0,7) and
    // [7,20) -- exercises the non-trivial n1,n2>0 merge branch (cpp:39-46) with the
    // per-element resqrt op order. _MeanValue/_StandardDeviation are Eigen 1xD row
    // vectors; dumped as 1x50 matrices, _NbOfValues (long) as a 1x1 matrix.
    {
        Eigen::MatrixXd synth(20, 50);
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 50; ++j) {
                synth(i, j) = ((i * 7 + j * 13) % 100) / 100.0;
            }
        }

        InputStatistics batch(synth);
        Matrix2BinaryFile(out + "stats_batch_mean.bin", batch._MeanValue);
        ++dumps;
        Matrix2BinaryFile(out + "stats_batch_std.bin", batch._StandardDeviation);
        ++dumps;
        {
            Eigen::MatrixXd n(1, 1);
            n(0, 0) = (double) batch._NbOfValues;
            Matrix2BinaryFile(out + "stats_batch_n.bin", n);
            ++dumps;
        }

        Eigen::MatrixXd part1 = synth.block(0, 0, 7, 50);
        Eigen::MatrixXd part2 = synth.block(7, 0, 13, 50);
        InputStatistics merged(part1);
        InputStatistics other(part2);
        merged.update(other);
        Matrix2BinaryFile(out + "stats_merged_mean.bin", merged._MeanValue);
        ++dumps;
        Matrix2BinaryFile(out + "stats_merged_std.bin", merged._StandardDeviation);
        ++dumps;
        {
            Eigen::MatrixXd n(1, 1);
            n(0, 0) = (double) merged._NbOfValues;
            Matrix2BinaryFile(out + "stats_merged_n.bin", n);
            ++dumps;
        }
    }

    // --- Task 11: end-to-end feature parity gate -----------------------------
    // legacy: BLSTMSpectralSegmenter::initSpectralAnalysis (:194-314) for the param
    // derivation + getBLSTMInputSequence (:561-591) for the assembly. Per variant:
    // parse the config with the real ConfigFile, derive params, decode a FRESH
    // AudioStruct (offset 0.35, dur 2.0), applyPreemph per config (0.97), skip
    // noise (seed 0), build the periodogram per channel, build the mel bank per
    // config, assemble per getBLSTMInputSequence, and dump inputseq_<v>_chan{1,2}
    // + params_<v> (1 x 8 doubles: order, window_size, bins, shift_frames, freq_beg,
    // freq_end, ltsv_half_window, ltsv_shift).
    //
    // DCT NOTE: the legacy computeSegmentPeriodogramEstimates applies the filterbank
    // + DCT internally via the Eigen GEMM. Per the Task 7 lock the DCT product is
    // replaced by the ascending triple loop (applyDCTLoop above), so here the
    // periodogram is computed with an EMPTY mel bank, then applyFilterBank + (when
    // nb_dct>0) applyDCTLoop are applied explicitly -- matching the Rust port's
    // periodogram -> apply_filter_bank -> apply_dct path.
    {
        const unsigned Min = 1;
        const unsigned Max = 20;
        Loki::Factory<AbstractFFT<double>, unsigned int> gfft_factory;
        FactoryInit<GFFTList<GFFT, Min, Max>::Result>::apply(gfft_factory);

        const char* variants[] = {"mfcc_deltas", "mfcc_sdc", "logmel", "rawband_ltsv"};
        for (const char* variant : variants) {
            std::string cfgPath = out + "variant_" + variant + ".config";
            ConfigFile conf(cfgPath);
            FeatureCfg c = readFeatureCfg(conf, "BLSTM");

            CorpusItem vitem(wav, "", "RUS", "RU", 0, 0, 1.0);
            AudioStruct vaudio(OFFSET_SEC, MAX_DUR_SEC, 0, vitem);
            const double RATE = (double)vaudio.getFrameRate();
            SpectralP s = deriveSpectral(c, RATE, Max);

            if (c.preemph > 0) vaudio.applyPreemph(c.preemph);
            // noise skipped: noise_seed == 0 in every variant config (:222-227).

            // params dump (1 x 8 doubles).
            {
                Eigen::MatrixXd params(1, 8);
                params(0, 0) = (double)s.order;
                params(0, 1) = (double)s.window_size;
                params(0, 2) = (double)s.bins;
                params(0, 3) = (double)s.shift_frames;
                params(0, 4) = (double)s.freq_beg;
                params(0, 5) = (double)s.freq_end;
                params(0, 6) = (double)s.ltsv_half_window;
                params(0, 7) = (double)s.ltsv_shift;
                Matrix2BinaryFile(out + "params_" + variant + ".bin", params);
                ++dumps;
            }

            // Mel bank per config, built with the SNAPPED freq band (:241 passes the
            // reassigned _MinFreq/_MaxFreq) and spectrum_size = periodogram_length-1.
            MelFilterBank mel;
            if (c.nb_bins > 0) {
                mel = MelFilterBank(c.min_mel, c.max_mel, c.nb_bins, s.min_freq_snapped,
                                    s.max_freq_snapped, RATE, s.bins - 1, c.is_log, c.nb_dct,
                                    c.ignore_first, c.deltas_nb, c.dd_nb);
            }

            // DCT coeffs (reconstructed by the ctor formula, as the Task 7 block does).
            const double PI = 3.14159265358979323846264338327;
            long nbFilters = mel.notEmpty() ? (long)mel.getNbFilters() : 0;
            long nbDct = c.nb_dct;
            if (nbDct > nbFilters) nbDct = nbFilters;   // ctor clamp (MelFilterBank.cpp)
            Eigen::MatrixXd coeffs;
            if (c.nb_bins > 0 && c.nb_dct > 0) {
                coeffs = Eigen::MatrixXd::Zero(nbFilters, nbDct);
                for (long col = 0; col < nbFilters; ++col)
                    for (long row = 0; row < nbDct; ++row)
                        coeffs(col, row) = std::cos(PI / nbFilters * (col + 0.5) * row);
            }

            Eigen::MatrixXd win = getWindowingCoefficients(c.win_type, false, s.window_size + 1, c.win_param);
            Eigen::MatrixXd noConv;
            MelFilterBank emptyMel;
            const long long endFull = vaudio.getFrameCount() - 1;

            for (int chan = 0; chan < 2; ++chan) {
                // Periodogram with an EMPTY bank (see DCT NOTE); flag_DCOffset per config.
                vaudio.computeSegmentPeriodogramEstimates(s.order, s.shift_frames, chan, c.flag_dc,
                                                          win, emptyMel, gfft_factory, noConv, 0, endFull);
                Eigen::MatrixXd perio = vaudio._Periodogram;

                // getBLSTMInputSequence assembly (:568-576), spectral path priority
                // DCT -> filterbank -> raw-band log.
                Eigen::MatrixXd inputSeq;
                if (mel.notEmpty()) {
                    Eigen::MatrixXd fb = Eigen::MatrixXd::Zero(perio.rows(), mel.getNbFilters());
                    mel.applyFilterBank(perio, fb);
                    if (c.nb_dct > 0) {
                        inputSeq = applyDCTLoop(fb, coeffs, (int)nbDct, c.ignore_first, c.deltas_nb, c.dd_nb);
                    } else {
                        inputSeq = fb;
                    }
                } else {
                    // raw-band path: ln(block(:, freq_beg..freq_end) + 1e-24) (:575).
                    inputSeq = (((perio.block(0, s.freq_beg, perio.rows(),
                                              s.freq_end - s.freq_beg + 1).array() + 1e-24).log()).matrix());
                }

                // LTSV column (getLTSV, :678), appended when R >= 1. LOAD-BEARING band
                // asymmetry: initSpectralAnalysis RESETS freq_beg/freq_end to
                // (0, output_dim-1) inside the nb_bins>0 block (:270-271, output_dim =
                // nbDCT when nbDCT>0 else nbFilters), so for mel variants getLTSV runs
                // over the RAW periodogram's FIRST output_dim bins, NOT the spectral
                // band. For the raw-band variant (nb_bins==0) the band stays spectral
                // (s.freq_beg/s.freq_end). output_dim == inputSeq.cols() here.
                if (s.ltsv_half_window > 0) {
                    long ltsv_beg = mel.notEmpty() ? 0 : s.freq_beg;
                    long ltsv_end = mel.notEmpty() ? (long)inputSeq.cols() - 1 : s.freq_end;
                    Eigen::MatrixXd ltsv = getLTSV(perio, s.ltsv_half_window, s.ltsv_shift,
                                                   ltsv_beg, ltsv_end);
                    Eigen::MatrixXd merged(inputSeq.rows(), inputSeq.cols() + 1);
                    merged << inputSeq, ltsv;
                    inputSeq = merged;
                }

                Matrix2BinaryFile(out + "inputseq_" + variant + "_chan" +
                                      std::to_string(chan + 1) + ".bin", inputSeq);
                ++dumps;
            }
        }
    }

    // --- Phase 2 Task 1: real BLSTMNeuralNetwork construction + product probes -
    // legacy: BLSTMNeuralNetwork.cpp:26-153 (ctor) + :209-238 (setWeights /
    // getNbOfWeights) + LSTMLayer.cpp:162-207 (per-layer flat setWeights layout) +
    // NeuronLayer.cpp:69-86 (dense flat layout) + NeuralNetwork.hpp:69-85 (network
    // setWeights order). Construct the REAL net from the vendored config + weights,
    // then measure whether Eigen's blocked products diverge bitwise from ascending
    // loops on the actual NN shapes.
    //
    // Construction pattern verified against BLSTMSpectralSegmenter.cpp:739-740:
    //   ConfigFile conf(path, '_'); <clear BLSTM_weightsFile>;
    //   BLSTMNeuralNetwork<LSTMLayer> nn(conf, "BLSTM", true); nn.setWeights(flat);
    // Clearing BLSTM_weightsFile is MANDATORY: the config's value points to a dead
    // .mat path (1_worker_1.config:59); a nonempty value drives the ctor into
    // BinaryFile2Vector on that path -> exit(1). The ctor reads it via
    // conf.get<string>(prep+"_weightsFile","") and skips loading when size()==0.
    // NOTE: the brief's `set_val<string>(key,"")` cannot be used -- storing "" makes
    // get<string> take the present-key branch -> read<string>("") asserts (empty
    // stream extraction fails, String.hpp read()). ERASING the key instead makes
    // get<string>(key,"") return the default "" via the absent-key branch, which is
    // exactly the "no weights file" path the brief intends. weightsSetExternally=true
    // skips the per-layer config weight keys (whose _ConfigFileString collapses under
    // the iof shim -- irrelevant when that branch is never taken).
    {
        ConfigFile conf(nnConfigPath, '_');
        conf._Params.erase("BLSTM_weightsFile");
        BLSTMNeuralNetwork<LSTMLayer> nn(conf, "BLSTM", true);

        Eigen::VectorXd flat = BinaryFile2Vector(nnWeightsPath);
        const long nbWeights = nn.getNbOfWeights();
        if (nbWeights != 33671) {
            std::cerr << "FATAL: getNbOfWeights() = " << nbWeights << " != 33671\n";
            abort();
        }
        nn.setWeights(flat);
        std::cout << "NN_REAL ok weights=" << nbWeights << "\n";

        // Reconstruct the real first-layer weight matrices straight from the flat
        // vector at their known offsets (the layer members are protected). Layout
        // matches LSTMLayer::setWeights / NeuronLayer::setWeights / the network
        // ordering (Forward net first). Forward LSTM layer 0: _InputSize = 23*4 = 92
        // (NNetInputSize 23, LSTMSubSampling[0] 4), _OutputSize = 24 -> 4*Out = 96.
        //   _InputWeights   : 92 x 96, column-major, flat offset 0.
        //   _FeedbackWeights: 24 x 96, column-major, flat offset 92*96 = 8832.
        // Output MLP first NeuronLayer: _InputSize 48, _OutputSize 12 -> _Weights
        // 48 x 12, column-major, flat offset fwd(16512)+bwd(16512) = 33024.
        const int LSTM_IN = 92, LSTM_OUT = 24, LSTM_GATES = 96;
        const int DENSE_IN = 48, DENSE_OUT = 12;
        const long OFF_INPUTW = 0;
        const long OFF_FEEDBACKW = (long)LSTM_IN * LSTM_GATES;        // 8832
        const long OFF_DENSEW = 33024;

        Eigen::MatrixXd inputWeights(LSTM_IN, LSTM_GATES);           // 92 x 96
        for (int col = 0; col < LSTM_GATES; ++col)
            for (int row = 0; row < LSTM_IN; ++row)
                inputWeights(row, col) = flat(OFF_INPUTW + (long)col * LSTM_IN + row);

        Eigen::MatrixXd feedbackWeights(LSTM_OUT, LSTM_GATES);       // 24 x 96
        for (int col = 0; col < LSTM_GATES; ++col)
            for (int row = 0; row < LSTM_OUT; ++row)
                feedbackWeights(row, col) = flat(OFF_FEEDBACKW + (long)col * LSTM_OUT + row);

        Eigen::MatrixXd denseWeights(DENSE_IN, DENSE_OUT);           // 48 x 12
        for (int col = 0; col < DENSE_OUT; ++col)
            for (int row = 0; row < DENSE_IN; ++row)
                denseWeights(row, col) = flat(OFF_DENSEW + (long)col * DENSE_IN + row);

        // Deterministic input trajectory for the lstm_input_gemm probe (200 rows).
        // Same closed-form style as synth_20x50 above, recentered to [-0.5, 0.5) so
        // the k-sum spans signed real-valued operands. Row t, col j.
        const int T = 200;
        Eigen::MatrixXd traj(T, 23);                                 // T x 23 (raw input band)
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < 23; ++j)
                traj(t, j) = (double)(((t * 31 + j * 17) % 100)) / 100.0 - 0.5;

        // Probe 1: lstm_input_gemm -- (T x 23) * (23 x 96) on the real first-layer
        // input weights (top 23 rows = the un-subsampled raw input band).
        nnProbe("lstm_input_gemm", traj, inputWeights.topRows(23));

        // Probe 2: lstm_recurrence_gemv -- the (1 x 24) * (24 x 96) recurrent GEMV
        // (LSTMLayer.cpp:351 `outputSeq.row(row-1)*_FeedbackWeights`), accumulated
        // over >= 100 timesteps. A single k=24 GEMV rarely trips Eigen's blocking;
        // running a mock recurrence for T steps and comparing the FULL trajectory
        // bit-for-bit is what surfaces (or rules out) per-step divergence. Each step:
        // h_next = 0.1 * (h * W)[:, :24] folded back into a 24-vector (a bounded,
        // deterministic feedback so the state neither blows up nor decays to zero),
        // computed BOTH ways; any 1-ULP split propagates through the >=100 steps.
        {
            const int STEPS = 150;                                   // >= 100
            Eigen::MatrixXd hEig(1, LSTM_OUT);
            Eigen::MatrixXd hLoop(1, LSTM_OUT);
            for (int j = 0; j < LSTM_OUT; ++j) {
                double v = (double)((j * 13) % 100) / 100.0 - 0.5;
                hEig(0, j) = v;
                hLoop(0, j) = v;
            }
            int mismatches = 0, total = 0;
            int firstStep = -1, firstCol = -1;
            uint64_t firstEigenBits = 0, firstLoopBits = 0;
            for (int step = 0; step < STEPS; ++step) {
                Eigen::MatrixXd gEig = hEig * feedbackWeights;       // 1 x 96, Eigen
                Eigen::MatrixXd gLoop = matSeq(hLoop, feedbackWeights);  // 1 x 96, ascending
                for (int c = 0; c < LSTM_GATES; ++c) {
                    uint64_t eBits, lBits;
                    double eVal = gEig(0, c);
                    double lVal = gLoop(0, c);
                    std::memcpy(&eBits, &eVal, sizeof(double));
                    std::memcpy(&lBits, &lVal, sizeof(double));
                    ++total;
                    if (eBits != lBits) {
                        if (mismatches == 0) {
                            firstStep = step;
                            firstCol = c;
                            firstEigenBits = eBits;
                            firstLoopBits = lBits;
                        }
                        ++mismatches;
                    }
                }
                // Bounded feedback: next state = 0.1 * first 24 gate activations.
                for (int j = 0; j < LSTM_OUT; ++j) {
                    hEig(0, j) = 0.1 * gEig(0, j);
                    hLoop(0, j) = 0.1 * gLoop(0, j);
                }
            }
            const bool diverged = mismatches > 0;
            std::cout << "NN_PROBE site=lstm_recurrence_gemv"
                      << " diverged=" << (diverged ? 1 : 0)
                      << " mismatches=" << mismatches << " total=" << total
                      << " first=(" << firstStep << "," << firstCol << ")"
                      << " eigen=0x" << std::hex << firstEigenBits
                      << " loop=0x" << firstLoopBits << std::dec << "\n";
        }

        // Probe 3: dense_gemm -- (200 x 48) * (48 x 12) on the real output-MLP
        // first-layer weights (NeuronLayer.cpp:130 `InputSeq*_Weights`). Deterministic
        // 200 x 48 input in the same closed-form style.
        Eigen::MatrixXd denseIn(T, DENSE_IN);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < DENSE_IN; ++j)
                denseIn(t, j) = (double)(((t * 19 + j * 23) % 100)) / 100.0 - 0.5;
        nnProbe("dense_gemm", denseIn, denseWeights);

        // Probe 4: softmax_rowsum -- the row-sum reduction over exp(dense output)
        // (NeuronLayer.cpp:138-139 softmax normalizer). Measure whether the ascending
        // row-sum of exp diverges from Eigen's rowwise().sum(). Build it as a matrix
        // product against a 12 x 1 ones vector (that IS the row-sum) both ways: Eigen
        // (exp .rowwise().sum()) vs the ascending accumulation matSeq(exp, ones).
        {
            Eigen::MatrixXd denseOut = matSeq(denseIn, denseWeights);   // 200 x 12
            Eigen::MatrixXd expOut = denseOut.array().exp().matrix();   // 200 x 12
            Eigen::MatrixXd ones = Eigen::MatrixXd::Ones(DENSE_OUT, 1);
            Eigen::MatrixXd eig = expOut.rowwise().sum();               // 200 x 1, Eigen reduce
            Eigen::MatrixXd loop = matSeq(expOut, ones);               // 200 x 1, ascending
            const int total = static_cast<int>(eig.rows());
            int mismatches = 0, firstRow = -1;
            uint64_t firstEigenBits = 0, firstLoopBits = 0;
            for (int r = 0; r < eig.rows(); ++r) {
                uint64_t eBits, lBits;
                double eVal = eig(r, 0);
                double lVal = loop(r, 0);
                std::memcpy(&eBits, &eVal, sizeof(double));
                std::memcpy(&lBits, &lVal, sizeof(double));
                if (eBits != lBits) {
                    if (mismatches == 0) {
                        firstRow = r;
                        firstEigenBits = eBits;
                        firstLoopBits = lBits;
                    }
                    ++mismatches;
                }
            }
            const bool diverged = mismatches > 0;
            std::cout << "NN_PROBE site=softmax_rowsum"
                      << " diverged=" << (diverged ? 1 : 0)
                      << " mismatches=" << mismatches << " total=" << total
                      << " first=(" << firstRow << ",0)"
                      << " eigen=0x" << std::hex << firstEigenBits
                      << " loop=0x" << firstLoopBits << std::dec << "\n";
        }
    }

    // --- Phase 2 Task 2: activation sweep -------------------------------------
    // legacy: ActivationFunctions.h:40-49 (Logistic, INCLUSIVE saturation),
    // :229-238 (GatesFunction, 0.1 pre-scale, EXCLUSIVE saturation),
    // :158-160 (Maxmin2 = asinh). expLimit computed via Log.hpp:193-195
    // (std::log(std::numeric_limits<double>::max())), NOT hardcoded, so the
    // boundary probes are the exact doubles the C++ saturation branches see.
    // Dump 4 x N: row0 inputs, row1 GatesFunction::fn, row2 Logistic::fn,
    // row3 Maxmin2::fn (asinh), via the REAL header structs.
    {
        const double expLimit = Log<double>::expLimit;
        std::vector<double> xs = {
            0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0,
            expLimit / 0.1, -expLimit / 0.1,
            std::nextafter(expLimit / 0.1, 0.0) , -std::nextafter(expLimit / 0.1, 0.0),
            expLimit, -expLimit,
            std::nextafter(expLimit, 0.0), -std::nextafter(expLimit, 0.0),
            -1e4, 0.5, -0.5, 42.0,
            // Exact boundary inputs GatesFunction::fn actually saturates on: the
            // input x such that 0.1*x rounds to precisely +-expLimit (algebraic
            // x = 10*expLimit need not round-trip through 0.1*(10*expLimit) back
            // to expLimit; these two are the values the harness's own 0.1*x
            // comparison sees at the boundary).
            10.0 * expLimit, -10.0 * expLimit,
        };
        const int n = static_cast<int>(xs.size());
        Eigen::MatrixXd sweep(4, n);
        for (int k = 0; k < n; ++k) {
            double x = xs[static_cast<size_t>(k)];
            sweep(0, k) = x;
            sweep(1, k) = GatesFunction::fn(x);
            sweep(2, k) = Logistic::fn(x);
            sweep(3, k) = Maxmin2::fn(x);
        }
        Matrix2BinaryFile(out + "act_sweep.bin", sweep);
        ++dumps;
    }

    // --- Phase 2 Task 3: LstmLayer weight (de)serialization + chaining -------
    // legacy: LSTMLayer.cpp:6-27 (ctor resizes: _InputWeights I x 4O,
    // _FeedbackWeights O x 4O, _PeepWeight 12 x O, _Biaises 1 x 4O -- NOTE the
    // header's size comments at LSTMLayer.h:33-36 are stale/wrong; the ctor resizes
    // are authoritative), :162-207 (setWeights/getWeights: per-block COLUMN-major
    // element order, block order InputWeights -> FeedbackWeights -> PeepWeight ->
    // Biaises; setWeights consumes weights.head(getNbOfWeights()) and returns the
    // tail via weights.tail(...)). Construct a REAL LSTMLayer(conf, "SYNW", 0, 3, 2,
    // true) (weightsSetExternally=true skips the per-block config-key read path;
    // the SYNW-prefixed scalar keys -- MaxSaturation/IsCellsPeepholesActive/etc --
    // all have defaults, so the real ConfigFile from the Task 1 block above is
    // reusable here without adding any new keys). Feed a synthetic flat vector
    // (deterministic formula w[k] = ((k*11+3) % 97)/97.0 - 0.5) sized for BOTH
    // layers back-to-back (first layer's nb + second layer's nb), setWeights on
    // layer 0, dump the INPUT vector and layer 0's getWeights() output (must be
    // bit-identical -- pins the mirror), then setWeights the LEFTOVER TAIL into a
    // second chained layer (I=2, O=1) and dump ITS getWeights() output too (pins
    // head/tail chaining semantics across two layers sharing one flat vector).
    {
        ConfigFile conf(nnConfigPath, '_');
        conf._Params.erase("BLSTM_weightsFile");

        LSTMLayer layer0(conf, "SYNW", 0, 3, 2, true);
        LSTMLayer layer1(conf, "SYNW", 1, 2, 1, true);

        const long nb0 = layer0.getNbOfWeights(); // 4*3*2+4*2*2+12*2+4*2 = 24+16+24+8 = 72
        const long nb1 = layer1.getNbOfWeights(); // 4*2*1+4*1*1+12*1+4*1 = 8+4+12+4 = 28
        const long nbTotal = nb0 + nb1;

        Eigen::VectorXd flat(nbTotal);
        for (long k = 0; k < nbTotal; ++k) {
            flat(k) = (double)((k * 11 + 3) % 97) / 97.0 - 0.5;
        }
        Eigen::MatrixXd flatDump(nbTotal, 1);
        flatDump.col(0) = flat;
        Matrix2BinaryFile(out + "lstm_w_roundtrip_in.bin", flatDump);
        ++dumps;

        Eigen::VectorXd tail0 = layer0.setWeights(flat);
        Eigen::VectorXd got0 = layer0.getWeights();
        if (got0.size() != nb0) {
            std::cerr << "FATAL: layer0 getWeights() size " << got0.size() << " != " << nb0 << "\n";
            abort();
        }

        Eigen::VectorXd tail1 = layer1.setWeights(tail0);
        Eigen::VectorXd got1 = layer1.getWeights();
        if (got1.size() != nb1) {
            std::cerr << "FATAL: layer1 getWeights() size " << got1.size() << " != " << nb1 << "\n";
            abort();
        }
        if (tail1.size() != 0) {
            std::cerr << "FATAL: tail after both layers is " << tail1.size() << " != 0\n";
            abort();
        }

        // Dump layer0's getWeights() concatenated with layer1's getWeights() as one
        // column vector: rows 0..nb0 pin layer0's mirror, rows nb0..nb0+nb1 pin
        // layer1's mirror over the leftover tail. Both halves must equal the
        // corresponding slices of lstm_w_roundtrip_in.bin bit-for-bit.
        Eigen::MatrixXd outDump(nbTotal, 1);
        outDump.block(0, 0, nb0, 1) = got0;
        outDump.block(nb0, 0, nb1, 1) = got1;
        Matrix2BinaryFile(out + "lstm_w_roundtrip_out.bin", outDump);
        ++dumps;

        std::cout << "NN_REAL ok lstm_nb0=" << nb0 << " lstm_nb1=" << nb1 << "\n";
    }

    // --- Phase 2 Task 4: LSTM forward/reverse dumps + NN_TOL probe -----------
    // legacy: LSTMLayer.cpp:312-421 (feedForward / feedForwardReverse). For each
    // (shape, flags, direction) case: build synthetic weights via the Task 3 formula
    // (w[k] = ((k*11+3) % 97)/97.0 - 0.5) sized to nb_of_weights(I,O), feed them
    // through BOTH the REAL LSTMLayer::setWeights AND the reimpl's unpacked arrays
    // (unpackLstmWeights, same column-major layout), run the REAL feedForward beside
    // lstmForwardLoop on a deterministic input, dump the reimpl's outputs (T x O) +
    // post-activation gates cache (T x 4O), and accumulate the max ULP/abs gap
    // between the real layer's output and the reimpl's output across ALL cases into
    // one NN_TOL line. Flags gate peephole families: cells (rows 0,1,2), gates
    // (4,5,6,8,9,10), gates-rec (3,7,11). Width-mismatch cases (forward, all-on):
    // input cols I+2 (leftCols(I), :313-314) and I-1 (topRows(cols), :315-316).
    {
        ConfigFile conf(nnConfigPath, '_');
        conf._Params.erase("BLSTM_weightsFile");

        struct Shape { int I, O, T; };
        const Shape shapes[] = {{3, 2, 7}, {5, 4, 12}};
        struct Flags { const char* label; bool c, g, r; };
        const Flags flagSet[] = {
            {"allon", true, true, true},
            {"cells", true, false, false},
            {"gates", false, true, false},
            {"gatesrec", false, false, true},
            {"alloff", false, false, false},
        };

        double globalMaxAbs = 0.0;
        long globalMaxUlp = 0;

        auto probeGap = [&](const Eigen::MatrixXd& real, const Eigen::MatrixXd& reimpl) {
            for (int r = 0; r < real.rows(); ++r) {
                for (int c = 0; c < real.cols(); ++c) {
                    double a = real(r, c), b = reimpl(r, c);
                    double absGap = std::fabs(a - b);
                    if (absGap > globalMaxAbs) globalMaxAbs = absGap;
                    uint64_t ab, bb;
                    std::memcpy(&ab, &a, sizeof(double));
                    std::memcpy(&bb, &b, sizeof(double));
                    long ulp = (ab > bb) ? (long)(ab - bb) : (long)(bb - ab);
                    if (ulp > globalMaxUlp) globalMaxUlp = ulp;
                }
            }
        };

        // Deterministic closed-form input: signed values spanning [-0.5, 0.5).
        auto makeInput = [](int T, int cols) {
            Eigen::MatrixXd x(T, cols);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < cols; ++j)
                    x(t, j) = (double)(((t * 37 + j * 53 + 7) % 101)) / 101.0 - 0.5;
            return x;
        };

        // Run one case: real layer + reimpl on the same weights/input; dump reimpl
        // outputs + gates; probe the real-vs-reimpl output gap. `reverse` picks the
        // direction; `inCols` overrides the input width (default I).
        auto runCase = [&](const std::string& name, int I, int O, int T,
                           const Flags& fl, bool reverse, int inCols) {
            conf.set_val<bool>("SYNW_IsCellsPeepholesActive", fl.c);
            conf.set_val<bool>("SYNW_IsGatesPeepholesActive", fl.g);
            conf.set_val<bool>("SYNW_IsGatesRecurrentPeepholesActive", fl.r);
            LSTMLayer layer(conf, "SYNW", 0, (size_t)I, (size_t)O, true);
            const long nb = layer.getNbOfWeights();
            Eigen::VectorXd flat(nb);
            for (long k = 0; k < nb; ++k) flat(k) = (double)((k * 11 + 3) % 97) / 97.0 - 0.5;
            layer.setWeights(flat);

            Eigen::MatrixXd inputW, feedbackW, peep, bias;
            unpackLstmWeights(flat, I, O, inputW, feedbackW, peep, bias);

            Eigen::MatrixXd input = makeInput(T, inCols);
            Eigen::MatrixXd realOut(T, O), gates, reimplOut;
            if (reverse) {
                layer.feedForwardReverse(input, realOut, false);
                lstmForwardReverseLoop(input, inputW, feedbackW, peep, bias, O,
                                       fl.c, fl.g, fl.r, gates, reimplOut);
            } else {
                layer.feedForward(input, realOut, false);
                lstmForwardLoop(input, inputW, feedbackW, peep, bias, O,
                                fl.c, fl.g, fl.r, gates, reimplOut);
            }
            probeGap(realOut, reimplOut);

            Matrix2BinaryFile(out + "lstm_fwd_" + name + ".bin", reimplOut);
            Matrix2BinaryFile(out + "lstm_gates_" + name + ".bin", gates);
            dumps += 2;
        };

        for (const Shape& s : shapes) {
            for (const Flags& fl : flagSet) {
                std::ostringstream base;
                base << s.I << "x" << s.O << "_T" << s.T << "_" << fl.label;
                runCase(base.str() + "_fwd", s.I, s.O, s.T, fl, false, s.I);
                runCase(base.str() + "_rev", s.I, s.O, s.T, fl, true, s.I);
            }
        }
        // Width-mismatch (forward, all-on): input cols I+2 (leftCols) and I-1 (topRows).
        {
            const Shape s = shapes[0];  // I=3,O=2,T=7
            const Flags allon = flagSet[0];
            std::ostringstream b1, b2;
            b1 << s.I << "x" << s.O << "_T" << s.T << "_allon_wideP2_fwd";
            b2 << s.I << "x" << s.O << "_T" << s.T << "_allon_narrowM1_fwd";
            runCase(b1.str(), s.I, s.O, s.T, allon, false, s.I + 2);
            runCase(b2.str(), s.I, s.O, s.T, allon, false, s.I - 1);
        }

        std::cout << "NN_TOL site=lstm_forward max_ulp=" << globalMaxUlp
                  << " max_abs=" << std::scientific << std::setprecision(3)
                  << globalMaxAbs << "\n";
    }

    // --- Phase 2 Task 5: NeuronLayer (dense) forward + NN_TOL probe ----------
    // legacy: NeuronLayer.cpp:69-99 (setWeights/getWeights: flat order = weights
    // COLUMN-major -- per-neuron fan-in blocks contiguous, m(ii,jj) =
    // needed(jj*_InputSize+ii) -- THEN all biases; nb = O*(I+1)), :127-149
    // (feedForward: width-tolerant projection -- cols>I uses leftCols(I), cols<I
    // uses topRows(cols), else the plain product; bias .rowwise() BEFORE
    // activation; lastLayer && O>1 -> UNSTABILIZED softmax [no max-subtraction
    // overflow guard -- exp(a+b) computed directly, per-row sum via a SEQUENTIAL
    // rowwise().sum(), then per-COLUMN cwiseQuotient by that sum]; lastLayer &&
    // O==1 -> Logistic; else Maxmin2/asinh).
    {
        // denseForwardLoop is now a file-scope static (hoisted for Task 6 reuse);
        // the reimpl semantics (:127-149) are unchanged.
        ConfigFile conf(nnConfigPath, '_');
        conf._Params.erase("BLSTM_weightsFile");

        double globalMaxAbs = 0.0;
        long globalMaxUlp = 0;
        auto probeGap = [&](const Eigen::MatrixXd& real, const Eigen::MatrixXd& reimpl) {
            for (int r = 0; r < real.rows(); ++r) {
                for (int c = 0; c < real.cols(); ++c) {
                    double a = real(r, c), b = reimpl(r, c);
                    double absGap = std::fabs(a - b);
                    if (absGap > globalMaxAbs) globalMaxAbs = absGap;
                    uint64_t ab, bb;
                    std::memcpy(&ab, &a, sizeof(double));
                    std::memcpy(&bb, &b, sizeof(double));
                    long ulp = (ab > bb) ? (long)(ab - bb) : (long)(bb - ab);
                    if (ulp > globalMaxUlp) globalMaxUlp = ulp;
                }
            }
        };

        // Deterministic closed-form input: signed values spanning [-0.5, 0.5)
        // (same formula family as the Task 4 LSTM input).
        auto makeInput = [](int T, int cols) {
            Eigen::MatrixXd x(T, cols);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < cols; ++j)
                    x(t, j) = (double)(((t * 37 + j * 53 + 7) % 101)) / 101.0 - 0.5;
            return x;
        };

        // Run one case: real NeuronLayer + reimpl on the same synthetic weights and
        // input; dump the reimpl's output; probe the real-vs-reimpl gap.
        auto runCase = [&](const std::string& dumpName, int I, int O, int T,
                           bool lastLayer, int inCols) {
            NeuronLayer layer(conf, "SYNW", 0, (size_t)I, (size_t)O, true);
            const long nb = layer.getNbOfWeights(); // O*(I+1)
            Eigen::VectorXd flat(nb);
            for (long k = 0; k < nb; ++k) flat(k) = (double)((k * 11 + 3) % 97) / 97.0 - 0.5;
            layer.setWeights(flat);

            // Unpack the same flat vector into (weights I x O, bias 1 x O), matching
            // NeuronLayer::setWeights (:69-82): weights COLUMN-major, then all biases.
            Eigen::MatrixXd weights(I, O), bias(1, O);
            for (int jj = 0; jj < O; ++jj)
                for (int ii = 0; ii < I; ++ii) weights(ii, jj) = flat((long)jj * I + ii);
            for (int jj = 0; jj < O; ++jj) bias(0, jj) = flat((long)O * I + jj);

            Eigen::MatrixXd input = makeInput(T, inCols);
            Eigen::MatrixXd realOut(T, O);
            layer.feedForward(input, realOut, lastLayer);
            Eigen::MatrixXd reimplOut = denseForwardLoop(input, weights, bias, lastLayer);

            probeGap(realOut, reimplOut);
            Matrix2BinaryFile(out + dumpName, reimplOut);
            ++dumps;
        };

        // hidden: lastLayer=false, asinh, I=4 O=3 T=6.
        runCase("dense_hidden.bin", 4, 3, 6, false, 4);
        // softmax: lastLayer=true, O=3 (unstabilized softmax), I=4 T=6.
        runCase("dense_softmax.bin", 4, 3, 6, true, 4);
        // logistic: lastLayer=true, O=1, I=4 T=6.
        runCase("dense_logistic.bin", 4, 1, 6, true, 4);
        // width-mismatch, both directions, lastLayer=false: cols=I+2 and cols=I-1.
        runCase("dense_wide.bin", 4, 3, 6, false, 4 + 2);
        runCase("dense_narrow.bin", 4, 3, 6, false, 4 - 1);

        std::cout << "NN_TOL site=dense_forward max_ulp=" << globalMaxUlp
                  << " max_abs=" << std::scientific << std::setprecision(3)
                  << globalMaxAbs << "\n";
    }

    // --- Phase 2 Task 6: NeuralNetwork container (sub-sample stacking, chained
    // weights, double-input MLP) ---------------------------------------------
    // legacy: NeuralNetwork.hpp:25-249. Build the REAL NeuralNetwork<LSTMLayer> /
    // NeuralNetwork<NeuronLayer> from the reusable config, set one synthetic flat
    // vector (Task 3 formula) via NeuralNetwork::setWeights (which chains head/tail
    // through the layers), and compare its feedForward/feedForwardReverse/
    // feedForwardDouble against the container reimpl (netForwardLoop etc, wired to
    // lstmForwardLoop/denseForwardLoop) on the same inputs -> NN_TOL. Dump the reimpl
    // outputs as the goldens the Rust Network port reproduces. The container reimpl's
    // per-layer steps re-chain the SAME flat vector into per-layer weight slices,
    // matching setWeights's head/tail split (each layer consumes nb_of_weights()).
    {
        ConfigFile conf(nnConfigPath, '_');
        conf._Params.erase("BLSTM_weightsFile");
        // All-peephole-on for every LSTM layer (SYNW-prefixed flags apply uniformly).
        conf.set_val<bool>("SYNW_IsCellsPeepholesActive", true);
        conf.set_val<bool>("SYNW_IsGatesPeepholesActive", true);
        conf.set_val<bool>("SYNW_IsGatesRecurrentPeepholesActive", true);

        // The Task 3/4/5 synthetic flat-vector formula, length n.
        auto synthFlat = [](long n) {
            Eigen::VectorXd flat(n);
            for (long k = 0; k < n; ++k) flat(k) = (double)((k * 11 + 3) % 97) / 97.0 - 0.5;
            return flat;
        };
        // The Task 4/5 deterministic closed-form input.
        auto makeInput = [](int T, int cols) {
            Eigen::MatrixXd x(T, cols);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < cols; ++j)
                    x(t, j) = (double)(((t * 37 + j * 53 + 7) % 101)) / 101.0 - 0.5;
            return x;
        };
        // Max ULP/abs gap between the real network output and the reimpl output.
        auto probeGap = [](const Eigen::MatrixXd& real, const Eigen::MatrixXd& reimpl,
                           long& maxUlp, double& maxAbs) {
            for (int r = 0; r < real.rows(); ++r) {
                for (int c = 0; c < real.cols(); ++c) {
                    double a = real(r, c), b = reimpl(r, c);
                    double absGap = std::fabs(a - b);
                    if (absGap > maxAbs) maxAbs = absGap;
                    uint64_t ab, bb;
                    std::memcpy(&ab, &a, sizeof(double));
                    std::memcpy(&bb, &b, sizeof(double));
                    long ulp = (ab > bb) ? (long)(ab - bb) : (long)(bb - ab);
                    if (ulp > maxUlp) maxUlp = ulp;
                }
            }
        };

        // ---- 2-layer LSTM net [3,4,2] sub [2,1] on T=11 (odd -> dropped-tail floor).
        // Layer sizes: L0 = LSTMLayer(in=neuronNb[0]*sub[0]=6, out=neuronNb[1]=4);
        // L1 = LSTMLayer(in=neuronNb[1]*sub[1]=4, out=neuronNb[2]=2). SubSample(2) on
        // the 11-row input drops row 10 -> 5 rows.
        {
            const std::vector<std::vector<double>::size_type> neuronNb = {3, 4, 2};
            const std::vector<std::vector<double>::size_type> subSampling = {2, 1};
            const std::vector<long> nnL = {3, 4, 2};
            const std::vector<long> ssL = {2, 1};
            const int L0in = 6, L0out = 4, L1in = 4, L1out = 2;

            NeuralNetwork<LSTMLayer> net(conf, "SYNW", neuronNb, subSampling, true);
            const long nbTotal = net.getNbOfWeights();
            Eigen::VectorXd flat = synthFlat(nbTotal);
            net.setWeights(flat);

            // Re-chain the SAME flat vector into per-layer slices (head/tail split).
            LSTMLayer probe0(conf, "SYNW", 0, (size_t)L0in, (size_t)L0out, true);
            LSTMLayer probe1(conf, "SYNW", 1, (size_t)L1in, (size_t)L1out, true);
            const long nb0 = probe0.getNbOfWeights();
            const long nb1 = probe1.getNbOfWeights();
            Eigen::VectorXd flat0 = flat.head(nb0);
            Eigen::VectorXd flat1 = flat.segment(nb0, nb1);
            Eigen::MatrixXd iw0, fw0, pp0, bs0, iw1, fw1, pp1, bs1;
            unpackLstmWeights(flat0, L0in, L0out, iw0, fw0, pp0, bs0);
            unpackLstmWeights(flat1, L1in, L1out, iw1, fw1, pp1, bs1);

            Eigen::MatrixXd input = makeInput(11, 3);

            for (bool reverse : {false, true}) {
                std::vector<NetLayerStep> steps(2);
                if (!reverse) {
                    steps[0].forward = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool) {
                        Eigen::MatrixXd g;
                        lstmForwardLoop(in, iw0, fw0, pp0, bs0, L0out, true, true, true, g, outm);
                    };
                    steps[1].forward = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool) {
                        Eigen::MatrixXd g;
                        lstmForwardLoop(in, iw1, fw1, pp1, bs1, L1out, true, true, true, g, outm);
                    };
                } else {
                    steps[0].forward = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool) {
                        Eigen::MatrixXd g;
                        lstmForwardReverseLoop(in, iw0, fw0, pp0, bs0, L0out, true, true, true, g, outm);
                    };
                    steps[1].forward = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool) {
                        Eigen::MatrixXd g;
                        lstmForwardReverseLoop(in, iw1, fw1, pp1, bs1, L1out, true, true, true, g, outm);
                    };
                }
                const long outRows = 11 / 2;  // floor after SubSample(2)
                Eigen::MatrixXd reimplOut, realOut(outRows, L1out);
                netForwardLoop(nnL, ssL, steps, input, reimplOut);
                if (reverse) net.feedForwardReverse(input, realOut);
                else net.feedForward(input, realOut);

                long maxUlp = 0; double maxAbs = 0.0;
                probeGap(realOut, reimplOut, maxUlp, maxAbs);
                std::cout << "NN_TOL site=net_lstm_forward" << (reverse ? "_rev" : "_fwd")
                          << " max_ulp=" << maxUlp << " max_abs=" << std::scientific
                          << std::setprecision(3) << maxAbs << "\n";
                Matrix2BinaryFile(out + (reverse ? "net_lstm_rev.bin" : "net_lstm_fwd.bin"), reimplOut);
                ++dumps;
            }

            // Chained set_weights round-trip: dump the input flat vector and the
            // network's getWeights() output (must be bit-identical -- pure copy).
            Eigen::MatrixXd inDump(nbTotal, 1);
            inDump.col(0) = flat;
            Matrix2BinaryFile(out + "net_w_in.bin", inDump);
            Eigen::VectorXd got = net.getWeights();
            if (got.size() != nbTotal) {
                std::cerr << "FATAL: net getWeights() size " << got.size() << " != " << nbTotal << "\n";
                abort();
            }
            Eigen::MatrixXd outDump(nbTotal, 1);
            outDump.col(0) = got;
            Matrix2BinaryFile(out + "net_w_out.bin", outDump);
            dumps += 2;
        }

        // ---- Dense net [4,3,2] sub [1,2] on T=11. L0 = NeuronLayer(in=4, out=3)
        // lastLayer=false (asinh); SubSample(2) between layers drops row 10 -> 5 rows;
        // L1 = NeuronLayer(in=neuronNb[1]*sub[1]=6, out=2) lastLayer=true (softmax O=2).
        {
            const std::vector<std::vector<double>::size_type> neuronNb = {4, 3, 2};
            const std::vector<std::vector<double>::size_type> subSampling = {1, 2};
            const std::vector<long> nnD = {4, 3, 2};
            const std::vector<long> ssD = {1, 2};
            const int L0in = 4, L0out = 3, L1in = 6, L1out = 2;

            NeuralNetwork<NeuronLayer> net(conf, "SYNW", neuronNb, subSampling, true);
            const long nbTotal = net.getNbOfWeights();
            Eigen::VectorXd flat = synthFlat(nbTotal);
            net.setWeights(flat);

            const long nb0 = (long)L0out * (L0in + 1);
            Eigen::VectorXd flat0 = flat.head(nb0);
            Eigen::VectorXd flat1 = flat.segment(nb0, (long)L1out * (L1in + 1));
            Eigen::MatrixXd w0(L0in, L0out), b0(1, L0out), w1(L1in, L1out), b1(1, L1out);
            for (int jj = 0; jj < L0out; ++jj)
                for (int ii = 0; ii < L0in; ++ii) w0(ii, jj) = flat0((long)jj * L0in + ii);
            for (int jj = 0; jj < L0out; ++jj) b0(0, jj) = flat0((long)L0out * L0in + jj);
            for (int jj = 0; jj < L1out; ++jj)
                for (int ii = 0; ii < L1in; ++ii) w1(ii, jj) = flat1((long)jj * L1in + ii);
            for (int jj = 0; jj < L1out; ++jj) b1(0, jj) = flat1((long)L1out * L1in + jj);

            std::vector<NetLayerStep> steps(2);
            steps[0].forward = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool last) {
                outm = denseForwardLoop(in, w0, b0, last);
            };
            steps[1].forward = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool last) {
                outm = denseForwardLoop(in, w1, b1, last);
            };

            Eigen::MatrixXd input = makeInput(11, 4);
            const long outRows = 11 / 2;  // floor after SubSample(2) at layer 1
            Eigen::MatrixXd reimplOut, realOut(outRows, L1out);
            netForwardLoop(nnD, ssD, steps, input, reimplOut);
            net.feedForward(input, realOut);

            long maxUlp = 0; double maxAbs = 0.0;
            probeGap(realOut, reimplOut, maxUlp, maxAbs);
            std::cout << "NN_TOL site=net_dense_forward max_ulp=" << maxUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbs << "\n";
            Matrix2BinaryFile(out + "net_dense.bin", reimplOut);
            ++dumps;
        }

        // ---- feedForwardDouble on a dense net [10,4,2] sub [1,1] from two 5-col
        // halves (forward half LEFT). L0 = NeuronLayer(in=10, out=4) lastLayer=false;
        // L1 = NeuronLayer(in=4, out=2) lastLayer=true (softmax). Asymmetric halves so
        // an accidental first/second swap changes the hcat and the output.
        {
            const std::vector<std::vector<double>::size_type> neuronNb = {10, 4, 2};
            const std::vector<std::vector<double>::size_type> subSampling = {1, 1};
            const std::vector<long> nnD = {10, 4, 2};
            const std::vector<long> ssD = {1, 1};
            const int L0in = 10, L0out = 4, L1in = 4, L1out = 2;

            NeuralNetwork<NeuronLayer> net(conf, "SYNW", neuronNb, subSampling, true);
            const long nbTotal = net.getNbOfWeights();
            Eigen::VectorXd flat = synthFlat(nbTotal);
            net.setWeights(flat);

            const long nb0 = (long)L0out * (L0in + 1);
            Eigen::VectorXd flat0 = flat.head(nb0);
            Eigen::VectorXd flat1 = flat.segment(nb0, (long)L1out * (L1in + 1));
            Eigen::MatrixXd w0(L0in, L0out), b0(1, L0out), w1(L1in, L1out), b1(1, L1out);
            for (int jj = 0; jj < L0out; ++jj)
                for (int ii = 0; ii < L0in; ++ii) w0(ii, jj) = flat0((long)jj * L0in + ii);
            for (int jj = 0; jj < L0out; ++jj) b0(0, jj) = flat0((long)L0out * L0in + jj);
            for (int jj = 0; jj < L1out; ++jj)
                for (int ii = 0; ii < L1in; ++ii) w1(ii, jj) = flat1((long)jj * L1in + ii);
            for (int jj = 0; jj < L1out; ++jj) b1(0, jj) = flat1((long)L1out * L1in + jj);

            std::vector<NetLayerStep> steps(2);
            steps[0].forward = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool last) {
                outm = denseForwardLoop(in, w0, b0, last);
            };
            steps[1].forward = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outm, bool last) {
                outm = denseForwardLoop(in, w1, b1, last);
            };

            // Asymmetric halves: first = makeInput(T,5); second = makeInput shifted so
            // swapping first/second yields a different hcat (a swap must fail the test).
            const int T = 11;
            Eigen::MatrixXd first(T, 5), second(T, 5);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < 5; ++j) {
                    first(t, j) = (double)(((t * 37 + j * 53 + 7) % 101)) / 101.0 - 0.5;
                    second(t, j) = (double)(((t * 41 + j * 59 + 13) % 103)) / 103.0 - 0.5;
                }

            Eigen::MatrixXd reimplOut, realOut(T, L1out);
            netForwardDoubleLoop(nnD, ssD, steps, first, second, reimplOut);
            net.feedForwardDouble(first, second, realOut);

            long maxUlp = 0; double maxAbs = 0.0;
            probeGap(realOut, reimplOut, maxUlp, maxAbs);
            std::cout << "NN_TOL site=net_double max_ulp=" << maxUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbs << "\n";
            Matrix2BinaryFile(out + "net_double.bin", reimplOut);
            ++dumps;
        }
    }

    // --- Phase 2 Task 8: BLSTM input normalization + core bidirectional forward
    // + plain feedForwardBackward -------------------------------------------
    // legacy: BLSTMNeuralNetwork.cpp:419-437 (feedForward core), :711-751 (windowed
    // FFB entry -- normalization types 1/-1 mutate inputSeq IN PLACE, then dispatch),
    // :776-830 (plain feedForwardBackward -- type -2 into a COPY, then feedForward +
    // cost). This block reimpls the normalization + core-forward path with ascending
    // products (reusing lstmForwardLoop / netForwardLoop / netForwardDoubleLoop) and
    // probes it against the REAL BLSTMNeuralNetwork<LSTMLayer> on each dump.
    //
    // Dumps: a synthetic net LSTM [3,4,2] sub [2,1] / output [4,5,3] sub [1,1]
    // (output[0]==4==2*lstm.back()==2*2), T=12 closed-form input, one dump per
    // normalization type {1, -1, -2, 0} -> blstm_norm<type>_out.bin + the
    // (possibly-mutated) input blstm_norm<type>_input_after.bin. Plus the REAL net
    // (from the vendored config + weights) on a deterministic synthetic 200 x 23
    // input, plain FFB no targets -> blstm_real_fullseq_{out,fwd,bwd}.bin + its
    // NN_TOL (EXPECTED NONZERO: k=23 input GEMM + k=24 recurrence and layer-1 GEMMs + output-net (k=48/k=12) GEMM divergence).
    {
        // Reimpl of the whole-sequence self-normalization (type -1, in place;
        // BLSTMNeuralNetwork.cpp:737-744). mean = colSum/rows; center row-by-row;
        // std = sqrt((colSum(centered^2) + 1e-32)/rows) with 1e-32 added to the SUM;
        // then per row asinh(x/std). Keeps the row-loop op order.
        auto normalizeType_1 = [](Eigen::MatrixXd& in) {
            const int R = static_cast<int>(in.rows());
            const int C = static_cast<int>(in.cols());
            if (R == 0) return;
            std::vector<double> mean(C, 0.0);
            for (int c = 0; c < C; ++c) {
                double acc = 0.0;
                for (int r = 0; r < R; ++r) acc += in(r, c);
                mean[c] = acc / (double)R;
            }
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c) in(r, c) -= mean[c];   // :738-740 row-by-row
            std::vector<double> stdv(C, 0.0);
            for (int c = 0; c < C; ++c) {
                double acc = 0.0;
                for (int r = 0; r < R; ++r) acc += in(r, c) * in(r, c);
                stdv[c] = std::sqrt((acc + 1e-32) / (double)R);     // 1e-32 into the SUM
            }
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c) in(r, c) = Maxmin2::fn(in(r, c) / stdv[c]);
        };

        // Reimpl of external normalization (type 1, in place;
        // BLSTMNeuralNetwork.cpp:720-723). Per column jj < min(cols, mean.size()):
        // (col - mean_jj)/max(1e-12, std_jj); cols beyond mean.size() UNTOUCHED.
        auto normalizeType1 = [](Eigen::MatrixXd& in, const std::vector<double>& mean,
                                 const std::vector<double>& stdv) {
            const int R = static_cast<int>(in.rows());
            const long maxCol = std::min<long>(in.cols(), (long)mean.size());
            for (long jj = 0; jj < maxCol; ++jj) {
                double denom = std::max(1e-12, stdv[(size_t)jj]);
                for (int r = 0; r < R; ++r) in(r, jj) = (in(r, jj) - mean[(size_t)jj]) / denom;
            }
        };

        // Reimpl of per-window self-normalization (type -2, into a COPY;
        // BLSTMNeuralNetwork.cpp:778-781): replicate/whole-matrix form. Same formula
        // as type -1 but does NOT mutate the input; returns the normalized copy.
        auto normalizeType_2 = [](const Eigen::MatrixXd& in) {
            const int R = static_cast<int>(in.rows());
            const int C = static_cast<int>(in.cols());
            Eigen::MatrixXd meanRep(R, C), norm(R, C);
            for (int c = 0; c < C; ++c) {
                double acc = 0.0;
                for (int r = 0; r < R; ++r) acc += in(r, c);
                double m = acc / (double)R;
                for (int r = 0; r < R; ++r) meanRep(r, c) = m;
            }
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c) norm(r, c) = in(r, c) - meanRep(r, c);
            std::vector<double> stdv(C, 0.0);
            for (int c = 0; c < C; ++c) {
                double acc = 0.0;
                for (int r = 0; r < R; ++r) acc += norm(r, c) * norm(r, c);
                stdv[c] = std::sqrt((acc + 1e-32) / (double)R);
            }
            for (int r = 0; r < R; ++r)
                for (int c = 0; c < C; ++c) norm(r, c) = Maxmin2::fn(norm(r, c) / stdv[c]);
            return norm;
        };

        // Core feedForward reimpl (BLSTMNeuralNetwork.cpp:419-437): outputLength =
        // rows divided SEQUENTIALLY by each forward-net LSTM ratio; forward net
        // feedForward + backward net feedForwardReverse into (outputLength x lstmOut);
        // leftCols(fwdInputSize) gate when LSTMRatios[0] > 1 && input wider; output
        // net feedForwardDouble (forward LEFT). Fills outForward/outBackward + output.
        auto blstmFeedForward =
            [&](const std::vector<long>& lstmNeuronNb, const std::vector<long>& lstmSub,
                const std::vector<NetLayerStep>& fwdSteps, const std::vector<NetLayerStep>& bwdSteps,
                const std::vector<long>& outNeuronNb, const std::vector<long>& outSub,
                const std::vector<NetLayerStep>& outSteps, int fwdInputSize,
                const Eigen::MatrixXd& input, Eigen::MatrixXd& outForward,
                Eigen::MatrixXd& outBackward, Eigen::MatrixXd& output) {
                long outputLength = input.rows();
                for (size_t jj = 0; jj < lstmSub.size(); ++jj) outputLength /= lstmSub[jj];
                (void)outputLength;  // sizing is implicit in the reimpl steps
                Eigen::MatrixXd fwdIn = input, bwdIn = input;
                if (lstmSub[0] > 1 && fwdInputSize < input.cols()) {
                    fwdIn = input.leftCols(fwdInputSize);
                    bwdIn = input.leftCols(fwdInputSize);
                }
                netForwardLoop(lstmNeuronNb, lstmSub, fwdSteps, fwdIn, outForward);
                netForwardReverseLoop(lstmNeuronNb, lstmSub, bwdSteps, bwdIn, outBackward);
                netForwardDoubleLoop(outNeuronNb, outSub, outSteps, outForward, outBackward, output);
            };

        // Build the per-layer forward/reverse steps for an LSTM sub-network from its
        // flat weight slice (all peepholes on). Returns via out-params so the closures
        // capture stable references (the unpacked matrices live in the provided vecs).
        auto makeLstmSteps = [](const Eigen::VectorXd& flat,
                                const std::vector<int>& ins, const std::vector<int>& outs,
                                std::vector<Eigen::MatrixXd>& iw, std::vector<Eigen::MatrixXd>& fw,
                                std::vector<Eigen::MatrixXd>& pp, std::vector<Eigen::MatrixXd>& bs,
                                std::vector<NetLayerStep>& fwdSteps,
                                std::vector<NetLayerStep>& bwdSteps) {
                const size_t L = ins.size();
                iw.resize(L); fw.resize(L); pp.resize(L); bs.resize(L);
                fwdSteps.resize(L); bwdSteps.resize(L);
                long pos = 0;
                for (size_t jj = 0; jj < L; ++jj) {
                    int I = ins[jj], O = outs[jj];
                    long nb = 4L * I * O + 4L * O * O + 12L * O + 4L * O;
                    Eigen::VectorXd slice = flat.segment(pos, nb);
                    pos += nb;
                    unpackLstmWeights(slice, I, O, iw[jj], fw[jj], pp[jj], bs[jj]);
                }
                for (size_t jj = 0; jj < L; ++jj) {
                    int O = outs[jj];
                    fwdSteps[jj].forward = [&iw, &fw, &pp, &bs, jj, O](const Eigen::MatrixXd& in,
                                                                      Eigen::MatrixXd& outm, bool) {
                        Eigen::MatrixXd g;
                        lstmForwardLoop(in, iw[jj], fw[jj], pp[jj], bs[jj], O, true, true, true, g, outm);
                    };
                    bwdSteps[jj].forward = [&iw, &fw, &pp, &bs, jj, O](const Eigen::MatrixXd& in,
                                                                      Eigen::MatrixXd& outm, bool) {
                        Eigen::MatrixXd g;
                        lstmForwardReverseLoop(in, iw[jj], fw[jj], pp[jj], bs[jj], O, true, true, true, g, outm);
                    };
                }
                return pos;
            };

        // Build the per-layer steps for a dense (NeuronLayer) sub-network from its
        // flat weight slice. lastLayer flag is threaded by the driver.
        auto makeDenseSteps = [](const Eigen::VectorXd& flat,
                                 const std::vector<int>& ins, const std::vector<int>& outs,
                                 std::vector<Eigen::MatrixXd>& ws, std::vector<Eigen::MatrixXd>& bs,
                                 std::vector<NetLayerStep>& steps) {
                const size_t L = ins.size();
                ws.resize(L); bs.resize(L); steps.resize(L);
                long pos = 0;
                for (size_t jj = 0; jj < L; ++jj) {
                    int I = ins[jj], O = outs[jj];
                    long nb = (long)O * (I + 1);
                    Eigen::VectorXd slice = flat.segment(pos, nb);
                    pos += nb;
                    ws[jj].resize(I, O); bs[jj].resize(1, O);
                    for (int c = 0; c < O; ++c)
                        for (int r = 0; r < I; ++r) ws[jj](r, c) = slice((long)c * I + r);
                    for (int c = 0; c < O; ++c) bs[jj](0, c) = slice((long)O * I + c);
                }
                for (size_t jj = 0; jj < L; ++jj) {
                    steps[jj].forward = [&ws, &bs, jj](const Eigen::MatrixXd& in,
                                                       Eigen::MatrixXd& outm, bool last) {
                        outm = denseForwardLoop(in, ws[jj], bs[jj], last);
                    };
                }
                return pos;
            };

        // Max ULP/abs gap between the real BLSTM output and the reimpl output.
        auto probeGap = [](const Eigen::MatrixXd& real, const Eigen::MatrixXd& reimpl,
                           long& maxUlp, double& maxAbs) {
            for (int r = 0; r < real.rows(); ++r) {
                for (int c = 0; c < real.cols(); ++c) {
                    double a = real(r, c), b = reimpl(r, c);
                    double absGap = std::fabs(a - b);
                    if (absGap > maxAbs) maxAbs = absGap;
                    uint64_t ab, bb;
                    std::memcpy(&ab, &a, sizeof(double));
                    std::memcpy(&bb, &b, sizeof(double));
                    long ulp = (ab > bb) ? (long)(ab - bb) : (long)(bb - ab);
                    if (ulp > maxUlp) maxUlp = ulp;
                }
            }
        };

        // Same as probeGap, but identical-bit-pattern NaNs (incl. same-pattern NaN)
        // compare equal instead of registering as a mismatch -- matches the golden's
        // bitwise contract for the OverLap uncovered-rows 0/0->NaN case (Task 9) and
        // is used everywhere in Tasks 8/9 for uniformity, even where no NaN can occur.
        auto probeGapNaN = [](const Eigen::MatrixXd& real, const Eigen::MatrixXd& reimpl,
                              long& maxUlp, double& maxAbs) {
            if (real.rows() != reimpl.rows() || real.cols() != reimpl.cols()) {
                std::cerr << "probeGapNaN dims mismatch: real " << real.rows() << "x"
                          << real.cols() << " vs reimpl " << reimpl.rows() << "x"
                          << reimpl.cols() << "\n";
                std::abort();
            }
            for (int r = 0; r < real.rows(); ++r) {
                for (int c = 0; c < real.cols(); ++c) {
                    double a = real(r, c), b = reimpl(r, c);
                    uint64_t ab, bb;
                    std::memcpy(&ab, &a, sizeof(double));
                    std::memcpy(&bb, &b, sizeof(double));
                    if (ab == bb) continue;  // identical bits (incl. same-pattern NaN) -> equal
                    double absGap = std::fabs(a - b);
                    if (absGap > maxAbs) maxAbs = absGap;
                    long ulp = (ab > bb) ? (long)(ab - bb) : (long)(bb - ab);
                    if (ulp > maxUlp) maxUlp = ulp;
                }
            }
        };

        // ---- Synthetic net: LSTM [3,4,2] sub [2,1], output [4,5,3] sub [1,1] ----
        // A synthetic prefix "SYNB" is defined in the config with all the keys the
        // BLSTMNeuralNetwork ctor reads; only _InputNormalizationType varies per dump.
        {
            const std::vector<long> lstmNN = {3, 4, 2};
            const std::vector<long> lstmSS = {2, 1};
            const std::vector<long> outNN = {4, 5, 3};
            const std::vector<long> outSS = {1, 1};
            // Forward/backward LSTM per-layer (in, out): L0 in=3*2=6 out=4; L1 in=4*1=4 out=2.
            const std::vector<int> lstmIns = {6, 4}, lstmOuts = {4, 2};
            // Output dense per-layer: L0 in=4 out=5 (asinh); L1 in=5 out=3 (softmax).
            const std::vector<int> outIns = {4, 5}, outOuts = {5, 3};
            const int fwdInputSize = 3;  // neuronNb[0]
            const int T = 12;

            // Deterministic closed-form input (3 raw cols), same family as Task 4/5
            // but offset so the values are non-degenerate for normalization.
            Eigen::MatrixXd baseInput(T, 3);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < 3; ++j)
                    baseInput(t, j) = (double)(((t * 29 + j * 13 + 5) % 97)) / 97.0 - 0.5;

            // Synthetic BLSTM config prefix. Build once; flip InputNormalizationType
            // per dump. weightsSetExternally=true skips per-layer config weight reads.
            ConfigFile confB(nnConfigPath, '_');
            confB._Params.erase("BLSTM_weightsFile");
            confB.set_val<std::string>("SYNB_LSTMNeuronNb", "3,4,2");
            confB.set_val<std::string>("SYNB_LSTMSubSampling", "2,1");
            confB.set_val<std::string>("SYNB_OutputNeuronNb", "4,5,3");
            confB.set_val<std::string>("SYNB_OutputSubSampling", "1,1");
            confB.set_val<bool>("SYNB_TwoSweeps", false);
            confB.set_val<bool>("SYNB_BackPropagationActivated", false);
            // Peephole flags per direction all default true (unset keys).

            // nb of weights: forward + backward LSTM nets + output dense net + 2*3 tail.
            auto lstmNetNb = [&](const std::vector<int>& ins, const std::vector<int>& outs) {
                long s = 0;
                for (size_t jj = 0; jj < ins.size(); ++jj) {
                    int I = ins[jj], O = outs[jj];
                    s += 4L * I * O + 4L * O * O + 12L * O + 4L * O;
                }
                return s;
            };
            auto denseNetNb = [&](const std::vector<int>& ins, const std::vector<int>& outs) {
                long s = 0;
                for (size_t jj = 0; jj < ins.size(); ++jj) s += (long)outs[jj] * (ins[jj] + 1);
                return s;
            };
            const long fwdNb = lstmNetNb(lstmIns, lstmOuts);
            const long bwdNb = fwdNb;
            const long outNb = denseNetNb(outIns, outOuts);
            const long tailNb = 2 * fwdInputSize;
            const long nbTotal = fwdNb + bwdNb + outNb + tailNb;

            // Synthetic flat vector with a NONZERO mean/std tail (so type 1 actually
            // shifts/scales). Base formula for the body; the mean tail = 0.1*(idx+1),
            // the std tail = 1.0 + 0.05*idx (all > 0 so max(1e-12,std)=std).
            Eigen::VectorXd flat(nbTotal);
            for (long k = 0; k < nbTotal - tailNb; ++k)
                flat(k) = (double)((k * 11 + 3) % 97) / 97.0 - 0.5;
            std::vector<double> meanTail(fwdInputSize), stdTail(fwdInputSize);
            for (int j = 0; j < fwdInputSize; ++j) {
                meanTail[(size_t)j] = 0.1 * (j + 1);
                stdTail[(size_t)j] = 1.0 + 0.05 * j;
                flat(nbTotal - tailNb + j) = meanTail[(size_t)j];
                flat(nbTotal - tailNb + fwdInputSize + j) = stdTail[(size_t)j];
            }

            // Unpack the reimpl weight steps ONCE (shared across all norm dumps -- the
            // weights do not change, only the normalization type + input do).
            Eigen::VectorXd fwdSlice = flat.head(fwdNb);
            Eigen::VectorXd bwdSlice = flat.segment(fwdNb, bwdNb);
            Eigen::VectorXd outSlice = flat.segment(fwdNb + bwdNb, outNb);
            std::vector<Eigen::MatrixXd> fiw, ffw, fpp, fbs, biw, bfw, bpp, bbs, ows, obs;
            std::vector<NetLayerStep> fwdSteps, fwdStepsRev, bwdSteps, bwdStepsRev, outSteps;
            makeLstmSteps(fwdSlice, lstmIns, lstmOuts, fiw, ffw, fpp, fbs, fwdSteps, fwdStepsRev);
            makeLstmSteps(bwdSlice, lstmIns, lstmOuts, biw, bfw, bpp, bbs, bwdSteps, bwdStepsRev);
            makeDenseSteps(outSlice, outIns, outOuts, ows, obs, outSteps);

            for (int normType : {1, -1, -2, 0}) {
                confB.set_val<short>("SYNB_InputNormalizationType", (short)normType);
                BLSTMNeuralNetwork<LSTMLayer> nn(confB, "SYNB", true);
                nn.setWeights(flat);
                nn.setProcessingType(false, false);  // plain path (no truncate/overlap)

                // Real class: windowed FFB entry with no targets. window_size/shift
                // are unused on the plain (non-truncate) path.
                Eigen::MatrixXd realInput = baseInput;   // MUTATED for type 1/-1
                Eigen::MatrixXd realOut(6, 3);           // outputLength = 12/2/1 = 6, out 3
                Eigen::MatrixXd emptyTargets;
                nn.feedForwardBackward(realInput, 4, 2, realOut, emptyTargets);

                // Reimpl: mirror the windowed entry's normalization + dispatch to the
                // plain FFB. Type 1/-1 mutate the reimpl input in place; type -2
                // normalizes into a copy inside the plain FFB; type 0 does nothing.
                Eigen::MatrixXd reimplInput = baseInput;
                Eigen::MatrixXd reimplOut, outForward, outBackward;
                if (normType == 1) {
                    normalizeType1(reimplInput, meanTail, stdTail);
                    // analyseInputSeq updates _InputStatistics (no effect on output).
                    blstmFeedForward(lstmNN, lstmSS, fwdSteps, bwdStepsRev, outNN, outSS, outSteps,
                                     fwdInputSize, reimplInput, outForward, outBackward, reimplOut);
                } else if (normType == -1) {
                    normalizeType_1(reimplInput);
                    blstmFeedForward(lstmNN, lstmSS, fwdSteps, bwdStepsRev, outNN, outSS, outSteps,
                                     fwdInputSize, reimplInput, outForward, outBackward, reimplOut);
                } else if (normType == -2) {
                    Eigen::MatrixXd norm = normalizeType_2(reimplInput);  // input untouched
                    blstmFeedForward(lstmNN, lstmSS, fwdSteps, bwdStepsRev, outNN, outSS, outSteps,
                                     fwdInputSize, norm, outForward, outBackward, reimplOut);
                } else {
                    blstmFeedForward(lstmNN, lstmSS, fwdSteps, bwdStepsRev, outNN, outSS, outSteps,
                                     fwdInputSize, reimplInput, outForward, outBackward, reimplOut);
                }

                // Site tag: negative types use an 'm' prefix so the NN_TOL site name
                // stays \w-safe (the extractor regex is `site=(\w+)`).
                std::string tag = (normType < 0) ? ("m" + std::to_string(-normType))
                                                 : std::to_string(normType);

                long maxUlp = 0; double maxAbs = 0.0;
                probeGapNaN(realOut, reimplOut, maxUlp, maxAbs);
                std::cout << "NN_TOL site=blstm_norm" << tag
                          << " max_ulp=" << maxUlp << " max_abs=" << std::scientific
                          << std::setprecision(3) << maxAbs << "\n";

                // Verify the reimpl's in-place mutation matches the real class's
                // mutation (types 1/-1 mutate; -2/0 leave the input identical).
                long inUlp = 0; double inAbs = 0.0;
                probeGapNaN(realInput, reimplInput, inUlp, inAbs);
                std::cout << "NN_TOL site=blstm_norm" << tag << "_input"
                          << " max_ulp=" << inUlp << " max_abs=" << std::scientific
                          << std::setprecision(3) << inAbs << "\n";

                // _OutputForward/_OutputBackward ARE PUBLIC members (BLSTMNeuralNetwork.h:
                // 23-24, no accessor needed) -- probe the hidden states vs the reimpl's
                // local outForward/outBackward too, per normalization type.
                long fUlp = 0; double fAbs = 0.0;
                probeGapNaN(nn._OutputForward, outForward, fUlp, fAbs);
                std::cout << "NN_TOL site=blstm_norm" << tag << "_fwd"
                          << " max_ulp=" << fUlp << " max_abs=" << std::scientific
                          << std::setprecision(3) << fAbs << "\n";
                long bUlp = 0; double bAbs = 0.0;
                probeGapNaN(nn._OutputBackward, outBackward, bUlp, bAbs);
                std::cout << "NN_TOL site=blstm_norm" << tag << "_bwd"
                          << " max_ulp=" << bUlp << " max_abs=" << std::scientific
                          << std::setprecision(3) << bAbs << "\n";

                Matrix2BinaryFile(out + "blstm_norm" + tag + "_out.bin", reimplOut);
                Matrix2BinaryFile(out + "blstm_norm" + tag + "_input_after.bin", reimplInput);
                Matrix2BinaryFile(out + "blstm_norm" + tag + "_fwd.bin", outForward);
                Matrix2BinaryFile(out + "blstm_norm" + tag + "_bwd.bin", outBackward);
                dumps += 4;
            }
        }

        // ---- Real net full-sequence: plain FFB, no targets, synthetic 200 x 23 ----
        // The real BLSTMNeuralNetwork<LSTMLayer> (1_worker_1.config + NNweights, 33671)
        // has _InputNormalizationType == -1 -> whole-sequence self-norm in place. Run
        // the real windowed FFB (plain path) and reproduce it with the reimpl, dumping
        // the output + the forward/backward LSTM hidden states.
        {
            ConfigFile conf(nnConfigPath, '_');
            conf._Params.erase("BLSTM_weightsFile");
            BLSTMNeuralNetwork<LSTMLayer> nn(conf, "BLSTM", true);
            Eigen::VectorXd flat = BinaryFile2Vector(nnWeightsPath);
            nn.setWeights(flat);
            nn.setProcessingType(false, false);

            const int T = 200;
            Eigen::MatrixXd baseInput(T, 23);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < 23; ++j)
                    baseInput(t, j) = (double)(((t * 31 + j * 17) % 100)) / 100.0 - 0.5;

            // Real class (input MUTATED by the type -1 self-norm).
            Eigen::MatrixXd realInput = baseInput;
            const long outRows = T / 4;  // fwd LSTM sub 4*1 = 4 -> 200/4 = 50
            Eigen::MatrixXd realOut(outRows, 1);
            Eigen::MatrixXd emptyTargets;
            nn.feedForwardBackward(realInput, 4, 2, realOut, emptyTargets);

            // Reimpl. Real net: LSTM [23,24,24] sub [4,1]; output [48,12,1] sub [1,1].
            const std::vector<long> lstmNN = {23, 24, 24};
            const std::vector<long> lstmSS = {4, 1};
            const std::vector<long> outNN = {48, 12, 1};
            const std::vector<long> outSS = {1, 1};
            const std::vector<int> lstmIns = {92, 24}, lstmOuts = {24, 24};  // 23*4=92, 24*1=24
            const std::vector<int> outIns = {48, 12}, outOuts = {12, 1};
            const int fwdInputSize = 23;

            long fwdNb = 0;
            for (size_t jj = 0; jj < lstmIns.size(); ++jj) {
                int I = lstmIns[jj], O = lstmOuts[jj];
                fwdNb += 4L * I * O + 4L * O * O + 12L * O + 4L * O;
            }
            long outNbW = 0;
            for (size_t jj = 0; jj < outIns.size(); ++jj) outNbW += (long)outOuts[jj] * (outIns[jj] + 1);

            Eigen::VectorXd fwdSlice = flat.head(fwdNb);
            Eigen::VectorXd bwdSlice = flat.segment(fwdNb, fwdNb);
            Eigen::VectorXd outSlice = flat.segment(2 * fwdNb, outNbW);
            std::vector<Eigen::MatrixXd> fiw, ffw, fpp, fbs, biw, bfw, bpp, bbs, ows, obs;
            std::vector<NetLayerStep> fwdSteps, fwdStepsRev, bwdSteps, bwdStepsRev, outSteps;
            makeLstmSteps(fwdSlice, lstmIns, lstmOuts, fiw, ffw, fpp, fbs, fwdSteps, fwdStepsRev);
            makeLstmSteps(bwdSlice, lstmIns, lstmOuts, biw, bfw, bpp, bbs, bwdSteps, bwdStepsRev);
            makeDenseSteps(outSlice, outIns, outOuts, ows, obs, outSteps);

            // Type -1 self-norm in place, then core forward.
            Eigen::MatrixXd reimplInput = baseInput;
            normalizeType_1(reimplInput);
            Eigen::MatrixXd reimplOut, outForward, outBackward;
            blstmFeedForward(lstmNN, lstmSS, fwdSteps, bwdStepsRev, outNN, outSS, outSteps,
                             fwdInputSize, reimplInput, outForward, outBackward, reimplOut);

            long maxUlp = 0; double maxAbs = 0.0;
            probeGapNaN(realOut, reimplOut, maxUlp, maxAbs);
            std::cout << "NN_TOL site=blstm_real_fullseq max_ulp=" << maxUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbs << "\n";

            // _OutputForward/_OutputBackward ARE PUBLIC members (BLSTMNeuralNetwork.h:
            // 23-24, no accessor needed). EXPECTED NONZERO here too, same root cause as
            // the output site: k=23 input GEMM + k=24 recurrence and layer-1 GEMMs + output-net (k=48/k=12) GEMM divergence
            // (Task 1 lstm_input_gemm probe).
            long fUlp = 0; double fAbs = 0.0;
            probeGapNaN(nn._OutputForward, outForward, fUlp, fAbs);
            std::cout << "NN_TOL site=blstm_real_fullseq_fwd max_ulp=" << fUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << fAbs << "\n";
            long bUlp = 0; double bAbs = 0.0;
            probeGapNaN(nn._OutputBackward, outBackward, bUlp, bAbs);
            std::cout << "NN_TOL site=blstm_real_fullseq_bwd max_ulp=" << bUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << bAbs << "\n";

            Matrix2BinaryFile(out + "blstm_real_fullseq_out.bin", reimplOut);
            Matrix2BinaryFile(out + "blstm_real_fullseq_fwd.bin", outForward);
            Matrix2BinaryFile(out + "blstm_real_fullseq_bwd.bin", outBackward);
            dumps += 3;
        }

        // ================================================================
        // Phase 2 Task 9: the four windowed forward drivers.
        // legacy: BLSTMNeuralNetwork.cpp:488-546 (feedForwardBackwardTruncateSweep),
        // :548-590 (feedForwardBackwardTruncate / TwoSweeps), :592-681
        // (feedForwardBackwardOverLap), :683-709 (feedForwardBackwardMLPOverLap).
        // The reimpls transcribe those methods' integer arithmetic + write-back index
        // math verbatim, calling the plain-FFB reimpl (blstmFeedForward, no targets)
        // per window and stitching. Probed vs the REAL class per mode -- OUTPUT and
        // hidden states alike: _OutputForward/_OutputBackward ARE PUBLIC members
        // (BLSTMNeuralNetwork.h:23-24, no accessor needed), populated by
        // feedForwardBackward, so `nn._OutputForward`/`nn._OutputBackward` are read
        // directly and compared to the reimpl's fwd/bwd dumps (blstm_<mode>_fwd/_bwd
        // NN_TOL sites). MLPOverLap explicitly EMPTIES both members (no LSTM in the MLP
        // path); that site's fwd/bwd probes assert 0x0 on both sides instead. Dumps:
        // blstm_<mode>_{out,fwd,bwd}.bin (+ counts where illuminating).
        //
        // NaN handling: the OverLap uncovered-rows case divides 0/0 -> NaN (the legacy
        // reproduces this, no guard). probeGapNaN (defined above, shared with Task 8)
        // treats IDENTICAL NaN bit patterns as equal (bitwise) before the arithmetic
        // gap -- matching the golden's bitwise contract.

        // ---- Synthetic BLSTM net (same as Task 8): LSTM [3,4,2] sub [2,1], output
        // [4,5,3] sub [1,1]. Reuse the Task 8 makeLstmSteps/makeDenseSteps/blstmFeedForward
        // helpers (outer-scope lambdas). InputNormalizationType 0 (no mutation) keeps the
        // windowed-arithmetic reasoning independent of normalization.
        {
            const std::vector<long> lstmNN = {3, 4, 2};
            const std::vector<long> lstmSS = {2, 1};
            const std::vector<long> outNN = {4, 5, 3};
            const std::vector<long> outSS = {1, 1};
            const std::vector<int> lstmIns = {6, 4}, lstmOuts = {4, 2};
            const std::vector<int> outIns = {4, 5}, outOuts = {5, 3};
            const int fwdInputSize = 3;
            const long subRatio = 2;         // getSubSamplingRatio() = lstm(2)*out(1)
            const long outNetRatio = 1;      // _OutputNetwork.getSubSamplingRatio()
            const long lstmSubRatio = 2;     // _ForwardNetwork.getSubSamplingRatio()
            const int fwdOut = 2;            // last LSTM layer output size

            // Synthetic BLSTM config prefix (SYNB); topology fixed, only the per-mode
            // knobs (InputNormalizationType, TwoSweeps) flip below.
            ConfigFile confB(nnConfigPath, '_');
            confB._Params.erase("BLSTM_weightsFile");
            confB.set_val<std::string>("SYNB_LSTMNeuronNb", "3,4,2");
            confB.set_val<std::string>("SYNB_LSTMSubSampling", "2,1");
            confB.set_val<std::string>("SYNB_OutputNeuronNb", "4,5,3");
            confB.set_val<std::string>("SYNB_OutputSubSampling", "1,1");
            confB.set_val<bool>("SYNB_BackPropagationActivated", false);

            auto lstmNetNb = [&](const std::vector<int>& ins, const std::vector<int>& outs) {
                long s = 0;
                for (size_t jj = 0; jj < ins.size(); ++jj) {
                    int I = ins[jj], O = outs[jj];
                    s += 4L * I * O + 4L * O * O + 12L * O + 4L * O;
                }
                return s;
            };
            auto denseNetNb = [&](const std::vector<int>& ins, const std::vector<int>& outs) {
                long s = 0;
                for (size_t jj = 0; jj < ins.size(); ++jj) s += (long)outs[jj] * (ins[jj] + 1);
                return s;
            };
            const long fwdNb = lstmNetNb(lstmIns, lstmOuts);
            const long bwdNb = fwdNb;
            const long outNb = denseNetNb(outIns, outOuts);
            const long tailNb = 2 * fwdInputSize;
            const long nbTotal = fwdNb + bwdNb + outNb + tailNb;

            Eigen::VectorXd flat(nbTotal);
            for (long k = 0; k < nbTotal - tailNb; ++k)
                flat(k) = (double)((k * 11 + 3) % 97) / 97.0 - 0.5;
            for (int j = 0; j < fwdInputSize; ++j) {
                flat(nbTotal - tailNb + j) = 0.1 * (j + 1);
                flat(nbTotal - tailNb + fwdInputSize + j) = 1.0 + 0.05 * j;
            }

            Eigen::VectorXd fwdSlice = flat.head(fwdNb);
            Eigen::VectorXd bwdSlice = flat.segment(fwdNb, bwdNb);
            Eigen::VectorXd outSlice = flat.segment(fwdNb + bwdNb, outNb);
            std::vector<Eigen::MatrixXd> fiw, ffw, fpp, fbs, biw, bfw, bpp, bbs, ows, obs;
            std::vector<NetLayerStep> fwdSteps, fwdStepsRev, bwdSteps, bwdStepsRev, outSteps;
            makeLstmSteps(fwdSlice, lstmIns, lstmOuts, fiw, ffw, fpp, fbs, fwdSteps, fwdStepsRev);
            makeLstmSteps(bwdSlice, lstmIns, lstmOuts, biw, bfw, bpp, bbs, bwdSteps, bwdStepsRev);
            makeDenseSteps(outSlice, outIns, outOuts, ows, obs, outSteps);

            // Per-window plain FFB reimpl (no normalization/targets): forward only,
            // filling outF/outB (LSTM hidden states) + output. Wraps blstmFeedForward.
            auto plainWindow = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outF,
                                   Eigen::MatrixXd& outB, Eigen::MatrixXd& outp) {
                blstmFeedForward(lstmNN, lstmSS, fwdSteps, bwdStepsRev, outNN, outSS, outSteps,
                                 fwdInputSize, in, outF, outB, outp);
            };

            // Deterministic closed-form input (3 raw cols), T rows.
            auto makeInput = [&](int T) {
                Eigen::MatrixXd m(T, 3);
                for (int t = 0; t < T; ++t)
                    for (int j = 0; j < 3; ++j)
                        m(t, j) = (double)(((t * 29 + j * 13 + 5) % 97)) / 97.0 - 0.5;
                return m;
            };

            // Reimpl of feedForwardBackwardTruncateSweep (:488-546). outF/outB are the
            // caller-visible _OutputForward/_OutputBackward AFTER the sweep (stitched).
            auto truncateSweep = [&](const Eigen::MatrixXd& input, long window_size,
                                     Eigen::MatrixXd& outputSeq, Eigen::MatrixXd& outF,
                                     Eigen::MatrixXd& outB) {
                long lengthOutputLSTM = input.rows();
                for (long r : lstmSS) lengthOutputLSTM /= r;                 // :493-496
                Eigen::MatrixXd outputForward = Eigen::MatrixXd::Zero(lengthOutputLSTM, fwdOut);
                Eigen::MatrixXd outputBackward = Eigen::MatrixXd::Zero(lengthOutputLSTM, fwdOut);

                long length = window_size, lstmLenShort = length;           // :500-512
                for (long r : lstmSS) length /= r;
                lstmLenShort = length;
                for (long r : outSS) length /= r;
                long nominalLen = length, nominalLstm = lstmLenShort;

                for (long jj = 0; jj < input.rows(); jj += window_size) {    // :513
                    long begin = jj;
                    long end = jj + window_size - 1;
                    if (end >= input.rows()) end = input.rows() - 1;
                    long lengthSeq = end - begin + 1;
                    long lengthShort, lstmLength;
                    if (lengthSeq != window_size) {                         // :518-529
                        lengthShort = lengthSeq;
                        for (long r : lstmSS) lengthShort /= r;
                        lstmLength = lengthShort;
                        for (long r : outSS) lengthShort /= r;
                    } else {
                        lengthShort = nominalLen;
                        lstmLength = nominalLstm;
                    }
                    if (lengthShort > 0) {                                  // :534
                        Eigen::MatrixXd block = input.block(begin, 0, lengthSeq, input.cols());
                        Eigen::MatrixXd outShort, oF, oB;
                        plainWindow(block, oF, oB, outShort);
                        outputSeq.block(begin / subRatio, 0, lengthShort, outputSeq.cols()) = outShort;  // :539
                        outputForward.block(begin / lstmSubRatio, 0, lstmLength, fwdOut) = oF;           // :540
                        outputBackward.block(begin / lstmSubRatio, 0, lstmLength, fwdOut) = oB;          // :541
                    }
                }
                outF = outputForward;                                      // :544-545
                outB = outputBackward;
            };

            // ---- Mode A: TruncateSweep, T=19 window 6 (trailing chunk lengthShort==0).
            // T=19: chunks at jj=0,6,12 (full) then jj=18 (1 input row) -> lengthSeq=1,
            // lengthShort = 1/2/1 = 0 -> SILENTLY DROPPED. The first three chunks write
            // outputSeq rows [0,3),[3,6),[6,9); outputSeq is allocated with 10 rows and
            // PRE-SEEDED to zero, so ROW 9 (the dropped chunk's begin/subRatio=18/2=9
            // target) stays 0 in both the real class and the reimpl. Pre-seeding both
            // buffers identically makes the untouched row compare bit-equal.
            {
                const int T = 19;
                Eigen::MatrixXd input = makeInput(T);
                confB.set_val<short>("SYNB_InputNormalizationType", (short)0);
                confB.set_val<bool>("SYNB_TwoSweeps", false);
                BLSTMNeuralNetwork<LSTMLayer> nn(confB, "SYNB", true);
                nn.setWeights(flat);
                nn.setProcessingType(true, false);  // truncate, no overlap -> Truncate path

                // _OutputForward/_OutputBackward ARE PUBLIC members (BLSTMNeuralNetwork.h:
                // 23-24, no accessor needed) -- populated by feedForwardBackward, so both
                // the OUTPUT and the hidden states are probed against the real class here.
                Eigen::MatrixXd realInput = input;
                Eigen::MatrixXd realOut = Eigen::MatrixXd::Zero(10, 3);  // 10 rows, row 9 untouched
                Eigen::MatrixXd emptyTargets;
                nn.feedForwardBackward(realInput, 6, 3, realOut, emptyTargets);

                Eigen::MatrixXd reimplOut = Eigen::MatrixXd::Zero(10, 3);  // pre-seeded IDENTICALLY
                Eigen::MatrixXd reimplF, reimplB;
                truncateSweep(input, 6, reimplOut, reimplF, reimplB);

                long u = 0; double a = 0.0;
                probeGapNaN(realOut, reimplOut, u, a);
                std::cout << "NN_TOL site=blstm_truncate max_ulp=" << u
                          << " max_abs=" << std::scientific << std::setprecision(3) << a << "\n";

                long uF = 0; double aF = 0.0;
                probeGapNaN(nn._OutputForward, reimplF, uF, aF);
                std::cout << "NN_TOL site=blstm_truncate_fwd max_ulp=" << uF
                          << " max_abs=" << std::scientific << std::setprecision(3) << aF << "\n";
                long uB = 0; double aB = 0.0;
                probeGapNaN(nn._OutputBackward, reimplB, uB, aB);
                std::cout << "NN_TOL site=blstm_truncate_bwd max_ulp=" << uB
                          << " max_abs=" << std::scientific << std::setprecision(3) << aB << "\n";

                Matrix2BinaryFile(out + "blstm_truncate_out.bin", reimplOut);
                Matrix2BinaryFile(out + "blstm_truncate_fwd.bin", reimplF);
                Matrix2BinaryFile(out + "blstm_truncate_bwd.bin", reimplB);
                dumps += 3;
            }

            // ---- Mode B: TwoSweeps, T=20 window 8. shiftShort=(8/2)/2=2, shift=4,
            // window_sizeShort=8/2=4. Two front-padded sweeps averaged; _OutputForward/
            // _OutputBackward come out DOUBLE-WIDTH (hcat of the two sweeps' hidden
            // windows) -- asserted in-harness below.
            {
                const int T = 20;
                Eigen::MatrixXd input = makeInput(T);
                const long windowSize = 8;
                confB.set_val<short>("SYNB_InputNormalizationType", (short)0);
                confB.set_val<bool>("SYNB_TwoSweeps", true);
                BLSTMNeuralNetwork<LSTMLayer> nn(confB, "SYNB", true);
                nn.setWeights(flat);
                nn.setProcessingType(true, false);  // truncate -> Truncate path; TwoSweeps on

                Eigen::MatrixXd realInput = input;
                Eigen::MatrixXd realOut = Eigen::MatrixXd::Zero(T / subRatio, 3);  // 10 x 3
                Eigen::MatrixXd emptyTargets;
                nn.feedForwardBackward(realInput, windowSize, 4, realOut, emptyTargets);

                // Reimpl of feedForwardBackwardTruncate TwoSweeps branch (:549-586).
                const long shiftShort = (windowSize / 2) / subRatio;   // :550 window/2 FIRST
                const long shift = shiftShort * subRatio;              // :551
                const long windowSizeShort = windowSize / subRatio;    // :552
                const long outRows = realOut.rows();

                auto replicateEnds = [](const Eigen::MatrixXd& m, long front, long back) {
                    Eigen::MatrixXd out(front + m.rows() + back, m.cols());
                    out << m.row(0).replicate(front, 1), m, m.row(m.rows() - 1).replicate(back, 1);
                    return out;
                };
                Eigen::MatrixXd inputPadded = replicateEnds(input, windowSize, windowSize);   // :553-554
                Eigen::MatrixXd outputSeq = Eigen::MatrixXd::Zero(outRows, 3);
                Eigen::MatrixXd outputPadded =
                    replicateEnds(outputSeq, windowSizeShort, windowSizeShort);              // :555-556

                // Sweep 1: drop FRONT windowSize padding, KEEP back padding (:565).
                Eigen::MatrixXd sweep1In =
                    inputPadded.block(windowSize, 0, inputPadded.rows() - windowSize, inputPadded.cols());
                Eigen::MatrixXd sweep1Out =
                    outputPadded.block(windowSizeShort, 0, outputPadded.rows() - windowSizeShort, outputPadded.cols());
                Eigen::MatrixXd sF, sB;
                truncateSweep(sweep1In, windowSize, sweep1Out, sF, sB);

                Eigen::MatrixXd combined = sweep1Out.block(0, 0, outRows, sweep1Out.cols());   // :568
                long memRows = outRows * outNetRatio;
                Eigen::MatrixXd memF = sF.block(0, 0, memRows, sF.cols());                     // :569
                Eigen::MatrixXd memB = sB.block(0, 0, memRows, sB.cols());

                // Sweep 2: from block(shift), into block(shiftShort) of a zeroed pad (:572-576).
                Eigen::MatrixXd outputPadded2 = Eigen::MatrixXd::Zero(outputPadded.rows(), outputPadded.cols());
                Eigen::MatrixXd sweep2In =
                    inputPadded.block(shift, 0, inputPadded.rows() - shift, inputPadded.cols());
                Eigen::MatrixXd sweep2Out =
                    outputPadded2.block(shiftShort, 0, outputPadded2.rows() - shiftShort, outputPadded2.cols());
                Eigen::MatrixXd sF2, sB2;
                truncateSweep(sweep2In, windowSize, sweep2Out, sF2, sB2);
                outputPadded2.block(shiftShort, 0, outputPadded2.rows() - shiftShort, outputPadded2.cols()) = sweep2Out;

                combined += outputPadded2.block(windowSizeShort, 0, outRows, outputPadded2.cols());  // :578
                combined /= 2.0;                                                                     // :579
                long memOff = (windowSizeShort - shiftShort) * outNetRatio;                          // :580
                Eigen::MatrixXd memFs = sF2.block(memOff, 0, memRows, sF2.cols());
                Eigen::MatrixXd memBs = sB2.block(memOff, 0, memRows, sB2.cols());

                Eigen::MatrixXd reimplF(memF.rows(), memF.cols() + memFs.cols());   // :583-584 HCAT
                reimplF << memF, memFs;
                Eigen::MatrixXd reimplB(memB.rows(), memB.cols() + memBs.cols());
                reimplB << memB, memBs;

                // In-harness double-width assertion: the reimpl _OutputForward/
                // _OutputBackward are the HCAT of the two sweeps' hidden windows, so
                // both come out 2*fwdOut wide (the real class does the same; its PUBLIC
                // members are probed directly against the reimpl just below).
                if (reimplF.cols() != 2 * fwdOut || reimplB.cols() != 2 * fwdOut) {
                    std::cerr << "TwoSweeps double-width FAIL: reimplF.cols=" << reimplF.cols()
                              << " reimplB.cols=" << reimplB.cols() << " expected " << 2 * fwdOut << "\n";
                    return 1;
                }

                long u = 0; double a = 0.0;
                probeGapNaN(realOut, combined, u, a);
                std::cout << "NN_TOL site=blstm_twosweeps max_ulp=" << u
                          << " max_abs=" << std::scientific << std::setprecision(3) << a << "\n";

                // _OutputForward/_OutputBackward are PUBLIC (BLSTMNeuralNetwork.h:23-24);
                // probe the double-width hidden states against the real class too.
                long uF = 0; double aF = 0.0;
                probeGapNaN(nn._OutputForward, reimplF, uF, aF);
                std::cout << "NN_TOL site=blstm_twosweeps_fwd max_ulp=" << uF
                          << " max_abs=" << std::scientific << std::setprecision(3) << aF << "\n";
                long uB = 0; double aB = 0.0;
                probeGapNaN(nn._OutputBackward, reimplB, uB, aB);
                std::cout << "NN_TOL site=blstm_twosweeps_bwd max_ulp=" << uB
                          << " max_abs=" << std::scientific << std::setprecision(3) << aB << "\n";

                Matrix2BinaryFile(out + "blstm_twosweeps_out.bin", combined);
                Matrix2BinaryFile(out + "blstm_twosweeps_fwd.bin", reimplF);
                Matrix2BinaryFile(out + "blstm_twosweeps_bwd.bin", reimplB);
                dumps += 3;
            }

            // Reimpl of feedForwardBackwardOverLap (:592-681). Accumulate windowed sums +
            // per-row counts, then quotient (0/0 -> NaN on uncovered rows, no guard).
            auto overlap = [&](const Eigen::MatrixXd& input, long windowSize, long windowShift,
                               Eigen::MatrixXd& outputSeq, Eigen::MatrixXd& outF, Eigen::MatrixXd& outB,
                               Eigen::MatrixXd& outCount, Eigen::MatrixXd& outCountLSTM) {
                long lengthOutputLSTM = outputSeq.rows() * outNetRatio;                 // :596
                Eigen::MatrixXd outputForward = Eigen::MatrixXd::Zero(lengthOutputLSTM, fwdOut);
                Eigen::MatrixXd outputBackward = Eigen::MatrixXd::Zero(lengthOutputLSTM, fwdOut);
                outCount = Eigen::MatrixXd::Zero(outputSeq.rows(), 1);                  // :599
                outCountLSTM = Eigen::MatrixXd::Zero(lengthOutputLSTM, 1);              // :600
                Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(outputSeq.rows(), outputSeq.cols());

                long length = 2 * windowSize + 1;                                       // :602-609
                for (long r : lstmSS) length /= r;
                for (long r : outSS) length /= r;
                long nominalLen = length, nominalLstm = length;

                for (long jj = 0; jj < input.rows(); jj += windowShift) {               // :613
                    long begin = (jj < windowSize) ? 0 : jj - windowSize;               // :614-619
                    begin = (begin / subRatio) * subRatio;                              // :620 snap down
                    long end = begin + 2 * windowSize;                                  // :621
                    if (end >= input.rows()) end = input.rows() - 1;                    // :623-625
                    end = ((end + 1) / subRatio) * subRatio - 1;                        // :626 snap up
                    long lengthSeq = end - begin + 1;                                   // :627
                    long lengthShort, lengthShortLSTM;
                    if (lengthSeq != 2 * windowSize + 1) {                              // :628-639
                        lengthShort = lengthSeq;
                        lengthShortLSTM = lengthSeq;
                        for (long r : lstmSS) { lengthShort /= r; lengthShortLSTM /= r; }
                        for (long r : outSS) lengthShort /= r;
                    } else {
                        lengthShort = nominalLen;
                        lengthShortLSTM = nominalLstm;
                    }
                    if (lengthShort > 0) {                                              // :645
                        Eigen::MatrixXd block = input.block(begin, 0, lengthSeq, input.cols());
                        Eigen::MatrixXd outShort, oF, oB;
                        plainWindow(block, oF, oB, outShort);
                        sum.block(begin / subRatio, 0, lengthShort, sum.cols()) += outShort;              // :664
                        outCount.block(begin / subRatio, 0, lengthShort, 1) += Eigen::MatrixXd::Ones(lengthShort, 1); // :665
                        long lbeg = begin * outNetRatio / subRatio;                                       // :667-669
                        outputForward.block(lbeg, 0, lengthShortLSTM, fwdOut) += oF;
                        outputBackward.block(lbeg, 0, lengthShortLSTM, fwdOut) += oB;
                        outCountLSTM.block(lbeg, 0, lengthShortLSTM, 1) += Eigen::MatrixXd::Ones(lengthShortLSTM, 1);
                    }
                }
                for (long c = 0; c < outputSeq.cols(); ++c)                             // :674-675 (0/0->NaN)
                    for (long r = 0; r < outputSeq.rows(); ++r)
                        outputSeq(r, c) = sum(r, c) / outCount(r, 0);
                for (long c = 0; c < fwdOut; ++c)                                       // :677-679 bound=cols
                    for (long r = 0; r < lengthOutputLSTM; ++r) {
                        outputForward(r, c) /= outCountLSTM(r, 0);
                        outputBackward(r, c) /= outCountLSTM(r, 0);
                    }
                outF = outputForward;
                outB = outputBackward;
            };

            // ---- Mode C1: OverLap, T=20 window_size 4 shift 3 (grid snapping exercised;
            // FULL coverage, no NaN). begin snaps 5->4, 11->10; end snapped up per window.
            {
                const int T = 20;
                Eigen::MatrixXd input = makeInput(T);
                confB.set_val<short>("SYNB_InputNormalizationType", (short)0);
                confB.set_val<bool>("SYNB_TwoSweeps", false);
                BLSTMNeuralNetwork<LSTMLayer> nn(confB, "SYNB", true);
                nn.setWeights(flat);
                nn.setProcessingType(true, true);  // truncate + overlap -> OverLap path

                Eigen::MatrixXd realInput = input;
                Eigen::MatrixXd realOut = Eigen::MatrixXd::Zero(T / subRatio, 3);  // 10 x 3
                Eigen::MatrixXd emptyTargets;
                nn.feedForwardBackward(realInput, 4, 3, realOut, emptyTargets);

                Eigen::MatrixXd reimplOut = Eigen::MatrixXd::Zero(T / subRatio, 3);
                Eigen::MatrixXd reimplF, reimplB, cnt, cntL;
                overlap(input, 4, 3, reimplOut, reimplF, reimplB, cnt, cntL);

                long u = 0; double a = 0.0;
                probeGapNaN(realOut, reimplOut, u, a);
                std::cout << "NN_TOL site=blstm_overlap max_ulp=" << u
                          << " max_abs=" << std::scientific << std::setprecision(3) << a << "\n";

                // _OutputForward/_OutputBackward are PUBLIC (BLSTMNeuralNetwork.h:23-24).
                long uF = 0; double aF = 0.0;
                probeGapNaN(nn._OutputForward, reimplF, uF, aF);
                std::cout << "NN_TOL site=blstm_overlap_fwd max_ulp=" << uF
                          << " max_abs=" << std::scientific << std::setprecision(3) << aF << "\n";
                long uB = 0; double aB = 0.0;
                probeGapNaN(nn._OutputBackward, reimplB, uB, aB);
                std::cout << "NN_TOL site=blstm_overlap_bwd max_ulp=" << uB
                          << " max_abs=" << std::scientific << std::setprecision(3) << aB << "\n";

                Matrix2BinaryFile(out + "blstm_overlap_out.bin", reimplOut);
                Matrix2BinaryFile(out + "blstm_overlap_fwd.bin", reimplF);
                Matrix2BinaryFile(out + "blstm_overlap_bwd.bin", reimplB);
                Matrix2BinaryFile(out + "blstm_overlap_count.bin", cnt);
                dumps += 4;
            }

            // ---- Mode C2: OverLap uncovered-rows NaN case. window_size 3 shift 10 (>
            // 2*3+1=7 so windows leave gaps). T=20: jj=0 covers outputSeq[0,3); jj=10
            // covers [3,6); rows 6,7,8,9 are NEVER covered -> outCount 0 -> 0/0 = NaN.
            // The fwd/bwd hidden states (lengthOutputLSTM=10) likewise NaN on rows 6-9.
            // probeGapNaN treats identical-NaN bits as equal.
            {
                const int T = 20;
                Eigen::MatrixXd input = makeInput(T);
                confB.set_val<short>("SYNB_InputNormalizationType", (short)0);
                confB.set_val<bool>("SYNB_TwoSweeps", false);
                BLSTMNeuralNetwork<LSTMLayer> nn(confB, "SYNB", true);
                nn.setWeights(flat);
                nn.setProcessingType(true, true);

                Eigen::MatrixXd realInput = input;
                Eigen::MatrixXd realOut = Eigen::MatrixXd::Zero(T / subRatio, 3);  // 10 x 3
                Eigen::MatrixXd emptyTargets;
                nn.feedForwardBackward(realInput, 3, 10, realOut, emptyTargets);

                Eigen::MatrixXd reimplOut = Eigen::MatrixXd::Zero(T / subRatio, 3);
                Eigen::MatrixXd reimplF, reimplB, cnt, cntL;
                overlap(input, 3, 10, reimplOut, reimplF, reimplB, cnt, cntL);

                // probeGapNaN treats identical-NaN bits as equal; the uncovered rows
                // (6-9) are NaN in both the real class output and the reimpl.
                long u = 0; double a = 0.0;
                probeGapNaN(realOut, reimplOut, u, a);
                std::cout << "NN_TOL site=blstm_overlap_nan max_ulp=" << u
                          << " max_abs=" << std::scientific << std::setprecision(3) << a << "\n";

                // _OutputForward/_OutputBackward are PUBLIC (BLSTMNeuralNetwork.h:23-24);
                // uncovered rows are NaN in both the real class and the reimpl.
                long uF = 0; double aF = 0.0;
                probeGapNaN(nn._OutputForward, reimplF, uF, aF);
                std::cout << "NN_TOL site=blstm_overlap_nan_fwd max_ulp=" << uF
                          << " max_abs=" << std::scientific << std::setprecision(3) << aF << "\n";
                long uB = 0; double aB = 0.0;
                probeGapNaN(nn._OutputBackward, reimplB, uB, aB);
                std::cout << "NN_TOL site=blstm_overlap_nan_bwd max_ulp=" << uB
                          << " max_abs=" << std::scientific << std::setprecision(3) << aB << "\n";

                Matrix2BinaryFile(out + "blstm_overlap_nan_out.bin", reimplOut);
                Matrix2BinaryFile(out + "blstm_overlap_nan_fwd.bin", reimplF);
                Matrix2BinaryFile(out + "blstm_overlap_nan_bwd.bin", reimplB);
                dumps += 3;
            }
        }

        // ---- Mode D: MLPOverLap on an MLP-mode net: LSTM [0,...] (is_mlp), output
        // [5,4,1] sub [2,1]. getSubSamplingRatio() (MLP) = out net ratio = 2*1 = 2, so
        // the driver forces window_size = 2/2 = 1 (IGNORING the passed window_size),
        // length = 3, length_short = 1. T=12 window_shift 2: only exact 3-row windows
        // (jj=2,4,6,8,10) write output[jj/shift, 0]; jj=0 is an edge window (lengthSeq=2)
        // -> skipped, so output ROW 0 keeps the caller's zeros. _OutputForward/
        // _OutputBackward emptied.
        {
            const std::vector<long> outNN = {5, 4, 1};
            const std::vector<long> outSS = {2, 1};
            // Per-layer fan-in accounts for subsampling: L0 in = neuronNb[0]*sub[0] =
            // 5*2 = 10; L1 in = neuronNb[1]*sub[1] = 4*1 = 4 (Task 6 dense convention).
            const std::vector<int> outIns = {10, 4}, outOuts = {4, 1};

            long outNb = 0;
            for (size_t jj = 0; jj < outIns.size(); ++jj) outNb += (long)outOuts[jj] * (outIns[jj] + 1);
            const long tailNb = 2 * 5;  // MLP input_size = OutputNeuronNb[0] = 5
            const long nbTotal = outNb + tailNb;
            Eigen::VectorXd flat(nbTotal);
            for (long k = 0; k < nbTotal - tailNb; ++k)
                flat(k) = (double)((k * 11 + 3) % 97) / 97.0 - 0.5;
            for (int j = 0; j < 5; ++j) {
                flat(nbTotal - tailNb + j) = 0.1 * (j + 1);
                flat(nbTotal - tailNb + 5 + j) = 1.0 + 0.05 * j;
            }

            std::vector<Eigen::MatrixXd> ows, obs;
            std::vector<NetLayerStep> outSteps;
            makeDenseSteps(flat.head(outNb), outIns, outOuts, ows, obs, outSteps);

            // Plain MLP forward reimpl (feedForwardMLP :462-469): input width == net
            // input (5) so the leftCols gate is false; forward the output net directly.
            auto mlpForward = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outp) {
                netForwardLoop(outNN, outSS, outSteps, in, outp);
            };

            const int T = 12;
            const long windowShift = 2;
            const long subRatioMLP = 2;  // MLP getSubSamplingRatio() = out ratio
            const long mlpWindow = subRatioMLP / 2;  // forced = 1 (:686)
            Eigen::MatrixXd input(T, 5);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < 5; ++j)
                    input(t, j) = (double)(((t * 29 + j * 13 + 5) % 97)) / 97.0 - 0.5;

            ConfigFile confM(nnConfigPath, '_');
            confM._Params.erase("BLSTM_weightsFile");
            confM.set_val<std::string>("SYNM_LSTMNeuronNb", "0,3");
            confM.set_val<std::string>("SYNM_LSTMSubSampling", "1");
            confM.set_val<std::string>("SYNM_OutputNeuronNb", "5,4,1");
            confM.set_val<std::string>("SYNM_OutputSubSampling", "2,1");
            confM.set_val<short>("SYNM_InputNormalizationType", (short)0);
            confM.set_val<bool>("SYNM_TwoSweeps", false);
            BLSTMNeuralNetwork<LSTMLayer> nn(confM, "SYNM", true);
            nn.setWeights(flat);
            nn.setProcessingType(true, true);  // MLP + truncate + overlap -> MLPOverLap

            // Output rows: plain MLP would give floor(T/outRatio)=12/2=6; loop indexes
            // up to jj=10 -> jj/shift=5, so 6 rows. Row 0 (edge-skipped) keeps zeros.
            // Pass a DELIBERATELY WRONG window_size (99) to prove it is ignored (:686):
            // the driver overwrites it with getSubSamplingRatio()/2 regardless. The
            // Rust test re-runs with the CORRECT window and asserts the same output.
            Eigen::MatrixXd realInput = input;
            Eigen::MatrixXd realOut = Eigen::MatrixXd::Zero(6, 1);
            Eigen::MatrixXd emptyTargets;
            nn.feedForwardBackward(realInput, 99, windowShift, realOut, emptyTargets);

            // Reimpl of feedForwardBackwardMLPOverLap (:683-709).
            Eigen::MatrixXd reimplOut = Eigen::MatrixXd::Zero(6, 1);  // row 0 stays zero
            for (long jj = 0; jj < input.rows(); jj += windowShift) {
                long begin = (jj < mlpWindow) ? 0 : jj - mlpWindow;    // :690-695
                long end = jj + mlpWindow;                              // :696
                if (end >= input.rows()) end = input.rows() - 1;       // :697
                long lengthSeq = end - begin + 1;
                if (lengthSeq == 2 * mlpWindow + 1) {                   // :699 exact windows only
                    Eigen::MatrixXd block = input.block(begin, 0, lengthSeq, input.cols());
                    Eigen::MatrixXd outShort;
                    mlpForward(block, outShort);                        // :703
                    reimplOut(jj / windowShift, 0) = outShort(0, 0);    // :704
                }
            }

            long u = 0; double a = 0.0;
            probeGapNaN(realOut, reimplOut, u, a);
            std::cout << "NN_TOL site=blstm_mlpoverlap max_ulp=" << u
                      << " max_abs=" << std::scientific << std::setprecision(3) << a << "\n";

            // _OutputForward/_OutputBackward are PUBLIC (BLSTMNeuralNetwork.h:23-24), but
            // feedForwardBackwardMLPOverLap explicitly EMPTIES them (BLSTMNeuralNetwork.cpp:
            // 707-708: assignment to a default-constructed Eigen::MatrixXd, i.e. 0x0) since
            // the MLP path has no LSTM hidden states. Assert both are 0x0 on the real class
            // and print a zero-mismatch probe line for uniformity with the other four sites.
            if (nn._OutputForward.rows() != 0 || nn._OutputForward.cols() != 0 ||
                nn._OutputBackward.rows() != 0 || nn._OutputBackward.cols() != 0) {
                std::cerr << "MLPOverLap emptied-members FAIL: _OutputForward="
                          << nn._OutputForward.rows() << "x" << nn._OutputForward.cols()
                          << " _OutputBackward=" << nn._OutputBackward.rows() << "x"
                          << nn._OutputBackward.cols() << " expected 0x0\n";
                return 1;
            }
            std::cout << "NN_TOL site=blstm_mlpoverlap_fwd max_ulp=0 max_abs="
                      << std::scientific << std::setprecision(3) << 0.0 << "\n";
            std::cout << "NN_TOL site=blstm_mlpoverlap_bwd max_ulp=0 max_abs="
                      << std::scientific << std::setprecision(3) << 0.0 << "\n";

            Matrix2BinaryFile(out + "blstm_mlpoverlap_out.bin", reimplOut);
            dumps += 1;
        }

        // ================================================================
        // Phase 2 Task 10: scoring feedForward (BLSTMNeuralNetwork.cpp:843-929).
        // The scoring feedForward builds a per-frame target sequence from
        // (targetIndex, targetModifier), runs the windowed feedForwardBackward, and
        // for a BINARY net (outputSize == 1) with targets expands the single posterior
        // column into [1-p, p]. Reimpl transcribes the target-construction counter
        // scheme + soft-target (isCostModified ? 0.1*modifier : 0.0) + binary
        // expansion, calling blstmFeedForward (plain path, InputNormalizationType 0)
        // per frame; probed vs the REAL class's scoring feedForward
        // (BLSTMNeuralNetwork.h:195). isCostModified maps to CostLaw::_BackPropWER
        // (CostLaw.cpp:110-112), driven by the synthetic config's SYNS_BackPropWER key
        // (>= 0 -> modified). _Cost/_NbOfClassif are read from the REAL class via the
        // public getCost()/getNbOfClassif() accessors (BLSTMNeuralNetwork.h:157-158)
        // and printed as parseable BLSTM_SCORING lines for the manifest.
        //
        // Synthetic BINARY net: LSTM [3,4,2] sub [2,1], output [4,5,1] sub [1,1]
        // (output ENDS AT 1). T=12 -> length = 12/2/1 = 6. Cases sweep {enforcement
        // step 0, 2} x {targetModifier 1.0, 2.0} x {cost-modified false, true}, all at
        // targetIndex 1 (the SPEECH class in the binary VAD convention).
        {
            const std::vector<long> lstmNN = {3, 4, 2};
            const std::vector<long> lstmSS = {2, 1};
            const std::vector<long> outNN = {4, 5, 1};
            const std::vector<long> outSS = {1, 1};
            const std::vector<int> lstmIns = {6, 4}, lstmOuts = {4, 2};
            const std::vector<int> outIns = {4, 5}, outOuts = {5, 1};
            const int fwdInputSize = 3;
            const long subRatio = 2;   // lstm(2)*out(1)
            const int T = 12;
            const long length = 6;     // 12/2/1

            auto lstmNetNb = [&](const std::vector<int>& ins, const std::vector<int>& outs) {
                long s = 0;
                for (size_t jj = 0; jj < ins.size(); ++jj) {
                    int I = ins[jj], O = outs[jj];
                    s += 4L * I * O + 4L * O * O + 12L * O + 4L * O;
                }
                return s;
            };
            auto denseNetNb = [&](const std::vector<int>& ins, const std::vector<int>& outs) {
                long s = 0;
                for (size_t jj = 0; jj < ins.size(); ++jj) s += (long)outs[jj] * (ins[jj] + 1);
                return s;
            };
            const long fwdNb = lstmNetNb(lstmIns, lstmOuts);
            const long bwdNb = fwdNb;
            const long outNb = denseNetNb(outIns, outOuts);
            const long tailNb = 2 * fwdInputSize;
            const long nbTotal = fwdNb + bwdNb + outNb + tailNb;

            Eigen::VectorXd flat(nbTotal);
            for (long k = 0; k < nbTotal - tailNb; ++k)
                flat(k) = (double)((k * 11 + 3) % 97) / 97.0 - 0.5;
            for (int j = 0; j < fwdInputSize; ++j) {
                flat(nbTotal - tailNb + j) = 0.1 * (j + 1);
                flat(nbTotal - tailNb + fwdInputSize + j) = 1.0 + 0.05 * j;
            }

            Eigen::VectorXd fwdSlice = flat.head(fwdNb);
            Eigen::VectorXd bwdSlice = flat.segment(fwdNb, bwdNb);
            Eigen::VectorXd outSlice = flat.segment(fwdNb + bwdNb, outNb);
            std::vector<Eigen::MatrixXd> fiw, ffw, fpp, fbs, biw, bfw, bpp, bbs, ows, obs;
            std::vector<NetLayerStep> fwdSteps, fwdStepsRev, bwdSteps, bwdStepsRev, outSteps;
            makeLstmSteps(fwdSlice, lstmIns, lstmOuts, fiw, ffw, fpp, fbs, fwdSteps, fwdStepsRev);
            makeLstmSteps(bwdSlice, lstmIns, lstmOuts, biw, bfw, bpp, bbs, bwdSteps, bwdStepsRev);
            makeDenseSteps(outSlice, outIns, outOuts, ows, obs, outSteps);

            Eigen::MatrixXd baseInput(T, 3);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < 3; ++j)
                    baseInput(t, j) = (double)(((t * 29 + j * 13 + 5) % 97)) / 97.0 - 0.5;

            // Reimpl of the scoring target construction + binary expansion (:868-926).
            // outputSize == 1 (binary), so only the binary branch (:887-903) runs.
            auto scoringReimpl = [&](int step, double modifier, bool costModified,
                                     Eigen::MatrixXd& outputSeq) {
                const int targetIndex = 1;
                double target = costModified ? 0.1 * modifier : 0.0;   // :870-871
                Eigen::MatrixXd targetSeq(length, 1);                   // :888
                int counter = 0;
                for (long ii = 0; ii < length; ++ii) {
                    if (counter >= step) {                              // :890
                        counter = 0;
                        targetSeq(ii, 0) = (targetIndex == 1) ? (1.0 - target) : target;  // :893-897
                    } else {
                        ++counter;
                        targetSeq(ii, 0) = -0.5;                        // :901
                    }
                }
                // :920 windowed FFB -> plain path (InputNormalizationType 0, no norm).
                Eigen::MatrixXd oF, oB;
                Eigen::MatrixXd raw(length, 1);
                blstmFeedForward(lstmNN, lstmSS, fwdSteps, bwdStepsRev, outNN, outSS, outSteps,
                                 fwdInputSize, baseInput, oF, oB, raw);
                // :922-926 binary expansion into [1-p, p].
                outputSeq = Eigen::MatrixXd(length, 2);
                for (long ii = 0; ii < length; ++ii) {
                    outputSeq(ii, 0) = 1.0 - raw(ii, 0);
                    outputSeq(ii, 1) = raw(ii, 0);
                }
            };

            struct ScoreCase { int step; double modifier; bool costModified; const char* tag; };
            const ScoreCase cases[] = {
                {0, 1.0, false, "step0_mod1_plain"},
                {2, 1.0, false, "step2_mod1_plain"},
                {0, 2.0, true,  "step0_mod2_mod"},
                {2, 2.0, true,  "step2_mod2_mod"},
            };

            for (const ScoreCase& sc : cases) {
                // Synthetic BINARY config prefix (SYNS). InputNormalizationType 0.
                ConfigFile confS(nnConfigPath, '_');
                confS._Params.erase("BLSTM_weightsFile");
                confS.set_val<std::string>("SYNS_LSTMNeuronNb", "3,4,2");
                confS.set_val<std::string>("SYNS_LSTMSubSampling", "2,1");
                confS.set_val<std::string>("SYNS_OutputNeuronNb", "4,5,1");
                confS.set_val<std::string>("SYNS_OutputSubSampling", "1,1");
                confS.set_val<short>("SYNS_InputNormalizationType", (short)0);
                confS.set_val<bool>("SYNS_TwoSweeps", false);
                confS.set_val<bool>("SYNS_BackPropagationActivated", false);
                confS.set_val<int>("SYNS_TargetEnforcementStep", sc.step);
                // isCostModified() == CostLaw::_BackPropWER (>= 0 -> modified).
                confS.set_val<double>("SYNS_BackPropWER", sc.costModified ? 0.0 : -1.0);
                BLSTMNeuralNetwork<LSTMLayer> nn(confS, "SYNS", true);
                nn.setWeights(flat);
                nn.setProcessingType(false, false);  // plain path

                // Real class scoring feedForward (BLSTMNeuralNetwork.h:195): input is
                // Eigen::Ref (mutated for norm 1/-1; norm 0 here leaves it alone).
                Eigen::MatrixXd realInput = baseInput;
                Eigen::MatrixXd realOut = nn.feedForward(realInput, 4, 2, 1, sc.modifier);

                Eigen::MatrixXd reimplOut;
                scoringReimpl(sc.step, sc.modifier, sc.costModified, reimplOut);

                long u = 0; double a = 0.0;
                probeGapNaN(realOut, reimplOut, u, a);
                std::cout << "NN_TOL site=blstm_scoring_" << sc.tag << " max_ulp=" << u
                          << " max_abs=" << std::scientific << std::setprecision(3) << a << "\n";

                // _Cost/_NbOfClassif from the REAL class (public accessors). Print as
                // hex bits (parseable, bit-exact) alongside the decimal for the manifest.
                double cost = nn.getCost();
                double nbClassif = nn.getNbOfClassif();
                uint64_t costBits;
                std::memcpy(&costBits, &cost, sizeof(double));
                std::cout << "BLSTM_SCORING case=" << sc.tag
                          << " cost=0x" << std::hex << costBits << std::dec
                          << " cost_dec=" << std::scientific << std::setprecision(17) << cost
                          << " nb_of_classif=" << (long long)nbClassif << "\n";

                Matrix2BinaryFile(out + "blstm_scoring_" + sc.tag + "_out.bin", reimplOut);
                dumps += 1;
            }
        }
    }

    // --- Phase 2 Task 10: END-TO-END real-net gate leg -----------------------
    // legacy: the Phase 1 feature front-end (BLSTMSpectralSegmenter::initSpectralAnalysis
    // param derivation + getBLSTMInputSequence assembly, transcribed above in the Task 11
    // feature block) driven by the REAL config's BLSTM_* DSP keys on the excerpt wav
    // (chan 1, offset 0.35 dur 2.0), then the real-net blstm reimpl forward TWICE:
    //   (a) full-sequence (plain FFB, no targets) -> e2e_out_full + e2e_fwd_full + e2e_bwd_full,
    //   (b) OverLap window_size 25 shift 12 (EXPLICIT constants; getBLSTMParam derivation
    //       is Phase 2b) -> e2e_out_overlap + e2e_fwd_overlap + e2e_bwd_overlap.
    // Also dumps e2e_input.bin (the assembled inputSeq -- revalidates Phase 1 under the
    // REAL config for the Rust gate). NN_TOL probes vs the REAL class on both forwards
    // (EXPECTED NONZERO per the established topRows(11) input slice + k=24 recurrence and layer-1 GEMMs + output-net (k=48/k=12) GEMM divergence pattern; measured recorded).
    //
    // Real config keys: preemph_ratio -0.97 (< 0 -> preemph SKIPPED, matching
    // BLSTMSpectralSegmenter.cpp:216 `_PreemphRatio > 0`); spectrum_order 10; nb_bins 20,
    // nb_DCT 4, is_log_mel true, IgnoreFirstDCT true, ComputeDeltasNb 5,
    // ComputeDeltaDeltasNb 3 -> mel + DCT path; LTSVwindow 0 -> R == 0 -> NO LTSV. The
    // assembled input width D is whatever the DCT branch produces; the net input is 23,
    // and D != 23 exercises the feedForward width tolerance (leftCols when D > 23 &&
    // LSTMRatios[0] > 1, else the per-layer topRows in the LSTM/dense forward). D is
    // dumped in the manifest via e2e_input.bin's column count.
    {
        // Local copies of the Task 8/9 block-scoped helpers (makeLstmSteps /
        // makeDenseSteps / probeGapNaN), needed here because this leg lives OUTSIDE
        // the Task 8/9 block that defines them. Identical logic; kept in sync.
        auto makeLstmSteps = [](const Eigen::VectorXd& flat,
                                const std::vector<int>& ins, const std::vector<int>& outs,
                                std::vector<Eigen::MatrixXd>& iw, std::vector<Eigen::MatrixXd>& fw,
                                std::vector<Eigen::MatrixXd>& pp, std::vector<Eigen::MatrixXd>& bs,
                                std::vector<NetLayerStep>& fwdSteps,
                                std::vector<NetLayerStep>& bwdSteps) {
            const size_t L = ins.size();
            iw.resize(L); fw.resize(L); pp.resize(L); bs.resize(L);
            fwdSteps.resize(L); bwdSteps.resize(L);
            long pos = 0;
            for (size_t jj = 0; jj < L; ++jj) {
                int I = ins[jj], O = outs[jj];
                long nb = 4L * I * O + 4L * O * O + 12L * O + 4L * O;
                Eigen::VectorXd slice = flat.segment(pos, nb);
                pos += nb;
                unpackLstmWeights(slice, I, O, iw[jj], fw[jj], pp[jj], bs[jj]);
            }
            for (size_t jj = 0; jj < L; ++jj) {
                int O = outs[jj];
                fwdSteps[jj].forward = [&iw, &fw, &pp, &bs, jj, O](const Eigen::MatrixXd& in,
                                                                  Eigen::MatrixXd& outm, bool) {
                    Eigen::MatrixXd g;
                    lstmForwardLoop(in, iw[jj], fw[jj], pp[jj], bs[jj], O, true, true, true, g, outm);
                };
                bwdSteps[jj].forward = [&iw, &fw, &pp, &bs, jj, O](const Eigen::MatrixXd& in,
                                                                  Eigen::MatrixXd& outm, bool) {
                    Eigen::MatrixXd g;
                    lstmForwardReverseLoop(in, iw[jj], fw[jj], pp[jj], bs[jj], O, true, true, true, g, outm);
                };
            }
            return pos;
        };
        auto makeDenseSteps = [](const Eigen::VectorXd& flat,
                                 const std::vector<int>& ins, const std::vector<int>& outs,
                                 std::vector<Eigen::MatrixXd>& ws, std::vector<Eigen::MatrixXd>& bs,
                                 std::vector<NetLayerStep>& steps) {
            const size_t L = ins.size();
            ws.resize(L); bs.resize(L); steps.resize(L);
            long pos = 0;
            for (size_t jj = 0; jj < L; ++jj) {
                int I = ins[jj], O = outs[jj];
                long nb = (long)O * (I + 1);
                Eigen::VectorXd slice = flat.segment(pos, nb);
                pos += nb;
                ws[jj].resize(I, O); bs[jj].resize(1, O);
                for (int cc = 0; cc < O; ++cc)
                    for (int r = 0; r < I; ++r) ws[jj](r, cc) = slice((long)cc * I + r);
                for (int cc = 0; cc < O; ++cc) bs[jj](0, cc) = slice((long)O * I + cc);
            }
            for (size_t jj = 0; jj < L; ++jj) {
                steps[jj].forward = [&ws, &bs, jj](const Eigen::MatrixXd& in,
                                                   Eigen::MatrixXd& outm, bool last) {
                    outm = denseForwardLoop(in, ws[jj], bs[jj], last);
                };
            }
            return pos;
        };
        auto probeGapNaN = [](const Eigen::MatrixXd& real, const Eigen::MatrixXd& reimpl,
                              long& maxUlp, double& maxAbs) {
            if (real.rows() != reimpl.rows() || real.cols() != reimpl.cols()) {
                std::cerr << "probeGapNaN dims mismatch: real " << real.rows() << "x"
                          << real.cols() << " vs reimpl " << reimpl.rows() << "x"
                          << reimpl.cols() << "\n";
                std::abort();
            }
            for (int r = 0; r < real.rows(); ++r) {
                for (int cc = 0; cc < real.cols(); ++cc) {
                    double av = real(r, cc), bv = reimpl(r, cc);
                    uint64_t ab, bb;
                    std::memcpy(&ab, &av, sizeof(double));
                    std::memcpy(&bb, &bv, sizeof(double));
                    if (ab == bb) continue;
                    double absGap = std::fabs(av - bv);
                    if (absGap > maxAbs) maxAbs = absGap;
                    long ulp = (ab > bb) ? (long)(ab - bb) : (long)(bb - ab);
                    if (ulp > maxUlp) maxUlp = ulp;
                }
            }
        };

        const unsigned Min = 1;
        const unsigned Max = 20;
        Loki::Factory<AbstractFFT<double>, unsigned int> gfft_factory;
        FactoryInit<GFFTList<GFFT, Min, Max>::Result>::apply(gfft_factory);

        // Parse the REAL config with the same ConfigFile the feature block uses.
        ConfigFile conf(nnConfigPath);
        FeatureCfg c = readFeatureCfg(conf, "BLSTM");

        CorpusItem eitem(wav, "", "RUS", "RU", 0, 0, 1.0);
        AudioStruct eaudio(OFFSET_SEC, MAX_DUR_SEC, 0, eitem);
        const double RATE = (double)eaudio.getFrameRate();
        SpectralP s = deriveSpectral(c, RATE, Max);

        if (c.preemph > 0) eaudio.applyPreemph(c.preemph);   // -0.97 -> SKIPPED
        // noise skipped: real config noise_seed -3 (< 0 -> BLSTMSpectralSegmenter.cpp:222
        // `_NoiseSeed > 0` false), and the feature transcription follows suit.

        MelFilterBank mel;
        if (c.nb_bins > 0) {
            mel = MelFilterBank(c.min_mel, c.max_mel, c.nb_bins, s.min_freq_snapped,
                                s.max_freq_snapped, RATE, s.bins - 1, c.is_log, c.nb_dct,
                                c.ignore_first, c.deltas_nb, c.dd_nb);
        }

        const double PI = 3.14159265358979323846264338327;
        long nbFilters = mel.notEmpty() ? (long)mel.getNbFilters() : 0;
        long nbDct = c.nb_dct;
        if (nbDct > nbFilters) nbDct = nbFilters;
        Eigen::MatrixXd coeffs;
        if (c.nb_bins > 0 && c.nb_dct > 0) {
            coeffs = Eigen::MatrixXd::Zero(nbFilters, nbDct);
            for (long col = 0; col < nbFilters; ++col)
                for (long row = 0; row < nbDct; ++row)
                    coeffs(col, row) = std::cos(PI / nbFilters * (col + 0.5) * row);
        }

        Eigen::MatrixXd win = getWindowingCoefficients(c.win_type, false, s.window_size + 1, c.win_param);
        Eigen::MatrixXd noConv;
        MelFilterBank emptyMel;
        const long long endFull = eaudio.getFrameCount() - 1;

        // chan 1 (0-indexed 0), per the brief.
        const int chan = 0;
        eaudio.computeSegmentPeriodogramEstimates(s.order, s.shift_frames, chan, c.flag_dc,
                                                  win, emptyMel, gfft_factory, noConv, 0, endFull);
        Eigen::MatrixXd perio = eaudio._Periodogram;

        Eigen::MatrixXd inputSeq;
        if (mel.notEmpty()) {
            Eigen::MatrixXd fb = Eigen::MatrixXd::Zero(perio.rows(), mel.getNbFilters());
            mel.applyFilterBank(perio, fb);
            if (c.nb_dct > 0) {
                inputSeq = applyDCTLoop(fb, coeffs, (int)nbDct, c.ignore_first, c.deltas_nb, c.dd_nb);
            } else {
                inputSeq = fb;
            }
        } else {
            inputSeq = (((perio.block(0, s.freq_beg, perio.rows(),
                                      s.freq_end - s.freq_beg + 1).array() + 1e-24).log()).matrix());
        }
        if (s.ltsv_half_window > 0) {
            long ltsv_beg = mel.notEmpty() ? 0 : s.freq_beg;
            long ltsv_end = mel.notEmpty() ? (long)inputSeq.cols() - 1 : s.freq_end;
            Eigen::MatrixXd ltsv = getLTSV(perio, s.ltsv_half_window, s.ltsv_shift, ltsv_beg, ltsv_end);
            Eigen::MatrixXd merged(inputSeq.rows(), inputSeq.cols() + 1);
            merged << inputSeq, ltsv;
            inputSeq = merged;
        }

        Matrix2BinaryFile(out + "e2e_input.bin", inputSeq);
        dumps += 1;
        std::cout << "E2E_INPUT rows=" << inputSeq.rows() << " cols=" << inputSeq.cols() << "\n";

        // --- Real-net blstm reimpl forward (mirrors the Task 8 real-fullseq block). --
        // Rebuild the makeLstmSteps/makeDenseSteps/blstmFeedForward plumbing here (those
        // Task 8/9 lambdas are scoped to the block above). Real net: LSTM [23,24,24] sub
        // [4,1]; output [48,12,1] sub [1,1]; InputNormalizationType -1 (self-norm).
        const std::vector<long> lstmNN = {23, 24, 24};
        const std::vector<long> lstmSS = {4, 1};
        const std::vector<long> outNN = {48, 12, 1};
        const std::vector<long> outSS = {1, 1};
        const std::vector<int> lstmIns = {92, 24}, lstmOuts = {24, 24};   // 23*4, 24*1
        const std::vector<int> outIns = {48, 12}, outOuts = {12, 1};
        const int fwdInputSize = 23;
        const long subRatio = 4;   // lstm(4)*out(1)

        ConfigFile nconf(nnConfigPath, '_');
        nconf._Params.erase("BLSTM_weightsFile");
        BLSTMNeuralNetwork<LSTMLayer> nn(nconf, "BLSTM", true);
        Eigen::VectorXd flat = BinaryFile2Vector(nnWeightsPath);
        nn.setWeights(flat);

        long fwdNb = 0;
        for (size_t jj = 0; jj < lstmIns.size(); ++jj) {
            int I = lstmIns[jj], O = lstmOuts[jj];
            fwdNb += 4L * I * O + 4L * O * O + 12L * O + 4L * O;
        }
        long outNbW = 0;
        for (size_t jj = 0; jj < outIns.size(); ++jj) outNbW += (long)outOuts[jj] * (outIns[jj] + 1);
        Eigen::VectorXd fwdSlice = flat.head(fwdNb);
        Eigen::VectorXd bwdSlice = flat.segment(fwdNb, fwdNb);
        Eigen::VectorXd outSlice = flat.segment(2 * fwdNb, outNbW);
        std::vector<Eigen::MatrixXd> fiw, ffw, fpp, fbs, biw, bfw, bpp, bbs, ows, obs;
        std::vector<NetLayerStep> fwdSteps, fwdStepsRev, bwdSteps, bwdStepsRev, outSteps;
        makeLstmSteps(fwdSlice, lstmIns, lstmOuts, fiw, ffw, fpp, fbs, fwdSteps, fwdStepsRev);
        makeLstmSteps(bwdSlice, lstmIns, lstmOuts, biw, bfw, bpp, bbs, bwdSteps, bwdStepsRev);
        makeDenseSteps(outSlice, outIns, outOuts, ows, obs, outSteps);

        // Reimpl type -1 self-norm in place, then core forward (leftCols gate:
        // LSTMRatios[0]==4 > 1 && fwdInputSize(23) < D crops leftCols(23) when D > 23;
        // when D < 23 the full input flows and the per-layer topRows tolerance applies).
        auto normalizeType_1 = [](Eigen::MatrixXd& in) {
            const int R = (int)in.rows(), C = (int)in.cols();
            if (R == 0) return;
            std::vector<double> mean(C, 0.0);
            for (int cc = 0; cc < C; ++cc) {
                double acc = 0.0;
                for (int r = 0; r < R; ++r) acc += in(r, cc);
                mean[cc] = acc / (double)R;
            }
            for (int r = 0; r < R; ++r)
                for (int cc = 0; cc < C; ++cc) in(r, cc) -= mean[cc];
            std::vector<double> stdv(C, 0.0);
            for (int cc = 0; cc < C; ++cc) {
                double acc = 0.0;
                for (int r = 0; r < R; ++r) acc += in(r, cc) * in(r, cc);
                stdv[cc] = std::sqrt((acc + 1e-32) / (double)R);
            }
            for (int r = 0; r < R; ++r)
                for (int cc = 0; cc < C; ++cc) in(r, cc) = Maxmin2::fn(in(r, cc) / stdv[cc]);
        };

        // (a) full-sequence: plain FFB path, no targets. window_size/shift unused there.
        {
            long outRows = inputSeq.rows();
            for (long r : lstmSS) outRows /= r;   // fwd LSTM ratio 4*1 -> rows/4
            Eigen::MatrixXd realInput = inputSeq;  // MUTATED by the type -1 self-norm
            Eigen::MatrixXd realOut(outRows, 1);
            Eigen::MatrixXd emptyTargets;
            nn.setProcessingType(false, false);
            nn.feedForwardBackward(realInput, 4, 2, realOut, emptyTargets);

            Eigen::MatrixXd reimplInput = inputSeq;
            normalizeType_1(reimplInput);
            Eigen::MatrixXd reimplOut, outForward, outBackward;
            // Core forward (leftCols(23) gate lives inside blstmFeedForward).
            {
                Eigen::MatrixXd fwdIn = reimplInput, bwdIn = reimplInput;
                if (lstmSS[0] > 1 && fwdInputSize < reimplInput.cols()) {
                    fwdIn = reimplInput.leftCols(fwdInputSize);
                    bwdIn = reimplInput.leftCols(fwdInputSize);
                }
                netForwardLoop(lstmNN, lstmSS, fwdSteps, fwdIn, outForward);
                netForwardReverseLoop(lstmNN, lstmSS, bwdStepsRev, bwdIn, outBackward);
                netForwardDoubleLoop(outNN, outSS, outSteps, outForward, outBackward, reimplOut);
            }

            long u = 0; double a = 0.0;
            probeGapNaN(realOut, reimplOut, u, a);
            std::cout << "NN_TOL site=e2e_full max_ulp=" << u
                      << " max_abs=" << std::scientific << std::setprecision(3) << a << "\n";
            long uF = 0; double aF = 0.0;
            probeGapNaN(nn._OutputForward, outForward, uF, aF);
            std::cout << "NN_TOL site=e2e_full_fwd max_ulp=" << uF
                      << " max_abs=" << std::scientific << std::setprecision(3) << aF << "\n";
            long uB = 0; double aB = 0.0;
            probeGapNaN(nn._OutputBackward, outBackward, uB, aB);
            std::cout << "NN_TOL site=e2e_full_bwd max_ulp=" << uB
                      << " max_abs=" << std::scientific << std::setprecision(3) << aB << "\n";

            Matrix2BinaryFile(out + "e2e_out_full.bin", reimplOut);
            Matrix2BinaryFile(out + "e2e_fwd_full.bin", outForward);
            Matrix2BinaryFile(out + "e2e_bwd_full.bin", outBackward);
            dumps += 3;
        }

        // (b) OverLap window_size 25 shift 12 (EXPLICIT constants). Real net _Overlaps.
        {
            const long WINDOW_SIZE = 25;
            const long WINDOW_SHIFT = 12;
            const long outNetRatio = 1;   // _OutputNetwork.getSubSamplingRatio()
            const int fwdOut = 24;        // last LSTM layer output size

            long outRows = inputSeq.rows() / subRatio;   // whole-BLSTM ratio 4
            Eigen::MatrixXd realInput = inputSeq;
            Eigen::MatrixXd realOut = Eigen::MatrixXd::Zero(outRows, 1);
            Eigen::MatrixXd emptyTargets;
            nn.setProcessingType(true, true);  // truncate + overlap -> OverLap path
            nn.feedForwardBackward(realInput, WINDOW_SIZE, WINDOW_SHIFT, realOut, emptyTargets);

            // Reimpl OverLap (BLSTMNeuralNetwork.cpp:592-681), self-norm in place first.
            Eigen::MatrixXd reimplInput = inputSeq;
            normalizeType_1(reimplInput);

            auto plainWindow = [&](const Eigen::MatrixXd& in, Eigen::MatrixXd& outF,
                                   Eigen::MatrixXd& outB, Eigen::MatrixXd& outp) {
                Eigen::MatrixXd fwdIn = in, bwdIn = in;
                if (lstmSS[0] > 1 && fwdInputSize < in.cols()) {
                    fwdIn = in.leftCols(fwdInputSize);
                    bwdIn = in.leftCols(fwdInputSize);
                }
                netForwardLoop(lstmNN, lstmSS, fwdSteps, fwdIn, outF);
                netForwardReverseLoop(lstmNN, lstmSS, bwdStepsRev, bwdIn, outB);
                netForwardDoubleLoop(outNN, outSS, outSteps, outF, outB, outp);
            };

            Eigen::MatrixXd reimplOut = Eigen::MatrixXd::Zero(outRows, 1);
            long lengthOutputLSTM = reimplOut.rows() * outNetRatio;                  // :596
            Eigen::MatrixXd outputForward = Eigen::MatrixXd::Zero(lengthOutputLSTM, fwdOut);
            Eigen::MatrixXd outputBackward = Eigen::MatrixXd::Zero(lengthOutputLSTM, fwdOut);
            Eigen::MatrixXd outCount = Eigen::MatrixXd::Zero(reimplOut.rows(), 1);   // :599
            Eigen::MatrixXd outCountLSTM = Eigen::MatrixXd::Zero(lengthOutputLSTM, 1); // :600
            Eigen::MatrixXd sum = Eigen::MatrixXd::Zero(reimplOut.rows(), reimplOut.cols());

            long length = 2 * WINDOW_SIZE + 1;                                       // :602-609
            for (long r : lstmSS) length /= r;
            for (long r : outSS) length /= r;
            long nominalLen = length, nominalLstm = length;

            for (long jj = 0; jj < reimplInput.rows(); jj += WINDOW_SHIFT) {         // :613
                long begin = (jj < WINDOW_SIZE) ? 0 : jj - WINDOW_SIZE;              // :614-619
                begin = (begin / subRatio) * subRatio;                              // :620
                long end = begin + 2 * WINDOW_SIZE;                                  // :621
                if (end >= reimplInput.rows()) end = reimplInput.rows() - 1;         // :623-625
                end = ((end + 1) / subRatio) * subRatio - 1;                         // :626
                long lengthSeq = end - begin + 1;                                    // :627
                long lengthShort, lengthShortLSTM;
                if (lengthSeq != 2 * WINDOW_SIZE + 1) {                              // :628-639
                    lengthShort = lengthSeq;
                    lengthShortLSTM = lengthSeq;
                    for (long r : lstmSS) { lengthShort /= r; lengthShortLSTM /= r; }
                    for (long r : outSS) lengthShort /= r;
                } else {
                    lengthShort = nominalLen;
                    lengthShortLSTM = nominalLstm;
                }
                if (lengthShort > 0) {                                              // :645
                    Eigen::MatrixXd block = reimplInput.block(begin, 0, lengthSeq, reimplInput.cols());
                    Eigen::MatrixXd outShort, oF, oB;
                    plainWindow(block, oF, oB, outShort);
                    sum.block(begin / subRatio, 0, lengthShort, sum.cols()) += outShort;   // :664
                    outCount.block(begin / subRatio, 0, lengthShort, 1) += Eigen::MatrixXd::Ones(lengthShort, 1); // :665
                    long lbeg = begin * outNetRatio / subRatio;                     // :667-669
                    outputForward.block(lbeg, 0, lengthShortLSTM, fwdOut) += oF;
                    outputBackward.block(lbeg, 0, lengthShortLSTM, fwdOut) += oB;
                    outCountLSTM.block(lbeg, 0, lengthShortLSTM, 1) += Eigen::MatrixXd::Ones(lengthShortLSTM, 1);
                }
            }
            for (long cc = 0; cc < reimplOut.cols(); ++cc)                          // :674-675
                for (long r = 0; r < reimplOut.rows(); ++r)
                    reimplOut(r, cc) = sum(r, cc) / outCount(r, 0);
            for (long cc = 0; cc < fwdOut; ++cc)                                    // :677-679
                for (long r = 0; r < lengthOutputLSTM; ++r) {
                    outputForward(r, cc) /= outCountLSTM(r, 0);
                    outputBackward(r, cc) /= outCountLSTM(r, 0);
                }

            long u = 0; double a = 0.0;
            probeGapNaN(realOut, reimplOut, u, a);
            std::cout << "NN_TOL site=e2e_overlap max_ulp=" << u
                      << " max_abs=" << std::scientific << std::setprecision(3) << a << "\n";
            long uF = 0; double aF = 0.0;
            probeGapNaN(nn._OutputForward, outputForward, uF, aF);
            std::cout << "NN_TOL site=e2e_overlap_fwd max_ulp=" << uF
                      << " max_abs=" << std::scientific << std::setprecision(3) << aF << "\n";
            long uB = 0; double aB = 0.0;
            probeGapNaN(nn._OutputBackward, outputBackward, uB, aB);
            std::cout << "NN_TOL site=e2e_overlap_bwd max_ulp=" << uB
                      << " max_abs=" << std::scientific << std::setprecision(3) << aB << "\n";

            Matrix2BinaryFile(out + "e2e_out_overlap.bin", reimplOut);
            Matrix2BinaryFile(out + "e2e_fwd_overlap.bin", outputForward);
            Matrix2BinaryFile(out + "e2e_bwd_overlap.bin", outputBackward);
            dumps += 3;
        }
    }

    // --- Phase 2b Task 2: results2segmentation (conv + updateSegmentation) ----
    // legacy: Segmenter::buildFromConf (:73-148, real _ConvolutionCoeff via the
    // BLSTM_convolution_window_size=9 -> 19-tap hHCw kernel), Segmenter::
    // results2segmentation (:1113-1126: conv gate `_ConvolutionCoeff.cols()>1`,
    // in-place Convolution on results, then updateSegmentation). Constructs a
    // REAL Segmentation from the harness's own AudioStruct (the only real ctor;
    // it needs a live AudioStruct, so `Segmentation seg(6.0)` from the plan is
    // shorthand -- there is no such scalar ctor, see Segmentation.h:106).
    {
        ConfigFile conf(nnConfigPath, '_');
        SegProbe probe;
        probe.buildFromConf(conf, "BLSTM", false, false);

        // conv coeff golden: BLSTM_convolution_window_size=9 -> 2*9+1=19 taps.
        Matrix2BinaryFile(out + "r2s_conv_coeff.bin", probe._ConvolutionCoeff);

        // Deterministic synthetic result row spanning the real thresholds
        // (rising ~0.752, falling ~0.374): r[k] = 0.2 + 0.7*((k*13)%17)/16.
        const int N = 60;
        Eigen::MatrixXd results(1, N);
        for (int k = 0; k < N; ++k) {
            results(0, k) = 0.2 + 0.7 * ((k * 13) % 17) / 16.0;
        }
        Eigen::MatrixXd targets = Eigen::MatrixXd::Zero(1, N);

        // Real Segmentation, seeded from the harness's own AudioStruct (2ch,
        // real duration from OFFSET_SEC/MAX_DUR_SEC truncation); pruning thresh
        // is unused by results2segmentation/updateSegmentation.
        Segmentation seg(audio, 0.5);

        probe.results2segmentation(seg, 0.04, 0.0, results, targets, 0, SPEECH);
        Matrix2BinaryFile(out + "r2s_convolved.bin", results);

        // Walk the channel-0 hypothesis boundary list into an Nx2 begin/type dump.
        {
            const auto& segs = seg._Classification.at(0);
            Eigen::MatrixXd boundaries((long) segs.size(), 2);
            for (std::vector<double>::size_type ii = 0; ii < segs.size(); ++ii) {
                boundaries((long) ii, 0) = segs[ii]._BeginTime;
                boundaries((long) ii, 1) = (double) segs[ii]._Type;
            }
            Matrix2BinaryFile(out + "r2s_boundaries.bin", boundaries);
        }
        dumps += 3;

        // conv=none variant: BLSTM_convolution_window_size -> "0" via the
        // established _Params erase+set pattern, then rebuild _ConvolutionCoeff
        // via a fresh buildFromConf. Assert in-harness that results are UNCHANGED
        // (the `_ConvolutionCoeff.cols() > 1` gate is false -> no Convolution call).
        {
            ConfigFile confNoConv(nnConfigPath, '_');
            confNoConv._Params["BLSTM_convolution_window_size"] = "0";
            SegProbe probeNoConv;
            probeNoConv.buildFromConf(confNoConv, "BLSTM", false, false);

            Eigen::MatrixXd resultsNoConv(1, N);
            for (int k = 0; k < N; ++k) {
                resultsNoConv(0, k) = 0.2 + 0.7 * ((k * 13) % 17) / 16.0;
            }
            Eigen::MatrixXd resultsNoConvCopy = resultsNoConv;
            Eigen::MatrixXd targetsNoConv = Eigen::MatrixXd::Zero(1, N);

            Segmentation segNoConv(audio, 0.5);
            probeNoConv.results2segmentation(segNoConv, 0.04, 0.0, resultsNoConv, targetsNoConv, 0, SPEECH);

            if (resultsNoConv != resultsNoConvCopy) {
                std::cerr << "ERROR: noconv variant mutated results (expected untouched)\n";
                return 1;
            }

            const auto& segsNoConv = segNoConv._Classification.at(0);
            Eigen::MatrixXd boundariesNoConv((long) segsNoConv.size(), 2);
            for (std::vector<double>::size_type ii = 0; ii < segsNoConv.size(); ++ii) {
                boundariesNoConv((long) ii, 0) = segsNoConv[ii]._BeginTime;
                boundariesNoConv((long) ii, 1) = (double) segsNoConv[ii]._Type;
            }
            Matrix2BinaryFile(out + "r2s_boundaries_noconv.bin", boundariesNoConv);
            ++dumps;
        }
    }

    // --- Phase 2b Task 4: TdcSegmenter (Algo 1) ------------------------------
    // legacy: TimeDomainCorrel::getSegmentation (TimeDomainCorrel.cpp:93-259). No
    // NN in the chain (autocorrelation is cwise+sum, only libm is fmath::log/cos
    // via classifySequence + the hamming window coeffs), so the REAL compiled
    // getSegmentation is called AS-IS -- no transcription. A second, independent
    // replication of the framing loop (using the real public classifySequence +
    // AudioStruct::getSequence) recovers the PRE-convolution result_vec for the
    // tdc_result_chan{N}.bin dump (that value is a local inside getSegmentation,
    // not otherwise observable). Uses its OWN DEDICATED AudioStruct (audioTdc,
    // same offset/duration recipe as the shared `audio`, built fresh here) rather
    // than the harness's shared `audio`: the shared object already had
    // applyPreemph(0.97) called on it at the top of main() for earlier phase-1/2
    // stages and is never reset, so reusing it here would double-apply preemph
    // (a harness-local artifact of shared mutable state across stages, not a
    // legacy quirk to reproduce).
    {
        ConfigFile confTdc(tdcConfigPath, '_');
        AudioStruct audioTdc(OFFSET_SEC, MAX_DUR_SEC, 0, item);

        // --- File 1: real getSegmentation on a fresh TdcProbe ------------------
        TdcProbe probe(confTdc);
        Segmentation seg(audioTdc, 0.5);
        probe.getSegmentation(audioTdc, seg);
        seg.compute_errors();

        const long rate = audioTdc.getFrameRate();
        const long long frameCount = audioTdc.getFrameCount();

        // Independent re-derivation of window_size/window_shift/min_lag/max_lag,
        // matching TimeDomainCorrel.cpp:100-111 exactly (TDC_lags re-read here
        // since _MinMaxLag is private even to a derived class -- see TdcProbe).
        std::vector<double>::size_type window_size =
            (std::vector<double>::size_type) boost::math::round(probe._WindowSize * rate / 2.0);
        std::vector<double>::size_type full_window_size = 2 * window_size + 1;
        std::vector<double> minMaxLag = confTdc.get_list<double>("TDC_lags");
        for (auto &v : minMaxLag) {
            if (v < 0.0) v = 0.0;
        }
        long min_lag = (long) boost::math::round(minMaxLag[0] * rate);
        long max_lag = (long) boost::math::round(minMaxLag[1] * rate);
        if (max_lag >= (long) full_window_size) max_lag = full_window_size - 1;

        // getSegmentation already quantized probe._WindowShift as a side effect
        // (the stateful member); window_shift in FRAMES is recovered from it.
        long window_shift = (long) boost::math::round(probe._WindowShift * rate);

        Eigen::MatrixXd windowing_coeff =
            getWindowingCoefficients(probe._WindowingType, false, full_window_size, probe._WindowingParam);

        std::vector<double>::size_type vec_size = (std::vector<double>::size_type) frameCount;
        if ((std::vector<double>::size_type) (vec_size / window_shift) * window_shift == vec_size) {
            vec_size = (std::vector<double>::size_type) (vec_size / window_shift);
        } else {
            vec_size = (std::vector<double>::size_type) (vec_size / window_shift + 1);
        }

        for (int chan = 0; chan < audioTdc.getChannelCount(); ++chan) {
            Eigen::MatrixXd windowed_signal = Eigen::MatrixXd::Zero(1, full_window_size);
            Eigen::MatrixXd result_vec = Eigen::MatrixXd::Zero(1, vec_size);
            for (std::vector<double>::size_type jj = 0; jj < (std::vector<double>::size_type) frameCount; jj += window_shift) {
                audioTdc.getSequence(jj, window_size, chan, probe._FlagDCOffset, windowing_coeff, windowed_signal);
                result_vec(0, jj / window_shift) = probe.classifySequence(min_lag, max_lag, windowed_signal);
            }
            std::ostringstream bufResult;
            bufResult << "tdc_result_chan" << chan + 1 << ".bin";
            Matrix2BinaryFile(out + bufResult.str(), result_vec);
            ++dumps;
        }

        // Convolved rows + boundaries: read back off the REAL post-getSegmentation
        // Segmentation. The convolved result_vec itself is also a getSegmentation
        // local, so it is independently rebuilt here via results2segmentation on a
        // COPY of the pre-conv rows just dumped (same probe -- exercises the
        // identical Segmenter::results2segmentation the real getSegmentation calls).
        for (int chan = 0; chan < audioTdc.getChannelCount(); ++chan) {
            std::ostringstream bufResult;
            bufResult << "tdc_result_chan" << chan + 1 << ".bin";
            Eigen::MatrixXd resultVec;
            {
                long long r, c;
                std::ifstream f(out + bufResult.str(), std::ios::binary);
                f.read((char*) &r, sizeof(long long));
                f.read((char*) &c, sizeof(long long));
                resultVec.resize(r, c);
                f.read((char*) resultVec.data(), sizeof(double) * r * c);
            }
            Eigen::MatrixXd targetSeq = Eigen::MatrixXd::Zero(resultVec.cols(), 1);
            Segmentation segConv(audioTdc, 0.5);
            probe.results2segmentation(segConv, probe._WindowShift, 0.0, resultVec, targetSeq, chan, SPEECH);
            std::ostringstream bufConv;
            bufConv << "tdc_convolved_chan" << chan + 1 << ".bin";
            Matrix2BinaryFile(out + bufConv.str(), resultVec);
            ++dumps;

            std::ostringstream bufBound;
            bufBound << "tdc_boundaries_chan" << chan + 1 << ".bin";
            const auto& segs = segConv._Classification.at(chan);
            Eigen::MatrixXd boundaries((long) segs.size(), 2);
            for (std::vector<double>::size_type ii = 0; ii < segs.size(); ++ii) {
                boundaries((long) ii, 0) = segs[ii]._BeginTime;
                boundaries((long) ii, 1) = (double) segs[ii]._Type;
            }
            Matrix2BinaryFile(out + bufBound.str(), boundaries);
            ++dumps;
        }

        // VRCTS bytes: REAL toFile_VRCTS, 0b-ii byte-equivalence closure golden vs
        // the Rust `to_vrcts_string` writer. The Rust `Segmentation` container has
        // NO `_AudioOffset` field (a documented structural simplification -- offset
        // folds to 0.0), so "identical Segmentations" (spec S10) requires an
        // AudioStruct built with audio_offset=0.0 here too -- audioTdc above uses
        // OFFSET_SEC=0.35, which would bake a `+0.35` into every stime/etime/sigdur
        // that the Rust side cannot reproduce. A dedicated zero-offset AudioStruct
        // on the SAME wav sidesteps the gap entirely (2-channel -> filenames get a
        // _chan_N suffix; only chan 1 is dumped per the brief). Uses `itemVrcts`
        // (stable-path CorpusItem, Finding 3) so the dumped `path=` attribute no
        // longer churns across regenerations.
        {
            AudioStruct audioZeroOff(0.0, MAX_DUR_SEC, 0, itemVrcts);
            ConfigFile confTdcV(tdcConfigPath, '_');
            TdcProbe probeV(confTdcV);
            Segmentation segV(audioZeroOff, 0.5);
            probeV.getSegmentation(audioZeroOff, segV);
            segV.compute_errors();

            std::string base = out + "tdc_vrcts";
            segV.toFile_VRCTS(base);
            std::ifstream vf(base + "_chan_1.xml", std::ios::binary);
            std::ostringstream vs;
            vs << vf.rdbuf();
            std::ofstream outv(out + "tdc_vrcts_chan1.xml", std::ios::binary);
            outv << vs.str();
        }
        ++dumps;

        // compute_errors vs a programmatic reference: build a FRESH Segmentation
        // (own dedicated AudioStruct-derived duration), populate _Reference
        // directly (public member) with two SPEECH spans via label_segment, run
        // the REAL getSegmentation to get a hypothesis, then compute_errors() with
        // the reference branch active. Constants recorded in the manifest.
        {
            AudioStruct audioScore(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            ConfigFile confTdc2(tdcConfigPath, '_');
            TdcProbe probeScore(confTdc2);
            Segmentation segScore(audioScore, 0.5);
            // _Reference starts empty (no .stm/.csv/.xml _RefSegFilename): seed it
            // directly (public member) with the seeded pair (Other@0, End@dur) then
            // two SPEECH spans via the real label_segment, per channel.
            double audioDuration = ((double) audioScore.getFrameCount() - 1) / audioScore.getFrameRate();
            for (int chan = 0; chan < audioScore.getChannelCount(); ++chan) {
                segScore._Reference.push_back(std::deque<Segment> {Segment(), Segment(audioDuration, END)});
                segScore.label_segment(segScore._Reference.at(chan), 0.4, 0.9, SPEECH);
                segScore.label_segment(segScore._Reference.at(chan), 1.2, 1.6, SPEECH);
            }
            probeScore.getSegmentation(audioScore, segScore);
            segScore.compute_errors();

            Eigen::MatrixXd scores(audioScore.getChannelCount(), 3);
            for (int chan = 0; chan < audioScore.getChannelCount(); ++chan) {
                scores(chan, 0) = segScore._ClassificationErrors.at(chan)[SPEECH]._Pfa;
                scores(chan, 1) = segScore._ClassificationErrors.at(chan)[SPEECH]._Pmiss;
                scores(chan, 2) = segScore._ClassificationErrors.at(chan)[SPEECH]._ErrorRate;
            }
            Matrix2BinaryFile(out + "tdc_scores.bin", scores);
            ++dumps;
        }

        // --- TWO-FILES golden: run the SAME probe object again on a FRESH
        // dedicated AudioStruct -- pins the _WindowShift quantization lifecycle
        // (idempotent: already quantized to a grid point of 1/rate, so a second
        // run is a no-op on the member, but the boundary LIST regenerates fresh
        // on a NEW Segmentation/AudioStruct pair, matching a second per-file call
        // in the real corpus driver).
        {
            AudioStruct audioTdc2(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            Segmentation seg2(audioTdc2, 0.5);
            probe.getSegmentation(audioTdc2, seg2);
            seg2.compute_errors();
            const auto& segs2 = seg2._Classification.at(0);
            Eigen::MatrixXd boundaries2((long) segs2.size(), 2);
            for (std::vector<double>::size_type ii = 0; ii < segs2.size(); ++ii) {
                boundaries2((long) ii, 0) = segs2[ii]._BeginTime;
                boundaries2((long) ii, 1) = (double) segs2[ii]._Type;
            }
            Matrix2BinaryFile(out + "tdc_boundaries_file2_chan1.bin", boundaries2);
            ++dumps;
        }
    }

    // --- Phase 2b Task 5: LtsvSegmenter (Algo 2) -----------------------------
    // legacy: LongTermSpectralVariation::getSegmentation (LongTermSpectralVariation.
    // cpp:130-407). PRIMARY config (ltsv.config, nb_DCT=0): the mel/DCT branch calls
    // applyFilterBank only (no DCT -> no Eigen GEMM), so the REAL compiled
    // getSegmentation is the bit-golden AS-IS, same strength oracle as TdcProbe.
    // SECONDARY config (ltsv_dct.config, nb_DCT=4): applyDCT's Eigen GEMM would
    // diverge from the Rust ascending-loop port at this shape, so a dedicated
    // in-process reimplementation of getSegmentation's body is run instead, with
    // ONLY the applyDCT call swapped for applyDCTLoop (already defined above for
    // Phase 1/2's own DCT golden) -- everything else (periodogram, mel filterbank,
    // LTSV classifySequence, results2segmentation) is the REAL compiled machinery.
    {
        ConfigFile confLtsv(ltsvConfigPath, '_');
        AudioStruct audioLtsv(OFFSET_SEC, MAX_DUR_SEC, 0, item);

        // --- File 1: real getSegmentation on a fresh LtsvProbe (nb_DCT=0) ------
        LtsvProbe probe(confLtsv);
        Segmentation seg(audioLtsv, 0.5);
        probe.getSegmentation(audioLtsv, seg);
        seg.compute_errors();

        const long rate = audioLtsv.getFrameRate();
        const long long frameCount = audioLtsv.getFrameCount();

        // Independent re-derivation matching LongTermSpectralVariation.cpp:145-307
        // exactly, to recover the pre-convolution decimated result_vec (a
        // getSegmentation-local, not otherwise observable) via the REAL public
        // classifySequence.
        long spectrumOrder = probe._SpectrumOrder;
        if (spectrumOrder > 19) spectrumOrder = 19;
        std::vector<double>::size_type windowSize = 1 << spectrumOrder;
        std::vector<double>::size_type fullSignalWindowSize = windowSize + 1;
        std::vector<double>::size_type periodogramLength = (1 << (spectrumOrder - 1)) + 1;
        // getSegmentation already quantized probe._SpectrumShift as a side effect.
        long spectrumShift = (long) boost::math::round(probe._SpectrumShift * rate);

        Eigen::MatrixXd windowingCoeff =
            getWindowingCoefficients(probe._WindowingType, false, fullSignalWindowSize, probe._WindowingParam);

        // Freq band: the LTSV.cpp:200-209 clamp-order VARIANT.
        std::vector<double>::size_type freqBeg = 0;
        std::vector<double>::size_type freqEnd = periodogramLength - 1;
        double freqStep = ((double) rate) / 2 / freqEnd;
        std::vector<double>::size_type tmp = (std::vector<double>::size_type) floor(probe._MinFreq / freqStep);
        if (freqBeg < tmp) freqBeg = tmp;
        tmp = (std::vector<double>::size_type) ceil(probe._MaxFreq / freqStep);
        if (freqEnd > tmp) freqEnd = tmp;
        if (freqBeg > freqEnd) freqBeg = freqEnd;

        MelFilterBank melFilters;
        if (probe._NbBins > 0) {
            melFilters = MelFilterBank(probe._MinMelFreq, probe._MaxMelFreq, probe._NbBins,
                                        freqBeg * freqStep, freqEnd * freqStep, rate,
                                        periodogramLength - 1, probe._IsLog, probe._NbDCT,
                                        probe._IgnoreFirstDCT, probe._ComputeDeltasNb,
                                        probe._ComputeDeltaDeltasNb);
            periodogramLength = melFilters.getNbFilters();
            freqBeg = 0;
            freqEnd = periodogramLength - 1;
        }

        Eigen::MatrixXd temporalConvCoeff =
            getWindowingCoefficients(probe._TemporalConvolutionType, true, 2 * probe._TemporalConvolutionSize + 1);

        std::vector<double>::size_type ltsvWindowSize =
            (std::vector<double>::size_type) boost::math::round(probe._WindowSize * rate / 2.0 / spectrumShift);
        if (ltsvWindowSize < 1) ltsvWindowSize = 1;
        // getSegmentation already quantized probe._WindowShift as a side effect;
        // ltsvWindowShift in PERIODOGRAM FRAMES is recovered from it.
        long ltsvWindowShift = (long) boost::math::round(probe._WindowShift * rate / spectrumShift);

        long long beginFrame = 0;
        long long endFrame = frameCount;
        std::vector<double>::size_type vecSize = (std::vector<double>::size_type) (endFrame - beginFrame + 1);
        if ((std::vector<double>::size_type) (vecSize / spectrumShift) * spectrumShift == vecSize) {
            vecSize = (std::vector<double>::size_type) (vecSize / spectrumShift);
        } else {
            vecSize = (std::vector<double>::size_type) (vecSize / spectrumShift + 1);
        }
        std::vector<double>::size_type realVecSize = vecSize;
        if ((std::vector<double>::size_type) (realVecSize / ltsvWindowShift) * ltsvWindowShift == realVecSize) {
            realVecSize = (std::vector<double>::size_type) (realVecSize / ltsvWindowShift);
        } else {
            realVecSize = (std::vector<double>::size_type) (realVecSize / ltsvWindowShift + 1);
        }

        Loki::Factory<AbstractFFT<double>,unsigned int> gfftFactoryLtsv;
        FactoryInit<GFFTList<GFFT,1,20>::Result>::apply(gfftFactoryLtsv);

        for (int chan = 0; chan < audioLtsv.getChannelCount(); ++chan) {
            audioLtsv.computeSegmentPeriodogramEstimates(spectrumOrder, spectrumShift, chan, probe._FlagDCOffset,
                                                          windowingCoeff, melFilters, gfftFactoryLtsv,
                                                          temporalConvCoeff, beginFrame, endFrame);
            const Eigen::MatrixXd& periodogramSrc =
                melFilters.notEmpty() ? audioLtsv._FilterBankedPeriodogram : audioLtsv._Periodogram;

            Eigen::MatrixXd resultVec = Eigen::MatrixXd::Zero(1, realVecSize);
            for (std::vector<double>::size_type jj = 0; jj < vecSize; jj += ltsvWindowShift) {
                resultVec(0, jj / ltsvWindowShift) =
                    probe.classifySequence(jj, freqBeg, freqEnd, ltsvWindowSize, periodogramLength, vecSize, periodogramSrc);
            }
            std::ostringstream bufResult;
            bufResult << "ltsv_result_chan" << chan + 1 << ".bin";
            Matrix2BinaryFile(out + bufResult.str(), resultVec);
            ++dumps;
        }

        // Convolved rows + boundaries: read back off the REAL post-getSegmentation
        // Segmentation, and independently rebuild via results2segmentation on a COPY
        // of the pre-conv rows just dumped (same probe -- exercises the identical
        // Segmenter::results2segmentation the real getSegmentation calls).
        for (int chan = 0; chan < audioLtsv.getChannelCount(); ++chan) {
            std::ostringstream bufResult;
            bufResult << "ltsv_result_chan" << chan + 1 << ".bin";
            Eigen::MatrixXd resultVec;
            {
                long long r, c;
                std::ifstream f(out + bufResult.str(), std::ios::binary);
                f.read((char*) &r, sizeof(long long));
                f.read((char*) &c, sizeof(long long));
                resultVec.resize(r, c);
                f.read((char*) resultVec.data(), sizeof(double) * r * c);
            }
            Eigen::MatrixXd targetSeq = Eigen::MatrixXd::Zero(resultVec.cols(), 1);
            Segmentation segConv(audioLtsv, 0.5);
            probe.results2segmentation(segConv, probe._WindowShift, 0.0, resultVec, targetSeq, chan, SPEECH);
            std::ostringstream bufConv;
            bufConv << "ltsv_convolved_chan" << chan + 1 << ".bin";
            Matrix2BinaryFile(out + bufConv.str(), resultVec);
            ++dumps;

            std::ostringstream bufBound;
            bufBound << "ltsv_boundaries_chan" << chan + 1 << ".bin";
            const auto& segs = segConv._Classification.at(chan);
            Eigen::MatrixXd boundaries((long) segs.size(), 2);
            for (std::vector<double>::size_type ii = 0; ii < segs.size(); ++ii) {
                boundaries((long) ii, 0) = segs[ii]._BeginTime;
                boundaries((long) ii, 1) = (double) segs[ii]._Type;
            }
            Matrix2BinaryFile(out + bufBound.str(), boundaries);
            ++dumps;
        }

        // VRCTS bytes: REAL toFile_VRCTS, on a zero-offset AudioStruct (same
        // rationale as tdc_vrcts_chan1.xml -- the Rust Segmentation container has no
        // _AudioOffset field). Uses `itemVrcts` (stable-path CorpusItem, Finding 3)
        // so the dumped `path=` attribute no longer churns across regenerations.
        {
            AudioStruct audioZeroOff(0.0, MAX_DUR_SEC, 0, itemVrcts);
            ConfigFile confLtsvV(ltsvConfigPath, '_');
            LtsvProbe probeV(confLtsvV);
            Segmentation segV(audioZeroOff, 0.5);
            probeV.getSegmentation(audioZeroOff, segV);
            segV.compute_errors();

            std::string base = out + "ltsv_vrcts";
            segV.toFile_VRCTS(base);
            std::ifstream vf(base + "_chan_1.xml", std::ios::binary);
            std::ostringstream vs;
            vs << vf.rdbuf();
            std::ofstream outv(out + "ltsv_vrcts_chan1.xml", std::ios::binary);
            outv << vs.str();
        }
        ++dumps;

        // compute_errors vs a programmatic reference: same recipe as TdcSegmenter.
        {
            AudioStruct audioScore(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            ConfigFile confLtsv2(ltsvConfigPath, '_');
            LtsvProbe probeScore(confLtsv2);
            Segmentation segScore(audioScore, 0.5);
            double audioDuration = ((double) audioScore.getFrameCount() - 1) / audioScore.getFrameRate();
            for (int chan = 0; chan < audioScore.getChannelCount(); ++chan) {
                segScore._Reference.push_back(std::deque<Segment> {Segment(), Segment(audioDuration, END)});
                segScore.label_segment(segScore._Reference.at(chan), 0.4, 0.9, SPEECH);
                segScore.label_segment(segScore._Reference.at(chan), 1.2, 1.6, SPEECH);
            }
            probeScore.getSegmentation(audioScore, segScore);
            segScore.compute_errors();

            Eigen::MatrixXd scores(audioScore.getChannelCount(), 3);
            for (int chan = 0; chan < audioScore.getChannelCount(); ++chan) {
                scores(chan, 0) = segScore._ClassificationErrors.at(chan)[SPEECH]._Pfa;
                scores(chan, 1) = segScore._ClassificationErrors.at(chan)[SPEECH]._Pmiss;
                scores(chan, 2) = segScore._ClassificationErrors.at(chan)[SPEECH]._ErrorRate;
            }
            Matrix2BinaryFile(out + "ltsv_scores.bin", scores);
            ++dumps;
        }

        // --- TWO-FILES golden: run the SAME probe object again on a FRESH
        // dedicated AudioStruct -- pins the _WindowShift/_SpectrumShift quantization
        // lifecycle (see the Rust two-files test's scoped-down determinism note).
        {
            AudioStruct audioLtsv2(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            Segmentation seg2(audioLtsv2, 0.5);
            probe.getSegmentation(audioLtsv2, seg2);
            seg2.compute_errors();
            const auto& segs2 = seg2._Classification.at(0);
            Eigen::MatrixXd boundaries2((long) segs2.size(), 2);
            for (std::vector<double>::size_type ii = 0; ii < segs2.size(); ++ii) {
                boundaries2((long) ii, 0) = segs2[ii]._BeginTime;
                boundaries2((long) ii, 1) = (double) segs2[ii]._Type;
            }
            Matrix2BinaryFile(out + "ltsv_boundaries_file2_chan1.bin", boundaries2);
            ++dumps;
        }
    }

    // --- Phase 2b Task 5: LtsvSegmenter SECONDARY variant (nb_DCT=4) ---------
    // applyDCT's Eigen GEMM diverges from the ascending-loop port at this shape (the
    // Phase 1 GEMM_CHECK pre-check established this for the same nb_bins/nb_DCT
    // family), so this variant runs an IN-PROCESS reimplementation of
    // getSegmentation's body with ONLY the applyDCT call swapped for applyDCTLoop
    // (defined above, T7). Everything else -- periodogram, mel filterbank, LTSV
    // classifySequence, results2segmentation -- is the REAL compiled machinery,
    // called through the SAME LtsvProbe/protected-member surface as the primary
    // variant.
    {
        ConfigFile confLtsvDct(ltsvDctConfigPath, '_');
        AudioStruct audioLtsvDct(OFFSET_SEC, MAX_DUR_SEC, 0, item);
        LtsvProbe probe(confLtsvDct);

        const long rate = audioLtsvDct.getFrameRate();
        const long long frameCount = audioLtsvDct.getFrameCount();

        long spectrumOrder = probe._SpectrumOrder;
        if (spectrumOrder > 19) spectrumOrder = 19;
        std::vector<double>::size_type periodogramLength = (1 << (spectrumOrder - 1)) + 1;
        long spectrumShift = (long) boost::math::round(probe._SpectrumShift * rate);
        probe._SpectrumShift = ((double) spectrumShift) / rate;
        std::vector<double>::size_type fullSignalWindowSize = (1u << spectrumOrder) + 1;

        Eigen::MatrixXd windowingCoeff =
            getWindowingCoefficients(probe._WindowingType, false, fullSignalWindowSize, probe._WindowingParam);

        if (probe._PreemphRatio > 0) audioLtsvDct.applyPreemph(probe._PreemphRatio);
        if (probe._NoiseSeed > 0) audioLtsvDct.applyNoise(probe._NoiseRatio);

        std::vector<double>::size_type freqBeg = 0;
        std::vector<double>::size_type freqEnd = periodogramLength - 1;
        double freqStep = ((double) rate) / 2 / freqEnd;
        std::vector<double>::size_type tmp = (std::vector<double>::size_type) floor(probe._MinFreq / freqStep);
        if (freqBeg < tmp) freqBeg = tmp;
        tmp = (std::vector<double>::size_type) ceil(probe._MaxFreq / freqStep);
        if (freqEnd > tmp) freqEnd = tmp;
        if (freqBeg > freqEnd) freqBeg = freqEnd;

        MelFilterBank melFilters(probe._MinMelFreq, probe._MaxMelFreq, probe._NbBins,
                                  freqBeg * freqStep, freqEnd * freqStep, rate,
                                  periodogramLength - 1, probe._IsLog, probe._NbDCT,
                                  probe._IgnoreFirstDCT, probe._ComputeDeltasNb,
                                  probe._ComputeDeltaDeltasNb);
        int nbFiltersForDct = (int) melFilters.getNbFilters();
        int nbDctClamped = probe._NbDCT;
        if (nbDctClamped > nbFiltersForDct) nbDctClamped = nbFiltersForDct;
        periodogramLength = melFilters.getNbDCT();
        freqBeg = 0;
        freqEnd = periodogramLength - 1;

        // _CoeffsDCT is private (no getter); reconstruct with the exact ctor
        // formula (MelFilterBank.cpp:108-127), same recipe as the Phase 1 DCT
        // GEMM_CHECK block above: NB_FILTERS x NB_DCT, PI = the legacy _PI literal.
        const double PI_LTSV = 3.14159265358979323846264338327;
        Eigen::MatrixXd dctCoeffs = Eigen::MatrixXd::Zero(nbFiltersForDct, nbDctClamped);
        for (int col = 0; col < nbFiltersForDct; ++col) {
            for (int row = 0; row < nbDctClamped; ++row) {
                dctCoeffs(col, row) = std::cos(PI_LTSV / nbFiltersForDct * (col + 0.5) * row);
            }
        }

        Eigen::MatrixXd temporalConvCoeff =
            getWindowingCoefficients(probe._TemporalConvolutionType, true, 2 * probe._TemporalConvolutionSize + 1);

        std::vector<double>::size_type ltsvWindowSize =
            (std::vector<double>::size_type) boost::math::round(probe._WindowSize * rate / 2.0 / spectrumShift);
        if (ltsvWindowSize < 1) ltsvWindowSize = 1;
        long ltsvWindowShift =
            (long) boost::math::round(probe._WindowShift * rate / spectrumShift);
        if (ltsvWindowShift < 1) ltsvWindowShift = 1;
        probe._WindowShift = ((double) ltsvWindowShift * spectrumShift) / rate;

        long long beginFrame = 0;
        long long endFrame = frameCount;
        std::vector<double>::size_type vecSize = (std::vector<double>::size_type) (endFrame - beginFrame + 1);
        if ((std::vector<double>::size_type) (vecSize / spectrumShift) * spectrumShift == vecSize) {
            vecSize = (std::vector<double>::size_type) (vecSize / spectrumShift);
        } else {
            vecSize = (std::vector<double>::size_type) (vecSize / spectrumShift + 1);
        }
        std::vector<double>::size_type realVecSize = vecSize;
        if ((std::vector<double>::size_type) (realVecSize / ltsvWindowShift) * ltsvWindowShift == realVecSize) {
            realVecSize = (std::vector<double>::size_type) (realVecSize / ltsvWindowShift);
        } else {
            realVecSize = (std::vector<double>::size_type) (realVecSize / ltsvWindowShift + 1);
        }

        Loki::Factory<AbstractFFT<double>,unsigned int> gfftFactoryDct;
        FactoryInit<GFFTList<GFFT,1,20>::Result>::apply(gfftFactoryDct);

        Segmentation seg(audioLtsvDct, 0.5);
        for (int chan = 0; chan < audioLtsvDct.getChannelCount(); ++chan) {
            audioLtsvDct.computeSegmentPeriodogramEstimates(spectrumOrder, spectrumShift, chan, probe._FlagDCOffset,
                                                             windowingCoeff, melFilters, gfftFactoryDct,
                                                             temporalConvCoeff, beginFrame, endFrame);
            // applyDCT substitution: the REAL MelFilterBank::applyFilterBank output
            // (audioLtsvDct._FilterBankedPeriodogram) feeds applyDCTLoop (T7's
            // ascending-loop port of applyDCT) instead of the real (GEMM-based)
            // MelFilterBank::applyDCT.
            Eigen::MatrixXd dctOut = applyDCTLoop(audioLtsvDct._FilterBankedPeriodogram, dctCoeffs,
                                                  nbDctClamped, probe._IgnoreFirstDCT, probe._ComputeDeltasNb,
                                                  probe._ComputeDeltaDeltasNb);

            Eigen::MatrixXd resultVec = Eigen::MatrixXd::Zero(1, realVecSize);
            for (std::vector<double>::size_type jj = 0; jj < vecSize; jj += ltsvWindowShift) {
                resultVec(0, jj / ltsvWindowShift) =
                    probe.classifySequence(jj, freqBeg, freqEnd, ltsvWindowSize, periodogramLength, vecSize, dctOut);
            }
            std::ostringstream bufResult;
            bufResult << "ltsv_dct_result_chan" << chan + 1 << ".bin";
            Matrix2BinaryFile(out + bufResult.str(), resultVec);
            ++dumps;

            Eigen::MatrixXd targetSeq = Eigen::MatrixXd::Zero(resultVec.cols(), 1);
            probe.results2segmentation(seg, probe._WindowShift, 0.0, resultVec, targetSeq, chan, SPEECH);
            std::ostringstream bufConv;
            bufConv << "ltsv_dct_convolved_chan" << chan + 1 << ".bin";
            Matrix2BinaryFile(out + bufConv.str(), resultVec);
            ++dumps;
        }
        seg.compute_errors();
        {
            const auto& segs = seg._Classification.at(0);
            Eigen::MatrixXd boundaries((long) segs.size(), 2);
            for (std::vector<double>::size_type ii = 0; ii < segs.size(); ++ii) {
                boundaries((long) ii, 0) = segs[ii]._BeginTime;
                boundaries((long) ii, 1) = (double) segs[ii]._Type;
            }
            Matrix2BinaryFile(out + "ltsv_dct_boundaries_chan1.bin", boundaries);
            ++dumps;
        }
    }

    // --- Phase 2b Task 5: LtsvSegmenter tiny-window floor-to-1 quirk ---------
    // ltsv_tiny.config: LTSVwindow 0.001 -> round(0.001*8000/2/80) == 0, floored to
    // 1 (LongTermSpectralVariation.cpp:257-258). Real getSegmentation call only --
    // no independent result-row replication needed, this just proves the driver
    // does not panic/divide-by-zero and produces a Segmentation.
    {
        ConfigFile confLtsvTiny(ltsvTinyConfigPath, '_');
        AudioStruct audioLtsvTiny(OFFSET_SEC, MAX_DUR_SEC, 0, item);
        LtsvProbe probeTiny(confLtsvTiny);
        Segmentation segTiny(audioLtsvTiny, 0.5);
        probeTiny.getSegmentation(audioLtsvTiny, segTiny);
        segTiny.compute_errors();
        const auto& segsTiny = segTiny._Classification.at(0);
        Eigen::MatrixXd boundariesTiny((long) segsTiny.size(), 2);
        for (std::vector<double>::size_type ii = 0; ii < segsTiny.size(); ++ii) {
            boundariesTiny((long) ii, 0) = segsTiny[ii]._BeginTime;
            boundariesTiny((long) ii, 1) = (double) segsTiny[ii]._Type;
        }
        Matrix2BinaryFile(out + "ltsv_tiny_boundaries_chan1.bin", boundariesTiny);
        ++dumps;
    }

    // --- Phase 2b Task 5 review, Finding 1: power-scale (is_log_mel=false) variant,
    // NON-VACUOUS decision-layer coverage --------------------------------------
    // ltsv.config (is_log_mel=true) drives ltsv_classify_sequence's score to
    // ~1e51 (see the IMPROVEMENTS.md log-mel-blowup entry): every column sits far
    // above _DecisionThreshRising (0.6), so update_segmentation/smooth_segmentation
    // never see a threshold crossing and the golden collapses to one
    // always-SPEECH span -- vacuous for the hysteresis/smoothing/suppression/
    // padding logic. ltsv_powermel.config is identical to ltsv.config except
    // is_log_mel=false (the power-scale periodogram the LTSV formula's 1e-12 mean
    // floor actually assumes) and smaller (but still nonzero) speech_padding/
    // min_speech/min_silence (0.05/0.1/0.1 vs 0.1/0.2/0.2) -- config keys only, no
    // code changes. Under this config the REAL compiled getSegmentation produces
    // genuine threshold crossings on channel 1: two SPEECH spans separated by an
    // OTHER gap survive smoothing -- the gap is SPEECH[0,0.9892)/OTHER[0.9892,
    // 1.3472)/SPEECH[1.3472,2.0), i.e. 1.3472-0.9892 = 0.358s, which is GREATER
    // than min_silence (0.1s), so suppress_short's "erase short OTHER" rule does
    // NOT fire and the gap survives as its own segment -- SPEECH, OTHER, SPEECH,
    // END. Channel 2 stays a single
    // always-SPEECH span (its own periodogram content never drops below
    // _DecisionThreshFalling long enough to accrue an ending area) -- both channels
    // are dumped so the Rust golden can assert channel 1's non-vacuous structure
    // directly and channel 2's differing (still-trivial) outcome honestly, rather
    // than cherry-picking only the channel that crosses.
    // Same GEMM-free strength oracle as the primary block (nb_DCT=0): the REAL
    // compiled getSegmentation is the bit-golden AS-IS. Dumps the same standard set
    // as the primary block (result/convolved/boundaries rows, both channels),
    // reusing the IDENTICAL independent-replication recipe (periodogram/mel/
    // classifySequence chain via the REAL public classifySequence) verbatim.
    {
        ConfigFile confLtsvPowermel(ltsvPowermelConfigPath, '_');
        AudioStruct audioLtsvPowermel(OFFSET_SEC, MAX_DUR_SEC, 0, item);

        LtsvProbe probePowermel(confLtsvPowermel);
        Segmentation segPowermel(audioLtsvPowermel, 0.5);
        probePowermel.getSegmentation(audioLtsvPowermel, segPowermel);
        segPowermel.compute_errors();

        const long ratePowermel = audioLtsvPowermel.getFrameRate();
        const long long frameCountPowermel = audioLtsvPowermel.getFrameCount();

        long spectrumOrderPm = probePowermel._SpectrumOrder;
        if (spectrumOrderPm > 19) spectrumOrderPm = 19;
        std::vector<double>::size_type windowSizePm = 1 << spectrumOrderPm;
        std::vector<double>::size_type fullSignalWindowSizePm = windowSizePm + 1;
        std::vector<double>::size_type periodogramLengthPm = (1 << (spectrumOrderPm - 1)) + 1;
        long spectrumShiftPm = (long) boost::math::round(probePowermel._SpectrumShift * ratePowermel);

        Eigen::MatrixXd windowingCoeffPm = getWindowingCoefficients(
            probePowermel._WindowingType, false, fullSignalWindowSizePm, probePowermel._WindowingParam);

        std::vector<double>::size_type freqBegPm = 0;
        std::vector<double>::size_type freqEndPm = periodogramLengthPm - 1;
        double freqStepPm = ((double) ratePowermel) / 2 / freqEndPm;
        std::vector<double>::size_type tmpPm = (std::vector<double>::size_type) floor(probePowermel._MinFreq / freqStepPm);
        if (freqBegPm < tmpPm) freqBegPm = tmpPm;
        tmpPm = (std::vector<double>::size_type) ceil(probePowermel._MaxFreq / freqStepPm);
        if (freqEndPm > tmpPm) freqEndPm = tmpPm;
        if (freqBegPm > freqEndPm) freqBegPm = freqEndPm;

        MelFilterBank melFiltersPm;
        if (probePowermel._NbBins > 0) {
            melFiltersPm = MelFilterBank(probePowermel._MinMelFreq, probePowermel._MaxMelFreq, probePowermel._NbBins,
                                          freqBegPm * freqStepPm, freqEndPm * freqStepPm, ratePowermel,
                                          periodogramLengthPm - 1, probePowermel._IsLog, probePowermel._NbDCT,
                                          probePowermel._IgnoreFirstDCT, probePowermel._ComputeDeltasNb,
                                          probePowermel._ComputeDeltaDeltasNb);
            periodogramLengthPm = melFiltersPm.getNbFilters();
            freqBegPm = 0;
            freqEndPm = periodogramLengthPm - 1;
        }

        Eigen::MatrixXd temporalConvCoeffPm = getWindowingCoefficients(
            probePowermel._TemporalConvolutionType, true, 2 * probePowermel._TemporalConvolutionSize + 1);

        std::vector<double>::size_type ltsvWindowSizePm = (std::vector<double>::size_type) boost::math::round(
            probePowermel._WindowSize * ratePowermel / 2.0 / spectrumShiftPm);
        if (ltsvWindowSizePm < 1) ltsvWindowSizePm = 1;
        long ltsvWindowShiftPm =
            (long) boost::math::round(probePowermel._WindowShift * ratePowermel / spectrumShiftPm);

        long long beginFramePm = 0;
        long long endFramePm = frameCountPowermel;
        std::vector<double>::size_type vecSizePm = (std::vector<double>::size_type) (endFramePm - beginFramePm + 1);
        if ((std::vector<double>::size_type) (vecSizePm / spectrumShiftPm) * spectrumShiftPm == vecSizePm) {
            vecSizePm = (std::vector<double>::size_type) (vecSizePm / spectrumShiftPm);
        } else {
            vecSizePm = (std::vector<double>::size_type) (vecSizePm / spectrumShiftPm + 1);
        }
        std::vector<double>::size_type realVecSizePm = vecSizePm;
        if ((std::vector<double>::size_type) (realVecSizePm / ltsvWindowShiftPm) * ltsvWindowShiftPm == realVecSizePm) {
            realVecSizePm = (std::vector<double>::size_type) (realVecSizePm / ltsvWindowShiftPm);
        } else {
            realVecSizePm = (std::vector<double>::size_type) (realVecSizePm / ltsvWindowShiftPm + 1);
        }

        Loki::Factory<AbstractFFT<double>,unsigned int> gfftFactoryPm;
        FactoryInit<GFFTList<GFFT,1,20>::Result>::apply(gfftFactoryPm);

        for (int chan = 0; chan < audioLtsvPowermel.getChannelCount(); ++chan) {
            audioLtsvPowermel.computeSegmentPeriodogramEstimates(
                spectrumOrderPm, spectrumShiftPm, chan, probePowermel._FlagDCOffset, windowingCoeffPm, melFiltersPm,
                gfftFactoryPm, temporalConvCoeffPm, beginFramePm, endFramePm);
            const Eigen::MatrixXd& periodogramSrcPm =
                melFiltersPm.notEmpty() ? audioLtsvPowermel._FilterBankedPeriodogram : audioLtsvPowermel._Periodogram;

            Eigen::MatrixXd resultVecPm = Eigen::MatrixXd::Zero(1, realVecSizePm);
            for (std::vector<double>::size_type jj = 0; jj < vecSizePm; jj += ltsvWindowShiftPm) {
                resultVecPm(0, jj / ltsvWindowShiftPm) = probePowermel.classifySequence(
                    jj, freqBegPm, freqEndPm, ltsvWindowSizePm, periodogramLengthPm, vecSizePm, periodogramSrcPm);
            }
            std::ostringstream bufResultPm;
            bufResultPm << "ltsv_powermel_result_chan" << chan + 1 << ".bin";
            Matrix2BinaryFile(out + bufResultPm.str(), resultVecPm);
            ++dumps;
        }

        for (int chan = 0; chan < audioLtsvPowermel.getChannelCount(); ++chan) {
            std::ostringstream bufResultPm;
            bufResultPm << "ltsv_powermel_result_chan" << chan + 1 << ".bin";
            Eigen::MatrixXd resultVecPm;
            {
                long long r, c;
                std::ifstream f(out + bufResultPm.str(), std::ios::binary);
                f.read((char*) &r, sizeof(long long));
                f.read((char*) &c, sizeof(long long));
                resultVecPm.resize(r, c);
                f.read((char*) resultVecPm.data(), sizeof(double) * r * c);
            }
            Eigen::MatrixXd targetSeqPm = Eigen::MatrixXd::Zero(resultVecPm.cols(), 1);
            Segmentation segConvPm(audioLtsvPowermel, 0.5);
            probePowermel.results2segmentation(segConvPm, probePowermel._WindowShift, 0.0, resultVecPm, targetSeqPm,
                                                chan, SPEECH);
            std::ostringstream bufConvPm;
            bufConvPm << "ltsv_powermel_convolved_chan" << chan + 1 << ".bin";
            Matrix2BinaryFile(out + bufConvPm.str(), resultVecPm);
            ++dumps;

            std::ostringstream bufBoundPm;
            bufBoundPm << "ltsv_powermel_boundaries_chan" << chan + 1 << ".bin";
            const auto& segsPm = segConvPm._Classification.at(chan);
            Eigen::MatrixXd boundariesPm((long) segsPm.size(), 2);
            for (std::vector<double>::size_type ii = 0; ii < segsPm.size(); ++ii) {
                boundariesPm((long) ii, 0) = segsPm[ii]._BeginTime;
                boundariesPm((long) ii, 1) = (double) segsPm[ii]._Type;
            }
            Matrix2BinaryFile(out + bufBoundPm.str(), boundariesPm);
            ++dumps;
        }
    }

    // --- Phase 2b Task 6: BlstmSignalSegmenter (Algo 4) ----------------------
    // legacy: BLSTMSignalSegmenter::getSegmentation (BLSTMSignalSegmenter.cpp:93-398,
    // LIVE code only). The FIRST driver with the NN in the chain: result_vec comes
    // from _BLSTMNeuralNetwork.feedForwardBackward (:260), whose real Eigen GEMMs
    // diverge from the ascending-loop port. Each variant's PRIMARY golden dumps are
    // produced by a TRANSCRIPTION of getSegmentation with ONLY that FFB call swapped
    // for signalReimplFFB (the reimpl family above); everything else -- the param
    // math (:96-108, :221-254), the results2segmentation call, the compute_errors --
    // is faithful. The SECONDARY probe runs the REAL BLSTMSignalSegmenter::
    // getSegmentation beside it: segment count + types must match EXACTLY (abort on
    // mismatch, spec decision 3), boundary max-delta recorded as
    //   SEG_STRUCT site=signal_<variant> ok=1 max_dt=<measured>.
    //
    // The real net (1_worker_1.config topology, 33671 weights) expects 23 inputs;
    // signal mode feeds a 1-column input (audio._Data.row(chan).transpose()), which
    // the LSTM input projection accepts via inputW.topRows(cols) width tolerance --
    // the legacy's own behavior, kept.
    //
    // Three variants (real net + real config keys, overridden window/shift):
    //   window0   -- BLSTM_window 0        (full-sequence, plain FFB path)
    //   overlap   -- BLSTM_window 0.01, shift 0.0005  (overlap FFB path; window_shift
    //                4 == ssr, the ONLY non-broken overlap regime -- the brief's
    //                original 0.5/0.1 params are broken-as-committed, see the
    //                signal-overlap-oob note below the runVariant calls)
    //   noOverlap -- BLSTM_window 0.5, shift 0    (truncate FFB path; the `=0.0`
    //                _WindowShift poisoning + the ssr-division sizing gate)
    {
        Eigen::VectorXd flatSignal = BinaryFile2Vector(nnWeightsPath);
        T6Blstm reimplNet(flatSignal);

        // Boundary-dump helper: (begin, type) rows off a Segmentation's channel-0
        // hypothesis list.
        auto dumpBoundaries = [&](const Segmentation& seg, const std::string& name) {
            const auto& segs = seg._Classification.at(0);
            Eigen::MatrixXd b((long) segs.size(), 2);
            for (std::vector<double>::size_type ii = 0; ii < segs.size(); ++ii) {
                b((long) ii, 0) = segs[ii]._BeginTime;
                b((long) ii, 1) = (double) segs[ii]._Type;
            }
            Matrix2BinaryFile(out + name, b);
        };

        // Run ONE variant end to end: transcribe getSegmentation (reimpl FFB) for the
        // golden dumps, then the REAL getSegmentation as a structural SECONDARY probe.
        // `windowOverride`/`shiftOverride` are the raw BLSTM_window / BLSTM_shift
        // config values for the variant. `chan` is fixed to 0 (chan 1) for the dumps.
        auto runVariant = [&](const std::string& tag, const std::string& windowOverride,
                              const std::string& shiftOverride) {
            // --- Transcription (reimpl FFB) on a FRESH probe + FRESH audio -------
            ConfigFile confT(signalConfigPath, '_');
            confT._Params.erase("BLSTM_weightsFile");
            confT.set_val<std::string>("BLSTM_window", windowOverride);
            confT.set_val<std::string>("BLSTM_shift", shiftOverride);
            AudioStruct audioT(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            SignalProbe probe(confT, flatSignal);

            const long rate = audioT.getFrameRate();
            const long long frameCount = audioT.getFrameCount();
            const long ssr = probe._BLSTMNeuralNetwork.getSubSamplingRatio();

            // signalraw: audio._Data BEFORE preemph/noise (:133-134). The signal
            // timing divergence golden vs signal (post).
            Matrix2BinaryFile(out + "signal_" + tag + "_signalraw_chan1.bin",
                              audioT._Data.row(0));

            // Window/shift derivation in SIGNAL samples (:96-108).
            std::vector<double>::size_type BLSTM_window_size =
                (std::vector<double>::size_type) boost::math::round(probe._WindowSize * rate / 2.0);
            if ((BLSTM_window_size != 0) && (BLSTM_window_size < (std::vector<double>::size_type) ssr))
                BLSTM_window_size = ssr;
            std::vector<double>::size_type BLSTM_full_window_size = 2 * BLSTM_window_size + 1;
            long BLSTM_window_shift = (long) boost::math::round(probe._WindowShift * rate);
            bool noOverlap = false;
            if ((BLSTM_window_size != 0) && (BLSTM_window_shift < 1)) {
                noOverlap = true;
                BLSTM_window_size =
                    (((std::vector<double>::size_type) boost::math::round(probe._WindowSize * rate)) / ssr) * ssr;
                if (BLSTM_window_size < 10 * (std::vector<double>::size_type) ssr)
                    BLSTM_window_size = 10 * ssr;
                BLSTM_full_window_size = BLSTM_window_size;
            }
            if ((BLSTM_window_size == 0) || (BLSTM_window_shift < 1)) BLSTM_window_shift = 1;
            probe._WindowShift = ((double) BLSTM_window_shift) / rate;   // MEMBER MUTATION (:108)

            // preemph -> noise (:137-148). windowing_coeff computed (:149) but never
            // applied (dump-only dead weight); getWindowingCoefficients returns 0 cols
            // for full==1 (window==0), matching the Rust windowing_coefficients(size<=1)
            // -> None no-op.
            if (probe._PreemphRatio > 0) audioT.applyPreemph(probe._PreemphRatio);
            if (probe._NoiseSeed > 0) audioT.applyNoise(probe._NoiseRatio);
            Eigen::MatrixXd windowing_coeff =
                getWindowingCoefficients(probe._WindowingType, false, BLSTM_full_window_size, probe._WindowingParam);

            // signal: audio._Data AFTER preemph/noise (:174-175).
            Matrix2BinaryFile(out + "signal_" + tag + "_signal_chan1.bin", audioT._Data.row(0));

            probe._BLSTMNeuralNetwork.resetWeightsDerivatives();  // :170

            const int chan = 0;
            Eigen::MatrixXd inputSeq = audioT._Data.row(chan).transpose();  // :188 Nx1

            // Result-vec sizing (:221-240): unconditional ceil-division by
            // BLSTM_window_shift, then ssr-division GATED on (window==0 || noOverlap).
            std::vector<double>::size_type vec_size = (std::vector<double>::size_type) frameCount;
            std::vector<double>::size_type real_vec_size = vec_size;
            if ((std::vector<double>::size_type) (real_vec_size / BLSTM_window_shift) * BLSTM_window_shift == real_vec_size) {
                real_vec_size = (std::vector<double>::size_type) (real_vec_size / BLSTM_window_shift);
            } else {
                real_vec_size = (std::vector<double>::size_type) (real_vec_size / BLSTM_window_shift + 1);
            }
            if ((BLSTM_window_size == 0) || (noOverlap)) {
                if (ssr > 1) {
                    std::vector<std::vector<double>::size_type> LSTMRatios =
                        probe._BLSTMNeuralNetwork.getLSTMSubSampling();
                    for (std::vector<double>::size_type jj = 0; jj < LSTMRatios.size(); ++jj)
                        real_vec_size /= LSTMRatios[jj];
                    std::vector<std::vector<double>::size_type> OutputRatios =
                        probe._BLSTMNeuralNetwork.getOutputSubSampling();
                    for (std::vector<double>::size_type jj = 0; jj < OutputRatios.size(); ++jj)
                        real_vec_size /= OutputRatios[jj];
                }
            }
            Eigen::MatrixXd result_vec = Eigen::MatrixXd::Zero(real_vec_size, 1);

            // timeStep/timeOffset (:244-254). _WindowShift here is the POST-:108 value.
            double timeStep = probe._WindowShift * ssr;
            double timeOffset = timeStep / 2 - probe._WindowShift / 2;
            if (BLSTM_window_size > 0) {
                if (noOverlap) {
                    timeStep = probe._WindowShift * ssr;
                    timeOffset = timeStep / 2 - probe._WindowShift / 2;
                } else {
                    timeStep = probe._WindowShift;
                    timeOffset = 0.0;
                }
            }

            // NN invocation SWAPPED for the reimpl (:259-260). No targets (no
            // reference set on this Segmentation). Input mutated in place by the type
            // -1 self-normalization inside signalReimplFFB, matching the real class.
            Eigen::MatrixXd reimplInput = inputSeq;
            signalReimplFFB(reimplNet, reimplInput, (long) BLSTM_window_size, BLSTM_window_shift,
                            noOverlap, result_vec);

            Eigen::MatrixXd result_vec2 = result_vec.transpose();  // :315 ROW vector
            Matrix2BinaryFile(out + "signal_" + tag + "_result_chan1.bin", result_vec2);

            // results2segmentation (:344-345): targetSeqTmp = result_vec copy quirk.
            Eigen::MatrixXd targetSeqTmp = result_vec;
            Segmentation seg(audioT, 0.5);
            probe.results2segmentation(seg, timeStep, timeOffset, result_vec2, targetSeqTmp, chan, SPEECH);
            seg.compute_errors();
            Matrix2BinaryFile(out + "signal_" + tag + "_convolved_chan1.bin", result_vec2);
            dumpBoundaries(seg, "signal_" + tag + "_boundaries_chan1.bin");
            dumps += 5;   // signalraw, signal, result, convolved, boundaries

            // --- SECONDARY structural probe: REAL getSegmentation on FRESH audio ---
            ConfigFile confR(signalConfigPath, '_');
            confR._Params.erase("BLSTM_weightsFile");
            confR.set_val<std::string>("BLSTM_window", windowOverride);
            confR.set_val<std::string>("BLSTM_shift", shiftOverride);
            AudioStruct audioR(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            SignalProbe probeR(confR, flatSignal);
            Segmentation segR(audioR, 0.5);
            probeR.getSegmentation(audioR, segR);   // REAL Eigen FFB
            segR.compute_errors();

            const auto& segsReimpl = seg._Classification.at(chan);
            const auto& segsReal = segR._Classification.at(chan);
            if (segsReimpl.size() != segsReal.size()) {
                std::cerr << "SEG_STRUCT site=signal_" << tag << " ABORT: count "
                          << segsReimpl.size() << " (reimpl) != " << segsReal.size()
                          << " (real)\n";
                std::abort();
            }
            double maxDt = 0.0;
            for (std::vector<double>::size_type ii = 0; ii < segsReimpl.size(); ++ii) {
                if (segsReimpl[ii]._Type != segsReal[ii]._Type) {
                    std::cerr << "SEG_STRUCT site=signal_" << tag << " ABORT: type mismatch at "
                              << ii << " (" << (int) segsReimpl[ii]._Type << " vs "
                              << (int) segsReal[ii]._Type << ")\n";
                    std::abort();
                }
                double dt = std::fabs(segsReimpl[ii]._BeginTime - segsReal[ii]._BeginTime);
                if (dt > maxDt) maxDt = dt;
            }
            std::cout << "SEG_STRUCT site=signal_" << tag << " ok=1 max_dt="
                      << std::scientific << std::setprecision(3) << maxDt << "\n";

            // VRCTS bytes: REAL toFile_VRCTS on the reimpl-driven Segmentation, on a
            // zero-offset AudioStruct (same rationale as tdc/ltsv_vrcts_chan1.xml).
            {
                ConfigFile confV(signalConfigPath, '_');
                confV._Params.erase("BLSTM_weightsFile");
                confV.set_val<std::string>("BLSTM_window", windowOverride);
                confV.set_val<std::string>("BLSTM_shift", shiftOverride);
                AudioStruct audioZeroOff(0.0, MAX_DUR_SEC, 0, itemVrcts);
                SignalProbe probeV(confV, flatSignal);
                const long rateV = audioZeroOff.getFrameRate();
                const long long frameCountV = audioZeroOff.getFrameCount();
                const long ssrV = probeV._BLSTMNeuralNetwork.getSubSamplingRatio();

                std::vector<double>::size_type wsz =
                    (std::vector<double>::size_type) boost::math::round(probeV._WindowSize * rateV / 2.0);
                if ((wsz != 0) && (wsz < (std::vector<double>::size_type) ssrV)) wsz = ssrV;
                long wsh = (long) boost::math::round(probeV._WindowShift * rateV);
                bool noOv = false;
                if ((wsz != 0) && (wsh < 1)) {
                    noOv = true;
                    wsz = (((std::vector<double>::size_type) boost::math::round(probeV._WindowSize * rateV)) / ssrV) * ssrV;
                    if (wsz < 10 * (std::vector<double>::size_type) ssrV) wsz = 10 * ssrV;
                }
                if ((wsz == 0) || (wsh < 1)) wsh = 1;
                probeV._WindowShift = ((double) wsh) / rateV;
                if (probeV._PreemphRatio > 0) audioZeroOff.applyPreemph(probeV._PreemphRatio);
                if (probeV._NoiseSeed > 0) audioZeroOff.applyNoise(probeV._NoiseRatio);
                probeV._BLSTMNeuralNetwork.resetWeightsDerivatives();

                std::vector<double>::size_type vszV = (std::vector<double>::size_type) frameCountV;
                std::vector<double>::size_type rvszV = vszV;
                if ((std::vector<double>::size_type) (rvszV / wsh) * wsh == rvszV) rvszV = rvszV / wsh;
                else rvszV = rvszV / wsh + 1;
                if ((wsz == 0) || (noOv)) {
                    if (ssrV > 1) {
                        auto LR = probeV._BLSTMNeuralNetwork.getLSTMSubSampling();
                        for (auto r : LR) rvszV /= r;
                        auto OR = probeV._BLSTMNeuralNetwork.getOutputSubSampling();
                        for (auto r : OR) rvszV /= r;
                    }
                }
                Eigen::MatrixXd rvV = Eigen::MatrixXd::Zero(rvszV, 1);
                double tsV = probeV._WindowShift * ssrV;
                double toV = tsV / 2 - probeV._WindowShift / 2;
                if (wsz > 0) {
                    if (noOv) { tsV = probeV._WindowShift * ssrV; toV = tsV / 2 - probeV._WindowShift / 2; }
                    else { tsV = probeV._WindowShift; toV = 0.0; }
                }
                Eigen::MatrixXd inV = audioZeroOff._Data.row(0).transpose();
                signalReimplFFB(reimplNet, inV, (long) wsz, wsh, noOv, rvV);
                Eigen::MatrixXd rvV2 = rvV.transpose();
                Eigen::MatrixXd ttV = rvV;
                Segmentation segV(audioZeroOff, 0.5);
                probeV.results2segmentation(segV, tsV, toV, rvV2, ttV, 0, SPEECH);
                segV.compute_errors();

                std::string base = out + "signal_" + tag + "_vrcts";
                segV.toFile_VRCTS(base);
                std::ifstream vf(base + "_chan_1.xml", std::ios::binary);
                std::ostringstream vs;
                vs << vf.rdbuf();
                std::ofstream outv(out + "signal_" + tag + "_vrcts_chan1.xml", std::ios::binary);
                outv << vs.str();
            }
            ++dumps;

            // compute_errors vs a programmatic two-span reference (same recipe as
            // TDC/LTSV): the transcription with a reference set, so getTargets fires.
            {
                ConfigFile confS(signalConfigPath, '_');
                confS._Params.erase("BLSTM_weightsFile");
                confS.set_val<std::string>("BLSTM_window", windowOverride);
                confS.set_val<std::string>("BLSTM_shift", shiftOverride);
                AudioStruct audioS(OFFSET_SEC, MAX_DUR_SEC, 0, item);
                SignalProbe probeS(confS, flatSignal);
                Segmentation segS(audioS, 0.5);
                double audioDuration = ((double) audioS.getFrameCount() - 1) / audioS.getFrameRate();
                for (int ch = 0; ch < audioS.getChannelCount(); ++ch) {
                    segS._Reference.push_back(std::deque<Segment> {Segment(), Segment(audioDuration, END)});
                    segS.label_segment(segS._Reference.at(ch), 0.4, 0.9, SPEECH);
                    segS.label_segment(segS._Reference.at(ch), 1.2, 1.6, SPEECH);
                }
                // Real getSegmentation (with reference -> getTargets active); the score
                // is a structural/decision-layer golden, downstream of the same decision
                // machinery, so the real Eigen FFB is fine here (the boundaries come from
                // update_segmentation, pinned structurally by SEG_STRUCT above).
                probeS.getSegmentation(audioS, segS);
                segS.compute_errors();
                Eigen::MatrixXd scores(audioS.getChannelCount(), 3);
                for (int ch = 0; ch < audioS.getChannelCount(); ++ch) {
                    scores(ch, 0) = segS._ClassificationErrors.at(ch)[SPEECH]._Pfa;
                    scores(ch, 1) = segS._ClassificationErrors.at(ch)[SPEECH]._Pmiss;
                    scores(ch, 2) = segS._ClassificationErrors.at(ch)[SPEECH]._ErrorRate;
                }
                Matrix2BinaryFile(out + "signal_" + tag + "_scores.bin", scores);
                ++dumps;
            }
        };

        runVariant("window0", "0", "0.1");
        // NOTE (legacy bug, IMPROVEMENTS.md signal-overlap-oob): the brief's overlap
        // params (window 0.5, shift 0.1) crash the REAL getSegmentation with an Eigen
        // block-bounds assertion -- the overlap result-vec sizing (ceil(frameCount/
        // shift)) is far too small for the OverLap FFB driver's write index
        // (begin/ssr + lengthShort ~ frameCount/ssr). It only survives when
        // window_shift <= ssr (== 4 samples here), i.e. shift <= 0.0005s. We use the
        // working degenerate regime (window 0.01 -> window_size 40, shift 0.0005 ->
        // window_shift 4 == ssr) so the OverLap path IS exercised end-to-end; the
        // brief's 0.5/0.1 is a broken-as-committed legacy path (verified: real
        // getSegmentation aborts on it under -DEIGEN assertions, heap-corrupts under
        // release -DNDEBUG), documented not shipped.
        runVariant("overlap", "0.01", "0.0005");
        runVariant("noOverlap", "0.5", "0");

        // --- TWO-FILES golden (noOverlap only): the REAL lifecycle. Unlike TDC/LTSV
        // where the shift re-quantization is idempotent, signal noOverlap assigns
        // _WindowShift = 0.0 mid-run (:376, the noOverlap poisoning). So on file 2 the
        // SAME SignalProbe instance re-enters getSegmentation with _WindowShift == 0.0:
        // :99 rounds 0.0*rate -> 0 -> shift<1 -> noOverlap re-triggers, :107 clamps
        // shift back to 1, :108 rewrites _WindowShift = 1/rate. The dump lets the Rust
        // two-files test pin this genuine round trip (both files' boundaries dumped so
        // the test can compare, not assume, file1 == file2).
        {
            ConfigFile conf2(signalConfigPath, '_');
            conf2._Params.erase("BLSTM_weightsFile");
            conf2.set_val<std::string>("BLSTM_window", "0.5");
            conf2.set_val<std::string>("BLSTM_shift", "0");
            SignalProbe probe2(conf2, flatSignal);

            // File 1: real getSegmentation, dump boundaries (file1 reference).
            AudioStruct audioF1(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            Segmentation segF1(audioF1, 0.5);
            probe2.getSegmentation(audioF1, segF1);
            segF1.compute_errors();
            dumpBoundaries(segF1, "signal_noOverlap_boundaries_file1_chan1.bin");
            ++dumps;

            // File 2: SAME probe2 (now carrying _WindowShift == 0.0 from :376), fresh
            // audio -> re-enters the noOverlap path deterministically.
            AudioStruct audioF2(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            Segmentation segF2(audioF2, 0.5);
            probe2.getSegmentation(audioF2, segF2);
            segF2.compute_errors();
            dumpBoundaries(segF2, "signal_noOverlap_boundaries_file2_chan1.bin");
            ++dumps;
        }
    }

    // --- Phase 2b Task 7: BlstmSpectralSegmenter (Algo 3, no pitch pass) ------
    // legacy: BLSTMSpectralSegmenter::getSegmentation (BLSTMSpectralSegmenter.cpp:
    // 593-887, MINUS the pitch second pass :757-805 which the Task-7 configs
    // short-circuit -- TDCwindow 0 -> TDC_window_size 0 -> the `if (TDC_window_size
    // > 0)` gate at :757 is false). This is the REAL 1_worker_1.config's algorithm.
    // The spectral result_vec comes from _BLSTMNeuralNetwork.feedForwardBackward
    // (:740), whose real Eigen GEMMs diverge from the ascending-loop port; the PRIMARY
    // golden dumps are produced by a TRANSCRIPTION of getSegmentation with ONLY that
    // FFB call swapped for signalReimplFFB (the Task 6 reimpl family, shared -- the
    // plain/truncate/overlap windowed drivers are input-column-agnostic and the
    // leftCols crop stays FALSE at D=11<23). Everything else -- the spectral param
    // derivation (initSpectralAnalysis/getWindowingCoeff/getTemporalConvolution/
    // getLTSVParam/getTDCParam/getBLSTMParam/getLTSV/getBLSTMInputSequence), the
    // results2segmentation call, the compute_errors -- calls the REAL compiled
    // protected helpers via the SpectralProbe `using` exposures.
    //
    // CROSS-CHANNEL REUSE QUIRK (the load-bearing layout fact): result_vec is
    // allocated ONCE by getBLSTMParam (:631) and REUSED across channels (:740). Under
    // the overlap FFB (accumulate-in-place, :664 += then /= count) channel 2 SEEDS
    // from channel 1's post-division contents. Reproduced by holding one result_vec
    // across the channel loop (matching the real class); the overlap variant dumps
    // BOTH channels so the Rust cross-channel golden can pin it.
    //
    // SECONDARY probe (spec decision 3): the REAL compiled getSegmentation runs beside
    // the transcription; segment count + types must match EXACTLY (abort on mismatch),
    // boundary max-delta recorded as SEG_STRUCT site=spectral_<variant> ok=1
    // max_dt=<measured>.
    //
    // The real net (33671 weights) expects 23 inputs; the spectral input width is
    // D == 11 (3*nb_DCT-1 with IgnoreFirstDCT), 11 < 23 -> the LSTM input projection's
    // inputW.topRows(cols) width tolerance applies (the E2E gate's D=11 leg), the
    // leftCols crop never fires.
    //
    // Three variants (real net + real config keys, BLSTM_shift overridden):
    //   real      -- 1_worker_1.config AS-IS (BLSTM_window 3.25, shift 0.8 -> overlap
    //                FFB; BLSTM_window_size 163, window_shift 80, real_vec_size 50)
    //   overlap   -- BLSTM_shift 0.01 (heavier overlap; window_shift 1, real_vec_size
    //                50 -- verified no OOB: max write index 50 == buffer rows)
    //   noOverlap -- BLSTM_shift 0 (truncate FFB via the noOverlap poisoning;
    //                window_size floored to (round(3.25*8000/80)/4)*4 = 324,
    //                window_shift clamped to 1, real_vec_size 50)
    {
        Eigen::VectorXd flatSpectral = BinaryFile2Vector(nnWeightsPath);
        T6Blstm reimplNetSpec(flatSpectral);

        auto dumpBoundariesSpec = [&](const Segmentation& seg, int chan, const std::string& name) {
            const auto& segs = seg._Classification.at(chan);
            Eigen::MatrixXd b((long) segs.size(), 2);
            for (std::vector<double>::size_type ii = 0; ii < segs.size(); ++ii) {
                b((long) ii, 0) = segs[ii]._BeginTime;
                b((long) ii, 1) = (double) segs[ii]._Type;
            }
            Matrix2BinaryFile(out + name, b);
        };

        // TRANSCRIPTION of getSegmentation (:593-887 minus the pitch block). The FFB
        // call is swapped for signalReimplFFB (the ascending-loop NN), AND the input
        // sequence is built with the ASCENDING-LOOP feature pipeline (deriveSpectral +
        // computeSegmentPeriodogramEstimates with an EMPTY mel bank + applyFilterBank +
        // applyDCTLoop + getLTSV), NOT the real compiled getBLSTMInputSequence -- because
        // the real applyDCT is an Eigen GEMM that diverges from the ascending-loop DCT
        // the Phase 1 goldens + the Rust port reproduce (same rationale as the E2E leg,
        // and confirmed here: the real getBLSTMInputSequence diverges from the Rust
        // build_input_sequence by up to ~19k ULP). The stateful param derivation
        // (getBLSTMParam :439-500, _SpectrumShift/_WindowShift/_SpectrumShiftInFrames) is
        // transcribed inline (pure integer/round math, no Eigen) mutating the probe's
        // members faithfully; the Segmenter machinery (getTargets/results2segmentation)
        // is called REAL via the probe. `setReference` sets a two-span programmatic
        // reference (getTargets fires). result_vec is allocated ONCE and reused across
        // channels (the cross-channel reuse quirk). The caller owns probe/audio/seg so
        // the two-files test can reuse a probe.
        // `pitchPass` (Task 8) enables the pitch-homothety SECOND pass
        // (BLSTMSpectralSegmenter.cpp:757-805): gated on TDC_window_size > 0, run
        // getPitch REAL over the pass-1 segmentation, warp _Periodogram by pitch/300,
        // re-filterbank/DCT, rebuild inputSeq with the OLD (pass-1) LTSV column
        // (:792 -- the LTSV matrix is NOT recomputed, a load-bearing quirk), re-forward
        // via the reimpl FFB, clearClassification, re-results2segmentation, and OVERWRITE
        // the error/classif slots. `pitchOut` (when non-null) receives the chan-0 pitch.
        // The DUMP QUIRK (:848-850): result_vec2 is dumped BEFORE the pitch pass rewrites
        // result_vec, so the result golden preserves the PASS-1 result even though the
        // final boundaries are PASS-2 -- reproduced here (dumps stay at the pass-1 point).
        auto transcribeSpectral =
            [&](SpectralProbe& probe, AudioStruct& audio, Segmentation& seg, bool setReference,
                const std::string& tag, bool dumpChan1, bool dumpBothChannels,
                bool pitchPass, double* pitchOut) {
                const unsigned Max = 20;
                Loki::Factory<AbstractFFT<double>, unsigned int> gfft_factory;
                FactoryInit<GFFTList<GFFT, 1, Max>::Result>::apply(gfft_factory);

                if (setReference) {
                    double audioDuration = ((double) audio.getFrameCount() - 1) / audio.getFrameRate();
                    for (int ch = 0; ch < audio.getChannelCount(); ++ch) {
                        seg._Reference.push_back(std::deque<Segment>{Segment(), Segment(audioDuration, END)});
                        seg.label_segment(seg._Reference.at(ch), 0.4, 0.9, SPEECH);
                        seg.label_segment(seg._Reference.at(ch), 1.2, 1.6, SPEECH);
                    }
                }

                const double rate = (double) audio.getFrameRate();
                const long ssr = probe._BLSTMNeuralNetwork.getSubSamplingRatio();

                // --- Feature-pipeline params via the ascending-loop reimpl (matches the
                // E2E leg + the Rust SpectralParams::derive). No Eigen DCT here. The
                // feature params (spectrum_order/shift, mel, freq band, LTSV) are
                // INDEPENDENT of the BLSTM_shift override (which only drives _WindowShift,
                // read from the probe below), so a fresh readFeatureCfg on the real
                // config is exact regardless of the variant. ---
                ConfigFile featConf(nnConfigPath, '_');
                featConf._Params.erase("BLSTM_weightsFile");
                FeatureCfg c = readFeatureCfg(featConf, "BLSTM");
                SpectralP s = deriveSpectral(c, rate, Max);

                // Stateful _SpectrumShiftInFrames/_SpectrumShift (:208-210), set on the
                // probe so getTargets/results2segmentation + the timeStep math see them.
                probe._SpectrumShiftInFrames = s.shift_frames;
                probe._SpectrumShift = (double) s.shift_frames / rate;

                // preemph -> noise (:216-227). Applied ONCE, directly on the audio.
                if (c.preemph > 0) audio.applyPreemph(c.preemph);
                if (c.noise_seed > 0) audio.applyNoise(c.noise_ratio);

                // getLTSVParam re-quantization of _LTSVWindowShift (:302-306); dead for
                // TDCwindow-0/LTSVwindow-0 configs but tracked for the stateful map.
                {
                    long ltsv_ws = (long) boost::math::round(c.ltsv_shift * rate / (double) s.shift_frames);
                    if (ltsv_ws < 1) ltsv_ws = 1;
                    (void) ltsv_ws;
                }

                // Mel bank + DCT coeffs for the ascending pipeline (the E2E-leg recipe).
                MelFilterBank mel;
                if (c.nb_bins > 0) {
                    mel = MelFilterBank(c.min_mel, c.max_mel, c.nb_bins, s.min_freq_snapped,
                                        s.max_freq_snapped, rate, s.bins - 1, c.is_log, c.nb_dct,
                                        c.ignore_first, c.deltas_nb, c.dd_nb);
                }
                const double PI = 3.14159265358979323846264338327;
                long nbFilters = mel.notEmpty() ? (long) mel.getNbFilters() : 0;
                long nbDct = c.nb_dct;
                if (nbDct > nbFilters) nbDct = nbFilters;
                Eigen::MatrixXd coeffs;
                if (c.nb_bins > 0 && c.nb_dct > 0) {
                    coeffs = Eigen::MatrixXd::Zero(nbFilters, nbDct);
                    for (long col = 0; col < nbFilters; ++col)
                        for (long row = 0; row < nbDct; ++row)
                            coeffs(col, row) = std::cos(PI / nbFilters * (col + 0.5) * row);
                }
                Eigen::MatrixXd win = getWindowingCoefficients(c.win_type, false, s.window_size + 1, c.win_param);
                Eigen::MatrixXd noConv;
                MelFilterBank emptyMel;

                // getTDCParam REAL (:344-369): derives TDC_window_size/full/shift/min_lag/
                // max_lag (NN-free round math) and the TDC windowing coeffs. Mutates the
                // probe's _TDCWindowShift/_MinMaxLag members as real side effects, reading
                // _TDCWindowSize/_Balance/_TDCWindowingType/Param + _SpectrumShiftInFrames
                // (set above). For the pitch config TDCwindow 0.032 -> TDC_window_size 128;
                // for the Task-7 configs TDCwindow 0 -> TDC_window_size 0 (pitch gate off).
                std::stringstream tdcLog;
                std::vector<double>::size_type TDC_window_size = 0, TDC_full_window_size = 0;
                long TDC_window_shift = 0, min_lag = 0, max_lag = 0;
                Eigen::MatrixXd TDC_windowing_coeff = probe.getTDCParam(
                    audio, tdcLog, TDC_window_size, TDC_full_window_size, TDC_window_shift, min_lag, max_lag);

                // --- getBLSTMParam (:439-500) transcribed inline (integer/round math,
                // no Eigen). Mutates probe._WindowShift (the stateful _WindowShift). ---
                const long ssif = s.shift_frames;
                std::vector<double>::size_type BLSTM_window_size =
                    (std::vector<double>::size_type) boost::math::round(probe._WindowSize * rate / 2.0 / ssif);
                if ((BLSTM_window_size != 0) && (BLSTM_window_size < (std::vector<double>::size_type) ssr))
                    BLSTM_window_size = ssr;
                long BLSTM_window_shift = (long) boost::math::round(probe._WindowShift * rate / ssif);
                bool noOverlap = false;
                if ((BLSTM_window_size != 0) && (BLSTM_window_shift < 1)) {
                    noOverlap = true;
                    BLSTM_window_size = (((std::vector<double>::size_type) boost::math::round(probe._WindowSize * rate / ssif)) / ssr) * ssr;
                    if (BLSTM_window_size < 10 * (std::vector<double>::size_type) ssr)
                        BLSTM_window_size = 10 * ssr;
                }
                if ((BLSTM_window_size == 0) || (BLSTM_window_shift < 1)) BLSTM_window_shift = 1;
                probe._WindowShift = ((double) BLSTM_window_shift * ssif) / rate;  // :453 MEMBER MUTATION
                // result_vec sizing (:475-499): ceil(frameCount/ssif), then UNCONDITIONAL
                // sequential ssr-division (the `if noOverlap` gate at :487 is commented).
                std::vector<double>::size_type vec_size = (std::vector<double>::size_type) audio.getFrameCount();
                if ((vec_size / ssif) * ssif == vec_size) vec_size = vec_size / ssif;
                else vec_size = vec_size / ssif + 1;
                std::vector<double>::size_type real_vec_size = vec_size;
                if (ssr > 1) {
                    for (auto r : probe._BLSTMNeuralNetwork.getLSTMSubSampling()) real_vec_size /= r;
                    for (auto r : probe._BLSTMNeuralNetwork.getOutputSubSampling()) real_vec_size /= r;
                }
                // result_vec allocated ONCE (:631), reused across channels -- the
                // load-bearing cross-channel reuse quirk.
                Eigen::MatrixXd result_vec = Eigen::MatrixXd::Zero(real_vec_size, 1);
                Eigen::MatrixXd targetSeq = Eigen::MatrixXd();

                probe._BLSTMNeuralNetwork.resetWeightsDerivatives();  // :473

                // timeStep/timeOffset (:724-734). Overlap OVERRIDE with _SpectrumShift.
                double timeStep = probe._WindowShift * ssr;
                double timeOffset = timeStep / 2 - probe._WindowShift / 2;
                if (BLSTM_window_size > 0) {
                    if (noOverlap) {
                        timeStep = probe._WindowShift * ssr;
                        timeOffset = timeStep / 2 - probe._WindowShift / 2;
                    } else {
                        timeStep = probe._SpectrumShift * ssr;
                        timeOffset = timeStep / 2 - probe._SpectrumShift / 2;
                    }
                }

                const int channelCount = audio.getChannelCount();
                const long long endFull = audio.getFrameCount() - 1;
                for (int chan = 0; chan < channelCount; ++chan) {
                    // Periodogram with an EMPTY mel bank -> _Periodogram is raw (NOT run
                    // through the real Eigen applyDCT), matching the E2E leg + Rust port.
                    audio.computeSegmentPeriodogramEstimates(s.order, s.shift_frames, chan, c.flag_dc,
                                                             win, emptyMel, gfft_factory, noConv, 0, endFull);
                    Eigen::MatrixXd perio = audio._Periodogram;

                    Eigen::MatrixXd inputSeq;
                    if (mel.notEmpty()) {
                        Eigen::MatrixXd fb = Eigen::MatrixXd::Zero(perio.rows(), mel.getNbFilters());
                        mel.applyFilterBank(perio, fb);
                        if (c.nb_dct > 0) {
                            inputSeq = applyDCTLoop(fb, coeffs, (int) nbDct, c.ignore_first, c.deltas_nb, c.dd_nb);
                        } else {
                            inputSeq = fb;
                        }
                    } else {
                        inputSeq = (((perio.block(0, s.freq_beg, perio.rows(),
                                                  s.freq_end - s.freq_beg + 1).array() + 1e-24).log()).matrix());
                    }
                    // The pass-1 LTSV column, hoisted so the pitch pass can REUSE it
                    // verbatim (:792 rebuilds inputSeq with the OLD LTSV -- NOT recomputed
                    // on the warped periodogram; a load-bearing quirk).
                    Eigen::MatrixXd ltsvPass1;
                    if (s.ltsv_half_window > 0) {
                        long ltsv_beg = mel.notEmpty() ? 0 : s.freq_beg;
                        long ltsv_end = mel.notEmpty() ? (long) inputSeq.cols() - 1 : s.freq_end;
                        ltsvPass1 = getLTSV(perio, s.ltsv_half_window, s.ltsv_shift, ltsv_beg, ltsv_end);
                        Eigen::MatrixXd merged(inputSeq.rows(), inputSeq.cols() + 1);
                        merged << inputSeq, ltsvPass1;
                        inputSeq = merged;
                    }
                    if (chan == 0 && dumpChan1) {
                        Matrix2BinaryFile(out + "spectral_" + tag + "_inputseq_chan1.bin", inputSeq);
                        ++dumps;
                    }

                    if (seg._Reference.size() > 0) {
                        targetSeq = result_vec;
                        probe.getTargets(seg, timeStep, timeOffset, targetSeq, chan, SPEECH);
                    }
                    probe._BLSTMNeuralNetwork.setProcessingType((BLSTM_window_size > 0), !noOverlap);

                    // NN invocation SWAPPED for the reimpl (:740). Input mutated in
                    // place by the type -1 self-normalization inside signalReimplFFB.
                    // result_vec is the SHARED buffer (reused across channels).
                    Eigen::MatrixXd reimplInput = inputSeq;
                    signalReimplFFB(reimplNetSpec, reimplInput, (long) BLSTM_window_size, BLSTM_window_shift, noOverlap, result_vec);

                    Eigen::MatrixXd result_vec2 = result_vec.transpose();  // :744 ROW vector
                    if (chan == 0 && dumpChan1) {
                        Matrix2BinaryFile(out + "spectral_" + tag + "_result_chan1.bin", result_vec2);
                        ++dumps;
                    } else if (dumpBothChannels && chan == 1) {
                        Matrix2BinaryFile(out + "spectral_" + tag + "_result_chan2.bin", result_vec2);
                        ++dumps;
                    }

                    // results2segmentation (:750-751): targetSeqTmp = result_vec copy
                    // quirk (dead param in the live results2segmentation).
                    Eigen::MatrixXd targetSeqTmp = result_vec;
                    probe.results2segmentation(seg, timeStep, timeOffset, result_vec2, targetSeqTmp, chan, SPEECH);
                    seg._CumulativeError[chan] = 0.0;   // no NN cost accumulated (no targets on the reimpl path)
                    seg._NbOfClassif[chan] = 0;

                    if (chan == 0 && dumpChan1) {
                        Matrix2BinaryFile(out + "spectral_" + tag + "_convolved_chan1.bin", result_vec2);
                        ++dumps;
                    } else if (dumpBothChannels && chan == 1) {
                        Matrix2BinaryFile(out + "spectral_" + tag + "_convolved_chan2.bin", result_vec2);
                        ++dumps;
                    }

                    // --- PITCH SECOND PASS (:757-805) ---------------------------------
                    // Gated on TDC_window_size > 0 (pitch config: 128; Task-7 configs: 0).
                    if (pitchPass && TDC_window_size > 0) {
                        // getPitch REAL (:758) over the PASS-1 seg (just written above by
                        // results2segmentation). NN-free (getSequence + computePitch = an
                        // autocorrelation argmax, no NN, no fmath::log). Uses _FlagDCOffset.
                        double pitch = probe.getPitch(audio, seg, tdcLog, chan, TDC_window_size,
                                                      TDC_full_window_size, TDC_window_shift, min_lag,
                                                      max_lag, TDC_windowing_coeff);
                        if (chan == 0 && pitchOut) *pitchOut = pitch;

                        // Homothety warp of _Periodogram by coeff = pitch/300 (:760-775).
                        // The legacy warps audio._Periodogram in place from a copy
                        // (periodogramMem). Here `perio` IS that copy (== audio._Periodogram
                        // for this channel); applyHomothety returns the warped matrix.
                        double coeff_homo = pitch / 300.0;
                        Eigen::MatrixXd warped = applyHomothety(perio, coeff_homo);
                        // The raw warped periodogram is NOT dumped (it is 201x513, ~825KB);
                        // its effect is fully pinned downstream by the pass-2 inputseq golden
                        // (warp -> filterbank -> DCT -> LTSV), which fails if the warp drifts.

                        // Re-filterbank/DCT on the WARPED periodogram (:777-787). Same
                        // ascending pipeline as pass 1 (mel.applyFilterBank REAL, applyDCTLoop
                        // for the DCT -- the real Eigen applyDCT diverges, same rationale).
                        Eigen::MatrixXd inputSeqP;
                        if (mel.notEmpty()) {
                            Eigen::MatrixXd fbP = Eigen::MatrixXd::Zero(warped.rows(), mel.getNbFilters());
                            mel.applyFilterBank(warped, fbP);
                            if (c.nb_dct > 0) {
                                inputSeqP = applyDCTLoop(fbP, coeffs, (int) nbDct, c.ignore_first, c.deltas_nb, c.dd_nb);
                            } else {
                                inputSeqP = fbP;
                            }
                        } else {
                            inputSeqP = (((warped.block(0, s.freq_beg, warped.rows(),
                                                        s.freq_end - s.freq_beg + 1).array() + 1e-24).log()).matrix());
                        }
                        // Rebuild inputSeq WITH THE OLD (pass-1) LTSV column (:792 -- NOT
                        // recomputed on the warped periodogram; the load-bearing quirk).
                        if (s.ltsv_half_window > 0) {
                            Eigen::MatrixXd merged(inputSeqP.rows(), inputSeqP.cols() + 1);
                            merged << inputSeqP, ltsvPass1;
                            inputSeqP = merged;
                        }
                        if (chan == 0 && dumpChan1) {
                            Matrix2BinaryFile(out + "spectral_" + tag + "_inputseq_pass2_chan1.bin", inputSeqP);
                            ++dumps;
                        }

                        // Re-forward (:793) into the SAME result_vec (:793 reuses it). The
                        // pitch>0 guard (:791) short-circuits the whole re-seg when pitch==0.
                        if (pitch > 0) {
                            Eigen::MatrixXd reimplInputP = inputSeqP;
                            signalReimplFFB(reimplNetSpec, reimplInputP, (long) BLSTM_window_size, BLSTM_window_shift, noOverlap, result_vec);
                            Eigen::MatrixXd result_vec2P = result_vec.transpose();  // :797
                            if (chan == 0 && dumpChan1) {
                                Matrix2BinaryFile(out + "spectral_" + tag + "_result_pass2_chan1.bin", result_vec2P);
                                ++dumps;
                            }
                            // clearClassification (:800) then re-results2segmentation (:801)
                            // OVERWRITE the pass-1 boundaries. targetSeqTmp copy quirk (:799).
                            Eigen::MatrixXd targetSeqTmpP = result_vec;
                            seg.clearClassification(chan);
                            probe.results2segmentation(seg, timeStep, timeOffset, result_vec2P, targetSeqTmpP, chan, SPEECH);
                            // error/classif slots OVERWRITTEN (:802-803); no NN cost on the
                            // reimpl path, so they land back at 0/0 (same as pass 1).
                            seg._CumulativeError[chan] = 0.0;
                            seg._NbOfClassif[chan] = 0;
                            if (chan == 0 && dumpChan1) {
                                Matrix2BinaryFile(out + "spectral_" + tag + "_convolved_pass2_chan1.bin", result_vec2P);
                                ++dumps;
                            }
                        }
                    }
                }
                seg.compute_errors();
                if (dumpChan1) {
                    dumpBoundariesSpec(seg, 0, "spectral_" + tag + "_boundaries_chan1.bin");
                    ++dumps;
                }
                if (dumpBothChannels) {
                    dumpBoundariesSpec(seg, 1, "spectral_" + tag + "_boundaries_chan2.bin");
                    ++dumps;
                }
                if (noOverlap) probe._WindowShift = 0.0;  // :885 (post-dump order, cosmetic)
                return std::pair<double, long>(seg._CumulativeError[0], seg._NbOfClassif[0]);
            };

        // Run ONE variant end to end: transcribe (reimpl FFB) for the golden dumps,
        // then the REAL getSegmentation as a structural SECONDARY probe, then VRCTS +
        // scores.
        auto runSpectralVariant = [&](const std::string& tag, const std::string& shiftOverride,
                                      bool dumpBothChannels) {
            // --- Transcription (reimpl FFB) on a FRESH probe + FRESH audio ------
            ConfigFile confT(nnConfigPath, '_');
            confT._Params.erase("BLSTM_weightsFile");
            confT.set_val<std::string>("BLSTM_shift", shiftOverride);
            AudioStruct audioT(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            SpectralProbe probe(confT);
            probe.setWeights(flatSpectral);
            Segmentation seg(audioT, 0.5);
            transcribeSpectral(probe, audioT, seg, /*setReference=*/false, tag, /*dumpChan1=*/true, dumpBothChannels, /*pitchPass=*/false, /*pitchOut=*/nullptr);

            // --- SECONDARY structural probe: REAL getSegmentation on FRESH audio ---
            ConfigFile confR(nnConfigPath, '_');
            confR._Params.erase("BLSTM_weightsFile");
            confR.set_val<std::string>("BLSTM_shift", shiftOverride);
            AudioStruct audioR(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            SpectralProbe probeR(confR);
            probeR.setWeights(flatSpectral);
            Segmentation segR(audioR, 0.5);
            probeR.getSegmentation(audioR, segR);  // REAL Eigen FFB
            segR.compute_errors();

            const int chanCmp = 0;
            const auto& segsReimpl = seg._Classification.at(chanCmp);
            const auto& segsReal = segR._Classification.at(chanCmp);
            if (segsReimpl.size() != segsReal.size()) {
                std::cerr << "SEG_STRUCT site=spectral_" << tag << " ABORT: count "
                          << segsReimpl.size() << " (reimpl) != " << segsReal.size()
                          << " (real)\n";
                std::abort();
            }
            double maxDt = 0.0;
            for (std::vector<double>::size_type ii = 0; ii < segsReimpl.size(); ++ii) {
                if (segsReimpl[ii]._Type != segsReal[ii]._Type) {
                    std::cerr << "SEG_STRUCT site=spectral_" << tag << " ABORT: type mismatch at "
                              << ii << " (" << (int) segsReimpl[ii]._Type << " vs "
                              << (int) segsReal[ii]._Type << ")\n";
                    std::abort();
                }
                double dt = std::fabs(segsReimpl[ii]._BeginTime - segsReal[ii]._BeginTime);
                if (dt > maxDt) maxDt = dt;
            }
            std::cout << "SEG_STRUCT site=spectral_" << tag << " ok=1 max_dt="
                      << std::scientific << std::setprecision(3) << maxDt << "\n";

            // VRCTS bytes: REAL toFile_VRCTS on the reimpl-driven Segmentation, on a
            // zero-offset AudioStruct (same rationale as signal_*_vrcts_chan1.xml).
            {
                ConfigFile confV(nnConfigPath, '_');
                confV._Params.erase("BLSTM_weightsFile");
                confV.set_val<std::string>("BLSTM_shift", shiftOverride);
                AudioStruct audioZeroOff(0.0, MAX_DUR_SEC, 0, itemVrcts);
                SpectralProbe probeV(confV);
                probeV.setWeights(flatSpectral);
                Segmentation segV(audioZeroOff, 0.5);
                transcribeSpectral(probeV, audioZeroOff, segV, /*setReference=*/false, tag, /*dumpChan1=*/false, /*dumpBothChannels=*/false, /*pitchPass=*/false, /*pitchOut=*/nullptr);
                std::string base = out + "spectral_" + tag + "_vrcts";
                segV.toFile_VRCTS(base);
                std::ifstream vf(base + "_chan_1.xml", std::ios::binary);
                std::ostringstream vs;
                vs << vf.rdbuf();
                std::ofstream outv(out + "spectral_" + tag + "_vrcts_chan1.xml", std::ios::binary);
                outv << vs.str();
            }
            ++dumps;

            // compute_errors vs a programmatic two-span reference (getTargets fires):
            // the transcription with a reference set (chan-0 scores dumped as Nx3).
            {
                ConfigFile confS(nnConfigPath, '_');
                confS._Params.erase("BLSTM_weightsFile");
                confS.set_val<std::string>("BLSTM_shift", shiftOverride);
                AudioStruct audioS(OFFSET_SEC, MAX_DUR_SEC, 0, item);
                SpectralProbe probeS(confS);
                probeS.setWeights(flatSpectral);
                Segmentation segS(audioS, 0.5);
                transcribeSpectral(probeS, audioS, segS, /*setReference=*/true, tag, /*dumpChan1=*/false, /*dumpBothChannels=*/false, /*pitchPass=*/false, /*pitchOut=*/nullptr);
                Eigen::MatrixXd scores(audioS.getChannelCount(), 3);
                for (int ch = 0; ch < audioS.getChannelCount(); ++ch) {
                    scores(ch, 0) = segS._ClassificationErrors.at(ch)[SPEECH]._Pfa;
                    scores(ch, 1) = segS._ClassificationErrors.at(ch)[SPEECH]._Pmiss;
                    scores(ch, 2) = segS._ClassificationErrors.at(ch)[SPEECH]._ErrorRate;
                }
                Matrix2BinaryFile(out + "spectral_" + tag + "_scores.bin", scores);
                ++dumps;
            }
        };

        // real (1_worker_1.config AS-IS: BLSTM_shift 0.8 -> overlap window_shift 80),
        // overlap (shift 0.01 -> window_shift 1, heavier overlap, cross-channel golden),
        // noOverlap (shift 0 -> truncate).
        runSpectralVariant("real", "8.000000000000000e-01", /*dumpBothChannels=*/false);
        runSpectralVariant("overlap", "0.01", /*dumpBothChannels=*/true);
        runSpectralVariant("noOverlap", "0", /*dumpBothChannels=*/false);

        // --- PITCH VARIANT (Task 8): the pitch-homothety second pass ---------------
        // Config = the "real" spectral base (BLSTM_shift 0.8 -> overlap) + TDC keys that
        // activate the pitch pass (brief-mandated): TDCwindow 0.032 -> TDC_window_size
        // 128 > 0 (gate on), TDCshift 0.01, lags (0.002,0.016), balance 0.7, hamming-257
        // window with param 0.8. The pass-1 seg (from the reimpl FFB) crosses the rising
        // threshold on chan-1 ([SPEECH@0, OTHER@~1.18, END@2.0]), so getPitch walks a real
        // SPEECH segment and returns a nonzero pitch -> the warp does real work and pass-2
        // boundaries can differ from pass-1. DUMP QUIRK reproduced: the pass-1 result/
        // convolved dumps are preserved (dumped before the pitch pass), while the pass-2
        // result/inputseq/boundaries are dumped separately; the final boundaries dump
        // reflects pass-2 (post-clearClassification).
        {
            auto setPitchOverrides = [&](ConfigFile& conf) {
                conf.set_val<std::string>("BLSTM_shift", "8.000000000000000e-01");
                conf.set_val<std::string>("BLSTM_TDCwindow", "0.032");
                conf.set_val<std::string>("BLSTM_TDCshift", "0.01");
                conf.set_val<std::string>("BLSTM_TDC_lags", "0.002,0.016");
                conf.set_val<std::string>("BLSTM_TDC_balance", "0.7");
                conf.set_val<std::string>("BLSTM_TDC_windowing_type", "hamming");
                conf.set_val<std::string>("BLSTM_TDC_windowing_param", "0.8");
            };
            const std::string tag = "pitch";

            // --- Transcription (reimpl FFB, pitch pass active) on FRESH probe+audio ---
            ConfigFile confT(nnConfigPath, '_');
            confT._Params.erase("BLSTM_weightsFile");
            setPitchOverrides(confT);
            AudioStruct audioT(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            SpectralProbe probe(confT);
            probe.setWeights(flatSpectral);
            Segmentation seg(audioT, 0.5);
            double measuredPitch = -1.0;
            transcribeSpectral(probe, audioT, seg, /*setReference=*/false, tag, /*dumpChan1=*/true, /*dumpBothChannels=*/false, /*pitchPass=*/true, &measuredPitch);
            {
                Eigen::MatrixXd pitchMat(1, 1);
                pitchMat(0, 0) = measuredPitch;
                Matrix2BinaryFile(out + "spectral_pitch_pitch_chan1.bin", pitchMat);
                ++dumps;
            }
            std::cout << "SPECTRAL_PITCH measured_pitch_chan1=" << std::setprecision(17)
                      << measuredPitch << "\n";

            // --- SECONDARY structural probe: REAL getSegmentation (its OWN pitch pass)
            // on FRESH audio. The reimpl pass-2 structure must match the real Eigen
            // pass-2 structure (count + types) or abort. ---
            ConfigFile confR(nnConfigPath, '_');
            confR._Params.erase("BLSTM_weightsFile");
            setPitchOverrides(confR);
            AudioStruct audioR(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            SpectralProbe probeR(confR);
            probeR.setWeights(flatSpectral);
            Segmentation segR(audioR, 0.5);
            probeR.getSegmentation(audioR, segR);  // REAL Eigen FFB + REAL pitch pass
            segR.compute_errors();

            const int chanCmp = 0;
            const auto& segsReimpl = seg._Classification.at(chanCmp);
            const auto& segsReal = segR._Classification.at(chanCmp);
            if (segsReimpl.size() != segsReal.size()) {
                std::cerr << "SEG_STRUCT site=spectral_" << tag << " ABORT: count "
                          << segsReimpl.size() << " (reimpl) != " << segsReal.size()
                          << " (real)\n";
                std::abort();
            }
            double maxDt = 0.0;
            for (std::vector<double>::size_type ii = 0; ii < segsReimpl.size(); ++ii) {
                if (segsReimpl[ii]._Type != segsReal[ii]._Type) {
                    std::cerr << "SEG_STRUCT site=spectral_" << tag << " ABORT: type mismatch at "
                              << ii << " (" << (int) segsReimpl[ii]._Type << " vs "
                              << (int) segsReal[ii]._Type << ")\n";
                    std::abort();
                }
                double dt = std::fabs(segsReimpl[ii]._BeginTime - segsReal[ii]._BeginTime);
                if (dt > maxDt) maxDt = dt;
            }
            std::cout << "SEG_STRUCT site=spectral_" << tag << " ok=1 max_dt="
                      << std::scientific << std::setprecision(3) << maxDt << "\n";

            // VRCTS bytes: REAL toFile_VRCTS on the reimpl-driven (pass-2) Segmentation,
            // zero-offset audio (same rationale as the other spectral vrcts goldens).
            {
                ConfigFile confV(nnConfigPath, '_');
                confV._Params.erase("BLSTM_weightsFile");
                setPitchOverrides(confV);
                AudioStruct audioZeroOff(0.0, MAX_DUR_SEC, 0, itemVrcts);
                SpectralProbe probeV(confV);
                probeV.setWeights(flatSpectral);
                Segmentation segV(audioZeroOff, 0.5);
                transcribeSpectral(probeV, audioZeroOff, segV, /*setReference=*/false, tag, /*dumpChan1=*/false, /*dumpBothChannels=*/false, /*pitchPass=*/true, /*pitchOut=*/nullptr);
                std::string base = out + "spectral_" + tag + "_vrcts";
                segV.toFile_VRCTS(base);
                std::ifstream vf(base + "_chan_1.xml", std::ios::binary);
                std::ostringstream vs;
                vs << vf.rdbuf();
                std::ofstream outv(out + "spectral_" + tag + "_vrcts_chan1.xml", std::ios::binary);
                outv << vs.str();
            }
            ++dumps;
        }

        // --- TWO-FILES golden (noOverlap only): the stateful lifecycle. The spectral
        // noOverlap assigns _WindowShift = 0.0 at :885 (post-dump). So on file 2 the
        // SAME probe re-enters getBLSTMParam with _WindowShift == 0.0: :445 rounds
        // 0.0 -> 0, :446 -> noOverlap re-triggers, :452 clamps shift to 1, :453
        // rewrites _WindowShift = 1*80/rate. Dump both files' boundaries + file2's
        // params row so the Rust two-files test pins the round trip.
        {
            ConfigFile conf2(nnConfigPath, '_');
            conf2._Params.erase("BLSTM_weightsFile");
            conf2.set_val<std::string>("BLSTM_shift", "0");
            SpectralProbe probe2(conf2);
            probe2.setWeights(flatSpectral);

            AudioStruct audioF1(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            Segmentation segF1(audioF1, 0.5);
            transcribeSpectral(probe2, audioF1, segF1, /*setReference=*/false, "noOverlap_file1", /*dumpChan1=*/false, /*dumpBothChannels=*/false, /*pitchPass=*/false, /*pitchOut=*/nullptr);
            dumpBoundariesSpec(segF1, 0, "spectral_noOverlap_boundaries_file1_chan1.bin");
            ++dumps;

            // File 2: SAME probe2 (now carrying _WindowShift == 0.0 from :885), fresh
            // audio -> re-enters the noOverlap path deterministically. Dump file2's
            // params row (window_shift, window_size, spectrum_shift_in_frames,
            // real_vec_size) so the two-files test can pin the quantization lifecycle.
            AudioStruct audioF2(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            Segmentation segF2(audioF2, 0.5);
            transcribeSpectral(probe2, audioF2, segF2, /*setReference=*/false, "noOverlap_file2", /*dumpChan1=*/false, /*dumpBothChannels=*/false, /*pitchPass=*/false, /*pitchOut=*/nullptr);
            dumpBoundariesSpec(segF2, 0, "spectral_noOverlap_boundaries_file2_chan1.bin");
            ++dumps;

            // File2 params row: re-derive via getBLSTMParam on a THIRD fresh audio with
            // probe2's post-file2 state (already at _WindowShift == 0.0 again -> the
            // same noOverlap re-derivation). Dumped as a 1x4 row for the Rust golden.
            AudioStruct audioF3(OFFSET_SEC, MAX_DUR_SEC, 0, item);
            std::stringstream ls3;
            const unsigned Max3 = 20;
            MelFilterBank mf3;
            std::vector<double>::size_type fw3 = 0, fb3 = 0, fe3 = 0;
            probe2.initSpectralAnalysis(audioF3, ls3, Max3, fw3, fb3, fe3, mf3);
            std::vector<double>::size_type bws3 = 0;
            long bwsh3 = 0;
            bool noOv3 = false;
            Eigen::MatrixXd rv3 = probe2.getBLSTMParam(audioF3, ls3, bws3, bwsh3, noOv3);
            Eigen::MatrixXd paramsRow(1, 4);
            paramsRow(0, 0) = (double) bwsh3;
            paramsRow(0, 1) = (double) bws3;
            paramsRow(0, 2) = (double) probe2._SpectrumShiftInFrames;
            paramsRow(0, 3) = (double) rv3.rows();
            Matrix2BinaryFile(out + "spectral_noOverlap_params_file2.bin", paramsRow);
            ++dumps;
        }
    }

    // =====================================================================
    // --- Phase 3 Task 1: BACKWARD reimpl NN_TOL/NN_PROBE probes -----------
    // For each site, construct the REAL compiled layer/net, run its forward THEN
    // backward on the same operands, harvest getWeightsDerivatives().col(0) (the
    // summed deriv), and compare it element-by-element (ULP) against the ascending-
    // loop reimpl's col(0). Synthetic small shapes MUST be max_ulp=0 (the extractor
    // asserts it); the real-net site (blstm_real_backward, k>=23) records nonzero as
    // Phase 4 calibration. The reimpl-produced Nx2 derivs ARE the goldens (Tasks 3-6
    // consume them), dumped here -- NOT the real-Eigen derivs.
    {
        ConfigFile conf(nnConfigPath, '_');
        conf._Params.erase("BLSTM_weightsFile");
        conf.set_val<bool>("SYNW_IsCellsPeepholesActive", true);
        conf.set_val<bool>("SYNW_IsGatesPeepholesActive", true);
        conf.set_val<bool>("SYNW_IsGatesRecurrentPeepholesActive", true);

        auto synthFlat = [](long n) {
            Eigen::VectorXd flat(n);
            for (long k = 0; k < n; ++k) flat(k) = (double)((k * 11 + 3) % 97) / 97.0 - 0.5;
            return flat;
        };
        auto makeInput = [](int T, int cols) {
            Eigen::MatrixXd x(T, cols);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < cols; ++j)
                    x(t, j) = (double)(((t * 37 + j * 53 + 7) % 101)) / 101.0 - 0.5;
            return x;
        };
        // Deterministic delta seed (the incoming gradient a layer/net receives).
        auto makeDeltas = [](int T, int O) {
            Eigen::MatrixXd dd(T, O);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < O; ++j)
                    dd(t, j) = (double)(((t * 29 + j * 41 + 5) % 83)) / 83.0 - 0.5;
            return dd;
        };
        // ULP gap over col(0) of two Nx2 deriv matrices (the summed derivative).
        auto derivGap = [](const Eigen::MatrixXd& real, const Eigen::MatrixXd& reimpl,
                           long& maxUlp, double& maxAbs) {
            long rows = std::min(real.rows(), reimpl.rows());
            for (long r = 0; r < rows; ++r) {
                double a = real(r, 0), b = reimpl(r, 0);
                double absGap = std::fabs(a - b);
                if (absGap > maxAbs) maxAbs = absGap;
                uint64_t ab, bb;
                std::memcpy(&ab, &a, sizeof(double));
                std::memcpy(&bb, &b, sizeof(double));
                long ulp = (ab > bb) ? (long)(ab - bb) : (long)(bb - ab);
                if (ulp > maxUlp) maxUlp = ulp;
            }
        };

        // Full-matrix ULP/abs gap over an arbitrary (T x C) tensor (deltas_out), not
        // just col 0 -- used by the Phase 3 Task 5 net-backward deltas_out comparison.
        auto deltasGap = [](const Eigen::MatrixXd& real, const Eigen::MatrixXd& reimpl,
                            long& maxUlp, double& maxAbs) {
            long rows = std::min(real.rows(), reimpl.rows());
            long cols = std::min(real.cols(), reimpl.cols());
            for (long r = 0; r < rows; ++r) {
                for (long c = 0; c < cols; ++c) {
                    double a = real(r, c), b = reimpl(r, c);
                    double absGap = std::fabs(a - b);
                    if (absGap > maxAbs) maxAbs = absGap;
                    uint64_t ab, bb;
                    std::memcpy(&ab, &a, sizeof(double));
                    std::memcpy(&bb, &b, sizeof(double));
                    long ulp = (ab > bb) ? (long)(ab - bb) : (long)(bb - ab);
                    if (ulp > maxUlp) maxUlp = ulp;
                }
            }
        };

        // ---- Site lstm_backward: standalone LSTMLayer::feedBackward -----------
        // Grid {I=3,O=2,T=7; I=5,O=4,T=6}. Build the real layer, run feedForward to
        // populate _Gates/_CellStates/_CellsIn, then feedBackward(deltas). Mirror with
        // lstmForwardLoopCache + lstmBackwardLoop on the same weights/input/deltas.
        {
            long maxUlp = 0; double maxAbs = 0.0;
            struct S { int I, O, T; } grid[] = {{3, 2, 7}, {5, 4, 6}};
            for (const S& s : grid) {
                LSTMLayer layer(conf, "SYNW", 0, (size_t)s.I, (size_t)s.O, true);
                Eigen::VectorXd flat = synthFlat(layer.getNbOfWeights());
                layer.setWeights(flat);
                layer.resetWeightsDerivatives();
                Eigen::MatrixXd input = makeInput(s.T, s.I);
                Eigen::MatrixXd realOut(s.T, s.O);
                layer.feedForward(input, realOut, false);
                Eigen::MatrixXd deltas = makeDeltas(s.T, s.O);
                layer.feedBackward(input, realOut, deltas, 1, false);
                Eigen::MatrixXd realDerivs = layer.getWeightsDerivatives();

                Eigen::MatrixXd iw, fw, pp, bs, g, o, cs, ci;
                unpackLstmWeights(flat, s.I, s.O, iw, fw, pp, bs);
                lstmForwardLoopCache(input, iw, fw, pp, bs, s.O, true, true, true, g, o, cs, ci);
                LstmDerivs d; d.init(s.I, s.O);
                lstmBackwardLoop(input, o, deltas, iw, fw, pp, s.I, s.O, true, true, true, g, cs, ci, 1, d);
                LstmSubNet single;   // reuse flatDerivs assembly for one layer
                single.derivs = {d};
                Eigen::MatrixXd reimplDerivs = single.flatDerivs();
                derivGap(realDerivs, reimplDerivs, maxUlp, maxAbs);
                if (s.I == 3) { Matrix2BinaryFile(out + "bwd_lstm_derivs.bin", reimplDerivs); ++dumps; }
            }
            std::cout << "NN_TOL site=lstm_backward max_ulp=" << maxUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbs << "\n";
        }

        // ---- Site lstm_backward_variants: Task 4 per-variant goldens ----------
        // Synthetic I=2,O=2,T=5 (odd length). Dumps lstm_bwd_deltasprev_<v>.bin
        // (T x I) + lstm_bwd_derivs_<v>.bin (Nx2) for peep_all / peep_none / reverse
        // / subsample, plus lstm_bwd_signal_derivs.bin (the 1-col signal_width case,
        // dead weight rows accumulate exactly 0 -- spec S11.5). The reimpl (dumped)
        // is checked bit-for-bit (max_ulp=0) against the REAL LSTMLayer::feedBackward
        // / feedBackwardReverse (LSTMLayer.cpp:518-732).
        {
            long maxUlp = 0; double maxAbs = 0.0;
            const int I = 2, O = 2, T = 5;

            // deltasPreviousLayer ULP gap over the full T x I matrix.
            auto dplGap = [&](const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
                long rows = std::min(a.rows(), b.rows()), cols = std::min(a.cols(), b.cols());
                for (long r = 0; r < rows; ++r)
                    for (long c = 0; c < cols; ++c) {
                        double x = a(r, c), y = b(r, c);
                        double g = std::fabs(x - y);
                        if (g > maxAbs) maxAbs = g;
                        uint64_t xb, yb; std::memcpy(&xb, &x, 8); std::memcpy(&yb, &y, 8);
                        long u = (xb > yb) ? (long)(xb - yb) : (long)(yb - xb);
                        if (u > maxUlp) maxUlp = u;
                    }
            };

            struct V { const char* tag; bool cp, gp, grp; bool reverse; long ratio; int inCols; };
            V variants[] = {
                {"peep_all",  true,  true,  true,  false, 1, I},
                {"peep_none", false, false, false, false, 1, I},
                {"reverse",   true,  true,  true,  true,  1, I},
                {"subsample", true,  true,  true,  false, 2, I},
                {"signal",    true,  true,  true,  false, 1, 1},   // width tolerance, 1-col input
            };
            for (const V& v : variants) {
                conf.set_val<bool>("SYNW_IsCellsPeepholesActive", v.cp);
                conf.set_val<bool>("SYNW_IsGatesPeepholesActive", v.gp);
                conf.set_val<bool>("SYNW_IsGatesRecurrentPeepholesActive", v.grp);
                LSTMLayer layer(conf, "SYNW", 0, (size_t)I, (size_t)O, true);
                Eigen::VectorXd flat = synthFlat(layer.getNbOfWeights());
                layer.setWeights(flat);
                layer.resetWeightsDerivatives();
                Eigen::MatrixXd input = makeInput(T, v.inCols);
                Eigen::MatrixXd realOut(T, O);
                Eigen::MatrixXd realDpl;
                if (v.reverse) {
                    layer.feedForwardReverse(input, realOut, false);
                    Eigen::MatrixXd deltas = makeDeltas(T, O);
                    realDpl = layer.feedBackwardReverse(input, realOut, deltas, (size_t)v.ratio, false);
                } else {
                    layer.feedForward(input, realOut, false);
                    Eigen::MatrixXd deltas = makeDeltas(T, O);
                    realDpl = layer.feedBackward(input, realOut, deltas, (size_t)v.ratio, false);
                }
                Eigen::MatrixXd realDerivs = layer.getWeightsDerivatives();

                Eigen::MatrixXd iw, fw, pp, bs, g, o, cs, ci;
                unpackLstmWeights(flat, I, O, iw, fw, pp, bs);
                Eigen::MatrixXd deltas = makeDeltas(T, O);
                LstmDerivs d; d.init(I, O);
                Eigen::MatrixXd reimplDpl;
                if (v.reverse) {
                    // feedForwardReverse: forward on the reversed input, caches stay reversed.
                    Eigen::MatrixXd inRev = input.colwise().reverse(), oRev;
                    lstmForwardLoopCache(inRev, iw, fw, pp, bs, O, v.cp, v.gp, v.grp, g, oRev, cs, ci);
                    reimplDpl = lstmBackwardReverseLoop(input, oRev.colwise().reverse(), deltas,
                                                        iw, fw, pp, I, O, v.cp, v.gp, v.grp,
                                                        g, cs, ci, v.ratio, d);
                } else {
                    lstmForwardLoopCache(input, iw, fw, pp, bs, O, v.cp, v.gp, v.grp, g, o, cs, ci);
                    reimplDpl = lstmBackwardLoop(input, o, deltas, iw, fw, pp, I, O,
                                                 v.cp, v.gp, v.grp, g, cs, ci, v.ratio, d);
                }
                LstmSubNet single; single.derivs = {d};
                Eigen::MatrixXd reimplDerivs = single.flatDerivs();
                derivGap(realDerivs, reimplDerivs, maxUlp, maxAbs);
                dplGap(realDpl, reimplDpl);

                std::string t = v.tag;
                if (std::string(v.tag) == "signal") {
                    Matrix2BinaryFile(out + "lstm_bwd_signal_derivs.bin", reimplDerivs); ++dumps;
                } else {
                    Matrix2BinaryFile(out + "lstm_bwd_deltasprev_" + t + ".bin", reimplDpl); ++dumps;
                    Matrix2BinaryFile(out + "lstm_bwd_derivs_" + t + ".bin", reimplDerivs); ++dumps;
                }
            }
            // Restore the all-on flags for any later stages reusing conf.
            conf.set_val<bool>("SYNW_IsCellsPeepholesActive", true);
            conf.set_val<bool>("SYNW_IsGatesPeepholesActive", true);
            conf.set_val<bool>("SYNW_IsGatesRecurrentPeepholesActive", true);
            std::cout << "NN_TOL site=lstm_backward_variants max_ulp=" << maxUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbs << "\n";
        }

        // ---- Site dense_backward: standalone NeuronLayer::feedBackward --------
        // Grid: {I=4,O=3 hidden; I=4,O=3 last; I=4,O=1 last(logistic)}. The last-layer
        // case proves NO activation deriv on the output (fusion in CostLaw); the hidden
        // case exercises the asinh deriv on the layer input. Task 3 also dumps
        // deltas_out (T x I) per variant -- the double-count trap fixture (spec S11.3):
        // deltas_out for the last-layer sites is EXACTLY deltas*W^T with no activation
        // Jacobian folded in, which the Rust golden test asserts by hand.
        {
            long maxUlp = 0; double maxAbs = 0.0;
            long maxUlpDeltasOut = 0; double maxAbsDeltasOut = 0.0;
            struct S { int I, O, T; bool last; const char* tag; } grid[] = {
                {4, 3, 6, false, "hidden"}, {4, 3, 6, true, "last"}, {4, 1, 6, true, "logistic"}};
            for (const S& s : grid) {
                NeuronLayer layer(conf, "SYNW", 0, (size_t)s.I, (size_t)s.O, true);
                Eigen::VectorXd flat = synthFlat(layer.getNbOfWeights());
                layer.setWeights(flat);
                layer.resetWeightsDerivatives();
                Eigen::MatrixXd input = makeInput(s.T, s.I);
                Eigen::MatrixXd realOut(s.T, s.O);
                layer.feedForward(input, realOut, s.last);
                Eigen::MatrixXd deltas = makeDeltas(s.T, s.O);
                Eigen::MatrixXd realDeltasOut = layer.feedBackward(input, realOut, deltas, 1, s.last);
                Eigen::MatrixXd realDerivs = layer.getWeightsDerivatives();

                Eigen::MatrixXd w(s.I, s.O), b(1, s.O);
                for (int c = 0; c < s.O; ++c)
                    for (int r = 0; r < s.I; ++r) w(r, c) = flat((long)c * s.I + r);
                for (int c = 0; c < s.O; ++c) b(0, c) = flat((long)s.O * s.I + c);
                DenseDerivs d; d.init(s.I, s.O);
                Eigen::MatrixXd reimplDeltasOut = denseBackwardLoop(input, deltas, w, s.I, s.O, 1, s.last, d);
                DenseSubNet single; single.derivs = {d};
                Eigen::MatrixXd reimplDerivs = single.flatDerivs();
                derivGap(realDerivs, reimplDerivs, maxUlp, maxAbs);

                // ULP/abs gap over the FULL deltas_out matrix (not just col 0).
                long rows = std::min(realDeltasOut.rows(), reimplDeltasOut.rows());
                long cols = std::min(realDeltasOut.cols(), reimplDeltasOut.cols());
                for (long r = 0; r < rows; ++r) {
                    for (long c = 0; c < cols; ++c) {
                        double a = realDeltasOut(r, c), b2 = reimplDeltasOut(r, c);
                        double absGap = std::fabs(a - b2);
                        if (absGap > maxAbsDeltasOut) maxAbsDeltasOut = absGap;
                        uint64_t ab, bb;
                        std::memcpy(&ab, &a, sizeof(double));
                        std::memcpy(&bb, &b2, sizeof(double));
                        long ulp = (ab > bb) ? (long)(ab - bb) : (long)(bb - ab);
                        if (ulp > maxUlpDeltasOut) maxUlpDeltasOut = ulp;
                    }
                }

                Matrix2BinaryFile(out + "bwd_dense_" + s.tag + "_derivs.bin", reimplDerivs);
                Matrix2BinaryFile(out + "bwd_dense_" + s.tag + "_deltasout.bin", reimplDeltasOut);
                ++dumps;
                ++dumps;
            }
            std::cout << "NN_TOL site=dense_backward max_ulp=" << maxUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbs << "\n";
            std::cout << "NN_TOL site=dense_backward_deltasout max_ulp=" << maxUlpDeltasOut
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbsDeltasOut << "\n";
        }

        // ---- Sites net_lstm_backward_{fwd,rev}: NeuralNetwork<LSTMLayer> ------
        // 2-layer LSTM net [3,4,2] sub [2,1] on T=11 (SubSample interaction). Real net
        // feedForward(+Reverse) THEN feedBackward(+Reverse) on seed deltas sized to the
        // decimated output rows (11/2 = 5). Mirror with LstmSubNet forwardCapture +
        // backward. The reverse site pins the time-reversed cache consumption.
        {
            const std::vector<std::vector<double>::size_type> neuronNb = {3, 4, 2};
            const std::vector<std::vector<double>::size_type> subSampling = {2, 1};
            const std::vector<long> nnL = {3, 4, 2}, ssL = {2, 1};
            Eigen::MatrixXd input = makeInput(11, 3);
            const long outRows = 11 / 2;
            Eigen::MatrixXd seedDeltas = makeDeltas((int)outRows, 2);
            for (bool reverse : {false, true}) {
                NeuralNetwork<LSTMLayer> net(conf, "SYNW", neuronNb, subSampling, true);
                Eigen::VectorXd flat = synthFlat(net.getNbOfWeights());
                net.setWeights(flat);
                net.resetWeightsDerivatives();
                Eigen::MatrixXd realOut(outRows, 2);
                Eigen::MatrixXd realDeltasOut;
                if (reverse) { net.feedForwardReverse(input, realOut); realDeltasOut = net.feedBackwardReverse(input, realOut, seedDeltas); }
                else { net.feedForward(input, realOut); realDeltasOut = net.feedBackward(input, realOut, seedDeltas); }
                Eigen::MatrixXd realDerivs = net.getWeightsDerivatives();

                LstmSubNet sub;
                sub.build(flat, nnL, ssL, reverse);
                Eigen::MatrixXd fout = sub.forwardCapture(input);
                Eigen::MatrixXd reimplDeltasOut = sub.backward(input, fout, seedDeltas);
                Eigen::MatrixXd reimplDerivs = sub.flatDerivs();
                long maxUlp = 0; double maxAbs = 0.0;
                derivGap(realDerivs, reimplDerivs, maxUlp, maxAbs);
                std::cout << "NN_TOL site=net_lstm_backward" << (reverse ? "_rev" : "_fwd")
                          << " max_ulp=" << maxUlp << " max_abs=" << std::scientific
                          << std::setprecision(3) << maxAbs << "\n";
                // Phase 3 Task 5: also compare + dump the returned deltas_out (the layer-0
                // InvSubSample'd deltasPreviousLayer, NeuralNetwork.hpp:271-272/299). Row
                // count is floor(T/R)*R (11/2*2 = 10), NOT T -- the trailing SubSample row
                // is dropped by the forward floor and never restored by InvSubSample. The
                // reimpl IS the golden; deltas_out is `_InputWeights*testM` (LSTMLayer.cpp
                // :707), a GEMM with k=4*O (16 for this [3,4,2] layer-0, O=4) -- NOT k=O as
                // a prior comment here mis-stated (fix-wave 1, review finding 3) -- so the
                // REAL Eigen path diverges in the last ULP (recorded as a SEPARATE
                // calibration site, NOT gated to 0 -- unlike the k=1 outer-product derivs
                // above).
                long maxUlpDeltasOut = 0; double maxAbsDeltasOut = 0.0;
                deltasGap(realDeltasOut, reimplDeltasOut, maxUlpDeltasOut, maxAbsDeltasOut);
                std::cout << "NN_TOL site=net_lstm_backward" << (reverse ? "_rev" : "_fwd")
                          << "_deltasout max_ulp=" << maxUlpDeltasOut << " max_abs="
                          << std::scientific << std::setprecision(3) << maxAbsDeltasOut << "\n";
                Matrix2BinaryFile(out + (reverse ? "bwd_net_lstm_rev_derivs.bin" : "bwd_net_lstm_fwd_derivs.bin"), reimplDerivs);
                Matrix2BinaryFile(out + (reverse ? "bwd_net_lstm_rev_deltasout.bin" : "bwd_net_lstm_fwd_deltasout.bin"), reimplDeltasOut);
                ++dumps;
                ++dumps;
            }
        }

        // ---- Site net_dense_backward: NeuralNetwork<NeuronLayer> -------------
        // 2-layer dense net [4,3,2] sub [1,2] on T=11 (SubSample(2) between layers ->
        // 5 rows). Real net feedForward THEN feedBackward on seed deltas (5 x 2);
        // mirror with DenseSubNet.
        {
            const std::vector<std::vector<double>::size_type> neuronNb = {4, 3, 2};
            const std::vector<std::vector<double>::size_type> subSampling = {1, 2};
            const std::vector<long> nnD = {4, 3, 2}, ssD = {1, 2};
            Eigen::MatrixXd input = makeInput(11, 4);
            const long outRows = 11 / 2;   // decimated by the layer-1 SubSample(2)
            Eigen::MatrixXd seedDeltas = makeDeltas((int)outRows, 2);
            NeuralNetwork<NeuronLayer> net(conf, "SYNW", neuronNb, subSampling, true);
            Eigen::VectorXd flat = synthFlat(net.getNbOfWeights());
            net.setWeights(flat);
            net.resetWeightsDerivatives();
            Eigen::MatrixXd realOut(outRows, 2);
            net.feedForward(input, realOut);
            Eigen::MatrixXd realDeltasOut = net.feedBackward(input, realOut, seedDeltas);
            Eigen::MatrixXd realDerivs = net.getWeightsDerivatives();

            DenseSubNet sub;
            sub.build(flat, nnD, ssD);
            Eigen::MatrixXd fout = sub.forwardCapture(input);
            Eigen::MatrixXd reimplDeltasOut = sub.backward(input, fout, seedDeltas);
            Eigen::MatrixXd reimplDerivs = sub.flatDerivs();
            long maxUlp = 0; double maxAbs = 0.0;
            derivGap(realDerivs, reimplDerivs, maxUlp, maxAbs);
            std::cout << "NN_TOL site=net_dense_backward max_ulp=" << maxUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbs << "\n";
            // Phase 3 Task 5: also compare + dump the returned deltas_out. Layer-1
            // SubSample(2) decimates to 5 rows, InvSubSample restores to 10; layer 0
            // (no subsample) back-projects at deltas.rows()=10 while its count is
            // input.rows()=11 (NeuronLayer.cpp:199) -- deltas_out is 10 x 4. The
            // deltas*W^T GEMM (k=O) may diverge from the real Eigen path in the last
            // ULP -- a SEPARATE calibration site (the reimpl dump is the golden).
            long maxUlpDeltasOut = 0; double maxAbsDeltasOut = 0.0;
            deltasGap(realDeltasOut, reimplDeltasOut, maxUlpDeltasOut, maxAbsDeltasOut);
            std::cout << "NN_TOL site=net_dense_backward_deltasout max_ulp=" << maxUlpDeltasOut
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbsDeltasOut << "\n";
            Matrix2BinaryFile(out + "bwd_net_dense_derivs.bin", reimplDerivs);
            Matrix2BinaryFile(out + "bwd_net_dense_deltasout.bin", reimplDeltasOut);
            ++dumps;
            ++dumps;
        }

        // ---- Sites net_single_layer_backward_{subsample,plain}: NeuralNetwork
        //      <LSTMLayer> with neuronNb.size()==2 (a SINGLE layer, NeuralNetwork.hpp
        //      :255-263) ------------------------------------------------------------
        // Fix-wave 1 (review finding 1): the two net_lstm_backward_{fwd,rev} +
        // net_dense_backward sites above are BOTH neuronNb.size()==3 (2 hidden layers),
        // so they only ever exercise the multi-layer loop (:264-296) -- never the
        // single-layer branch (:255-263), which is structurally distinct: it reads
        // `deltas`/`deltas.rows()` (the SEED) directly, not the running `deltasOut`
        // that feeds the multi-layer jj==0 arm on every iteration but the last. LSTM
        // [2,2] sub [2] (subsample arm, the brief's dropped fixture) and LSTM [2,2]
        // sub [1] (plain arm, legacy :262, "if cheaply constructible" per the finding)
        // at T=7 (odd length, also exercising the SubSample floor quirk on this arm:
        // floor(7/2)=3 decimated rows). Real net feedForward THEN feedBackward on seed
        // deltas sized to the (decimated, for the subsample variant) output rows;
        // mirror with LstmSubNet (L==1 already routes through netBackwardLoop's L==2
        // branch, same code path exercised end-to-end since Task 1).
        {
            const std::vector<std::vector<double>::size_type> neuronNb = {2, 2};
            const std::vector<long> nnL = {2, 2};
            struct V { std::vector<std::vector<double>::size_type> ss; std::vector<long> ssL; long outRows; const char* tag; } variants[] = {
                {{2}, {2}, 7 / 2, "subsample"},
                {{1}, {1}, 7, "plain"},
            };
            for (const V& v : variants) {
                Eigen::MatrixXd input = makeInput(7, 2);
                Eigen::MatrixXd seedDeltas = makeDeltas((int)v.outRows, 2);
                NeuralNetwork<LSTMLayer> net(conf, "SYNW", neuronNb, v.ss, true);
                Eigen::VectorXd flat = synthFlat(net.getNbOfWeights());
                net.setWeights(flat);
                net.resetWeightsDerivatives();
                Eigen::MatrixXd realOut(v.outRows, 2);
                net.feedForward(input, realOut);
                Eigen::MatrixXd realDeltasOut = net.feedBackward(input, realOut, seedDeltas);
                Eigen::MatrixXd realDerivs = net.getWeightsDerivatives();

                LstmSubNet sub;
                sub.build(flat, nnL, v.ssL, false);
                Eigen::MatrixXd fout = sub.forwardCapture(input);
                Eigen::MatrixXd reimplDeltasOut = sub.backward(input, fout, seedDeltas);
                Eigen::MatrixXd reimplDerivs = sub.flatDerivs();

                long maxUlp = 0; double maxAbs = 0.0;
                derivGap(realDerivs, reimplDerivs, maxUlp, maxAbs);
                std::cout << "NN_TOL site=net_single_layer_backward_" << v.tag
                          << " max_ulp=" << maxUlp << " max_abs=" << std::scientific
                          << std::setprecision(3) << maxAbs << "\n";

                long maxUlpDeltasOut = 0; double maxAbsDeltasOut = 0.0;
                deltasGap(realDeltasOut, reimplDeltasOut, maxUlpDeltasOut, maxAbsDeltasOut);
                std::cout << "NN_TOL site=net_single_layer_backward_" << v.tag
                          << "_deltasout max_ulp=" << maxUlpDeltasOut << " max_abs="
                          << std::scientific << std::setprecision(3) << maxAbsDeltasOut << "\n";

                Matrix2BinaryFile(out + "bwd_net_single_" + v.tag + "_derivs.bin", reimplDerivs);
                Matrix2BinaryFile(out + "bwd_net_single_" + v.tag + "_deltasout.bin", reimplDeltasOut);
                ++dumps;
                ++dumps;
            }
        }

        // ---- Site blstm_feedbackward: synthetic BLSTMNeuralNetwork ------------
        // Small BINARY-multiclass net (LSTM [3,4,2] sub [2,1], output [4,5,2] sub
        // [1,1], T=12 -> length 6), backprop active, a multiclass (softmax+CE) target.
        // The seed deltas come from the REAL CostLaw::computeDeltas (bit-portable for
        // the polynomial law the golden config uses). feedForwardBackward with targets
        // drives the WHOLE real backward; harvest getWeightsDerivatives. Mirror with
        // BlstmBack (forward capture -> CostLaw seed -> backward). This is the double-
        // count-safe fusion site: NeuronLayer applies no output Jacobian.
        {
            const std::vector<long> lstmNN = {3, 4, 2}, lstmSS = {2, 1};
            const std::vector<long> outNN = {4, 5, 2}, outSS = {1, 1};
            ConfigFile bconf(nnConfigPath, '_');
            bconf._Params.erase("BLSTM_weightsFile");
            bconf.set_val<std::string>("SYNB_LSTMNeuronNb", "3,4,2");
            bconf.set_val<std::string>("SYNB_LSTMSubSampling", "2,1");
            bconf.set_val<std::string>("SYNB_OutputNeuronNb", "4,5,2");
            bconf.set_val<std::string>("SYNB_OutputSubSampling", "1,1");
            bconf.set_val<short>("SYNB_InputNormalizationType", (short)0);
            bconf.set_val<bool>("SYNB_TwoSweeps", false);
            bconf.set_val<bool>("SYNB_BackPropagationActivated", true);
            bconf.set_val<bool>("SYNB_BackPropOutputNetworkOnly", false);
            bconf.set_val<int>("SYNB_TargetEnforcementStep", 0);
            BLSTMNeuralNetwork<LSTMLayer> nn(bconf, "SYNB", true);
            Eigen::VectorXd flat = synthFlat(nn.getNbOfWeights());
            nn.setWeights(flat);
            nn.resetWeightsDerivatives();

            Eigen::MatrixXd input = makeInput(12, 3);
            const long outLen = 12 / 2;   // fwd LSTM ratio 2*1
            Eigen::MatrixXd realOut(outLen, 2);
            // Multiclass one-hot-ish targets: class 0 or 1 per row (>0.5 target).
            Eigen::MatrixXd targets = Eigen::MatrixXd::Zero(outLen, 2);
            for (long r = 0; r < outLen; ++r) targets(r, (r % 2)) = 1.0;
            nn.setProcessingType(false, false);
            nn.feedForwardBackward(input, 0, 0, realOut, targets);
            Eigen::MatrixXd realDerivs = nn.getWeightsDerivatives();

            // Reimpl: forward capture, then seed deltas via the REAL CostLaw on the
            // reimpl output (bit-portable polynomial/softmax-CE), then backward.
            BlstmBack bb;
            bb.build(flat, lstmNN, lstmSS, outNN, outSS);
            Eigen::MatrixXd oF, oB, hcat, reimplOut;
            bb.forward(input, oF, oB, reimplOut, hcat);
            CostLaw cost(bconf, "SYNB");
            Eigen::MatrixXd seed = Eigen::MatrixXd::Zero(targets.rows(), targets.cols());
            cost.computeDeltas(reimplOut, targets, seed);
            bb.backward(input, oF, oB, hcat, reimplOut, seed);
            Eigen::MatrixXd reimplDerivs = bb.flatDerivs();
            long maxUlp = 0; double maxAbs = 0.0;
            derivGap(realDerivs, reimplDerivs, maxUlp, maxAbs);
            std::cout << "NN_TOL site=blstm_feedbackward max_ulp=" << maxUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbs << "\n";
            Matrix2BinaryFile(out + "bwd_blstm_derivs.bin", reimplDerivs);
            ++dumps;
        }

        // ---- Site blstm_real_backward: the REAL 33,671-weight net ------------
        // Reuse e2e_input.bin (the assembled real-config inputSeq, 201 x 11) + the real
        // net (1_worker_1.config + NNweights_config1.bin, InputNormalizationType -1).
        // Backprop must be turned ON (the real config has it false + no RpropInit key) via
        // set_val. Multiclass output is O=1 (binary VAD via the scalar Logistic head), so
        // the SCALAR CostLaw path (computeUnitaryDeltas with the folded logistic deriv)
        // seeds the deltas. EXPECTED NONZERO (k=23/24/48 GEMM divergence) -> the reimpl
        // derivs ARE the golden; the delta is recorded as Phase 4 calibration.
        {
            const std::vector<long> lstmNN = {23, 24, 24}, lstmSS = {4, 1};
            const std::vector<long> outNN = {48, 12, 1}, outSS = {1, 1};
            Eigen::MatrixXd inputSeq = readBinMatrix(out + "e2e_input.bin");
            ConfigFile nconf(nnConfigPath, '_');
            nconf._Params.erase("BLSTM_weightsFile");
            nconf.set_val<bool>("BLSTM_BackPropagationActivated", true);
            nconf.set_val<bool>("BLSTM_BackPropOutputNetworkOnly", false);
            BLSTMNeuralNetwork<LSTMLayer> nn(nconf, "BLSTM", true);
            Eigen::VectorXd flat = BinaryFile2Vector(nnWeightsPath);
            nn.setWeights(flat);
            nn.resetWeightsDerivatives();

            const long outLen = inputSeq.rows() / 4;   // ssr 4
            Eigen::MatrixXd realOut(outLen, 1);
            // Scalar VAD targets: deterministic 0/1 per decimated row.
            Eigen::MatrixXd targets(outLen, 1);
            for (long r = 0; r < outLen; ++r) targets(r, 0) = ((r % 3) == 0) ? 1.0 : 0.0;
            Eigen::MatrixXd realInput = inputSeq;   // MUTATED by type -1 self-norm
            nn.setProcessingType(false, false);
            nn.feedForwardBackward(realInput, 0, 0, realOut, targets);
            Eigen::MatrixXd realDerivs = nn.getWeightsDerivatives();

            // Reimpl: type -1 self-norm in place, forward capture, CostLaw seed, backward.
            Eigen::MatrixXd reimplInput = inputSeq;
            {
                const int R = (int)reimplInput.rows(), C = (int)reimplInput.cols();
                std::vector<double> mean(C, 0.0);
                for (int c = 0; c < C; ++c) { double a = 0.0; for (int r = 0; r < R; ++r) a += reimplInput(r, c); mean[c] = a / (double)R; }
                for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) reimplInput(r, c) -= mean[c];
                std::vector<double> stdv(C, 0.0);
                for (int c = 0; c < C; ++c) { double a = 0.0; for (int r = 0; r < R; ++r) a += reimplInput(r, c) * reimplInput(r, c); stdv[c] = std::sqrt((a + 1e-32) / (double)R); }
                for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) reimplInput(r, c) = Maxmin2::fn(reimplInput(r, c) / stdv[c]);
            }
            BlstmBack bb;
            bb.build(flat, lstmNN, lstmSS, outNN, outSS);
            Eigen::MatrixXd oF, oB, hcat, reimplOut;
            bb.forward(reimplInput, oF, oB, reimplOut, hcat);
            CostLaw cost(nconf, "BLSTM");
            Eigen::MatrixXd seed = Eigen::MatrixXd::Zero(targets.rows(), targets.cols());
            cost.computeDeltas(reimplOut, targets, seed);
            bb.backward(reimplInput, oF, oB, hcat, reimplOut, seed);
            Eigen::MatrixXd reimplDerivs = bb.flatDerivs();
            long maxUlp = 0; double maxAbs = 0.0;
            derivGap(realDerivs, reimplDerivs, maxUlp, maxAbs);
            std::cout << "NN_TOL site=blstm_real_backward max_ulp=" << maxUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbs << "\n";
            std::cout << "NN_REAL ok backward_nb_derivs=" << reimplDerivs.rows() << "\n";
            Matrix2BinaryFile(out + "bwd_blstm_real_derivs.bin", reimplDerivs);
            ++dumps;
        }

        // ---- NN_PROBE backward: pure-arithmetic outer-product (0 ULP at synthetic k)
        // The weight-deriv outer product input.col(row)*deltas.row(row) (small k) and
        // the deltas*W^T back-projection (k=O) are ascending-loop products; at synthetic
        // shapes Eigen and matSeq agree bit-for-bit. Probe both against nnProbe.
        {
            Eigen::MatrixXd inCol = makeInput(1, 4);        // 1 x 4 (input.col row)
            Eigen::MatrixXd dRow = makeDeltas(1, 3);        // 1 x 3 (deltas.row)
            nnProbe("bwd_outer_product", inCol.transpose(), dRow);   // (4x1)*(1x3)
            Eigen::MatrixXd deltas = makeDeltas(6, 3);      // 6 x 3
            Eigen::MatrixXd wT = makeInput(3, 4);           // 3 x 4 (W^T, O x I)
            nnProbe("bwd_backproj", deltas, wT);            // (6x3)*(3x4)
        }
    }

    // =====================================================================
    // --- Phase 3 Task 2: CostLaw backward REAL-probe goldens --------------
    // Task 2 VALIDATES the already-ported cost.rs backward chain (deriv() per
    // law, computeUnitaryDeltas, computeDeltas) against the REAL compiled
    // CostLaw (CostLaw.cpp:278-445, CostLaw.h:6-213) -- not a transcription: the
    // REAL class is called directly as the golden (S2 tier 2). Three CostLaw
    // instances built from the real 1_worker_1.config (BLSTM_ prefix) via the
    // same ConfigFile + set_val override pattern as the Task 1 stage above:
    //   - polyConf:  CostLawSpeech/NoSpeech = "square" (pure arithmetic; the
    //     below/above-thresh switch + the output*(1-output) logistic fold are
    //     exercised bit-exactly on every platform).
    //   - logConf:   the config's OWN default ("log"/"log") -- libm-dependent
    //     (ln), canary-gated on the Rust side.
    //   - sqrtConf:  CostLawSpeech/NoSpeech = "sqrt" -- libm-dependent (sqrt),
    //     canary-gated on the Rust side.
    // `conf.set_val<T>(key, "")` would ABORT (empty-string parse), so no key is
    // ever erased to "unset" it; only real values are written, matching the
    // brief's constraint note.
    {
        auto buildCostConf = [&](const char* speechLaw, const char* noSpeechLaw) {
            ConfigFile conf(nnConfigPath, '_');
            conf._Params.erase("BLSTM_weightsFile");
            conf.set_val<std::string>("BLSTM_CostLawSpeech", std::string(speechLaw));
            conf.set_val<std::string>("BLSTM_CostLawNoSpeech", std::string(noSpeechLaw));
            return conf;
        };

        // ---- Scalar VAD sweep: cost_deriv_scalar_{speech,other}.bin ----------
        // Deterministic grid output = k/64, k in [0,64] (65 points), for
        // target=1.0 (speech) and target=0.0 (other). Dumped as Nx2 [output|delta]
        // via the REAL computeUnitaryDeltas, for the square (strict-bits) law.
        // The polynomial dump also exercises the BackPropWER scaling branch by
        // being re-dumped under a WER-enabled config variant (see below).
        auto dumpScalarSweep = [&](CostLaw& law, const std::string& fname) {
            const int N = 65;
            Eigen::MatrixXd mat(N, 2);
            for (int k = 0; k <= 64; ++k) {
                double output = (double)k / 64.0;
                mat(k, 0) = output;
                mat(k, 1) = law.computeUnitaryDeltas(output, 1.0);
            }
            Matrix2BinaryFile(out + fname + "_speech.bin", mat);
            for (int k = 0; k <= 64; ++k) {
                double output = (double)k / 64.0;
                mat(k, 0) = output;
                mat(k, 1) = law.computeUnitaryDeltas(output, 0.0);
            }
            Matrix2BinaryFile(out + fname + "_other.bin", mat);
            ++dumps;
            ++dumps;
        };

        {
            ConfigFile polyConf = buildCostConf("square", "square");
            CostLaw polyLaw(polyConf, "BLSTM");
            dumpScalarSweep(polyLaw, "cost_deriv_scalar_poly");
        }
        {
            // Config's own default (log/log) -- CANARY-GATED on the Rust side.
            ConfigFile logConf(nnConfigPath, '_');
            logConf._Params.erase("BLSTM_weightsFile");
            CostLaw logLaw(logConf, "BLSTM");
            dumpScalarSweep(logLaw, "cost_deriv_scalar_log");
        }
        {
            ConfigFile sqrtConf = buildCostConf("sqrt", "sqrt");
            CostLaw sqrtLaw(sqrtConf, "BLSTM");
            dumpScalarSweep(sqrtLaw, "cost_deriv_scalar_sqrt");
        }
        // The brief's literal alias names (cost_deriv_scalar_{speech,other}.bin) are
        // now produced as a byte copy of the poly dump by the Python extractor
        // (scripts/extract_phase3_fixtures.py), NOT re-dumped here -- a second
        // independent C++ dump block ~20 lines from the first is a manual
        // lock-step trap (fix-wave 1 review finding 2): the two blocks could
        // silently diverge if only one were edited later.

        // ---- Mid-thresh above-branch coverage: cubic/sqrt deriv, non-degenerate
        // logistic fold (fix-wave 1 review finding 1) ------------------------
        // The real config's CostLawThreshSpeech=1 / CostLawThreshNoSpeech=0
        // (tests/reference_data/phase0/1_worker_1.config:64-65) means the
        // above-thresh branch (CostLaw.cpp:297,324 speech; :307,326 no-speech)
        // fires ONLY at output=1.0 (speech) / output=0.0 (no-speech) in every
        // sweep above -- exactly where the logistic chain-rule fold
        // `delta*output*(1-output)` (CostLaw.cpp:344) is IDENTICALLY ZERO,
        // which multiplies away any bug in AboveCubic/AboveSqrt::deriv. This
        // block overrides BOTH thresholds to 0.5 so the above-thresh branch
        // fires at output values AWAY from 0/1 (nonzero logistic fold), for
        // the cubic law (CostLaw.h:84-117 AboveThreshCubicLaw, routed via the
        // "cubic" name at CostLaw.cpp:300,328) and the sqrt law (CostLaw.h:
        // 182-213 AboveThreshSqrtLaw, routed via CostLaw.cpp:298,326).
        //
        // Above-branch condition (CostLaw.cpp:282,297 speech; :308,324
        // no-speech): speech is "above" when output >= SwitchingThreshSpeech;
        // no-speech is "above" when output <= SwitchingThreshNoSpeech (the
        // ELSE arm of `output > _SwitchingThreshNoSpeech`, after which the
        // law is evaluated on the flipped `1-output`). With both thresholds
        // at 0.5: speech points 0.6/0.75/0.9 and no-speech points
        // 0.4/0.25/0.1 all land in the above-thresh regime.
        //
        // ALSO override CostLawParamSpeech/NoSpeech (the real config's own
        // value is 0, CostLaw.cpp:14-19 "q") away from 0: AboveThreshCubicLaw's
        // A/B (CostLaw.h:88-93, the "cubic" name routes to the square/cubic
        // formula) and AboveThreshSqrtLaw's A/B (CostLaw.h:186-188, A=-q, B=q)
        // are BOTH directly proportional to q -- at q=0 the above-thresh deriv
        // is identically 0.0 regardless of the logistic fold, a second,
        // independent way to land a vacuous golden (caught by inspecting the
        // dumped bytes below before wiring the Rust side, same as the WER
        // vacuity note above).
        auto dumpMidThreshPoints = [&](CostLaw& law, const std::string& fname) {
            const double speechPts[3] = {0.6, 0.75, 0.9};
            const double noSpeechPts[3] = {0.4, 0.25, 0.1};
            Eigen::MatrixXd matSpeech(3, 2);
            for (int k = 0; k < 3; ++k) {
                matSpeech(k, 0) = speechPts[k];
                matSpeech(k, 1) = law.computeUnitaryDeltas(speechPts[k], 1.0);
            }
            Matrix2BinaryFile(out + fname + "_speech.bin", matSpeech);
            Eigen::MatrixXd matNoSpeech(3, 2);
            for (int k = 0; k < 3; ++k) {
                matNoSpeech(k, 0) = noSpeechPts[k];
                matNoSpeech(k, 1) = law.computeUnitaryDeltas(noSpeechPts[k], 0.0);
            }
            Matrix2BinaryFile(out + fname + "_other.bin", matNoSpeech);
            ++dumps;
            ++dumps;
        };
        auto buildMidThreshConf = [&](const char* speechLaw, const char* noSpeechLaw) {
            ConfigFile conf = buildCostConf(speechLaw, noSpeechLaw);
            conf.set_val<double>("BLSTM_CostLawThreshSpeech", 0.5);
            conf.set_val<double>("BLSTM_CostLawThreshNoSpeech", 0.5);
            conf.set_val<double>("BLSTM_CostLawParamSpeech", 0.3);
            conf.set_val<double>("BLSTM_CostLawParamNoSpeech", 0.3);
            return conf;
        };
        {
            ConfigFile cubicMidConf = buildMidThreshConf("cubic", "cubic");
            CostLaw cubicMidLaw(cubicMidConf, "BLSTM");
            dumpMidThreshPoints(cubicMidLaw, "cost_deriv_scalar_midthresh_cubic");
        }
        {
            ConfigFile sqrtMidConf = buildMidThreshConf("sqrt", "sqrt");
            CostLaw sqrtMidLaw(sqrtMidConf, "BLSTM");
            dumpMidThreshPoints(sqrtMidLaw, "cost_deriv_scalar_midthresh_sqrt");
        }

        // ---- Multiclass softmax+CE fusion: cost_deltas_multiclass.bin --------
        // n_frames=4 x n_classes=3, deterministic outputs (softmax-shaped, but the
        // fusion math doesn't require a true softmax row -- any [0,1] values probe
        // the branch), one-hot-ish targets exercising:
        //   row0: normal on-class (target col1 > 0.5) -> output-1 on-class, output off-class.
        //   row1: an ignore-masked row (ALL targets < 0) -> deltas(row1,*) == 0.0 exactly.
        //   row2: on-class at col0.
        //   row3: on-class at col2.
        // The CostLaw for this dump has NO BackPropWER, NO ponderations (the
        // "vanilla" fusion path -- CostLaw.cpp:403-418 else-branch, no-pond arm).
        {
            const int NF = 4, NC = 3;
            Eigen::MatrixXd outputs(NF, NC);
            double ov[4][3] = {
                {0.7, 0.2, 0.1},
                {0.5, 0.3, 0.2},
                {0.6, 0.1, 0.3},
                {0.25, 0.35, 0.4},
            };
            for (int j = 0; j < NF; ++j)
                for (int k = 0; k < NC; ++k) outputs(j, k) = ov[j][k];
            Eigen::MatrixXd targets = Eigen::MatrixXd::Zero(NF, NC);
            targets(0, 1) = 1.0;              // row0: on-class col1
            targets(1, 0) = -1.0; targets(1, 1) = -1.0; targets(1, 2) = -1.0; // row1: fully masked
            targets(2, 0) = 1.0;               // row2: on-class col0
            targets(3, 2) = 1.0;               // row3: on-class col2

            ConfigFile mcConf(nnConfigPath, '_');
            mcConf._Params.erase("BLSTM_weightsFile");
            CostLaw mcLaw(mcConf, "BLSTM");
            Eigen::MatrixXd deltas = Eigen::MatrixXd::Zero(NF, NC);
            mcLaw.computeDeltas(outputs, targets, deltas);
            Matrix2BinaryFile(out + "cost_deltas_multiclass.bin", deltas);
            ++dumps;

            // Ponderation variant: same outputs/targets, classes_ponderations set
            // (non-uniform, exercises CostLaw.cpp:410-417 whole-row scaling by the
            // ON-CLASS ponderation). Dumped alongside the unponderated deltas above
            // so the Rust test can assert the exact scale factor between them.
            ConfigFile pondConf(nnConfigPath, '_');
            pondConf._Params.erase("BLSTM_weightsFile");
            pondConf.set_val<std::string>("BLSTM_classes_ponderations", std::string("2.0,3.0,4.0"));
            CostLaw pondLaw(pondConf, "BLSTM");
            Eigen::MatrixXd pondDeltas = Eigen::MatrixXd::Zero(NF, NC);
            pondLaw.computeDeltas(outputs, targets, pondDeltas);
            Matrix2BinaryFile(out + "cost_deltas_multiclass_pond.bin", pondDeltas);
            ++dumps;
        }

        // ---- BackPropWER multiclass path: cost_deltas_wer.bin -----------------
        // SOFT targets (not exact 0/1): the WER scaling factors are
        // `10*(1-target)` on-class / `10*target` off-class (CostLaw.cpp:379-397),
        // which VANISH at target==1.0/0.0 exactly -- the engine's real soft-target
        // convention (BLSTMNeuralNetwork.cpp:871, `0.1*modifier`) is what makes the
        // scaling non-degenerate, so a golden built on pure one-hot targets would
        // be vacuous here (S11.9 non-vacuity). Row0/1 carry soft on-class targets
        // (0.9, 0.8) with soft off-class floors (0.05/0.1/0.15..); row2 is fully
        // masked (target<0 everywhere) to prove the ignore path composes with WER.
        // WITHOUT ponderations, then WITH (CostLaw.cpp:372-401, per-element
        // pond*10*... arm).
        {
            const int NF = 3, NC = 3;
            Eigen::MatrixXd outputs(NF, NC);
            double ov[3][3] = {
                {0.7, 0.2, 0.1},
                {0.5, 0.3, 0.2},
                {0.25, 0.35, 0.4},
            };
            for (int j = 0; j < NF; ++j)
                for (int k = 0; k < NC; ++k) outputs(j, k) = ov[j][k];
            Eigen::MatrixXd targets = Eigen::MatrixXd::Zero(NF, NC);
            targets(0, 0) = 0.05; targets(0, 1) = 0.9;  targets(0, 2) = 0.05;
            targets(1, 0) = 0.8;  targets(1, 1) = 0.1;  targets(1, 2) = 0.1;
            targets(2, 0) = -1.0; targets(2, 1) = -1.0; targets(2, 2) = -1.0;

            ConfigFile werConf(nnConfigPath, '_');
            werConf._Params.erase("BLSTM_weightsFile");
            werConf.set_val<double>("BLSTM_BackPropWER", 0.0);
            CostLaw werLaw(werConf, "BLSTM");
            Eigen::MatrixXd werDeltas = Eigen::MatrixXd::Zero(NF, NC);
            werLaw.computeDeltas(outputs, targets, werDeltas);
            Matrix2BinaryFile(out + "cost_deltas_wer.bin", werDeltas);
            ++dumps;

            ConfigFile werPondConf(nnConfigPath, '_');
            werPondConf._Params.erase("BLSTM_weightsFile");
            werPondConf.set_val<double>("BLSTM_BackPropWER", 0.0);
            werPondConf.set_val<std::string>("BLSTM_classes_ponderations", std::string("2.0,3.0,4.0"));
            CostLaw werPondLaw(werPondConf, "BLSTM");
            Eigen::MatrixXd werPondDeltas = Eigen::MatrixXd::Zero(NF, NC);
            werPondLaw.computeDeltas(outputs, targets, werPondDeltas);
            Matrix2BinaryFile(out + "cost_deltas_wer_pond.bin", werPondDeltas);
            ++dumps;
        }
    }

    // --- Phase 2b Task 1: faithful iof::fmtr self-test ------------------------
    // The Phase 2b segmenter linkage makes the REAL Segmentation::toFile_VRCTS a
    // byte golden, which routes through iof::fmtr (%f.Ns -> std::fixed +
    // setprecision(N); %s -> default insertion; tail-flush + raw passthrough).
    // Each case below runs the shim AND an independent std::ostringstream ground
    // truth built with fixed/setprecision directly, and prints a parseable
    // FMTR_CHECK line the extractor asserts (ok=0 -> SystemExit).
    {
        // Case fmtr_vrcts: the exact %s + %f.4s shape from toFile_VRCTS.
        {
            std::ostringstream got;
            got << iof::fmtr("<x a=\"%s\" b=\"%f.4s\"/>") << 3 << 1.25;
            std::ostringstream want;
            want << "<x a=\"" << 3 << "\" b=\""
                 << std::fixed << std::setprecision(4) << 1.25 << "\"/>";
            std::cout << "FMTR_CHECK case=fmtr_vrcts ok="
                      << (got.str() == want.str() ? 1 : 0) << "\n";
        }
        // Case fmtr_f2: a bare %f.2s (the sigdur/spdur/dur precision).
        {
            std::ostringstream got;
            got << iof::fmtr("%f.2s") << 120.0;
            std::ostringstream want;
            want << std::fixed << std::setprecision(2) << 120.0;
            std::cout << "FMTR_CHECK case=fmtr_f2 ok="
                      << (got.str() == want.str() ? 1 : 0) << "\n";
        }
        // Case fmtr_tail: tail-flush + raw passthrough (Segmenter.cpp:66-67 shape).
        {
            std::ostringstream got;
            got << iof::fmtr("warn %s") << "t" << "u";
            std::ostringstream want;
            want << "warn " << "t" << "u";
            std::cout << "FMTR_CHECK case=fmtr_tail ok="
                      << (got.str() == want.str() ? 1 : 0) << "\n";
        }
        // Case fmtr_f3: a %f.3s directive (preemph_ratio / cost precisions).
        {
            std::ostringstream got;
            got << iof::fmtr("v=%f.3s;") << 0.97;
            std::ostringstream want;
            want << "v=" << std::fixed << std::setprecision(3) << 0.97 << ";";
            std::cout << "FMTR_CHECK case=fmtr_f3 ok="
                      << (got.str() == want.str() ? 1 : 0) << "\n";
        }
    }

    std::cout << "OK: " << dumps << " dumps\n";
    return 0;
}
