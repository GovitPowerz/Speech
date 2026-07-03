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
#include "ConfigFile.h"
#include "CorpusItem.h"
#include "Helpers.hpp"
#include "InputStatistics.h"
#include "LSTMLayer.h"
#include "MelFilterBank.h"
#include "NeuronLayer.h"
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <output_dir>\n";
        return 1;
    }
    std::string out = argv[1];
    if (!out.empty() && out.back() != '/') out += '/';
    std::string wav = out + "excerpt_2ch_8k.wav";

    // Phase 2 Task 1: real-net config + weights paths. Supplied as argv[2]/argv[3]
    // (absolute, from the extractor) so they resolve regardless of the runtime cwd;
    // the repo-relative fallbacks assume the harness is run from tools/oracle_harness.
    std::string nnConfigPath = (argc > 2)
        ? std::string(argv[2])
        : std::string("../../tests/reference_data/phase0/1_worker_1.config");
    std::string nnWeightsPath = (argc > 3)
        ? std::string(argv[3])
        : std::string("../../tests/reference_data/phase0/NNweights_config1.bin");

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
    // NN_TOL (EXPECTED NONZERO: the k=23 input GEMM diverges per Task 1 probes).
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
                probeGap(realOut, reimplOut, maxUlp, maxAbs);
                std::cout << "NN_TOL site=blstm_norm" << tag
                          << " max_ulp=" << maxUlp << " max_abs=" << std::scientific
                          << std::setprecision(3) << maxAbs << "\n";

                // Verify the reimpl's in-place mutation matches the real class's
                // mutation (types 1/-1 mutate; -2/0 leave the input identical).
                long inUlp = 0; double inAbs = 0.0;
                probeGap(realInput, reimplInput, inUlp, inAbs);
                std::cout << "NN_TOL site=blstm_norm" << tag << "_input"
                          << " max_ulp=" << inUlp << " max_abs=" << std::scientific
                          << std::setprecision(3) << inAbs << "\n";

                Matrix2BinaryFile(out + "blstm_norm" + tag + "_out.bin", reimplOut);
                Matrix2BinaryFile(out + "blstm_norm" + tag + "_input_after.bin", reimplInput);
                dumps += 2;
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
            probeGap(realOut, reimplOut, maxUlp, maxAbs);
            std::cout << "NN_TOL site=blstm_real_fullseq max_ulp=" << maxUlp
                      << " max_abs=" << std::scientific << std::setprecision(3) << maxAbs << "\n";

            Matrix2BinaryFile(out + "blstm_real_fullseq_out.bin", reimplOut);
            Matrix2BinaryFile(out + "blstm_real_fullseq_fwd.bin", outForward);
            Matrix2BinaryFile(out + "blstm_real_fullseq_bwd.bin", outBackward);
            dumps += 3;
        }
    }

    std::cout << "OK: " << dumps << " dumps\n";
    return 0;
}
