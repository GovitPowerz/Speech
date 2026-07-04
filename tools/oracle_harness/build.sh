#!/bin/bash
# Build the Phase 1 oracle harness against the vendored legacy feature sources.
# Deliberately WITHOUT -ffast-math, with -ffp-contract=off (gcc contracts FMAs by
# default even without fast-math) and -DEIGEN_DONT_VECTORIZE, so a sequential Rust
# port can match the dumps bit-for-bit. Homebrew g++ is required: Apple clang
# lacks -fpermissive, which this 2013-era C++ needs.
#
# NOTE on -std: the brief specified -std=gnu++0x, but the current Homebrew Boost
# (1.90) and default Eigen (5.x) require >= C++14; gnu++0x no longer parses their
# headers. We bump to gnu++14 (the minimum that satisfies them) and pin Eigen 3.4
# via eigen@3. This does NOT relax the parity guarantee: bit-exactness comes from
# -fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE, which govern IEEE FP
# semantics independently of the language standard.
set -euo pipefail
cd "$(dirname "$0")"

BREW="$(brew --prefix)"

# Homebrew g++ (clang is not usable here).
GXX="$(ls "$BREW"/bin/g++-* 2>/dev/null | sort -V | tail -1 || true)"
if [ -z "$GXX" ]; then
  echo "ERROR: no Homebrew g++ found in $BREW/bin. Run: brew install gcc" >&2
  exit 1
fi

# Required Homebrew deps (headers pulled via -I "$BREW/include"):
#   gcc      - the g++-* compiler above
#   eigen@3  - Eigen 3.4 headers. The keg-only brew "eigen" is now 5.x, which
#              rejects -std=gnu++0x ("requires at least c++14") and would risk
#              numeric drift vs the 2013-era legacy; eigen@3 is the faithful match.
#   libsndfile - sndfile.h + libsndfile (wav decode)
#   libmatio - matio.h + libmatio (.mat, referenced by Helpers.hpp/AudioStruct)
#   boost    - boost/math/round + range/lexical_cast (Helpers.hpp)
#   png++    - png++/png.hpp (image dump helpers in Helpers.hpp, never called)
#   sse2neon - shims/x86intrin.h -> <sse2neon.h> so legacy fmath.hpp parses on arm64
#   libpng   - Segmenter.cpp's plotting members odr-use png++ symbols (runtime-dead
#              image dumps); linking the segmenter TUs needs -lpng. libpng is a
#              png++ dependency, so it is almost certainly already installed.
missing=""
for dep in gcc eigen@3 libsndfile libmatio boost png++ sse2neon libpng; do
  brew list --versions "$dep" >/dev/null 2>&1 || missing="$missing $dep"
done
if [ -n "$missing" ]; then
  echo "ERROR: missing Homebrew deps:$missing" >&2
  echo "Run: brew install$missing" >&2
  exit 1
fi

# eigen@3 is keg-only; its headers live under its own prefix, not $BREW/include.
EIGEN3="$(brew --prefix eigen@3)/include/eigen3"

SRC=../../legacy/src

# Legacy translation units required to compile+link the harness:
#   AudioStruct     - the audio ctor + applyPreemph/applyNoise under test
#   MelFilterBank   - AudioStruct references its methods (link)
#   InputStatistics - batch/merge stats stage (Task 10)
#   ConfigFile      - listed for later stages; harmless now
#   CorpusItem      - AudioStruct ctor calls its getters (link)  [added]
#   Timer           - AudioStruct uses Timer in periodogram paths (link)  [added]
#   tinythread      - AudioStruct pulls tthread symbols (link)  [added]
#   -- Phase 2 NN stack (Task 1): construct a real BLSTMNeuralNetwork<LSTMLayer> --
#   BLSTMNeuralNetwork - the templated NN under port; its ctor/getNbOfWeights/
#                        setWeights are exercised. Explicit template instantiations
#                        at BLSTMNeuralNetwork.cpp:960-962 force <SRNLayer>/<CWRNNLayer>
#                        symbols too, so SRNLayer.cpp + CWRNNLayer.cpp are LINK-required
#                        even though this harness only builds the <LSTMLayer> variant.
#   LSTMLayer          - the forward/backward LSTM layer (link + the probe shapes)
#   NeuronLayer        - the output-MLP dense layer (link + dense_gemm probe shape)
#   SRNLayer/CWRNNLayer- link-only, per the explicit instantiations above
#   CostLaw            - BLSTMNeuralNetwork ctor builds a CostLaw member (link)
#   Rprop              - BLSTMNeuralNetwork ctor builds an Rprop _Trainer member (link)
#   -- Phase 2b segmenter TUs (Task 1): link the real segmenter hierarchy so the
#      compiled Segmentation::toFile_VRCTS becomes a byte golden source --
#   Segmenter                  - segmenter base (buildFromConf, plotting members
#                                that odr-use png++ -> -lpng below)
#   Segmentation               - the VRCTS/STM/CSV writer + compute_errors under
#                                port; DEFINES the global exclude_nontrans (:13)
#   BLSTMSpectralSegmenter     - spectral segmenter (LTSV/TDC param derivation)
#   BLSTMSignalSegmenter       - signal-domain segmenter
#   LongTermSpectralVariation  - BLSTMSpectralSegmenter's base (link-required)
#   TimeDomainCorrel           - TDC segmenter
#
# -include boost/math/special_functions/round.hpp: modern Boost's tr1.hpp (pulled
# by Helpers.hpp) no longer re-exports boost::math::round, which getWindowingCoeff
# calls. AudioStruct.cpp includes this header directly, but the other TUs that pull
# Helpers.hpp do not; force-including it (a compiler flag, no legacy edit) restores
# the symbol for all TUs.
"$GXX" -O2 -std=gnu++14 -fpermissive -fno-fast-math -ffp-contract=off -DEIGEN_DONT_VECTORIZE \
  -include boost/math/special_functions/round.hpp \
  -I shims -I "$SRC" -I "$EIGEN3" -I "$BREW/include" \
  main.cpp \
  "$SRC/AudioStruct.cpp" "$SRC/MelFilterBank.cpp" "$SRC/InputStatistics.cpp" \
  "$SRC/ConfigFile.cpp" "$SRC/CorpusItem.cpp" "$SRC/Timer.cpp" "$SRC/tinythread.cpp" \
  "$SRC/BLSTMNeuralNetwork.cpp" "$SRC/LSTMLayer.cpp" "$SRC/NeuronLayer.cpp" \
  "$SRC/SRNLayer.cpp" "$SRC/CWRNNLayer.cpp" "$SRC/CostLaw.cpp" "$SRC/Rprop.cpp" \
  "$SRC/Segmenter.cpp" "$SRC/Segmentation.cpp" "$SRC/BLSTMSpectralSegmenter.cpp" \
  "$SRC/BLSTMSignalSegmenter.cpp" "$SRC/LongTermSpectralVariation.cpp" \
  "$SRC/TimeDomainCorrel.cpp" \
  -L "$BREW/lib" -lsndfile -lmatio -lpng -o oracle_harness

echo "OK: built oracle_harness with $GXX"
