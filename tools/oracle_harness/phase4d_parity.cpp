// Phase 4d parity oracle driver: the -ffast-math REBUILD leg of the split oracle.
//
// The resurrected 2015 x86_64 `Segment` binary (tools/fsp_runtime/bin/Segment) writes
// the VRCTS segmentation correctly but SIGSEGVs in the post-inference saveWeights path
// BEFORE saveResults runs, so it never writes MultiConfigResults.mat (see
// tools/fsp_runtime/README.md). This driver reproduces the SAME run through the REAL
// COMPILED legacy CorpusProcessor stack (the Phase 4a tier-1 pattern) in a fresh native
// build that exits cleanly, so saveResults writes MultiConfigResults.mat (the
// numeric-column oracle for Task 4's tolerance comparisons).
//
// Built by `build.sh --fastmath` with -O3 -ffast-math (WITHOUT the strict-IEEE flags)
// into a SEPARATE binary (oracle_harness_fastmath) that never clobbers the strict
// oracle_harness. It links the same legacy translation units the strict harness does.
//
// The body mirrors FastSpeechProcessing.cpp:42-71 for the -s/-m branches: build a
// single-config vector, set the exclude_nontrans global, then
// CorpusProcessor(configs, mode).run(). The only additions over the legacy main are the
// argv[1] workdir chdir (so the CorpusProcessor's cwd-relative outputs -- VRCTS xml +
// MultiConfigResults.mat -- land in the extractor's scratch dir) and the argv[3] mode
// select (Task 3 fix-wave):
//   s -- solo, unscored (BagOfProcessors.cpp's scored/unscored branch stays false, so
//        the cost/error columns are always 0 regardless of any reference; VRCTS is
//        ALWAYS written in this branch, dumpDir fallback to the audio filename). This is
//        the leg used for the real-vs-fastmath VRCTS structural-agreement calibration.
//   m -- multi, SCORED (requires a reference: BagOfProcessors.cpp:302 exit(1)s if
//        seg._ClassificationErrors is empty, which only happens when the .stm fails to
//        OPEN -- reference-less scoring, e.g. a .stm with zero matching lines in the
//        window, still yields an empty-but-present Reference and does not trip this
//        gate). The scored branch writes Pfa/Pmiss/globalError/cumulativeError/
//        NbOfClassif for real; VRCTS is written only if Dump_Directory is set (unset in
//        the committed tuple configs, so this leg produces no VRCTS xml -- expected, the
//        -s leg above remains the sole VRCTS source).

#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "ConfigFile.h"
#include "CorpusProcessor.h"

// Defined in Segmentation.cpp (the legacy TU), same as FastSpeechProcessing.cpp:29.
extern bool exclude_nontrans;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <workdir> <config> <mode: s|m>\n";
        return 1;
    }
    std::string workdir = argv[1];
    std::string configPath = argv[2];
    std::string modeArg = argv[3];

    if ((modeArg.size() != 1) || ((modeArg[0] != 's') && (modeArg[0] != 'm'))) {
        std::cerr << "phase4d_parity: mode must be 's' or 'm', got '" << modeArg << "'\n";
        return 1;
    }

    if (chdir(workdir.c_str()) != 0) {
        std::cerr << "phase4d_parity: chdir to " << workdir << " failed\n";
        return 1;
    }

    // FastSpeechProcessing.cpp:42-53 (the -s/-m branches): single config, default
    // readLineChar. -m in the real CLI accepts multiple config files (argv[2..]); this
    // driver always passes exactly one, matching the -s single-config shape.
    std::vector<ConfigFile> configs;
    ConfigFile conf(configPath);
    configs.push_back(conf);

    // FastSpeechProcessing.cpp:68.
    exclude_nontrans = configs[0].get<bool>("exclude_nontrans", false);

    // char[3] (writable): CorpusProcessor may rewrite mode[1] in the train/gradCheck
    // paths, but neither solo (-s) nor multi (-m) with zero training epochs / zero
    // gradient-check epsilon (both unset in the committed tuple configs) ever reaches
    // those rewrites -- CorpusProcessor::run() falls through to runSolo() either way,
    // which is what keeps this a single-pass run under both modes (see the file header).
    char mode[3] = {'-', modeArg[0], '\0'};

    // FastSpeechProcessing.cpp:70-71.
    CorpusProcessor processor(configs, mode);
    processor.run();

    return 0;
}
