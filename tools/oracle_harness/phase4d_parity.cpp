// Phase 4d parity oracle driver: the -ffast-math REBUILD leg of the split oracle.
//
// The resurrected 2015 x86_64 `Segment` binary (tools/fsp_runtime/bin/Segment) writes
// the VRCTS segmentation correctly but SIGSEGVs in the post-inference saveWeights path
// BEFORE saveResults runs, so it never writes MultiConfigResults.mat (see
// tools/fsp_runtime/README.md). This driver reproduces the SAME solo (-s) run through
// the REAL COMPILED legacy CorpusProcessor stack (the Phase 4a tier-1 pattern) in a
// fresh native build that exits cleanly, so saveResults writes MultiConfigResults.mat
// (the numeric-column oracle for Task 4's tolerance comparisons).
//
// Built by `build.sh --fastmath` with -O3 -ffast-math (WITHOUT the strict-IEEE flags)
// into a SEPARATE binary (oracle_harness_fastmath) that never clobbers the strict
// oracle_harness. It links the same legacy translation units the strict harness does.
//
// The body mirrors FastSpeechProcessing.cpp:42-71 for the solo (-s) branch: build a
// single-config vector, set the exclude_nontrans global, then
// CorpusProcessor(configs, "-s").run(). The only additions over the legacy main are the
// argv[1] workdir chdir (so the CorpusProcessor's cwd-relative outputs -- VRCTS xml +
// MultiConfigResults.mat -- land in the extractor's scratch dir) and the fixed -s mode.

#include <iostream>
#include <string>
#include <vector>

#include <unistd.h>

#include "ConfigFile.h"
#include "CorpusProcessor.h"

// Defined in Segmentation.cpp (the legacy TU), same as FastSpeechProcessing.cpp:29.
extern bool exclude_nontrans;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <workdir> <config>\n";
        return 1;
    }
    std::string workdir = argv[1];
    std::string configPath = argv[2];

    if (chdir(workdir.c_str()) != 0) {
        std::cerr << "phase4d_parity: chdir to " << workdir << " failed\n";
        return 1;
    }

    // FastSpeechProcessing.cpp:42-53 (the -s branch): single config, default readLineChar.
    std::vector<ConfigFile> configs;
    ConfigFile conf(configPath);
    configs.push_back(conf);

    // FastSpeechProcessing.cpp:68.
    exclude_nontrans = configs[0].get<bool>("exclude_nontrans", false);

    // Solo mode. char[3] (writable): CorpusProcessor may rewrite mode[1] in the train/
    // gradCheck paths, but solo (-s) leaves it untouched.
    char mode[3] = {'-', 's', '\0'};

    // FastSpeechProcessing.cpp:70-71.
    CorpusProcessor processor(configs, mode);
    processor.run();

    return 0;
}
