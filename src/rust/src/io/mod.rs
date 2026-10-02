//! On-disk interchange: weight `.bin` codec and `.mat` read/write.
//!
//! Ported from legacy C++: Helpers.hpp (BinaryFile2Vector/Matrix2BinaryFile/
//! Matrix2MatFile).

pub mod binary;
pub mod matfile;

/// `<dir>/<prefix><base>` for a `path` of `<dir>/<base>`: the artifact-naming
/// rule of every weight-save prefix (`bestNNWeight_<n>_`, `weights_`,
/// `weightsDerivatives_`, `LID_`). The legacy glued the prefix onto the WHOLE
/// string (`BagOfProcessors.cpp:463`, `BLSTMNeuralNetwork.cpp:319-321`,
/// `TwinBLSTMSpectralLID.cpp:84`), so a path with a directory composed
/// `weights_/abs/out.mat` and died; a bare basename -- the legacy's only
/// production shape -- composes byte-identically here (IMPROVEMENTS.md,
/// "`saveWeights` glues the prefix to the WHOLE filename", FIXED).
pub fn prefix_basename(prefix: &str, path: &str) -> String {
    match path.rfind(std::path::is_separator) {
        Some(i) => format!("{}{prefix}{}", &path[..=i], &path[i + 1..]),
        None => format!("{prefix}{path}"),
    }
}

#[cfg(test)]
mod tests {
    use super::prefix_basename;

    #[test]
    fn bare_basename_composes_as_the_legacy_glue_did() {
        assert_eq!(
            prefix_basename("weights_", "epoch1.mat"),
            "weights_epoch1.mat"
        );
        assert_eq!(
            prefix_basename("bestNNWeight_1_", "tier2_spectral.mat"),
            "bestNNWeight_1_tier2_spectral.mat"
        );
    }

    #[test]
    fn a_directory_stays_ahead_of_the_prefix() {
        assert_eq!(
            prefix_basename("weights_", "/abs/out/epoch1.mat"),
            "/abs/out/weights_epoch1.mat"
        );
        assert_eq!(
            prefix_basename("LID_", "rel/dir/x.mat"),
            "rel/dir/LID_x.mat"
        );
        assert_eq!(
            prefix_basename("weights_", "/abs/out/bestNNWeight_1_x.mat"),
            "/abs/out/weights_bestNNWeight_1_x.mat"
        );
    }
}
