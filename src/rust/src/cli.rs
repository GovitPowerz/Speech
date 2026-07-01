//! CLI mode parsing, mirroring the legacy `fsp` flags.
//!
//! Ported from legacy C++: FastSpeechProcessing.cpp `main`.

/// Run mode selected by the leading CLI flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// `-s` / `-S`: run one config over the corpus.
    Solo,
    /// `-i` / `-I`: solo run plus image/segmentation dump.
    Image,
    /// `-t` / `-T`: solo run plus finite-difference gradient unit test.
    UnitTest,
    /// `-m` / `-M`: multi-config run.
    Multi,
}

impl Mode {
    /// Parse a leading flag into a [`Mode`]. Returns `None` for unknown flags.
    pub fn from_flag(flag: &str) -> Option<Mode> {
        match flag {
            "-s" | "-S" => Some(Mode::Solo),
            "-i" | "-I" => Some(Mode::Image),
            "-t" | "-T" => Some(Mode::UnitTest),
            "-m" | "-M" => Some(Mode::Multi),
            _ => None,
        }
    }
}
