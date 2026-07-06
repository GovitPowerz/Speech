//! CLI mode parsing, mirroring the legacy `fsp` flags.
//!
//! Ported from legacy C++: FastSpeechProcessing.cpp `main`.

/// Run mode selected by the leading CLI flag, case dropped (the legacy `mode[1]`
/// letter carries both the mode AND its case, e.g. `T`/`M`/`I` gate the verbose
/// per-file log branches in `BagOfProcessors::SegmentationFunction`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeKind {
    /// `-s` / `-S`: run one config over the corpus.
    Solo,
    /// `-i` / `-I`: solo run plus image/segmentation dump.
    Image,
    /// `-t` / `-T`: solo run plus finite-difference gradient unit test.
    UnitTest,
    /// `-m` / `-M`: multi-config run.
    Multi,
}

/// Run mode: the letter (`kind`) plus its case (`verbose`, legacy `isupper(mode[1])`).
/// The legacy passes the mode as a bare `char*` flag string and re-checks
/// `mode[1] == 'x'`/`'X'` all over the place; this struct is that char pre-decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mode {
    pub kind: ModeKind,
    pub verbose: bool,
}

impl Mode {
    /// Parse a leading flag into a [`Mode`]. Returns `None` for unknown flags.
    pub fn from_flag(flag: &str) -> Option<Mode> {
        let (kind, verbose) = match flag {
            "-s" => (ModeKind::Solo, false),
            "-S" => (ModeKind::Solo, true),
            "-i" => (ModeKind::Image, false),
            "-I" => (ModeKind::Image, true),
            "-t" => (ModeKind::UnitTest, false),
            "-T" => (ModeKind::UnitTest, true),
            "-m" => (ModeKind::Multi, false),
            "-M" => (ModeKind::Multi, true),
            _ => return None,
        };
        Some(Mode { kind, verbose })
    }
}
