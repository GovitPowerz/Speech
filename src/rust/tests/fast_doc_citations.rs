//! The fast tree cites the exact tree by SYMBOL, never by line number (issues #96, #118).
//!
//! `fast/bicell.rs` carried `nn/blstm.rs` line spans that the Phase 9 retention work had
//! moved by 55 lines, plus a paragraph telling the reader to add 55; `fast/driver.rs` cited
//! `tasks/sad.rs` and `tasks/lid.rs` lines that landed on closing braces. A line citation
//! into a file this repo keeps editing decays by construction; a `module::Type::method`
//! path does not, and this test keeps it resolvable: every crate path named in `src/fast/`
//! must lead to a definition under `src/`, so a renamed exact symbol fails the suite
//! instead of rotting the doc.
//!
//! Two rules, one test each:
//!
//! 1. No `<file>.rs:<line>` citation in `src/fast/` names a file under `src/` outside
//!    `fast/` (the basenames are read off the tree, `mod.rs` by its directory). The
//!    `audio.rs::get_sequence` (`:NNN`) form is refused by the same scan, since the
//!    resolver never sees it.
//! 2. Every `root::a::B::c` path in `src/fast/` whose root is a top-level module of
//!    `src/lib.rs` other than `fast` resolves. Resolution is FILE-granular, a text scan
//!    rather than a parse: a module segment picks the file, every later segment must be
//!    defined somewhere in that file (an item, a `pub use` re-export, an enum variant or
//!    struct field line). `nn::layers::NeuronLayer::feed_forward` therefore resolves on any
//!    `fn feed_forward` in `layers.rs`, and a path wrapped across two doc lines is refused
//!    outright (its tail would go unchecked). A path with a prefix other than `crate::`
//!    (`std::io::`, `super::nn::`) is not a citation and is skipped.

use std::fs;
use std::path::{Path, PathBuf};

fn src_dir(sub: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(sub)
}

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_dir() {
            out.extend(rust_files(&p));
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
    out.sort();
    out
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `text` starts with `word` as a whole identifier.
fn starts_with_word(text: &str, word: &str) -> bool {
    text.strip_prefix(word)
        .is_some_and(|tail| !tail.starts_with(is_ident))
}

/// The name of every `.rs` file under `src/` outside `src/fast/` (`blstm.rs`, `sad.rs`,
/// `audio.rs`, ..., a `mod.rs` as `cells/mod.rs`): a `<name>:<digit>` (a line) or
/// `<name>::` (a file-path symbol, cited with a separate `:NNN`) anywhere in `fast/` is a
/// file citation into the exact tree.
fn exact_file_names() -> Vec<String> {
    let fast = src_dir("fast");
    rust_files(&src_dir(""))
        .iter()
        .filter(|p| !p.starts_with(&fast))
        .map(|p| {
            let name = p.file_name().unwrap().to_str().unwrap();
            if name == "mod.rs" {
                let dir = p.parent().unwrap().file_name().unwrap().to_str().unwrap();
                format!("{dir}/{name}")
            } else {
                name.to_owned()
            }
        })
        .collect()
}

/// The top-level modules of the crate (`pub mod audio;` ... in `src/lib.rs`) other than
/// `fast`: the roots a citation from `src/fast/` may start with.
fn crate_roots() -> Vec<String> {
    fs::read_to_string(src_dir("lib.rs"))
        .unwrap()
        .lines()
        .filter_map(|line| {
            let t = line.trim();
            let t = t.strip_prefix("pub ").unwrap_or(t);
            t.strip_prefix("mod ")?.strip_suffix(';').map(str::to_owned)
        })
        .filter(|m| m != "fast")
        .collect()
}

#[test]
fn fast_cites_no_exact_file_by_line_number() {
    let names = exact_file_names();
    assert!(names.iter().any(|n| n == "blstm.rs") && names.iter().any(|n| n == "sad.rs"));
    let mut hits = Vec::new();
    for file in rust_files(&src_dir("fast")) {
        let text = fs::read_to_string(&file).unwrap();
        for (n, line) in text.lines().enumerate() {
            for name in &names {
                let needle = format!("{name}:");
                let mut from = 0;
                while let Some(i) = line[from..].find(&needle) {
                    let at = from + i + needle.len();
                    if line[at..].starts_with(|c: char| c.is_ascii_digit() || c == ':') {
                        hits.push(format!("{}:{}: {}", file.display(), n + 1, line.trim()));
                        break;
                    }
                    from = at;
                }
            }
        }
    }
    assert!(
        hits.is_empty(),
        "file citations into src/ by line number (cite the `module::symbol` path):\n{}",
        hits.join("\n")
    );
}

/// Every `root::a::B::c` path mentioned in `text` (doc comment, code comment or code) whose
/// root is one of `roots`, as `(line, segments)` with the root as the first segment. A
/// `crate::` prefix is dropped; any other prefix (`std::io::`, `super::nn::`, `fast::nn::`)
/// makes the run something else than an exact-tree citation, and it is skipped. A path
/// ending in `::` at the end of its line is wrapped onto the next one; it is reported with
/// an empty last segment so the resolver refuses it.
fn symbol_paths_in(text: &str, roots: &[String]) -> Vec<(usize, Vec<String>)> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let mut rest = line;
        while !rest.is_empty() {
            let Some(start) = rest.find(|c: char| is_ident(c) || c == ':') else {
                break;
            };
            let run_len = rest[start..]
                .find(|c: char| !(is_ident(c) || c == ':'))
                .unwrap_or(rest.len() - start);
            let run = &rest[start..start + run_len];
            let after = &rest[start + run_len..];
            if run.contains("::") {
                let wrapped = run.ends_with("::") && after.trim().is_empty();
                let mut segs: Vec<String> = run
                    .split("::")
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect();
                if segs.first().is_some_and(|s| s == "crate") {
                    segs.remove(0);
                }
                if segs.first().is_some_and(|s| roots.contains(s)) {
                    if wrapped {
                        segs.push(String::new());
                    }
                    out.push((n + 1, segs));
                }
            }
            rest = after;
        }
    }
    out
}

/// `name` is defined in `text`: an item (`fn`/`struct`/`enum`/`trait`/`type`/`const`/`static`/
/// `mod name`), a re-export (`pub use ...name`), or an enum variant / struct field line. A
/// line that merely STARTS with `name` (`LstmLayer::feed_forward(..)`, `matmul_seq(`) is a
/// use, not a definition, so a moved symbol's old path does not resolve on its call sites.
fn defines(text: &str, name: &str) -> bool {
    let item = [
        "fn ", "struct ", "enum ", "trait ", "type ", "const ", "static ", "mod ",
    ];
    text.lines().any(|line| {
        let t = line.trim_start();
        let is_item = item.iter().any(|kw| {
            let mut rest = t;
            for vis in ["pub(crate) ", "pub(super) ", "pub "] {
                rest = rest.strip_prefix(vis).unwrap_or(rest);
            }
            rest.strip_prefix(kw)
                .is_some_and(|rest| starts_with_word(rest, name))
        });
        let reexport =
            t.starts_with("pub use ") && t.split(|c: char| !is_ident(c)).any(|w| w == name);
        let variant = t.strip_prefix(name).is_some_and(|tail| {
            tail.is_empty()
                || tail.starts_with([',', '{'])
                || tail.starts_with(": ")
                || (tail.starts_with('(') && t.ends_with(','))
        });
        is_item || reexport || variant
    })
}

/// Walk `segs` from `src/`: the root and every module segment descend (`nn` -> `nn/`,
/// `blstm` -> `blstm.rs`, `audio` -> `audio.rs`), the rest must be defined in the file
/// reached (`mod.rs` for a directory). An empty segment is a path wrapped across lines and
/// is refused.
fn resolve(segs: &[String]) -> Result<(), String> {
    if segs.iter().any(String::is_empty) {
        return Err("the path wraps onto the next line; keep a cited path on one line".into());
    }
    let mut dir = src_dir("");
    let mut i = 0;
    let file = loop {
        if i == segs.len() {
            break dir.join("mod.rs");
        }
        let sub = dir.join(&segs[i]);
        let f = dir.join(format!("{}.rs", segs[i]));
        if sub.is_dir() {
            dir = sub;
            i += 1;
        } else if f.is_file() {
            i += 1;
            break f;
        } else if i == 0 {
            return Err(format!("`{}` is not a module under src/", segs[0]));
        } else {
            break dir.join("mod.rs");
        }
    };
    let text = fs::read_to_string(&file).map_err(|e| format!("{}: {e}", file.display()))?;
    for s in &segs[i..] {
        if !defines(&text, s) {
            return Err(format!("`{s}` is not defined in {}", file.display()));
        }
    }
    Ok(())
}

#[test]
fn every_exact_symbol_cited_in_fast_resolves() {
    let roots = crate_roots();
    assert!(roots.iter().any(|r| r == "nn") && roots.iter().any(|r| r == "tasks"));
    let mut bad = Vec::new();
    let mut seen = 0usize;
    for file in rust_files(&src_dir("fast")) {
        let text = fs::read_to_string(&file).unwrap();
        for (line, segs) in symbol_paths_in(&text, &roots) {
            seen += 1;
            if let Err(why) = resolve(&segs) {
                bad.push(format!(
                    "{}:{line}: {} -- {why}",
                    file.display(),
                    segs.join("::")
                ));
            }
        }
    }
    // The scan found 195 paths at the #118 close; half of that is a broken
    // tokenizer, not a doc trim.
    assert!(
        seen > SEEN_FLOOR,
        "the scan found only {seen} crate paths in src/fast/; the tokenizer is broken"
    );
    assert!(
        bad.is_empty(),
        "unresolved exact-tree symbols cited from src/fast/:\n{}",
        bad.join("\n")
    );
}

const SEEN_FLOOR: usize = 95;

#[test]
fn the_resolver_rejects_a_renamed_symbol() {
    let path = |s: &str| s.split("::").map(str::to_owned).collect::<Vec<_>>();
    assert!(
        resolve(&path(
            "nn::blstm::BlstmNetwork::feed_forward_backward_overlap"
        ))
        .is_ok()
    );
    assert!(
        resolve(&path(
            "nn::blstm::BlstmNetwork::feed_forward_backward_overlapp"
        ))
        .is_err()
    );
    assert!(resolve(&path("nn::network::Network::feed_forward_double")).is_ok());
    assert!(resolve(&path("nn::cells::CellLayer")).is_ok());
    assert!(resolve(&path("nn::cells::slstm::SlstmLayer::feed_forward_reverse")).is_ok());
    assert!(resolve(&path("nn::blstm::Direction::Forward")).is_ok());
    assert!(
        resolve(&path(
            "tasks::sad::BlstmSpectralSegmenter::get_segmentation"
        ))
        .is_ok()
    );
    assert!(
        resolve(&path(
            "tasks::lid::TwinBlstmSpectralLid::get_segmentation_mode7"
        ))
        .is_ok()
    );
    assert!(resolve(&path("audio::get_sequence")).is_ok());
    assert!(resolve(&path("audio::get_sequencee")).is_err());
    assert!(resolve(&path("config::pack_lstm_layer")).is_ok());
    assert!(resolve(&path("features::mel::MelFilterBank::apply_dct")).is_ok());
    assert!(resolve(&path("nn::nowhere::Thing")).is_err());
    assert!(resolve(&path("nowhere::Thing")).is_err());
    // `network.rs` calls `LstmLayer::feed_forward(..)` but defines neither.
    assert!(resolve(&path("nn::network::LstmLayer::feed_forward")).is_err());
}

#[test]
fn the_tokenizer_keeps_crate_paths_and_skips_foreign_ones() {
    let roots = crate_roots();
    let segs = |s: &str| {
        symbol_paths_in(s, &roots)
            .into_iter()
            .map(|(_, v)| v)
            .collect::<Vec<_>>()
    };
    let p = |s: &str| s.split("::").map(str::to_owned).collect::<Vec<_>>();
    assert_eq!(
        segs("see `nn::blstm::BlstmNetwork`"),
        vec![p("nn::blstm::BlstmNetwork")]
    );
    assert_eq!(
        segs("[`crate::tasks::lid::Twin`]"),
        vec![p("tasks::lid::Twin")]
    );
    assert_eq!(
        segs("use crate::nn::blstm::{BlstmConfig, CellType};"),
        vec![p("nn::blstm")]
    );
    assert!(segs("std::io::Result, super::nn::Thing, fast::nn::FastNet").is_empty());
    assert!(segs("crate::fast::nn::FastNet, ndarray::Array2").is_empty());
}

#[test]
fn a_path_wrapped_across_lines_is_refused() {
    let roots = crate_roots();
    let text = "/// see `nn::blstm::BlstmNetwork::\n/// load_weights_file`";
    let wrapped = symbol_paths_in(text, &roots);
    assert_eq!(wrapped.len(), 1);
    assert!(resolve(&wrapped[0].1).unwrap_err().contains("wraps"));
    // A module path followed by code on the same line is not a wrap.
    let braces = symbol_paths_in("use crate::nn::blstm::{BlstmConfig, CellType};", &roots);
    assert_eq!(braces.len(), 1);
    assert!(resolve(&braces[0].1).is_ok());
}
