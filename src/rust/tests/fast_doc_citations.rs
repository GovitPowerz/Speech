//! The fast tree cites the exact tree by SYMBOL, never by line number (issue #96).
//!
//! `fast/bicell.rs` carried `nn/blstm.rs` line spans that the Phase 9 retention work had
//! moved by 55 lines, plus a paragraph telling the reader to add 55. A line citation into a
//! file this repo keeps editing decays by construction; a `module::Type::method` path does
//! not, and this test keeps it resolvable: every `nn::...` path named in `src/fast/` must
//! lead to a definition in `src/nn/`, so a rename fails the suite instead of rotting the doc.
//!
//! Scope: the `nn/` citations only. Citations into `legacy/` C++ sources (`:NNN` after a
//! `.cpp` name) never move and are not scanned; `tasks/`, `audio.rs` and `features/`
//! line citations in `fast/` are a named follow-on (RESULTS.md, "Symbol/anchor citations").

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

/// The basenames of every file under `src/nn/` (`blstm.rs`, `slstm.rs`, ...): a `<basename>:<digit>`
/// anywhere in `fast/` is a line citation into the exact tree.
fn nn_basenames() -> Vec<String> {
    rust_files(&src_dir("nn"))
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap().to_owned())
        .filter(|n| n != "mod.rs")
        .collect()
}

#[test]
fn fast_cites_no_nn_file_by_line_number() {
    let names = nn_basenames();
    let mut hits = Vec::new();
    for file in rust_files(&src_dir("fast")) {
        let text = fs::read_to_string(&file).unwrap();
        for (n, line) in text.lines().enumerate() {
            for name in &names {
                let needle = format!("{name}:");
                let mut from = 0;
                while let Some(i) = line[from..].find(&needle) {
                    let at = from + i + needle.len();
                    if line[at..].starts_with(|c: char| c.is_ascii_digit()) {
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
        "line citations into src/nn/ (cite the symbol):\n{}",
        hits.join("\n")
    );
}

/// Every `nn::a::b::C::d` path mentioned in `src/fast/` (doc comment, code comment or code).
/// A `super::nn::` / `fast::nn::` path is `fast::nn`, not the exact tree, and is skipped.
fn nn_paths_in(text: &str) -> Vec<(usize, Vec<String>)> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let mut from = 0;
        while let Some(i) = line[from..].find("nn::") {
            let start = from + i;
            let before = &line[..start];
            let skip = before.ends_with(|c: char| is_ident(c))
                || before.ends_with("super::")
                || before.ends_with("fast::");
            let end = start
                + line[start..]
                    .find(|c: char| !(is_ident(c) || c == ':'))
                    .unwrap_or(line.len() - start);
            if !skip {
                let segs: Vec<String> = line[start..end]
                    .split("::")
                    .skip(1)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect();
                out.push((n + 1, segs));
            }
            from = end.max(start + 4);
        }
    }
    out
}

/// `name` is defined in `text`: an item (`fn`/`struct`/`enum`/`trait`/`type`/`const`/`static`/
/// `mod name`), a re-export (`pub use ...name`), or an enum variant / struct field line.
fn defines(text: &str, name: &str) -> bool {
    let item = [
        "fn ", "struct ", "enum ", "trait ", "type ", "const ", "static ", "mod ",
    ];
    text.lines().any(|line| {
        let t = line.trim_start();
        let after = |prefix: &str| {
            t.strip_prefix(prefix).is_some_and(|rest| {
                rest.strip_prefix(name)
                    .is_some_and(|tail| !tail.starts_with(|c: char| is_ident(c)))
            })
        };
        let is_item = item.iter().any(|kw| {
            let mut rest = t;
            for vis in ["pub(crate) ", "pub(super) ", "pub ", "unsafe ", "async "] {
                rest = rest.strip_prefix(vis).unwrap_or(rest);
            }
            rest.strip_prefix(kw).is_some_and(|rest| {
                rest.strip_prefix(name)
                    .is_some_and(|tail| !tail.starts_with(|c: char| is_ident(c)))
            })
        });
        let reexport = t.starts_with("pub use ")
            && (after("pub use ")
                || t.contains(&format!("::{name};"))
                || t.contains(&format!("{name},"))
                || t.contains(&format!(" {name}}}")));
        let variant = t
            .strip_prefix(name)
            .is_some_and(|tail| tail.is_empty() || tail.starts_with([',', '(', '{', ':']));
        is_item || reexport || variant
    })
}

/// Walk `segs` from `src/nn/`: module segments descend (`blstm` -> `blstm.rs`, `cells` ->
/// `cells/`), the rest must be defined in the file reached (`mod.rs` for a directory).
fn resolve(segs: &[String]) -> Result<(), String> {
    let mut dir = src_dir("nn");
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
fn every_nn_symbol_cited_in_fast_resolves() {
    let mut bad = Vec::new();
    let mut seen = 0usize;
    for file in rust_files(&src_dir("fast")) {
        let text = fs::read_to_string(&file).unwrap();
        for (line, segs) in nn_paths_in(&text) {
            seen += 1;
            if let Err(why) = resolve(&segs) {
                bad.push(format!(
                    "{}:{line}: nn::{} -- {why}",
                    file.display(),
                    segs.join("::")
                ));
            }
        }
    }
    assert!(
        seen > 40,
        "the scan found only {seen} `nn::` paths in src/fast/; the tokenizer is broken"
    );
    assert!(
        bad.is_empty(),
        "unresolved exact-tree symbols cited from src/fast/:\n{}",
        bad.join("\n")
    );
}

#[test]
fn the_resolver_rejects_a_renamed_symbol() {
    assert!(
        resolve(&[
            "blstm".into(),
            "BlstmNetwork".into(),
            "feed_forward_backward_overlap".into()
        ])
        .is_ok()
    );
    assert!(
        resolve(&[
            "blstm".into(),
            "BlstmNetwork".into(),
            "feed_forward_backward_overlapp".into()
        ])
        .is_err()
    );
    assert!(
        resolve(&[
            "network".into(),
            "Network".into(),
            "feed_forward_double".into()
        ])
        .is_ok()
    );
    assert!(resolve(&["cells".into(), "CellLayer".into()]).is_ok());
    assert!(
        resolve(&[
            "cells".into(),
            "slstm".into(),
            "SlstmLayer".into(),
            "feed_forward_reverse".into()
        ])
        .is_ok()
    );
    assert!(resolve(&["blstm".into(), "Direction".into(), "Forward".into()]).is_ok());
    assert!(resolve(&["nowhere".into(), "Thing".into()]).is_err());
}
