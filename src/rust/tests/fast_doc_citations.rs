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
//! Three rules, one test each:
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
//! 3. A bare `:NNN` in a `src/fast/` comment is a LEGACY line, into a source under
//!    `legacy/` that never moves, and the comment must say which: the module doc block for
//!    a `//!` cite, the `///` block itself for a doc cite, and for a `//` body comment
//!    either that comment's own block or the doc block of the enclosing item (the nearest
//!    `fn` / `struct` / `impl` / ... line above) must name a `.cpp` / `.h` / `.m` file.
//!    `fast/cells.rs`'s LSTM step kernel set the convention under #96 ("the `:NNN` cites in
//!    this doc and the body are that legacy loop's lines"); this is it made executable.

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

/// `text` names a legacy source: `<name>.cpp`, `<name>.h` or `<name>.m` as a whole word.
fn names_legacy_file(text: &str) -> bool {
    text.split(|c: char| !(is_ident(c) || c == '.')).any(|w| {
        let Some((stem, ext)) = w.rsplit_once('.') else {
            return false;
        };
        !stem.is_empty() && stem.chars().all(is_ident) && matches!(ext, "cpp" | "h" | "m")
    })
}

/// The bare line citations in one comment's text: each `:NNN` (two or more digits, so a
/// `:1` type ascription or a `1:1` ratio is not one) whose colon is not glued to an
/// identifier, a number, a `.` or another `:` on its left -- a `sad.rs:1436` is a FILE
/// citation (rule 1), a `12:30` a time, a `a::b` a path.
fn bare_cites_in(comment: &str) -> Vec<String> {
    let chars: Vec<char> = comment.chars().collect();
    let mut out = Vec::new();
    for (i, &c) in chars.iter().enumerate() {
        if c != ':' {
            continue;
        }
        let glued = i > 0 && (is_ident(chars[i - 1]) || chars[i - 1] == '.' || chars[i - 1] == ':');
        let digits = chars[i + 1..]
            .iter()
            .take_while(|d| d.is_ascii_digit())
            .count();
        if !glued && digits >= 2 {
            out.push(chars[i..=i + digits].iter().collect());
        }
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum CommentKind {
    Module,
    Doc,
    Body,
}

/// `line` as (kind, comment text) if it carries a comment: a `//!` / `///` / `//` line, or
/// code with a trailing `//`.
fn comment_of(line: &str) -> Option<(CommentKind, &str)> {
    let t = line.trim_start();
    if let Some(c) = t.strip_prefix("//!") {
        Some((CommentKind::Module, c))
    } else if let Some(c) = t.strip_prefix("///") {
        Some((CommentKind::Doc, c))
    } else {
        line.find("//").map(|i| (CommentKind::Body, &line[i + 2..]))
    }
}

/// The contiguous run of lines of `kind` around `n` (full-line comments only), joined.
fn block_around(lines: &[&str], n: usize, kind: CommentKind) -> String {
    let is_kind = |i: usize| {
        let t = lines[i].trim_start();
        match kind {
            CommentKind::Module => t.starts_with("//!"),
            CommentKind::Doc => t.starts_with("///"),
            CommentKind::Body => {
                t.starts_with("//") && !t.starts_with("///") && !t.starts_with("//!")
            }
        }
    };
    let mut lo = n;
    while lo > 0 && is_kind(lo - 1) {
        lo -= 1;
    }
    let mut hi = n;
    while hi + 1 < lines.len() && is_kind(hi + 1) {
        hi += 1;
    }
    lines[lo..=hi].join("\n")
}

/// `line` opens an item: `fn` / `struct` / `enum` / `impl` / `trait` / `const` / `static` /
/// `type` / `mod`, after any visibility.
fn is_item_line(line: &str) -> bool {
    let t = line.trim_start();
    if t.starts_with("//") {
        return false;
    }
    let mut rest = t;
    for vis in ["pub(crate) ", "pub(super) ", "pub "] {
        rest = rest.strip_prefix(vis).unwrap_or(rest);
    }
    [
        "fn ", "struct ", "enum ", "impl ", "impl<", "trait ", "const ", "static ", "type ", "mod ",
    ]
    .iter()
    .any(|kw| rest.starts_with(kw))
}

/// The `///` block documenting the item that encloses line `n`: walk up to the nearest item
/// line, then over its attributes to the doc block above it. Empty when there is none.
fn enclosing_item_doc(lines: &[&str], n: usize) -> String {
    let Some(item) = (0..n).rev().find(|&i| is_item_line(lines[i])) else {
        return String::new();
    };
    let mut i = item;
    while i > 0 && lines[i - 1].trim_start().starts_with("#[") {
        i -= 1;
    }
    if i > 0 && lines[i - 1].trim_start().starts_with("///") {
        block_around(lines, i - 1, CommentKind::Doc)
    } else {
        String::new()
    }
}

/// The bare `:NNN` cites in `text` that no legacy file name covers (rule 3), as
/// `(line, cite)`.
fn undeclared_bare_cites(text: &str) -> Vec<(usize, String)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    for (n, line) in lines.iter().enumerate() {
        let Some((kind, comment)) = comment_of(line) else {
            continue;
        };
        let cites = bare_cites_in(comment);
        if cites.is_empty() {
            continue;
        }
        let declared = match kind {
            CommentKind::Module | CommentKind::Doc => {
                names_legacy_file(&block_around(&lines, n, kind))
            }
            CommentKind::Body => {
                names_legacy_file(comment)
                    || names_legacy_file(&block_around(&lines, n, CommentKind::Body))
                    || names_legacy_file(&enclosing_item_doc(&lines, n))
            }
        };
        if !declared {
            out.extend(cites.into_iter().map(|c| (n + 1, c)));
        }
    }
    out
}

#[test]
fn every_bare_line_cite_in_fast_names_its_legacy_source() {
    let mut hits = Vec::new();
    for file in rust_files(&src_dir("fast")) {
        let text = fs::read_to_string(&file).unwrap();
        for (line, cite) in undeclared_bare_cites(&text) {
            hits.push(format!("{}:{line}: `{cite}`", file.display()));
        }
    }
    assert!(
        hits.is_empty(),
        "bare `:NNN` cites with no legacy source named (a `.cpp` / `.h` / `.m` in the \
         module doc, the doc block, the comment's block or the enclosing item's doc); \
         a line into a live file is cited by symbol instead:\n{}",
        hits.join("\n")
    );
}

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

#[test]
fn the_bare_cite_rule_reads_the_declaring_block() {
    // A `///` block naming its legacy file covers its own cites.
    let ok_doc = "/// Port of `Foo.cpp:10-40`; the gate at `:12`.\nfn f() {}\n";
    assert!(undeclared_bare_cites(ok_doc).is_empty());
    // An undeclared doc cite is a hit.
    let bad_doc = "/// The gate at `:12`.\nfn f() {}\n";
    assert_eq!(undeclared_bare_cites(bad_doc), vec![(1, ":12".to_owned())]);
    // A body comment is covered by the enclosing item's doc, through attributes.
    let ok_body = concat!(
        "/// Port of `Foo.cpp:10-40`.\n#[inline]\nfn f() {\n",
        "    let x = 1; // :12 the gate\n    // :14-16 the loop\n}\n"
    );
    assert!(undeclared_bare_cites(ok_body).is_empty());
    // ... but not by another item's doc.
    let bad_body =
        "/// Port of `Foo.cpp:10-40`.\nfn g() {}\n\nfn f() {\n    // :14-16 the loop\n}\n";
    assert_eq!(undeclared_bare_cites(bad_body), vec![(5, ":14".to_owned())]);
    // A body comment's own block may declare the file.
    let ok_block = "fn f() {\n    // The exact `Bar.m` lines:\n    // :14-16 the loop\n}\n";
    assert!(undeclared_bare_cites(ok_block).is_empty());
    // A module doc must name the file in the same `//!` block.
    assert!(undeclared_bare_cites("//! `A.h` ctor `:3-9`.\n").is_empty());
    assert_eq!(
        undeclared_bare_cites("//! ctor `:30-90`.\n"),
        vec![(1, ":30".to_owned())]
    );
    // Not cites: a file citation (rule 1), a path, a time, a one-digit ascription.
    assert!(bare_cites_in("`sad.rs:1436`, `a::b`, 12:30, `x: 1`, `mel.rs:362`").is_empty());
    assert_eq!(bare_cites_in("(`:362`/`:383`)"), vec![":362", ":383"]);
}
