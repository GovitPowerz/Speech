//! Corpus + per-file records: parse language2classmapping + fileslisting CSVs,
//! class-balance coefficients.
//!
//! Ported from legacy C++: Corpus.*, CorpusItem.*.

use std::collections::BTreeMap;
use std::fs;

use anyhow::{Result, bail};
use indexmap::IndexMap;

/// One corpus entry (audio path + refs + language/dialect/class metadata).
///
/// legacy: CorpusItem.h:7-28 (field set), Corpus.cpp:177 (construction order).
#[derive(Debug, Clone, PartialEq)]
pub struct CorpusItem {
    pub file_name: String,
    pub ref_seg: String,
    pub language: String,
    pub dialect: String,
    pub class_index: i32,
    pub file_id: i32,
    pub weight: f64,
}

/// Read a string from a config value, with a default for a missing key (legacy
/// `conf.get<string>(name, default)`).
fn parse_string_default(m: &IndexMap<String, String>, key: &str, default: &str) -> String {
    m.get(key).cloned().unwrap_or_else(|| default.to_string())
}

/// Legacy `splitstr(line, ';')` (String.hpp:110-114): repeated `getline(ss, item,
/// delim)`. `getline` returns false (stopping the loop) exactly when it is
/// called with NOTHING left to read (stream already at EOF) -- so a delimiter
/// as the LAST character of the string is consumed to end the preceding
/// field, but produces no further (empty) token, since after consuming it the
/// stream is at EOF with no more delimiter/content to read. This drops AT
/// MOST ONE trailing empty field, not a run of them: "a;b;" -> ["a","b"], but
/// "a;b;;" -> ["a","b",""] (verified against a real `getline` harness). A
/// line with no delimiter at all yields the whole line as one token; an empty
/// line yields zero tokens.
fn splitstr(line: &str, delim: char) -> Vec<String> {
    if line.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<String> = line.split(delim).map(String::from).collect();
    if line.ends_with(delim) {
        out.pop();
    }
    out
}

/// Legacy `read<string>` (Helpers.hpp:1602-1608): `ss << data; ss >> val`
/// extracts the FIRST whitespace-delimited token (leading whitespace skipped,
/// content after the first token silently dropped); an empty or
/// whitespace-only string fails the extraction and hits `check` -> exit(1).
fn read_string(data: &str) -> Result<String> {
    match data.split_whitespace().next() {
        Some(tok) => Ok(tok.to_string()),
        None => bail!("cannot read string '{data}' into variable with type 'string'"),
    }
}

/// Legacy `ConfigFile::get_list<T>(name, delim)` (ConfigFile.h:60-68): empty
/// vector if the key is absent, else `split_with_repeat<string>`
/// (String.hpp:116-127). BOTH split levels there are the template `split<T>`
/// (String.hpp:86-98), which getline-splits (one-trailing-empty-drop, same
/// rule as [`splitstr`]) AND pushes each piece through `read<T>` -- so an
/// empty element ("a,,b" middle, or a piece of "*2") exits(1) in the legacy
/// before any further parsing (bail here). After the outer comma split, each
/// element splits on the `*` repeater: "x*3" pushes 3 copies of "x" ("x*"
/// getline-drops the empty count and pushes one copy). The repeat count goes
/// through `natural` = `boost::lexical_cast<size_t>` (whole-string strict),
/// whose failure is an UNCAUGHT throw in the legacy (std::terminate): panic
/// here. All verified against a compiled transcription of String.hpp:86-127.
fn get_list_string(m: &IndexMap<String, String>, key: &str, delim: char) -> Result<Vec<String>> {
    let mut vect = Vec::new();
    if let Some(v) = m.get(key) {
        for piece in splitstr(v, delim) {
            let s1 = read_string(&piece)?;
            let mut parts = Vec::new();
            for p in splitstr(&s1, '*') {
                parts.push(read_string(&p)?);
            }
            let num_repeats = if parts.len() == 1 {
                1
            } else {
                parts[1].parse::<usize>().unwrap_or_else(|_| {
                    panic!(
                        "bad repeat count '{}' (legacy: uncaught boost::bad_lexical_cast)",
                        parts[1]
                    )
                })
            };
            for _ in 0..num_repeats {
                vect.push(parts[0].clone());
            }
        }
    }
    Ok(vect)
}

/// Legacy `ConfigFile::get_list<T>(name, defaultVal, length, delim)`
/// (ConfigFile.h:70-77): falls back to the list from the 2-arg overload; if
/// that list is EMPTY (key absent, or present but empty string splitting to
/// nothing), resize to `length` copies of `defaultVal` (inserted VERBATIM, no
/// `read<>` validation). A non-empty list shorter than `length` is NOT padded
/// (mirrors `vector::resize` only firing on the `.empty()` branch).
fn get_list_string_default(
    m: &IndexMap<String, String>,
    key: &str,
    default: &str,
    length: usize,
    delim: char,
) -> Result<Vec<String>> {
    let vect = get_list_string(m, key, delim)?;
    if vect.is_empty() {
        Ok(vec![default.to_string(); length])
    } else {
        Ok(vect)
    }
}

/// Corpus of items with per-file class/language mapping and class-balance
/// counts.
///
/// legacy: Corpus.h:19-28 (`_Items`/`_NbFiles`/`_LanguageCount`/`_DialectCount`/
/// `_ClassCount`/`_Language2ClassMapping`). `_NbFiles` is redundant with
/// `items.len()` in this port (both incremented together in `add_item`, legacy
/// keeps them as separate counters); `nb_of_files` reads `items.len()`.
#[derive(Debug, Clone, Default)]
pub struct Corpus {
    items: Vec<CorpusItem>,
    language_count: BTreeMap<String, i32>,
    dialect_count: BTreeMap<String, i32>,
    class_count: BTreeMap<i32, i32>,
    mapping: BTreeMap<String, BTreeMap<String, i32>>,
}

impl Corpus {
    /// legacy: Corpus.cpp:7-131 (ctor). Console-only prints (:15, :35, :40,
    /// :102-130 incl. the "normalizing coefficients" block) are display-only
    /// (no state produced) and dropped; see module doc above and the note on
    /// `add_item` for the per-item warnings, which are also print-only.
    pub fn from_config(map: &IndexMap<String, String>) -> Result<Corpus> {
        let mut corpus = Corpus::default();

        // legacy: Corpus.cpp:16-39 (language2classmapping load).
        let mapping_file = parse_string_default(map, "language2classmapping", "");
        let mapping_text = match fs::read_to_string(&mapping_file) {
            Ok(text) => text,
            Err(_) => bail!("The language->class mapping file {mapping_file} cannot be opened."),
        };
        // legacy `while (getline(infile, line))` == splitstr(text, '\n'): keeps
        // a '\r' on CRLF files (Rust `.lines()` would strip it -- divergent).
        for line in splitstr(&mapping_text, '\n') {
            let tokens = splitstr(&line, ';');
            if !tokens.is_empty() && !tokens[0].is_empty() {
                let lang = tokens[0].clone();
                let mut dial = String::new();
                let mut class_index: i32 = -1;
                if tokens.len() > 1 {
                    if !tokens[1].is_empty() {
                        dial = tokens[1].clone();
                    }
                    if tokens.len() > 2 && !tokens[2].is_empty() {
                        // legacy: `::atof` -- never fails, non-numeric text and
                        // leading-numeric-prefix strings parse permissively;
                        // truncates toward zero to int (C `atof` + implicit
                        // double->int conversion).
                        class_index = atof(&tokens[2]) as i32;
                    }
                }
                corpus
                    .mapping
                    .entry(lang)
                    .or_default()
                    .insert(dial, class_index);
            }
        }

        // legacy: Corpus.cpp:42-101 (listing load, both branches).
        let listing_file = parse_string_default(map, "fileslisting", "");
        if listing_file.is_empty() {
            // legacy: Corpus.cpp:44-57 (no-listing fallback via comma-lists).
            let files_names = get_list_string(map, "files", ',')?;
            let ref_seg_files_names =
                get_list_string_default(map, "refsegfiles", "", files_names.len(), ',')?;
            let ref_lang_files =
                get_list_string_default(map, "reflangfiles", "", files_names.len(), ',')?;
            let ref_dial_files =
                get_list_string_default(map, "refdialfiles", "", files_names.len(), ',')?;
            for (ii, filename) in files_names.iter().enumerate() {
                let refseg = ref_seg_files_names.get(ii).cloned().unwrap_or_default();
                let lang = ref_lang_files
                    .get(ii)
                    .cloned()
                    .unwrap_or_else(|| "unk".to_string());
                let dial = ref_dial_files
                    .get(ii)
                    .cloned()
                    .unwrap_or_else(|| "unk".to_string());
                corpus.add_item(filename.clone(), refseg, lang, dial, 1.0, 1);
            }
        } else {
            // legacy: Corpus.cpp:59-100 (listing-file branch, nested token parse).
            let listing_text = match fs::read_to_string(&listing_file) {
                Ok(text) => text,
                Err(_) => bail!("The listing file {listing_file} cannot be opened."),
            };
            // legacy getline loop: see the mapping loop note on '\r'.
            for line in splitstr(&listing_text, '\n') {
                let tokens = splitstr(&line, ';');
                if !tokens.is_empty() && !tokens[0].is_empty() {
                    let filename = tokens[0].clone();
                    let mut refseg = String::new();
                    let mut lang = "unk".to_string();
                    let mut dial = "unk".to_string();
                    let mut weight = 1.0;
                    let mut file_id = 1;
                    // legacy: Corpus.cpp:74-96 -- nesting mirrors the source
                    // brace structure exactly (each `tokens.size() > N` gate
                    // is a SIBLING of the `tokens[N-1].size() > 0` check one
                    // level up, not nested inside it; see the
                    // `listing_nested_quirk` test). Collapsed by clippy's
                    // suggestion would blur that 1:1 correspondence.
                    #[allow(clippy::collapsible_if)]
                    if tokens.len() > 1 {
                        if !tokens[1].is_empty() {
                            refseg = tokens[1].clone();
                        }
                        if tokens.len() > 2 {
                            if !tokens[2].is_empty() {
                                lang = tokens[2].clone();
                            }
                            if tokens.len() > 3 {
                                if !tokens[3].is_empty() {
                                    dial = tokens[3].clone();
                                }
                                if tokens.len() > 4 {
                                    if !tokens[4].is_empty() {
                                        // legacy: `istringstream >> double`
                                        // accept-on-success -- extraction
                                        // failure keeps the 1.0 default.
                                        if let Some(w) = iss_extract_double(&tokens[4]) {
                                            weight = w;
                                        }
                                        if tokens.len() > 5 && !tokens[5].is_empty() {
                                            if let Some(id) = iss_extract_int(&tokens[5]) {
                                                file_id = id;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    corpus.add_item(filename, refseg, lang, dial, weight, file_id);
                }
            }
        }

        Ok(corpus)
    }

    /// legacy: Corpus.cpp:164-194 (`Corpus::addItem`).
    fn add_item(
        &mut self,
        filename: String,
        refseg: String,
        lang: String,
        dial: String,
        weight: f64,
        file_id: i32,
    ) {
        let class_index = match self.mapping.get(&lang) {
            Some(dial_map) => match dial_map.get(&dial) {
                Some(&idx) => idx,
                // legacy :169-172: unknown dialect under a known language --
                // print-only warning dropped, mapping mutated to cache -1.
                None => {
                    self.mapping
                        .entry(lang.clone())
                        .or_default()
                        .insert(dial.clone(), -1);
                    -1
                }
            },
            // legacy :173-176: unknown language -- print-only warning dropped,
            // mapping mutated to cache -1 (a SECOND file with the same unknown
            // language+dialect pair now hits this cached entry via the `Some`
            // arm above, not this one).
            None => {
                self.mapping
                    .entry(lang.clone())
                    .or_default()
                    .insert(dial.clone(), -1);
                -1
            }
        };

        self.items.push(CorpusItem {
            file_name: filename,
            ref_seg: refseg,
            language: lang.clone(),
            dialect: dial.clone(),
            class_index,
            file_id,
            weight,
        });

        *self.language_count.entry(lang.clone()).or_insert(0) += 1;
        *self
            .dialect_count
            .entry(format!("{lang}_{dial}"))
            .or_insert(0) += 1;
        *self.class_count.entry(class_index).or_insert(0) += 1;
    }

    pub fn nb_of_files(&self) -> usize {
        self.items.len()
    }

    pub fn item(&self, pos: usize) -> &CorpusItem {
        &self.items[pos]
    }

    pub fn class_count(&self) -> &BTreeMap<i32, i32> {
        &self.class_count
    }

    pub fn language_count(&self) -> &BTreeMap<String, i32> {
        &self.language_count
    }
}

/// Longest valid leading double prefix, C `strtod`/`atof` shape: leading
/// whitespace skipped, then optional sign + digits[.digits][exponent];
/// trailing garbage after the prefix is ignored ("0.5abc" -> 0.5, "1e+" ->
/// 1.0). `None` if no valid numeric prefix exists. This is the CLASSIFIER
/// for [`atof`] only -- `istringstream >>` extraction has DIFFERENT
/// semantics, see [`iss_extract_double`].
fn parse_double_prefix(s: &str) -> Option<f64> {
    let trimmed = s.trim_start();
    let bytes = trimmed.as_bytes();
    let mut end = 0usize;
    let mut i = 0usize;
    if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
        i += 1;
    }
    let mut saw_digit = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        saw_digit = true;
        end = i;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        let dot = i;
        i += 1;
        let mut saw_frac_digit = false;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
            saw_frac_digit = true;
        }
        if saw_digit || saw_frac_digit {
            end = i;
        } else {
            i = dot;
        }
    }
    if (saw_digit || end > 0) && i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let mut j = i + 1;
        if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        let mut saw_exp_digit = false;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
            saw_exp_digit = true;
        }
        if saw_exp_digit {
            end = j;
        }
    }
    if end == 0 {
        return None;
    }
    trimmed[..end].parse::<f64>().ok()
}

/// Port of C `atof` (Corpus.cpp:32 classIndex parse): same prefix scan as
/// [`parse_double_prefix`], but never fails -- no valid prefix yields 0.0.
fn atof(s: &str) -> f64 {
    parse_double_prefix(s).unwrap_or(0.0)
}

/// `istringstream >> double` (Corpus.cpp:82-84 weight parse). NOT a prefix
/// parse: num_get stage 2 accumulates every char from the float atom set --
/// digits, a-f/A-F, x/X, e/E, p/P (hexfloat chars!), sign only when first or
/// right after an exponent char, and AT MOST ONE '.' (a second dot stops
/// accumulation) -- then stage 3 must consume the WHOLE accumulated string
/// or the extraction fails. So "0.5abc" FAILS (a/b/c accumulate, strtod
/// can't consume them) while "0.5z" extracts 0.5 ('z' is not an atom).
/// Out-of-range (ERANGE, e.g. "1e999") sets failbit in C++11 -> None.
/// Behavior table pinned in `iss_extract_matches_istringstream_oracle`
/// against compiled `istringstream` probes on the oracle env (Apple libc++).
/// Not reproduced (cannot occur in real listings): stage-3 hexfloat
/// ("0x1p3") and the subnormal-underflow ERANGE corner.
fn iss_extract_double(s: &str) -> Option<f64> {
    let mut acc = String::new();
    let mut seen_dot = false;
    for c in s.trim_start().chars() {
        let ok = match c {
            // a-f covers 'e' (the exponent char is also a hex digit)
            '0'..='9' | 'a'..='f' | 'A'..='F' | 'x' | 'X' | 'p' | 'P' => true,
            '+' | '-' => {
                acc.is_empty() || matches!(acc.chars().last(), Some('e' | 'E' | 'p' | 'P'))
            }
            '.' => !seen_dot,
            _ => false,
        };
        if !ok {
            break;
        }
        if c == '.' {
            seen_dot = true;
        }
        acc.push(c);
    }
    match acc.parse::<f64>() {
        Ok(v) if v.is_finite() => Some(v),
        _ => None,
    }
}

/// `istringstream >> int` (Corpus.cpp:87-89 fileId parse): stage-2 atom set
/// for a decimal int is digits plus a sign in first position only ('x', '.'
/// etc. stop accumulation: "7x" -> 7, "7.5" -> 7, "0x10" -> 0); whole-string
/// stage-3 parse, overflow -> failbit (C++11) -> None.
fn iss_extract_int(s: &str) -> Option<i32> {
    let mut acc = String::new();
    for c in s.trim_start().chars() {
        let ok = match c {
            '0'..='9' => true,
            '+' | '-' => acc.is_empty(),
            _ => false,
        };
        if !ok {
            break;
        }
        acc.push(c);
    }
    acc.parse::<i32>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn atof_matches_c_atof_oracle() {
        // Cases + expected values captured from a compiled `atof` call
        // (glibc/libSystem `<cstdlib>`), covering the shapes Corpus.cpp:32
        // feeds it (classIndex token text).
        let cases: &[(&str, f64)] = &[
            ("0", 0.0),
            ("-1", -1.0),
            ("3", 3.0),
            ("3.5", 3.5),
            ("  42", 42.0),
            ("1e3", 1000.0),
            ("-2.5e-1", -0.25),
            ("abc", 0.0),
            ("", 0.0),
            ("1.2.3", 1.2),
            ("+5", 5.0),
            (".5", 0.5),
            ("5.", 5.0),
            ("1e", 1.0),
            ("1e+", 1.0),
        ];
        for (input, expected) in cases {
            assert_eq!(atof(input), *expected, "atof({input:?})");
        }
    }

    #[test]
    fn iss_extract_matches_istringstream_oracle() {
        // Pinned against compiled `istringstream >> double` / `>> int`
        // probes (Apple libc++, the oracle env).
        let dbl_cases: &[(&str, Option<f64>)] = &[
            ("0.5abc", None),
            ("abc", None),
            (".5", Some(0.5)),
            ("5.", Some(5.0)),
            (" 2 ", Some(2.0)),
            ("1e999", None),
            ("-.5x", None),
            ("+", None),
            ("0.5z", Some(0.5)),
            ("0.5 abc", Some(0.5)),
            ("0.5x", None),
            ("2,", Some(2.0)),
            ("1e3", Some(1000.0)),
            ("1e", None),
            ("1e+", None),
            ("1.2.3", Some(1.2)),
            ("0.5-", Some(0.5)),
            ("0.5+2", Some(0.5)),
            ("1d", None),
            ("1f", None),
        ];
        for (input, expected) in dbl_cases {
            assert_eq!(iss_extract_double(input), *expected, "double {input:?}");
        }
        let int_cases: &[(&str, Option<i32>)] = &[
            ("7x", Some(7)),
            ("7.5", Some(7)),
            ("-3", Some(-3)),
            ("x7", None),
            ("99999999999", None),
            ("7-3", Some(7)),
            ("7+", Some(7)),
            ("--5", None),
            ("+7", Some(7)),
            (" 8 ", Some(8)),
            ("0x10", Some(0)),
        ];
        for (input, expected) in int_cases {
            assert_eq!(iss_extract_int(input), *expected, "int {input:?}");
        }
    }

    fn write_temp(dir: &std::path::Path, name: &str, contents: &str) -> String {
        let path = dir.join(name);
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn mapping_parse() {
        let tmp = tempfile::tempdir().unwrap();
        // Blank line and the fieldless "spa;mx" (no class token, tokens.size()
        // == 2) and bare "gibberish" (tokens.size() == 1, no dial/class) lines
        // exercise the mapping parser's own field-count guards (Corpus.cpp:
        // 25-37); "spa;mx" inserts the "mx" dialect with classIndex staying
        // at its -1 default, "gibberish" inserts dial "" with classIndex -1.
        let mapping = write_temp(
            tmp.path(),
            "mapping.csv",
            "eng;us;0\n\nfra;unk;1\nspa;mx\ngibberish\n",
        );
        // splitstr("b.wav;;fra;;;", ';') yields 5 tokens (one trailing empty
        // dropped, not all): ["b.wav","","fra","",""] -- dial (token 3) is
        // empty so it keeps its "unk" listing-branch default.
        let listing = write_temp(
            tmp.path(),
            "listing.csv",
            "a.wav;;eng;us;;\nb.wav;;fra;;;\n",
        );
        let mut map = IndexMap::new();
        map.insert("language2classmapping".to_string(), mapping);
        map.insert("fileslisting".to_string(), listing);
        let corpus = Corpus::from_config(&map).unwrap();

        assert_eq!(corpus.nb_of_files(), 2);
        assert_eq!(corpus.item(0).class_index, 0);
        assert_eq!(corpus.item(1).dialect, "unk");
        assert_eq!(corpus.item(1).class_index, 1);
    }

    #[test]
    fn listing_nested_quirk() {
        // The nesting in Corpus.cpp:74-96 gates PURELY on `tokens.size() >
        // N` at each level; `tokens[k].size() > 0` (:75,77,79,81,86) is a
        // SIBLING check that only decides whether to overwrite that one
        // field's default -- it does NOT gate entry into the next
        // `tokens.size() > N+1` block (verified against a compiled
        // reimplementation of :74-96). So an empty dial token (index 3)
        // leaves `dial` at its "unk" default but does NOT block weight/
        // fileId parsing, since `tokens.size() > 4` is a sibling of
        // `tokens[3].size() > 0`, not nested inside it.
        let tmp = tempfile::tempdir().unwrap();
        let mapping = write_temp(tmp.path(), "mapping.csv", "eng;;0\n");
        // tokens: ["f.wav", "", "eng", "", "0.5"] -- dial (token 3) empty,
        // weight (token 4) "0.5" IS still parsed.
        let listing = write_temp(tmp.path(), "listing.csv", "f.wav;;eng;;0.5\n");
        let mut map = IndexMap::new();
        map.insert("language2classmapping".to_string(), mapping);
        map.insert("fileslisting".to_string(), listing);
        let corpus = Corpus::from_config(&map).unwrap();

        assert_eq!(corpus.nb_of_files(), 1);
        let item = corpus.item(0);
        assert_eq!(item.dialect, "unk", "dial token empty -> default kept");
        assert_eq!(
            item.weight, 0.5,
            "weight parse NOT gated by dial's emptiness"
        );
        assert_eq!(
            item.file_id, 1,
            "no 6th token present -> fileId default kept"
        );
    }

    #[test]
    fn listing_weight_parse_failure_keeps_default() {
        // istringstream >> double accept-on-success: non-numeric weight
        // token leaves the 1.0 default untouched.
        let tmp = tempfile::tempdir().unwrap();
        let mapping = write_temp(tmp.path(), "mapping.csv", "eng;unk;0\n");
        let listing = write_temp(tmp.path(), "listing.csv", "f.wav;;eng;;notanumber;7\n");
        let mut map = IndexMap::new();
        map.insert("language2classmapping".to_string(), mapping);
        map.insert("fileslisting".to_string(), listing);
        let corpus = Corpus::from_config(&map).unwrap();

        let item = corpus.item(0);
        assert_eq!(item.weight, 1.0);
        assert_eq!(item.file_id, 7);
    }

    #[test]
    fn unknown_language_mutates_mapping() {
        let tmp = tempfile::tempdir().unwrap();
        let mapping = write_temp(tmp.path(), "mapping.csv", "eng;;0\n");
        let listing = write_temp(
            tmp.path(),
            "listing.csv",
            "a.wav;;zzz;;1.0;1\nb.wav;;zzz;;1.0;1\n",
        );
        let mut map = IndexMap::new();
        map.insert("language2classmapping".to_string(), mapping);
        map.insert("fileslisting".to_string(), listing);
        let corpus = Corpus::from_config(&map).unwrap();

        assert_eq!(corpus.item(0).class_index, -1);
        assert_eq!(corpus.item(1).class_index, -1);
        // both files hit the "unknown" -1 class, one via the None-language arm,
        // the second via the now-cached dialect-map entry.
        assert_eq!(*corpus.class_count().get(&-1).unwrap(), 2);
    }

    #[test]
    fn no_listing_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let mapping = write_temp(tmp.path(), "mapping.csv", "unk;unk;-1\n");
        let mut map = IndexMap::new();
        map.insert("language2classmapping".to_string(), mapping);
        map.insert("files".to_string(), "a.wav,b.wav".to_string());
        map.insert("refsegfiles".to_string(), "a.seg,b.seg".to_string());
        let corpus = Corpus::from_config(&map).unwrap();

        assert_eq!(corpus.nb_of_files(), 2);
        let item0 = corpus.item(0);
        assert_eq!(item0.file_name, "a.wav");
        assert_eq!(item0.ref_seg, "a.seg");
        // legacy quirk (Corpus.cpp:44-56): reflangfiles/refdialfiles are ABSENT
        // here, so get_list's 4-arg overload pads with ITS OWN default value
        // "" (not "unk"), and that padded vector's size always equals
        // filesNames.size() -- so the loop's local "unk" initializer (:51-52)
        // never fires; lang/dial come out "", not "unk".
        assert_eq!(item0.language, "");
        assert_eq!(item0.dialect, "");
        assert_eq!(item0.weight, 1.0);
        assert_eq!(item0.file_id, 1);
    }

    #[test]
    fn no_listing_fallback_unk_when_reflangfiles_short() {
        // The loop's local "unk" default (Corpus.cpp:51-52) only fires when
        // `ii >= refLangFiles.size()`, which requires the KEY to be PRESENT
        // but its comma-split shorter than `files` (a present-but-empty
        // vector is never resized further by get_list's 4-arg overload).
        let tmp = tempfile::tempdir().unwrap();
        let mapping = write_temp(tmp.path(), "mapping.csv", "unk;unk;-1\neng;;0\n");
        let mut map = IndexMap::new();
        map.insert("language2classmapping".to_string(), mapping);
        map.insert("files".to_string(), "a.wav,b.wav".to_string());
        map.insert("reflangfiles".to_string(), "eng".to_string());
        let corpus = Corpus::from_config(&map).unwrap();

        assert_eq!(corpus.item(0).language, "eng");
        assert_eq!(corpus.item(1).language, "unk");
        // refdialfiles key is absent entirely here (same quirk as
        // no_listing_fallback): the padded-to-"" vector always covers both
        // indices, so dial is "" for both, never "unk".
        assert_eq!(corpus.item(1).dialect, "");
    }

    #[test]
    fn files_list_star_repeater() {
        // Pinned against a compiled transcription of String.hpp:86-127:
        // "x*3,b" -> [x,x,x,b]; "x*" -> [x]; trailing comma drops one empty;
        // whitespace around elements is eaten by read<string>.
        let m: IndexMap<String, String> =
            IndexMap::from([("files".to_string(), "x.wav*3,b.wav".to_string())]);
        assert_eq!(
            get_list_string(&m, "files", ',').unwrap(),
            vec!["x.wav", "x.wav", "x.wav", "b.wav"]
        );
        let m2: IndexMap<String, String> =
            IndexMap::from([("files".to_string(), "x.wav*".to_string())]);
        assert_eq!(get_list_string(&m2, "files", ',').unwrap(), vec!["x.wav"]);
        let m3: IndexMap<String, String> =
            IndexMap::from([("files".to_string(), " a , b ".to_string())]);
        assert_eq!(get_list_string(&m3, "files", ',').unwrap(), vec!["a", "b"]);
        // empty element -> legacy read<string> check/exit(1) -> bail
        let m4: IndexMap<String, String> =
            IndexMap::from([("files".to_string(), "a,,b".to_string())]);
        assert!(get_list_string(&m4, "files", ',').is_err());
    }

    #[test]
    fn class_count_accumulates() {
        let tmp = tempfile::tempdir().unwrap();
        let mapping = write_temp(tmp.path(), "mapping.csv", "eng;unk;0\nfra;unk;1\n");
        let listing = write_temp(
            tmp.path(),
            "listing.csv",
            "a.wav;;eng;;1.0;1\nb.wav;;eng;;1.0;1\nc.wav;;fra;;1.0;1\n",
        );
        let mut map = IndexMap::new();
        map.insert("language2classmapping".to_string(), mapping);
        map.insert("fileslisting".to_string(), listing);
        let corpus = Corpus::from_config(&map).unwrap();

        assert_eq!(*corpus.class_count().get(&0).unwrap(), 2);
        assert_eq!(*corpus.class_count().get(&1).unwrap(), 1);
        assert_eq!(*corpus.language_count().get("eng").unwrap(), 2);
        assert_eq!(*corpus.language_count().get("fra").unwrap(), 1);
    }
}
