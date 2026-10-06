//! Importer for the legacy whitespace .config format.
//!
//! Ported from legacy C++: ConfigFile.cpp. KEY value, last-wins, # comments.
//! The 0a keys do not use the _-continuation rule, so a simple split suffices.
//! Also hosts the shared `conf.get<T>` / `get_list<T>` readers (issues #32, #47, #50):
//! the f64 scalar family over [`parse_finite_f64`], and the one list grammar
//! ([`split_list`]) every comma-separated config list goes through.

use anyhow::{Result, anyhow, bail};
use indexmap::IndexMap;

/// `read<double>` on one config token: a finite f64, or an error naming the key and the
/// text. Non-finite is malformed: Rust parses `nan`/`inf`/`1e400`, the legacy `operator>>`
/// rejects them.
pub(crate) fn parse_finite_f64(s: &str, key: &str) -> Result<f64> {
    match s.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => Ok(v),
        Ok(v) => bail!("cannot read '{s}' as f64 for '{key}': non-finite ({v})"),
        Err(e) => bail!("cannot read '{s}' as f64 for '{key}': {e}"),
    }
}

/// `read<size_type>` on one config token: a whole-string (trimmed) `usize`, or an error
/// naming the key and the text.
pub(crate) fn parse_usize(s: &str, key: &str) -> Result<usize> {
    s.trim()
        .parse::<usize>()
        .map_err(|e| anyhow!("cannot read '{s}' as usize for '{key}': {e}"))
}

/// `conf.get<double>(name)` (required): missing key -> error naming the key; present but
/// malformed or non-finite -> error naming the key and the text.
pub(crate) fn get_f64(map: &IndexMap<String, String>, key: &str) -> Result<f64> {
    map.get(key)
        .ok_or_else(|| anyhow!("param '{key}' not found in config"))
        .and_then(|s| parse_finite_f64(s, key))
}

/// `conf.get<double>(name, default)`: missing key -> `default`; present but malformed or
/// non-finite -> error (issues #32, #47).
pub(crate) fn get_f64_default(
    map: &IndexMap<String, String>,
    key: &str,
    default: f64,
) -> Result<f64> {
    map.get(key)
        .map_or(Ok(default), |s| parse_finite_f64(s, key))
}

/// `conf.get<double>(name)` as `Option`: missing key -> `None`, present -> `Some`
/// (`Audio_fixed_gain`, where absence selects a MODE, not a default value).
pub(crate) fn get_f64_opt(map: &IndexMap<String, String>, key: &str) -> Result<Option<f64>> {
    map.get(key).map(|s| parse_finite_f64(s, key)).transpose()
}

/// Legacy `splitstr(line, delim)` (String.hpp:110-114): repeated `getline(ss, item,
/// delim)`, which returns false exactly when called with NOTHING left to read, so a
/// delimiter as the LAST character is consumed to end the preceding field but
/// produces no further (empty) token: AT MOST ONE trailing empty field is dropped --
/// "a;b;" -> ["a","b"], "a;b;;" -> ["a","b",""] (verified against a real `getline`
/// harness). No delimiter -> the whole line as one token; an empty line -> no token.
pub(crate) fn splitstr(line: &str, delim: char) -> Vec<String> {
    if line.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<String> = line.split(delim).map(String::from).collect();
    if line.ends_with(delim) {
        out.pop();
    }
    out
}

/// The one list grammar, the legacy `split_with_repeat<T>` (String.hpp:116-127) with
/// the element parser as an argument (issue #50): the value splits on `,` by
/// [`splitstr`] (empty value -> `[]`, one trailing empty piece dropped), each piece
/// splits on the `*` repeater (`2*3` -> three copies of `2`, `2*` -> one), and each
/// element goes through `parse(element, key)`. An empty piece (`1,,2`, `*3`) reaches
/// the element parser and errors as the legacy `read<T>` exits. The count is trimmed
/// like the element (the legacy first-token read makes ` 2*2 ` read; `2* 3` is a
/// leniency over its `lexical_cast`). Two deliberate tightenings: a bad repeat count
/// (`2*x`, an uncaught throw in the legacy) and a third `*` part (`2*3*4`, the legacy
/// ignores the `4`) are errors naming the key and the piece.
pub(crate) fn split_list<T: Clone>(
    value: &str,
    key: &str,
    parse: impl Fn(&str, &str) -> Result<T>,
) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for piece in splitstr(value, ',') {
        out.extend(expand_repeat(&piece, key, &parse)?);
    }
    Ok(out)
}

/// One comma piece of [`split_list`]: the `*` repeater, `count` copies of the parsed
/// element.
pub(crate) fn expand_repeat<T: Clone>(
    piece: &str,
    key: &str,
    parse: impl Fn(&str, &str) -> Result<T>,
) -> Result<Vec<T>> {
    let parts = splitstr(piece, '*');
    let count = match parts.len() {
        0 | 1 => 1,
        2 => parts[1]
            .trim()
            .parse::<usize>()
            .map_err(|e| anyhow!("bad repeat count in '{piece}' for '{key}': {e}"))?,
        _ => bail!("bad repeat form '{piece}' for '{key}'"),
    };
    let element = parse(parts.first().map_or("", String::as_str), key).map_err(|e| {
        if parts.len() == 2 {
            anyhow!("{e} (in '{piece}')")
        } else {
            e
        }
    })?;
    Ok(vec![element; count])
}

/// `conf.get_list<double>(name)` on a required key.
pub(crate) fn get_f64_list(map: &IndexMap<String, String>, key: &str) -> Result<Vec<f64>> {
    map.get(key)
        .ok_or_else(|| anyhow!("param '{key}' not found in config"))
        .and_then(|v| split_list(v, key, parse_finite_f64))
}

/// `conf.get_list<size_type>(name)` on a required key.
pub(crate) fn get_usize_list(map: &IndexMap<String, String>, key: &str) -> Result<Vec<usize>> {
    map.get(key)
        .ok_or_else(|| anyhow!("param '{key}' not found in config"))
        .and_then(|v| split_list(v, key, parse_usize))
}

/// Parse legacy KEY value config text; last duplicate key wins, # lines skipped.
pub fn parse_legacy_config(text: &str) -> IndexMap<String, String> {
    let mut out = IndexMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.splitn(2, char::is_whitespace);
        let name = it.next().unwrap_or("");
        if name.starts_with('#') || name.is_empty() {
            continue;
        }
        let val = it.next().unwrap_or("").trim().to_string();
        out.insert(name.to_string(), val);
    }
    out
}

#[cfg(test)]
mod f64_reader_tests {
    use super::*;

    fn map(key: &str, value: &str) -> IndexMap<String, String> {
        let mut m = IndexMap::new();
        m.insert(key.to_string(), value.to_string());
        m
    }

    /// Issue #32: missing -> default; present but malformed or non-finite -> error naming
    /// the key and the text.
    #[test]
    fn f64_default_missing_present_malformed() {
        let m = map("K", "1e-x");
        assert_eq!(get_f64_default(&m, "absent", 0.5).unwrap(), 0.5);
        assert_eq!(get_f64_default(&map("K", "1e-4"), "K", 0.5).unwrap(), 1e-4);
        assert_eq!(get_f64_default(&map("K", " 2 "), "K", 0.5).unwrap(), 2.0);
        let err = get_f64_default(&m, "K", 0.5).unwrap_err().to_string();
        assert!(err.contains("K") && err.contains("1e-x"), "{err}");
        for bad in ["nan", "NaN", "inf", "-infinity", "1e400"] {
            let err = get_f64_default(&map("K", bad), "K", 0.5)
                .unwrap_err()
                .to_string();
            assert!(err.contains("K") && err.contains(bad), "{bad}: {err}");
        }
    }

    /// Issue #50: the required and `Option` readers share the finite parse; a missing
    /// required key errors naming it in the `nn/blstm.rs` wording.
    #[test]
    fn f64_required_and_opt() {
        assert_eq!(get_f64(&map("K", " 2 "), "K").unwrap(), 2.0);
        let err = get_f64(&map("K", "1"), "absent").unwrap_err().to_string();
        assert_eq!(err, "param 'absent' not found in config");
        assert_eq!(get_f64_opt(&map("K", "1"), "absent").unwrap(), None);
        assert_eq!(get_f64_opt(&map("K", "0.5"), "K").unwrap(), Some(0.5));
        for bad in ["nan", "inf", "1e400", "1O", ""] {
            for err in [
                get_f64(&map("K", bad), "K").unwrap_err().to_string(),
                get_f64_opt(&map("K", bad), "K").unwrap_err().to_string(),
            ] {
                assert!(err.contains("K") && err.contains(bad), "{bad:?}: {err}");
            }
        }
    }

    /// Issue #50: the legacy `split_with_repeat` grammar -- one trailing empty piece
    /// dropped, the `*` repeater -- and the deliberate tightenings (an inner empty
    /// piece, a bad or extra repeat part, a non-finite element) each error naming the
    /// key.
    #[test]
    fn list_grammar() {
        let list = |v: &str| get_f64_list(&map("K", v), "K");
        assert_eq!(list("1,2,").unwrap(), vec![1.0, 2.0]);
        assert_eq!(list("2*3").unwrap(), vec![2.0; 3]);
        assert_eq!(list("1, 2*2 ,3").unwrap(), vec![1.0, 2.0, 2.0, 3.0]);
        assert_eq!(list("2*").unwrap(), vec![2.0]);
        assert_eq!(list("2*3*").unwrap(), vec![2.0; 3]);
        assert_eq!(list("").unwrap(), Vec::<f64>::new());
        for bad in ["1,,2", "*3", "2*x", "2*3*4", "1,nan", "1,2,,", "2 3"] {
            let err = list(bad).unwrap_err().to_string();
            assert!(err.contains("K"), "{bad:?}: {err}");
        }
        // an empty piece is the element parser's fault, not the repeater's
        for empty in ["1,,2", "*3"] {
            let err = list(empty).unwrap_err().to_string();
            assert!(err.contains("cannot read ''"), "{empty:?}: {err}");
        }
        let err = list("1,2*x").unwrap_err().to_string();
        assert!(err.contains("2*x"), "{err}");
        let err = list("1,*3").unwrap_err().to_string();
        assert!(
            err.contains("cannot read ''") && err.contains("'*3'"),
            "{err}"
        );
        let err = list("2*3*4").unwrap_err().to_string();
        assert!(err.contains("bad repeat form '2*3*4'"), "{err}");
        let err = get_f64_list(&map("K", "1"), "absent")
            .unwrap_err()
            .to_string();
        assert_eq!(err, "param 'absent' not found in config");
        assert_eq!(
            get_usize_list(&map("K", "0,3*2"), "K").unwrap(),
            vec![0, 3, 3]
        );
        for bad in ["1.5", "-1", "2*x", "1,,2"] {
            let err = get_usize_list(&map("K", bad), "K").unwrap_err().to_string();
            assert!(err.contains("K"), "{bad:?}: {err}");
        }
    }
}
