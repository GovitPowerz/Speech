//! Phase 4b Task 2: Audio LID plumbing (`Audio::lang_index`/`Audio::weight`)
//! and the multi-class corpus fixture.
//!
//! The setter path itself (`bag_of_processors::apply_corpus_item`, called from
//! `segmentation_function` right after `read_audio`) is pinned by an inline
//! `#[cfg(test)]` unit test in `src/engine/bag_of_processors.rs` (least-
//! invasive: it needs `CorpusItem`/`Audio` construction only, no fixture
//! plumbing). This file covers the OTHER half named in the brief: that the new
//! 3-class LID corpus fixture (`languagemapping.csv` + `corpus_lid/
//! fileslisting_lid.csv`) parses through the real `Corpus::from_config` into
//! the class indices/weights the relabeling intends -- the Task 4 driver will
//! later thread these same `CorpusItem`s through `segmentation_function` into
//! real `Audio` values; until then this is the honest "the plumbing's INPUT
//! side is correct" check.

use std::path::PathBuf;

use indexmap::IndexMap;
use speech::engine::corpus::Corpus;

fn ref_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_data/phase4b")
}

// === corpus_lid_listing_yields_lang_classes_and_weights =====================
// The relabeled 3-class listing (f1->fax/non, f2->chi/man, f3->spa/spa) against
// the verbatim-copied legacy languagemapping.csv (fax;non;0 / chi;man;1 /
// spa;spa;2) must yield class_index 0/1/2 in file order, plus the weights/
// file_ids carried in fileslisting_lid.csv (STRICT, per the brief).
#[test]
fn corpus_lid_listing_yields_lang_classes_and_weights() {
    let mut map = IndexMap::new();
    map.insert(
        "language2classmapping".to_string(),
        ref_dir()
            .join("languagemapping.csv")
            .to_str()
            .unwrap()
            .to_string(),
    );
    map.insert(
        "fileslisting".to_string(),
        ref_dir()
            .join("fileslisting_lid.csv")
            .to_str()
            .unwrap()
            .to_string(),
    );

    let corpus = Corpus::from_config(&map).unwrap();
    assert_eq!(corpus.nb_of_files(), 3);

    let f1 = corpus.item(0);
    assert_eq!(f1.file_name, "corpus_lid/f1.wav");
    assert_eq!(f1.language, "fax");
    assert_eq!(f1.dialect, "non");
    assert_eq!(f1.class_index, 0);
    assert_eq!(f1.weight, 0.5);
    assert_eq!(f1.file_id, 2);

    let f2 = corpus.item(1);
    assert_eq!(f2.language, "chi");
    assert_eq!(f2.dialect, "man");
    assert_eq!(f2.class_index, 1);
    assert_eq!(f2.weight, 0.75);
    assert_eq!(f2.file_id, 1, "no 6th token -> default fileId");

    let f3 = corpus.item(2);
    assert_eq!(f3.language, "spa");
    assert_eq!(f3.dialect, "spa");
    assert_eq!(f3.class_index, 2);
    assert_eq!(f3.weight, 1.0, "no weight token -> default 1.0");
    assert_eq!(f3.file_id, 1);

    // Non-vacuity: three distinct classes actually got populated, not e.g. all
    // collapsing to the unknown-language -1 bucket.
    assert_eq!(*corpus.class_count().get(&0).unwrap(), 1);
    assert_eq!(*corpus.class_count().get(&1).unwrap(), 1);
    assert_eq!(*corpus.class_count().get(&2).unwrap(), 1);
    assert!(
        corpus.class_count().get(&-1).is_none(),
        "no unknown-class fallback"
    );
}
