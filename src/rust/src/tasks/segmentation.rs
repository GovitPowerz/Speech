//! Segmentation boundary-list container, ported from legacy `Segmentation.cpp`.
//!
//! The model is a boundary list: each [`Segment`] stores the `begin` time of an
//! interval and its class; the interval runs until the next segment's `begin`.
//! The list is seeded as `[Other@0.0, End@audio_duration]` and always ends with
//! the `End` sentinel. `label_segment` overwrites a typed interval into the list
//! (`Segmentation.cpp:147-174`); `sanitize` snaps boundaries to the 1e-4 grid and
//! merges adjacent same-type segments (`Segmentation.cpp:176-194`).
//!
//! Legacy uses `std::deque<Segment>` with iterator arithmetic; here the container
//! is a `Vec<Segment>` and iterators become indices. The invariant that maps the
//! two: the list is never empty and always ends with the `End` sentinel, so the
//! last index is `segs.len() - 1`, and "iterator not at the last element"
//! (`it + 1 != end()`) becomes `i + 1 < segs.len()`.

/// Segment class codes. `#[repr(i32)]` values are load-bearing: they must match
/// the legacy `segment_class` enum (`Segmentation.h`) exactly, since they are
/// used as numeric labels in serialized output and count tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum SegClass {
    Other = 0,
    Speech = 1,
    Ring = 2,
    Dtmf0 = 3,
    Dtmf1 = 4,
    Dtmf2 = 5,
    Dtmf3 = 6,
    Dtmf4 = 7,
    Dtmf5 = 8,
    Dtmf6 = 9,
    Dtmf7 = 10,
    Dtmf8 = 11,
    Dtmf9 = 12,
    DtmfA = 13,
    DtmfB = 14,
    DtmfC = 15,
    DtmfD = 16,
    DtmfStar = 17,
    DtmfSharp = 18,
    Insertion = 19,
    Substitution = 20,
    Excluded = 21,
    End = 22,
}

/// A boundary in the segmentation: the interval `[begin, next.begin)` has class
/// `ty`. Mirrors the legacy `Segment { _BeginTime, _Type }`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    pub begin: f64,
    pub ty: SegClass,
}

/// Boundary-list segmentation over `[0.0, audio_duration]`.
pub struct Segmentation {
    segs: Vec<Segment>,
    audio_duration: f64,
}

impl Segmentation {
    /// Seed the list as `[Other@0.0, End@audio_duration]`
    /// (`Segmentation.cpp:64`).
    pub fn new(audio_duration: f64) -> Self {
        Self {
            segs: vec![
                Segment {
                    begin: 0.0,
                    ty: SegClass::Other,
                },
                Segment {
                    begin: audio_duration,
                    ty: SegClass::End,
                },
            ],
            audio_duration,
        }
    }

    pub fn segments(&self) -> &[Segment] {
        &self.segs
    }

    pub fn audio_duration(&self) -> f64 {
        self.audio_duration
    }

    /// Overwrite `[begin, end)` with `class`, splitting/erasing existing
    /// boundaries and re-closing the tail with the last overwritten type.
    /// Direct index-based port of `Segmentation::label_segment`
    /// (`Segmentation.cpp:147-174`).
    ///
    /// Legacy returns the iterator to the final position; that return value is
    /// only consumed by `addPadding` (Task 5), so it is dropped here.
    pub fn label_segment(&mut self, begin: f64, end: f64, class: SegClass) {
        let mut begin = begin;
        let mut previous_type = SegClass::Other;

        // if (endTime > it->_BeginTime)   with it = begin()
        if end <= self.segs[0].begin {
            return;
        }
        // clamp begin up to the list start
        if begin < self.segs[0].begin {
            begin = self.segs[0].begin;
        }
        // if (beginTime < (segmentList.end()-1)->_BeginTime)  -- sentinel begin
        let last = self.segs.len() - 1;
        if begin >= self.segs[last].begin {
            return;
        }

        // Advance to the insertion point.
        // while ((it+1 != end) && (it->_BeginTime < beginTime))
        let mut i = 0usize;
        while i + 1 < self.segs.len() && self.segs[i].begin < begin {
            previous_type = self.segs[i].ty;
            i += 1;
        }

        // it = insert(it, Segment(beginTime, classType)) + 1
        self.segs.insert(i, Segment { begin, ty: class });
        i += 1;

        // Erase overwritten boundaries.
        // while ((it+1 != end) && (it->_BeginTime < endTime))
        while i + 1 < self.segs.len() && self.segs[i].begin < end {
            previous_type = self.segs[i].ty;
            self.segs.remove(i);
        }

        // Re-close the tail. `it` is always < end() here (sentinel present), so
        // the outer `if (it != end)` guard always holds.
        //   if (it+1 != end)  -> insert unconditionally
        //   else (it == last) -> insert only if endTime < sentinel begin
        // Both branches insert the same segment, so they collapse to one guard.
        let not_at_sentinel = i + 1 < self.segs.len();
        if not_at_sentinel || end < self.segs[self.segs.len() - 1].begin {
            self.segs.insert(
                i,
                Segment {
                    begin: end,
                    ty: previous_type,
                },
            );
        }
    }

    /// Snap each non-terminal boundary to the 1e-4 grid and merge adjacent
    /// same-type segments. Direct port of `Segmentation::sanitize`
    /// (`Segmentation.cpp:176-194`).
    ///
    /// Legacy rounds with `boost::math::round`, which is round-half-away-from-
    /// zero; Rust's `f64::round` has the same tie-breaking rule, so
    /// `(t * 1e4).round() / 1e4` is a faithful port. The loop stops when
    /// `it_next` reaches the sentinel, so the terminal `End` boundary is never
    /// rounded.
    pub fn sanitize(&mut self) {
        let mut i = 0usize;
        // while ((it != end) && (it_next != end))  with it_next = it + 1
        while i < self.segs.len() && i + 1 < self.segs.len() {
            self.segs[i].begin = (self.segs[i].begin * 1.0e4).round() / 1.0e4;
            if self.segs[i].ty == self.segs[i + 1].ty {
                // erase(it_next); it = it_next - 1  -> i stays put, re-check pair
                self.segs.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }
}
