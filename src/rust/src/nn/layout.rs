//! The weight layout: the ordered block list of a pack (issue #27).
//!
//! Port-only, NO legacy source. A pack is the flat f64 vector [`super::blstm::BlstmNetwork::set_weights`]
//! consumes; a BLOCK is one named matrix or vector inside it, at an offset, with the shape
//! the flat run stores it in (ROW-major: output unit outer, source index inner -- what every
//! `for_each_slot` walk and `fill_col_major` call amounts to once the stored transposition is
//! absorbed) and the structural fans of the linear map it belongs to.
//!
//! The layout is COMPUTED FROM THE LAYERS, never transcribed: each layer type exposes a
//! `weight_layout()` that lists its blocks in the order its own `set_weights` consumes them,
//! and [`super::blstm::BlstmNetwork::weight_layout`] concatenates the stacks in the order
//! its own `set_weights` threads them. The per-cell pins in `cells/` assert, with a one-hot
//! probe per element, that each block's offset is where `set_weights` really lands it; the
//! per-net pin asserts `sum(len) == nb_of_weights`. Python reads the list through
//! `speech_rs.weight_layout` and draws per block kind (ADR-0005); it holds no offset
//! arithmetic and no per-cell geometry.
//!
//! What is deliberately NOT a field: the `adim_coeff` (applied once in `config.rs` at the
//! legacy config seam, not a layout fact) and a dead-block flag (a derived gradient fact,
//! pinned by tests, not a structural one).
//!
//! Exact-tree discipline: every `weight_layout` method is INTROSPECTION -- it reads layer
//! sizes and writes nothing -- the ADR-0002 touch class this issue adds.

/// What a block is, for a draw rule: a dense linear map (`Weight`), its additive term
/// (`Bias`), an elementwise scale (`Gain`), the legacy LSTM's 12-column peephole bundle
/// (`Peephole`), or a normalize-tail statistic (`Stat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    Weight,
    Bias,
    Gain,
    Peephole,
    Stat,
}

impl BlockKind {
    /// The seam spelling (`speech_rs.weight_layout`'s `kind` field).
    pub fn as_str(self) -> &'static str {
        match self {
            BlockKind::Weight => "weight",
            BlockKind::Bias => "bias",
            BlockKind::Gain => "gain",
            BlockKind::Peephole => "peephole",
            BlockKind::Stat => "stat",
        }
    }
}

/// Which sub-network of the pack a block sits in, in `set_weights` order: the forward
/// recurrent stack, the backward one (bidirectional only), the output MLP, the normalize
/// tail (`mean` then `std`, each `input_size` wide; present in MLP mode too, sized by the
/// output net's input width -- `BlstmNetwork::nb_of_weights` counts it there as well).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stack {
    Forward,
    Backward,
    Output,
    Tail,
}

impl Stack {
    /// The seam spelling (`speech_rs.weight_layout`'s `stack` field).
    pub fn as_str(self) -> &'static str {
        match self {
            Stack::Forward => "forward",
            Stack::Backward => "backward",
            Stack::Output => "output",
            Stack::Tail => "tail",
        }
    }
}

/// One block of ONE layer, before the net-level walk assigns its stack, layer and offset.
///
/// `shape` is the row-major STORAGE shape of the flat run (`[rows, cols]` for a matrix,
/// `[n]` for a vector); `fan_in` / `fan_out` are the structural fans of the linear map the
/// block belongs to (a bias carries its map's fans; a gain or a statistic carries its width
/// on both sides). `columns` names the peephole bundle's 12 columns and nothing else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerBlock {
    pub name: String,
    pub kind: BlockKind,
    pub shape: Vec<usize>,
    pub fan_in: usize,
    pub fan_out: usize,
    pub columns: Option<Vec<&'static str>>,
}

impl LayerBlock {
    /// A row-major `rows x cols` matrix block.
    pub fn matrix(
        name: impl Into<String>,
        kind: BlockKind,
        rows: usize,
        cols: usize,
        fan_in: usize,
        fan_out: usize,
    ) -> LayerBlock {
        LayerBlock {
            name: name.into(),
            kind,
            shape: vec![rows, cols],
            fan_in,
            fan_out,
            columns: None,
        }
    }

    /// An `n`-wide vector block.
    pub fn vector(
        name: impl Into<String>,
        kind: BlockKind,
        n: usize,
        fan_in: usize,
        fan_out: usize,
    ) -> LayerBlock {
        LayerBlock {
            name: name.into(),
            kind,
            shape: vec![n],
            fan_in,
            fan_out,
            columns: None,
        }
    }

    /// The element count: the product of `shape`.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// `true` for a zero-element block (an empty Mamba adapter, a 0-wide layer).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One block of a whole pack: a [`LayerBlock`] placed at `offset` in `stack` / `layer`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightBlock {
    pub stack: Stack,
    /// The layer index inside its stack (0 for the tail's two blocks).
    pub layer: usize,
    pub block: LayerBlock,
    pub offset: usize,
}

impl WeightBlock {
    /// The element count (the block's).
    pub fn len(&self) -> usize {
        self.block.len()
    }

    /// `true` for a zero-element block.
    pub fn is_empty(&self) -> bool {
        self.block.is_empty()
    }

    /// The half-open flat range `[offset, offset + len)`.
    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.offset + self.len()
    }
}

/// The `(name, [lo, hi))` ranges of one layer's blocks, offsets accumulated from 0 in
/// list order -- what a per-cell pin reads to place its one-hot probes.
pub fn block_ranges(blocks: &[LayerBlock]) -> Vec<(String, std::ops::Range<usize>)> {
    let mut offset = 0usize;
    blocks
        .iter()
        .map(|b| {
            let r = offset..offset + b.len();
            offset += b.len();
            (b.name.clone(), r)
        })
        .collect()
}

/// Place `layers`' blocks (one `Vec<LayerBlock>` per layer, in `set_weights` order) into
/// `stack`, continuing from `*offset`, which is advanced past them.
pub(crate) fn place(
    out: &mut Vec<WeightBlock>,
    stack: Stack,
    layers: impl IntoIterator<Item = Vec<LayerBlock>>,
    offset: &mut usize,
) {
    for (layer, blocks) in layers.into_iter().enumerate() {
        for block in blocks {
            let len = block.len();
            out.push(WeightBlock {
                stack,
                layer,
                block,
                offset: *offset,
            });
            *offset += len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn place_assigns_contiguous_offsets_per_layer_in_order() {
        let l0 = vec![
            LayerBlock::matrix("a", BlockKind::Weight, 2, 3, 3, 2),
            LayerBlock::vector("b", BlockKind::Bias, 2, 3, 2),
        ];
        let l1 = vec![LayerBlock::vector("c", BlockKind::Gain, 4, 4, 4)];
        let mut out = Vec::new();
        let mut off = 5;
        place(&mut out, Stack::Output, [l0, l1], &mut off);
        let got: Vec<(usize, &str, usize, usize)> = out
            .iter()
            .map(|b| (b.layer, b.block.name.as_str(), b.offset, b.len()))
            .collect();
        assert_eq!(got, vec![(0, "a", 5, 6), (0, "b", 11, 2), (1, "c", 13, 4)]);
        assert_eq!(off, 17);
        assert_eq!(out[2].range(), 13..17);
        assert!(out.iter().all(|b| b.stack == Stack::Output));
    }

    #[test]
    fn the_seam_spellings_are_lower_case_words() {
        assert_eq!(BlockKind::Peephole.as_str(), "peephole");
        assert_eq!(Stack::Backward.as_str(), "backward");
    }
}
