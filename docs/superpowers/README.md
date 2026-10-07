# Design specs and implementation plans

One dated design spec and one implementation plan per roadmap phase, in the order they were written. Each phase ran the same loop: a brainstormed spec (`specs/`), a task-by-task plan (`plans/`), implementation with a fresh implementer and reviewer per task, a mutation battery, and a whole-branch review before the merge. The **status** column is the document's own `Status` line where it has one; every phase listed here shipped, and `main` carries its merge. What each phase delivered is summarized in [docs/ROADMAP.md](../ROADMAP.md); its numbers are in [RESULTS.md](../../RESULTS.md); the legacy behaviours it reproduced or fixed are in [IMPROVEMENTS.md](../../IMPROVEMENTS.md); the decisions that still constrain new work are the [ADRs](../adr/).

## Roadmap 1: the port (legacy-truth, closed at tag `legacy-parity-v1`)

| phase | spec | plan | status |
|---|---|---|---|
| scaffolding | [Speech repo setup](specs/2026-07-01-speech-repo-setup-design.md) | [plan](plans/2026-07-01-speech-repo-scaffolding.md) | approved |
| 0a | [Byte-parity foundation](specs/2026-07-01-phase-0a-byte-parity-design.md) | [plan](plans/2026-07-01-phase-0a-byte-parity.md) | approved |
| 0b-i | [Cost laws + Segmentation container](specs/2026-07-01-phase-0b-i-cost-container-design.md) | [plan](plans/2026-07-01-phase-0b-i-cost-container.md) | approved |
| 0b-ii | [Segmenter decision + I/O + scoring](specs/2026-07-01-phase-0b-ii-segmenter-scoring-design.md) | [plan](plans/2026-07-01-phase-0b-ii-segmenter-scoring.md) | approved |
| 1 | [Features](specs/2026-07-02-phase-1-features-design.md) | [plan](plans/2026-07-02-phase-1-features.md) | approved |
| 2 | [NN forward](specs/2026-07-02-phase-2-nn-forward-design.md) | [plan](plans/2026-07-02-phase-2-nn-forward.md) | approved |
| 2b | [Segmenter wiring](specs/2026-07-03-phase-2b-segmenter-wiring-design.md) | [plan](plans/2026-07-03-phase-2b-segmenter-wiring.md) | approved |
| 3 | [Training: network-level BPTT + iRPROP-](specs/2026-07-06-phase-3-training-design.md) | [plan](plans/2026-07-06-phase-3-training.md) | - |
| 4a | [Engine drivers: Corpus / BagOfProcessors / CorpusProcessor / CLI / .mat](specs/2026-07-06-phase-4a-engine-drivers-design.md) | [plan](plans/2026-07-06-phase-4a-engine-drivers.md) | approved |
| 4b | [LID: Algo 5/6 + VRCTS adapter + LID corpus integration](specs/2026-07-07-phase-4b-lid-design.md) | [plan](plans/2026-07-07-phase-4b-lid.md) | approved |
| 4c | [PyO3 seam + Python optimizer + TOML canonical config](specs/2026-07-08-phase-4c-pyo3-optimizer-design.md) | [plan](plans/2026-07-08-phase-4c-pyo3-optimizer.md) | approved |
| 4d | [dataprep + end-to-end parity](specs/2026-07-09-phase-4d-dataprep-endtoend-design.md) | [plan](plans/2026-07-09-phase-4d-dataprep-endtoend.md) | - |

## Roadmap 2: the post-port era (port-truth)

| phase | spec | plan | status |
|---|---|---|---|
| 5 | [Un-quirk + training foundation](specs/2026-07-10-phase-5-unquirk-training-foundation-design.md) | [plan](plans/2026-07-10-phase-5-unquirk-training-foundation.md), [fix list](plans/2026-07-10-phase5-fixlist.md) | - |
| 6 | [Baseline training on real data](specs/2026-07-17-phase-6-baseline-training-design.md) | [plan](plans/2026-07-17-phase-6-baseline-training.md) | - |
| 7 | [The efficient binary (offline)](specs/2026-07-18-phase-7-efficient-binary-design.md) | [plan](plans/2026-07-18-phase-7-efficient-binary.md) | - |
| 8 | [Online/streaming mode](specs/2026-07-19-phase-8-streaming-design.md) | [plan](plans/2026-07-19-phase-8-streaming.md) | - |
| 9 | [New architectures: Mamba + sLSTM behind a cell abstraction](specs/2026-07-27-phase-9-new-architectures-design.md) | [plan](plans/2026-07-27-phase-9-new-architectures.md) | approved |
| 10 | [CfC + fast-tree completion + the v2 lineage](specs/2026-07-31-phase-10-cfc-fast-completion-design.md) | [plan](plans/2026-07-31-phase-10-cfc-fast-completion.md) | approved |
| 11 | [The transformer encoder: windowed causal attention behind the CellLayer seam](specs/2026-08-12-phase-11-transformer-design.md) | [plan](plans/2026-08-12-phase-11-transformer.md) | - |

## Issue-driven work

Smaller changes start from a GitHub issue instead of a spec: the issue carries a grilling session whose outcome is recorded as a comment, and the plan cites that comment as its source of truth.

| issue | plan | status |
|---|---|---|
| [#23 Typed channel result: one owner for the 18+N result-row columns](https://github.com/GovitPowerz/Speech/issues/23) | [plan](plans/2026-10-07-issue-23-typed-channel-result.md) | implemented |
| [#22 Fold run: one module for running the engine through the seam](https://github.com/GovitPowerz/Speech/issues/22) | none: implemented from the grilling comment, recorded in [ADR-0010](../adr/0010-the-seam-takes-a-config-map-and-one-module-owns-the-fold-run.md) | implemented |

The specs are the record of what was decided and why; where a spec's wording was later found wrong, the correction is dated in place (the Phase 11 spec carries three dated amendments) rather than rewritten, so the trail from premise to result stays readable.
