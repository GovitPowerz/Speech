# ADR-0010: The seam takes a config map, the process cwd is not part of the Python-side contract, and one module owns the fold run

**Status:** accepted | **Date:** 2026-10-07 (issue #22, grilling outcome recorded there); the fast guard retired 2026-10-09 (issue #62)

## Context

Running one fold through `speech_rs.Engine` was a six-step recipe written at eight production
sites and about fifty test sites: render the config dict as text, write it to a file in the
working directory, change directory and build the engine on the file name (the seam accepted
paths only), force `Neural_Networks_BackPropagation_Epochs 0` (the F11 rule), skip
`set_weights` on `Inference_Path fast`, read the results back. Two of the six steps had already
drifted between copies (`evaluate` did not force the epoch count until #45; the checkpoint
naming disagreed until #29), and deleting any one copy made the recipe reappear at that
caller. The `chdir` also made the Python side depend on process-global state, so two engines in
one process could not safely live in two directories.

## Decision

Three things, each owned once.

- **The seam takes a map.** `Engine.from_map(configs: list[dict[str, str]], mode)` mirrors the
  path constructor; the map is the flattened legacy key shape the `KEY_TABLE` already produces
  (ADR-0006, no second vocabulary) and dict insertion order is preserved into the `IndexMap`.
  One `load_config_map(path)` in `speech-py` replaces the three read-and-dispatch copies behind
  `Engine`, `StreamingSession` and `StreamingLidSession`. Rust path semantics are unchanged: a
  relative path still resolves against the process cwd, as for the binary.
- **The process cwd is not part of the Python-side contract.** `fold_run.FoldRun` resolves the
  six config keys that name a file or directory (`fileslisting`, `language2classmapping`,
  `BLSTM_weightsFile`, `BLSTM_LID_weightsFile`, `Dump_Directory`,
  `multiConfigResultsOutputFile`) against the run's working directory before the map crosses
  the seam; an absent `.mat` key defaults to the working directory's `MultiConfigResults.mat`,
  an absent or empty `Dump_Directory` stays as given (empty is the engine's no-dump
  sentinel). Production listings carry absolute rows (`dataprep/lre.py`), so nothing under
  `src/python/speech/` changes directory or writes a
  config file to run the engine. A committed fixture with relative listing rows keeps its cwd in
  the tests that use it.
- **One module owns the fold run.** `FoldRun(cfg, workdir, *, backprop, listing=None)` is built
  once per listing and `run(weights=None) -> FoldResult` is one fold at the given weights (the
  modern loop reuses one engine across all its SMORMS3 steps). It owns the backprop flags (SAD, plus
  LID on the Twin), the F11 rule unconditionally, the listing override, the path resolution,
  the fast guard (on `Inference_Path fast` the arrays are written as workdir packs and the weight
  keys repointed; `set_weights` is never called there), and the rendering (`config`,
  `config_text` through the one `_config_text`, workdir-independent).
  **Amendment (2026-10-09, issue #62):** the fast guard is gone. It existed because a fast
  processor had no settable weight surface (T6b made `set_weights` bail there), and the packs
  it wrote entered through the drivers' FILE seam, which reads an over-long pack head-first:
  a pack the exact tree refuses ran on fast. The fast drivers now take `set_weights` under the
  exact tree's exact-length contract, in the same words, so `run(weights)` is `set_weights`
  plus `run` on one engine on both trees and nothing is written to the workdir. `FoldResult` carries the
  channel results, the per-net derivatives (empty on a forward-only fold), the post-run weights,
  and over config 0 the costs `ComputeGradient.m` assembles (`nn_cost_seg`, `nn_cost_lid`,
  `nn_cost`). `engine.forward_backward` takes a `FoldRun`; the L2 term stays there. The
  parameter is `backprop: bool`, not a mode literal: it controls exactly one thing, and `Mode`
  already names the CLI flag axis in Rust.

Checkpoints stay outside the module: a `FoldRun` takes weights, a checkpoint is a directory of
packs (CONTEXT.md), and `resolve_checkpoint_packs` is the one resolver of the two naming
schemes; `evaluate(state, packs, scores_dir)` takes the packs.

## Consequences

- The eight production sites (`drivers/train.py`, `drivers/test.py`, `drivers/baseline.py`)
  call one implementation; the overlay builders (`_eval_overlay`, `_hyperparam_overlay`) return
  dicts and know nothing about flags, epochs or files.
- A mutation that removes the Epochs-0 forcing fails committed tests at the map
  (`tests/test_fold_run.py`) and on the engine (`tests/pyo3/test_fold_run.py`, which also pins
  what a gradient fold does to the engine's own pack: `run_solo` ends with the bag's iRPROP-
  step, so `FoldResult.weights` is one engine step past theta on a gradient fold).
- The exit gates and the subset gates are unchanged in outcome; the exact-tree goldens are
  byte-green (`speech-py` and the Python side are outside the frozen trees, ADR-0002).
- Every `forward_backward` caller moved with it, the gate preflights and the exit gate's own
  builders included. The raw `speech_rs.Engine(` test sites stay: `test_engine_smoke`,
  `test_seam_replay` and `test_channel_results` pin the path constructor itself, and the
  recipe-shaped helpers around the others (`test_phase5_init_weights_pyo3`,
  `test_phase11_init_pyo3`) are a follow-up; the module is the place they move to.
