"""Command-line entry for the Python optimizer/orchestrator.

Ported from the legacy MATLAB run scripts (`Init_BLSTM.m` / `Train_BLSTM.m` /
`ReTrain_BLSTM.m` / `Test_BLSTM.m`), which were separate top-level scripts; here they are
`init`/`train`/`retrain`/`test` subcommands dispatching to `speech.drivers`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from speech.drivers.baseline import build_parser as _baseline_parser
from speech.drivers.baseline import run_baseline
from speech.drivers.init import init_run
from speech.drivers.retrain import retrain
from speech.drivers.state import RunState
from speech.drivers.test import evaluate
from speech.drivers.train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="speech", description="BLSTM SAD/LID optimizer orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="parse a config into a run state")
    p_init.add_argument("config", type=Path)
    p_init.add_argument("out_dir", type=Path)

    for name in ("train", "retrain", "test"):
        p = sub.add_parser(name)
        p.add_argument("out_dir", type=Path, help="the run dir holding run_state.json")
        if name in ("retrain", "test"):
            p.add_argument("checkpoint", type=Path)
        if name in ("train", "retrain"):
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--particles", type=int, default=24)
            p.add_argument("--epochs", type=int, default=100)
            p.add_argument("--inner-steps", type=int, default=20)

    # Phase 6 baseline arms (from-scratch SAD/LID training). Reuses the baseline module's own
    # argument set so the flags stay defined in one place; `parents=` mounts them here.
    sub.add_parser("baseline", parents=[_baseline_parser()], add_help=False, help="from-scratch Phase 6 baseline training arms")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the init/train/retrain/test drivers. Returns a process exit code."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2

    if args.command == "init":
        init_run(args.config, args.out_dir)
        return 0

    if args.command == "baseline":
        run_baseline(
            args.arm,
            args.corpus_root,
            args.out_dir,
            resume=args.resume,
            lanes=args.lanes,
            subset=args.subset,
            dry_run=args.dry_run,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            steps_per_epoch=args.steps_per_epoch,
            init_scheme=args.init_scheme,
            lre_listing=args.lre_listing,
        )
        return 0

    state = RunState.load(Path(args.out_dir) / "run_state.json")
    if args.command == "train":
        train(state, args.seed, qpso_particles=args.particles, qpso_epochs=args.epochs, inner_steps=args.inner_steps)
        return 0
    if args.command == "retrain":
        retrain(state, args.checkpoint, args.seed, qpso_particles=args.particles, qpso_epochs=args.epochs, inner_steps=args.inner_steps)
        return 0
    if args.command == "test":
        evaluate(state, args.checkpoint)
        return 0
    return 1
