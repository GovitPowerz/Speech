"""Generate the SEAM-tier fixture family for the port-only cells (phase-9 spec S8.1 SEAM /
S8.3 LID wiring; EXTENDED by phase-10 spec S9.2 with the CfC rows -- same recipe, same
directory, same test module, so the family stays ONE thing rather than a per-phase pile).

UNLIKE every other `scripts/extract_*` in this directory, this one has NO ORACLE: these
cells are port-only (no legacy source, no `.m`, no compiled harness), so there is nothing to
dump from. What it writes is a deterministic, self-contained, SYNTHETIC gate corpus:

  <cell>_<direction>.config      9 legacy `.config` files -- 6 SAD (algo 3 spectral,
  <cell>_<direction>_seed.bin      {slstm,mamba,cfc} x {bidirectional,forward}) + 3 Twins
  twin_lid_slstm.config            (algo 6 with `BLSTM_LID_Cell_Type <cell>` and the SAD net
  twin_lid_slstm_seed.bin          left on the legacy LSTM: mode 5 wav, the gradcheck
  twin_mode7_lid_slstm.config      regime, and mode 7 phSeq, the LIVE phase-6 LID training
  twin_mode7_lid_slstm_seed.bin    regime -- the phase-10 CfC row takes mode 7 only, the
  twin_mode7_lid_cfc.config        live regime), each paired with the seeded weight pack
  twin_mode7_lid_cfc_seed.bin      `speech.init_weights` builds for it.
  manifest.json

The audio/listing/mapping the configs point at are NOT written here: they are the ALREADY
COMMITTED synthetic tier-2 (phase4a) and twin (phase4b) corpora, which
`tests/pyo3/test_phase9_seam.py` stages into a tmpdir exactly like `test_seam_replay.py`
does. Nothing corpus-derived is produced or committed (license hygiene).

Determinism: every pack is drawn from a per-fixture `np.random.default_rng(seed)` with the
seed recorded in the manifest, and the config text is generated (not hand-edited), so a
re-run reproduces every byte. `--check` re-generates into a tempdir and diffs.

TWO FIXTURE-DESIGN DECISIONS worth their ink (both measured, see the Task 5 report):

1. SQUARE cost laws, not the live `log` pair. NOT for the reason the phase-4b sibling
   `twin_gradcheck.config` gives (that one wants polynomial laws for platform bit-stability
   of its goldens), and NOT following the direct SAD ancestor `tier2_gradcheck.config`
   either -- that one uses `log`. The reason here is that a `log` fixture stops being a
   VALID gradcheck INSTRUMENT once the net's outputs approach `Law::Log`'s clamps, in two
   compounding ways (T5 review, measured -- exact recipes + numbers in the Task 5 report):
   the central difference loses its convergence plateau (forward and backward differences
   disagree by tens of percent, so the FD is no longer an oracle to compare anything
   against), and far enough in, F7's clamp-consistent derivative makes the folded analytic
   gradient EXACTLY ZERO while the frame counts stay positive. Neither is a property of the
   new cells; both make analytic-vs-FD meaningless. `square` is smooth everywhere on this
   corpus, so the comparison measures the CELL backward, which is the whole point.
   (The training-side consequence of the zero-gradient regime is a T8 watch item, report.)

2. NO `Neural_Networks_Gradient_Check_Epsilon` key. A positive value there routes
   `CorpusProcessor::run()` into the engine's OWN full-sweep gradCheck (`:208`), so a plain
   `Engine.run()` would silently stop being a forward pass. `Engine.grad_check(eps, cap)`
   takes epsilon as an argument, so the key is pure downside here.

Usage: uv run python scripts/extract_phase9_fixtures.py [--check]
"""

from __future__ import annotations

import argparse
import filecmp
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "python"))

from speech.config_bridge import nnet_spec, parse_legacy_config  # noqa: E402
from speech.init_weights import init_weights  # noqa: E402
from speech.weight_bridge import write_bin  # noqa: E402

OUT_DIR = REPO_ROOT / "tests" / "reference_data" / "phase9"
PHASE4B = REPO_ROOT / "tests" / "reference_data" / "phase4b"

# The SAD front-end, copied verbatim from the committed
# `tests/reference_data/phase4a/tier2_gradcheck.config` front matter (same synthetic tier-2
# corpus, same 23-dim spectral input) EXCEPT the two cost-law names (decision 1 above).
# `noise_ratio 0` keeps the audio a pure function of the file, which the FD needs.
SAD_FRONT = """BLSTM_decision_thresh_rising 0.6
BLSTM_decision_area_rising 0.05
BLSTM_decision_thresh_falling 0.3
BLSTM_decision_area_falling 0.1
BLSTM_windowing_type hamming
BLSTM_windowing_param 0.8
BLSTM_convolution_window_type hann
BLSTM_convolution_window_size 4
BLSTM_flag_DCOffset false
BLSTM_preemph_ratio 0.97
BLSTM_noise_seed 0
BLSTM_noise_ratio 0
BLSTM_speech_padding 0.1,0.1,0.1,0.1
BLSTM_min_silence 0.2,0.2
BLSTM_min_speech 0.2,0.2,0.2
BLSTM_BackPropWER -1.0
BLSTM_spectrum_order 10
BLSTM_spectrum_shift 1.000000000000000e-02
BLSTM_spectrum_temporal_convolution_type uniform
BLSTM_spectrum_temporal_convolution_size 0
BLSTM_LTSVwindow 0
BLSTM_LTSVshift 5.657759158904229e-02
BLSTM_TDCwindow 0
BLSTM_TDC_lags 0,0
BLSTM_TDC_balance 5.439183651566210e-01
BLSTM_TDCshift 2.000000000000000e-01
BLSTM_TDC_windowing_type uniform
BLSTM_TDC_windowing_param 5.534716888700572e-01
BLSTM_minFreq 100
BLSTM_maxFreq 3800
BLSTM_minMelFreq 1.861001763856132e+02
BLSTM_maxMelFreq 2.502400490895474e+03
BLSTM_nb_bins 20
BLSTM_is_log_mel true
BLSTM_nb_DCT 4
BLSTM_ComputeDeltasNb 5
BLSTM_ComputeDeltaDeltasNb 3
BLSTM_IgnoreFirstDCT true
BLSTM_TwoSweeps false
BLSTM_NNType 0
BLSTM_CostLawSpeech square
BLSTM_CostLawNoSpeech square
BLSTM_CostLawParamSpeech 0
BLSTM_CostLawParamNoSpeech 0
BLSTM_CostLawThreshSpeech 1
BLSTM_CostLawThreshNoSpeech 0
BLSTM_Forward_IsCellsPeepholesActive true
BLSTM_Backward_IsCellsPeepholesActive true
BLSTM_Forward_IsGatesPeepholesActive true
BLSTM_Backward_IsGatesPeepholesActive true
BLSTM_Forward_IsGatesRecurrentPeepholesActive true
BLSTM_Backward_IsGatesRecurrentPeepholesActive true
"""

# TINY by design: `grad_check` costs 2 corpus runs PER swept weight, and the block-probe
# leg costs 2 more per probed block. hidden 4 + subsampling 4 (50 recurrent timesteps per
# channel over the 2 s excerpt) keeps the whole seam suite at a couple of seconds.
SAD_HIDDEN = 4
SAD_INPUT = 23
SAD_SUBSAMPLING = 4
# Small mamba geometry (spec S3.3 defaults are 16/4/2 -- far too wide for a gradcheck net).
MAMBA_D_STATE = 4
MAMBA_D_CONV = 3
MAMBA_EXPAND = 2
# Small CfC geometry (phase-10 spec S1.4's SIZED default is B=45 -- same story: far too wide
# for a per-weight gradcheck). B=6 keeps the backbone genuinely wider than the 4-unit hidden
# state (so a B/H transposition could not pass unnoticed) at 666 weights per stack.
CFC_BACKBONE_UNITS = 6
CFC_BACKBONE_LAYERS = 1

# One seed per fixture, so a regenerated pack never silently inherits a neighbour's draw.
SEEDS = {
    "slstm_bidirectional": 920501,
    "slstm_forward": 920502,
    "mamba_bidirectional": 920503,
    "mamba_forward": 920504,
    "twin_lid_slstm": 920505,
    "twin_mode7_lid_slstm": 920506,
    # Phase 10 Task 3 (the CfC rows).
    "cfc_bidirectional": 1003001,
    "cfc_forward": 1003002,
    "twin_mode7_lid_cfc": 1003003,
}

#: Per-cell provenance, so the generated header names the task that owns the row. The
#: phase-9 rows must render EXACTLY as they were committed (`--check` is the guard).
PROVENANCE = {"slstm": "Phase 9 Task 5", "mamba": "Phase 9 Task 5", "cfc": "Phase 10 Task 3"}


def geometry_keys(cell: str) -> str:
    """The cell's own (UNPREFIXED) geometry keys, or `""` for a cell that has none."""
    if cell == "mamba":
        return f"Mamba_D_State {MAMBA_D_STATE}\nMamba_D_Conv {MAMBA_D_CONV}\nMamba_Expand {MAMBA_EXPAND}\n"
    if cell == "cfc":
        return f"Cfc_Backbone_Units {CFC_BACKBONE_UNITS}\nCfc_Backbone_Layers {CFC_BACKBONE_LAYERS}\n"
    return ""


def sad_config(cell: str, direction: str, name: str) -> str:
    """One SAD gate config: the shared front-end + the port-only architecture keys.

    `output_neuron_nb[0]` tracks the direction (`2*hidden` bidirectional, `hidden` forward)
    because `BlstmConfig::from_legacy` REQUIRES it -- the same derived key
    `drivers.baseline.cell_overlay` writes for the arm CLI.
    """
    out_in = SAD_HIDDEN * (2 if direction == "bidirectional" else 1)
    geom_keys = geometry_keys(cell)
    return (
        f"# {name}.config -- {PROVENANCE[cell]} seam-tier gate config ({cell}, {direction}).\n"
        "# GENERATED by scripts/extract_phase9_fixtures.py -- edit that, not this.\n"
        "# Algo 3 (spectral SAD) over the committed synthetic tier-2 corpus (phase4a), TINY\n"
        f"# net ({SAD_INPUT},{SAD_HIDDEN} recurrent / {out_in},1 output), square cost laws, no\n"
        "# windowing (plain whole-sequence FFB), no input self-normalization, no added noise.\n"
        "# See the generator docstring for why square and why no Gradient_Check_Epsilon key.\n"
        + SAD_FRONT
        + f"""Algo_choice 3
numOuterThreads 1
Audio_offset 0.0
Audio_max_duration 2.0
File_Type 0
BLSTM_window 0.0
BLSTM_shift 0.1
BLSTM_LSTMNeuronNb {SAD_INPUT},{SAD_HIDDEN}
BLSTM_LSTMSubSampling {SAD_SUBSAMPLING}
BLSTM_NNetInputSize {SAD_INPUT}
BLSTM_OutputNeuronNb {out_in},1
BLSTM_OutputSubSampling 1
BLSTM_InputNormalizationType 0
BLSTM_TargetEnforcementStep 0
BLSTM_BackPropagationActivated true
BLSTM_Cell_Type {cell}
BLSTM_Direction {direction}
{geom_keys}BLSTM_weightsFile {name}_seed.bin
language2classmapping language2classmapping.csv
fileslisting tier2_gc_fileslisting.csv
Neural_Networks_BackPropagation_Epochs 0
multiConfigResultsOutputFile {name}_results.mat
"""
    )


def twin_config(name: str) -> str:
    """The S8.3 LID mechanical-wiring fixture: the committed phase-4b `twin_gradcheck.config`
    (algo 6, mode 5, wav, BOTH nets backprop-active, square cost laws already) with the LID
    net's cell type and weight pack swapped, and the inherited `Gradient_Check_Epsilon` line
    STRIPPED. The SAD net stays on the legacy peephole LSTM, so a cell swap on ONE net of a
    two-net bag is what gets pinned.

    THE STRIP IS LOAD-BEARING (T5 review C1). The phase-4b source carries
    `Neural_Networks_Gradient_Check_Epsilon 1e-5`, which is exactly the hazard decision 2 of
    the module docstring names: with it set, `CorpusProcessor::run()` takes the
    `grad_check_full` branch (`:208`), and `grad_check_capped` restores `seam_derivs` at its
    epilogue -- so `Engine.run()` is a full uncapped weight sweep and the seam afterwards
    reads the RESET accumulator (an all-zero gradient). The 6 SAD configs are written from
    scratch and never had the key; only the two clone-based Twins could inherit it, and only
    this one did (`twin_train.config`, the mode-7 source, has no such key).
    """
    text = (PHASE4B / "twin_gradcheck.config").read_text()
    text = text.replace("multiConfigResultsOutputFile twin_gc.mat", f"multiConfigResultsOutputFile {name}_results.mat")
    text = text.replace("BLSTM_LID_weightsFile tiny_lid_seed.bin", f"BLSTM_LID_weightsFile {name}_seed.bin")
    assert f"{name}_seed.bin" in text, "the phase-4b LID weightsFile key moved -- re-point the replace"
    epsilon_line = "Neural_Networks_Gradient_Check_Epsilon 1e-5\n"
    assert text.count(epsilon_line) == 1, "the phase-4b gradient-check key moved -- re-point the strip (see the docstring)"
    text = text.replace(epsilon_line, "")
    header = (
        f"# {name}.config -- Phase 9 Task 5 LID mechanical-wiring fixture (spec S8.3).\n"
        "# GENERATED by scripts/extract_phase9_fixtures.py from the committed phase-4b\n"
        "# twin_gradcheck.config: SAME corpus, SAME SAD net (legacy peephole LSTM), only the\n"
        "# LID net swapped to the phase-9 sLSTM cell via BLSTM_LID_Cell_Type + its seed pack,\n"
        "# and the inherited Neural_Networks_Gradient_Check_Epsilon line REMOVED so run() is a\n"
        "# forward+backward fold at theta rather than the engine's own full gradCheck sweep.\n"
    )
    return header + text + "BLSTM_LID_Cell_Type slstm\n"


def twin_mode7_config(name: str, cell: str = "slstm") -> str:
    """The SECOND LID fixture: the same cell swap in the LIVE phase-6 regime (Mode 7 phSeq,
    the one `speech baseline lid-phseq` trains), cloned from the committed phase-4b
    `twin_train.config` with `Epochs 0` so `Engine.run()` is one forward+backward fold at
    theta rather than the engine-internal training loop (the F11 convention).

    It is kept on its own merits, NOT (as an earlier draft claimed) because mode 5 cannot
    reach the folded gradient -- that was the stripped-key artifact, see [`twin_config`].
    Mode 7 is the regime `speech baseline lid-phseq` actually trains, and it exercises a
    structurally different LID path: one-hot phSeq `external_features` instead of wav, and a
    FROZEN SAD net (`BLSTM_BackPropagationActivated false`), so `grad_check` visits the LID
    net alone. Mode 5 is the two-nets-live regime. Neither covers the other.

    THE TWO INHERITED-KEY HAZARDS, asserted rather than assumed (the phase-10 T3 brief's
    step 1, and the T5-review lesson [`twin_config`] records): the source must carry NO
    `Neural_Networks_Gradient_Check_Epsilon` (it would route `run()` into the engine's own
    full gradCheck and hand the seam a reset accumulator), and its inherited
    `Neural_Networks_BackPropagation_Epochs 6` must be OVERRIDDEN -- which the appended
    `Epochs 0` does under the last-wins parser, asserted here as "the appended line is the
    last occurrence" rather than trusted.
    """
    text = (PHASE4B / "twin_train.config").read_text()
    text = text.replace("multiConfigResultsOutputFile twin_train.mat", f"multiConfigResultsOutputFile {name}_results.mat")
    text = text.replace("BLSTM_LID_weightsFile tiny_lid_seed.bin", f"BLSTM_LID_weightsFile {name}_seed.bin")
    assert f"{name}_seed.bin" in text and f"{name}_results.mat" in text, "the phase-4b twin_train keys moved -- re-point the replaces"
    assert "Neural_Networks_Gradient_Check_Epsilon" not in text, "twin_train.config gained a gradient-check epsilon -- STRIP it (see the docstring)"
    cell_note = "phase-9 sLSTM" if cell == "slstm" else "phase-10 CfC"
    header = (
        f"# {name}.config -- {PROVENANCE[cell]} LID mechanical-wiring fixture, Mode 7 (spec S8.3).\n"
        "# GENERATED by scripts/extract_phase9_fixtures.py from the committed phase-4b\n"
        "# twin_train.config: SAME phSeq corpus, SAME frozen SAD net, LID net swapped to the\n"
        f"# {cell_note} cell. Epochs forced to 0 so run() is a single fold at theta (F11).\n"
    )
    out = header + text + geometry_keys(cell) + f"BLSTM_LID_Cell_Type {cell}\nNeural_Networks_BackPropagation_Epochs 0\n"
    epochs = [i for i, line in enumerate(out.splitlines()) if line.startswith("Neural_Networks_BackPropagation_Epochs")]
    assert out.splitlines()[epochs[-1]] == "Neural_Networks_BackPropagation_Epochs 0", "the inherited Epochs key is not overridden last"
    return out


def build(out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixtures: dict[str, object] = {}

    specs: list[tuple[str, str, str]] = [
        ("slstm_bidirectional", "slstm", "bidirectional"),
        ("slstm_forward", "slstm", "forward"),
        ("mamba_bidirectional", "mamba", "bidirectional"),
        ("mamba_forward", "mamba", "forward"),
        ("cfc_bidirectional", "cfc", "bidirectional"),
        ("cfc_forward", "cfc", "forward"),
    ]
    for name, cell, direction in specs:
        text = sad_config(cell, direction, name)
        (out_dir / f"{name}.config").write_text(text)
        spec = nnet_spec(parse_legacy_config(text))
        pack = init_weights(spec, np.random.default_rng(SEEDS[name]), "xavier", True)[0]
        write_bin(pack.shape[0], 1, pack, out_dir / f"{name}_seed.bin")
        fixtures[name] = {
            "config": f"{name}.config",
            "weights": f"{name}_seed.bin",
            "algo": 3,
            "cell_type": cell,
            "direction": direction,
            "seed": SEEDS[name],
            "init_scheme": "xavier",
            "forget_bias_one": True,
            "pack_length": int(pack.shape[0]),
            "nets": 1,
        }

    twins = (
        ("twin_lid_slstm", twin_config("twin_lid_slstm"), 5, "slstm"),
        ("twin_mode7_lid_slstm", twin_mode7_config("twin_mode7_lid_slstm"), 7, "slstm"),
        ("twin_mode7_lid_cfc", twin_mode7_config("twin_mode7_lid_cfc", "cfc"), 7, "cfc"),
    )
    for name, text, mode, lid_cell in twins:
        (out_dir / f"{name}.config").write_text(text)
        flat = parse_legacy_config(text)
        lid_pack = init_weights(nnet_spec(flat, "BLSTM_LID"), np.random.default_rng(SEEDS[name]), "xavier", True)[0]
        write_bin(lid_pack.shape[0], 1, lid_pack, out_dir / f"{name}_seed.bin")
        fixtures[name] = {
            "config": f"{name}.config",
            "weights": f"{name}_seed.bin",
            "algo": 6,
            "lid_mode": mode,
            "cell_type": "lstm",
            "lid_cell_type": lid_cell,
            "direction": "bidirectional",
            "seed": SEEDS[name],
            "init_scheme": "xavier",
            "forget_bias_one": True,
            "lid_pack_length": int(lid_pack.shape[0]),
            "sad_pack_length": 537,  # the committed phase-4b tiny_sad_seed.bin, reused as-is
            "nets": 2,
        }

    manifest = {
        "phase": 9,
        "task": 5,
        "extended": "phase 10 task 3 (the cfc_* SAD rows + twin_mode7_lid_cfc)",
        "generator": "scripts/extract_phase9_fixtures.py",
        "corpus": {
            "sad": "tests/reference_data/phase4a (tier2_gc_fileslisting.csv + corpus/f1.{wav,stm})",
            "twin": "tests/reference_data/phase4b (listing_gc_wav.csv + corpus_lid/f1.{wav,stm} + tiny_sad_seed.bin)",
            "twin_mode7": "tests/reference_data/phase4b (corpus_phseq/ + languagemapping_lid7.csv + tiny_sad_seed.bin)",
        },
        "geometry": {
            "sad_input": SAD_INPUT,
            "sad_hidden": SAD_HIDDEN,
            "sad_sub_sampling": SAD_SUBSAMPLING,
            "mamba_d_state": MAMBA_D_STATE,
            "mamba_d_conv": MAMBA_D_CONV,
            "mamba_expand": MAMBA_EXPAND,
            "cfc_backbone_units": CFC_BACKBONE_UNITS,
            "cfc_backbone_layers": CFC_BACKBONE_LAYERS,
        },
        "fixtures": fixtures,
        # MEASURED-then-pinned, this box (M4 Pro, macOS 25.5, python 3.14). `grad_check`
        # sweeps the first `max_weights` flat weights; `block_probe` is the per-block
        # targeted FD the test runs through `weights_derivatives` (see the test module
        # docstring). Every pin in the test is `measured * 10` and every value is < 1e-4.
        "measured": {
            "grad_check_max_weights": 12,
            # Per-fixture, because the three families' gradient scales differ by ~3 decades
            # each and so do their measured FD U-curve minima: SAD bottoms out at 1e-5
            # (|grad| ~ 1e-1), the mode-5 Twin at 1e-4 (~1e-3), the mode-7 Twin at 1e-3
            # (~1e-6, so its central difference needs the widest step to clear roundoff).
            "sad_epsilon": 1e-5,
            "twin_epsilon": 1e-4,
            "twin_mode7_epsilon": 1e-3,
            # The phase-10 CfC rows REUSE `sad_epsilon` (their own 5-point sweeps bottom
            # at 1e-6/1e-5 and read 3.14e-7 / 1.17e-8 at 1e-5 -- see the seam test's
            # GRAD_CHECK_PINS comment), but the mode-7 CfC Twin needs its OWN: its LID
            # gradient is ~5.6e-5, ~60x the sLSTM Twin's ~9e-7, so its FD optimum sits one
            # decade lower than `twin_mode7_epsilon`.
            "twin_mode7_cfc_epsilon": 1e-4,
        },
        "files": {},
    }
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out_dir.iterdir()) if p.name != "manifest.json"}
    manifest["files"] = digests
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="regenerate into a tempdir and diff against the committed fixtures")
    args = ap.parse_args()

    if not args.check:
        manifest = build(OUT_DIR)
        for name, info in manifest["fixtures"].items():  # type: ignore[union-attr]
            print(f"  {name}: {info}")
        print(f"wrote {OUT_DIR}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp) / "phase9"
        build(tmp_dir)
        names = sorted({p.name for p in tmp_dir.iterdir()} | {p.name for p in OUT_DIR.iterdir()})
        bad = [n for n in names if not (OUT_DIR / n).exists() or not (tmp_dir / n).exists() or not filecmp.cmp(OUT_DIR / n, tmp_dir / n, shallow=False)]
        if bad:
            print("REPRODUCIBILITY FAILURE:", bad)
            return 1
        print(f"reproducible: {len(names)} files byte-identical")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
