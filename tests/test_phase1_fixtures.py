"""Assert the Phase 1 audio-stage oracle fixtures are present and well-formed.

These fixtures are produced by ``scripts/extract_phase1_fixtures.py`` (which runs
the C++ oracle harness). Later Rust ports must match the ``.bin`` dumps bit-for-bit;
this test only guards presence, shape, and the strict-FP flag string in the manifest.
"""

import json
from pathlib import Path

import soundfile as sf
from speech.weight_bridge import read_bin

REF = Path("tests/reference_data/phase1")


def test_excerpt_wav_present() -> None:
    info = sf.info(REF / "excerpt_2ch_8k.wav")
    assert info.frames == 24000
    assert info.channels == 2
    assert info.samplerate == 8000


def test_manifest_dumps_present_with_declared_shapes() -> None:
    manifest = json.loads((REF / "manifest.json").read_text())
    assert manifest["dumps"], "manifest lists no dumps"
    for name, shape in manifest["dumps"].items():
        rows, cols, data = read_bin(REF / name)
        assert (rows, cols) == (shape["rows"], shape["cols"]), name
        assert data.shape[0] == rows * cols, name


def test_sig_norm_shape() -> None:
    rows, cols, _ = read_bin(REF / "sig_norm.bin")
    assert (rows, cols) == (2, 16001)


def test_manifest_records_strict_fp_flags() -> None:
    flags = json.loads((REF / "manifest.json").read_text())["flags"]
    for flag in ("-fno-fast-math", "-ffp-contract=off", "-DEIGEN_DONT_VECTORIZE"):
        assert flag in flags, flag
