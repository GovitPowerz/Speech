"""Issue #33 (ADR-0011): the live mutation-battery runner, `tests/mutations/run.py`.

The pure half (the patch header, the verdict read off a catcher's output, the catcher's file)
is pinned on strings; the tree half (apply, run, restore, check, selection) on a throwaway git
repository whose catchers are pytest node ids, so no cargo build runs here.
"""

from __future__ import annotations

import math
import struct
import subprocess
from pathlib import Path

import pytest

from tests._libm_gate import oracle_strict
from tests.mutations import run as runner

_DIFF = """diff --git a/pkg/mod.py b/pkg/mod.py
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""


def _patch(header: str, diff: str = _DIFF) -> str:
    return header + diff


# -- the header -------------------------------------------------------------------------------


def test_header_parses_every_key() -> None:
    m = runner.parse_patch(
        "b/1a",
        _patch(
            "mutation: VALUE off by one\n"
            "catcher: CAUGHT cargo --lib tasks::segmenter::tests::hyst_step\n"
            "catcher: MISSED cargo --test phase7_parity_sad sad_parity_exact_vs_fast\n"
            "catcher: MISSED pytest tests/test_mod.py::test_value\n"
            "requires: corpus\n"
            "requires: oracle-libm\n"
        ),
    )
    assert m.name == "b/1a"
    assert m.description == "VALUE off by one"
    assert m.requires == ("corpus", "oracle-libm")
    assert m.files == ("pkg/mod.py",)
    assert [(c.expected, c.kind, c.target, c.name) for c in m.catchers] == [
        ("CAUGHT", "cargo", "--lib", "tasks::segmenter::tests::hyst_step"),
        ("MISSED", "cargo", "--test phase7_parity_sad", "sad_parity_exact_vs_fast"),
        ("MISSED", "pytest", "", "tests/test_mod.py::test_value"),
    ]


def test_header_ends_at_the_first_diff_line_and_ignores_blank_lines() -> None:
    m = runner.parse_patch("b/1", _patch("mutation: m\n\ncatcher: CAUGHT pytest t.py::x\n"))
    assert len(m.catchers) == 1
    assert m.requires == ()


@pytest.mark.parametrize(
    ("header", "why"),
    [
        ("catcher: CAUGHT pytest t.py::x\n", "one mutation:"),
        ("mutation: m\n", "no catcher:"),
        ("mutation: m\ncatcher: FIRED pytest t.py::x\n", "verdict"),
        ("mutation: m\ncatcher: STALE pytest t.py::x\n", "verdict"),
        ("mutation: m\ncatcher: SKIPPED pytest t.py::x\n", "verdict"),
        ("mutation: m\ncatcher: INVALID pytest t.py::x\n", "verdict"),
        ("mutation: m\ncatcher: CAUGHT nextest t.py::x\n", "kind"),
        ("mutation: m\ncatcher: CAUGHT cargo --bench k x\n", "cargo target"),
        ("mutation: m\ncatcher: CAUGHT cargo --lib\n", "cargo target"),
        ("mutation: m\ncatcher: CAUGHT pytest t.py\n", "node id"),
        ("mutation: m\ncatcher: CAUGHT pytest t.py::x\nrequires: legacy\n", "requires"),
        ("mutation: m\ncatcher: CAUGHT pytest t.py::x\nnote: hi\n", "unknown key"),
        ("mutation: m\nmutation: n\ncatcher: CAUGHT pytest t.py::x\n", "one mutation:"),
    ],
)
def test_header_refuses_a_malformed_patch(header: str, why: str) -> None:
    with pytest.raises(runner.PatchError, match=why):
        runner.parse_patch("b/1", _patch(header))


def test_header_refuses_a_patch_with_no_diff() -> None:
    with pytest.raises(runner.PatchError, match="no diff"):
        runner.parse_patch("b/1", "mutation: m\ncatcher: CAUGHT pytest t.py::x\n")


# -- the verdict, read off a catcher's output -------------------------------------------------

_CARGO_FAILED = """running 1 test
test tasks::x::tests::pin ... FAILED

failures:

---- tasks::x::tests::pin stdout ----

thread 'tasks::x::tests::pin' panicked at src/tasks/x.rs:12:5:
assertion `left == right` failed
  left: 1
 right: 2

failures:
    tasks::x::tests::pin

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 254 filtered out; finished in 0.01s
"""


def test_cargo_failed_catcher_is_caught_with_its_panic_message() -> None:
    assert runner.cargo_verdict(_CARGO_FAILED) == ("CAUGHT", "assertion `left == right` failed")


def test_cargo_passing_catcher_is_missed() -> None:
    out = "running 1 test\ntest a ... ok\n\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 9 filtered out\n"
    assert runner.cargo_verdict(out) == ("MISSED", "")


@pytest.mark.parametrize(
    "counts",
    [
        "0 passed; 0 failed; 0 ignored",  # the filter matched nothing: cargo exits 0
        "0 passed; 0 failed; 1 ignored",  # an #[ignore] test never runs
        "2 passed; 0 failed; 0 ignored",
        "1 passed; 1 failed; 0 ignored",
    ],
    ids=["zero-match", "ignored", "two-passed", "one-each"],
)
def test_cargo_catcher_that_is_not_exactly_one_run_test_is_stale(counts: str) -> None:
    out = f"test result: ok. {counts}; 0 measured; 9 filtered out; finished in 0.00s\n"
    assert runner.cargo_verdict(out)[0] == "STALE"


def test_a_tree_that_does_not_build_is_invalid_with_its_first_error() -> None:
    out = "   Compiling speech v0.1.0\nerror[E0308]: mismatched types\n --> src/x.rs:3:5\nerror: could not compile `speech`\n"
    assert runner.build_verdict(out) == ("INVALID", "error[E0308]: mismatched types")


def test_a_build_naming_a_missing_test_target_is_stale() -> None:
    assert runner.build_verdict("error: no test target named `nope` in `speech` package\n")[0] == "STALE"


def test_cargo_output_with_no_result_line_is_stale() -> None:
    assert runner.cargo_verdict("error: no test target named `nope` in `speech` package\n")[0] == "STALE"


@pytest.mark.parametrize(
    ("code", "out", "verdict"),
    [
        (1, "E   assert 2 == 1\n1 failed in 0.02s\n", "CAUGHT"),
        (1, "E   ValueError: boom\n1 error in 0.02s\n", "CAUGHT"),
        (0, "1 passed in 0.01s\n", "MISSED"),
        (0, "1 skipped in 0.01s\n", "SKIPPED"),
        (4, "ERROR: not found: t.py::nope\nno tests ran in 0.01s\n", "STALE"),
        (5, "no tests ran in 0.01s\n", "STALE"),
        (0, "2 passed in 0.01s\n", "STALE"),
        (2, "E   SyntaxError: invalid syntax\nInterrupted: 1 error during collection\n1 error in 0.1s\n", "INVALID"),
        (4, "ERROR: found no collectors for t.py::x\n___ ERROR collecting t.py ___\nE   SyntaxError: x\n1 error in 0.3s\n", "INVALID"),
    ],
    ids=["failed", "error", "passed", "skipped", "not-found", "no-tests", "two-passed", "collection-interrupted", "collection-node-id"],
)
def test_pytest_verdict(code: int, out: str, verdict: str) -> None:
    assert runner.pytest_verdict(code, out)[0] == verdict


def test_pytest_caught_line_is_the_first_e_line() -> None:
    assert runner.pytest_verdict(1, "x\nE   assert 2 == 1\nE    +  where\n1 failed\n") == ("CAUGHT", "assert 2 == 1")


# -- the catcher's file (what `--changed-since` watches) ---------------------------------------


def test_catcher_file_resolves_a_lib_module_path_to_its_source(tmp_path: Path) -> None:
    src = tmp_path / "src/rust/src"
    (src / "tasks").mkdir(parents=True)
    (src / "tasks/segmenter.rs").touch()
    (src / "fast/plan").mkdir(parents=True)
    (src / "fast/plan/mod.rs").touch()
    (src / "lib.rs").touch()
    lib = runner.Catcher("CAUGHT", "cargo", "--lib", "tasks::segmenter::tests::hyst_step")
    nested = runner.Catcher("CAUGHT", "cargo", "--lib", "fast::plan::tests::timeline")
    root = runner.Catcher("CAUGHT", "cargo", "--lib", "tests::crate_level")
    assert runner.catcher_file(tmp_path, lib) == "src/rust/src/tasks/segmenter.rs"
    assert runner.catcher_file(tmp_path, nested) == "src/rust/src/fast/plan/mod.rs"
    assert runner.catcher_file(tmp_path, root) == "src/rust/src/lib.rs"


def test_catcher_file_of_an_integration_test_and_a_node_id(tmp_path: Path) -> None:
    integ = runner.Catcher("CAUGHT", "cargo", "--test phase7_parity_sad", "legs::sad_parity")
    node = runner.Catcher("CAUGHT", "pytest", "", "tests/pyo3/test_fold_run.py::test_x[a]")
    assert runner.catcher_file(tmp_path, integ) == "src/rust/tests/phase7_parity_sad.rs"
    assert runner.catcher_file(tmp_path, node) == "tests/pyo3/test_fold_run.py"


# -- the tree half, on a throwaway repository ----------------------------------------------------

_TEST_MOD = """import pytest

from pkg.mod import VALUE


def test_value():
    assert VALUE == 1


def test_other():
    assert True


@pytest.mark.skip(reason="precondition")
def test_skipped():
    pass

"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def _repo(tmp_path: Path, patches: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)
    # As the repository's own .gitignore: a catcher's bytecode is not a tree change.
    (root / ".gitignore").write_text("__pycache__/\n")
    (root / "pkg/__init__.py").touch()
    (root / "pkg/mod.py").write_text("VALUE = 1\n")
    (root / "conftest.py").touch()
    (root / "t").mkdir()
    (root / "t/test_mod.py").write_text(_TEST_MOD)
    (root / "t/test_far.py").write_text("def test_far():\n    assert True\n")
    for name, text in patches.items():
        path = root / "tests/mutations" / f"{name}.patch"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    return root


def _verdicts(root: Path, name: str, corpus: bool = False, oracle_libm: bool = False) -> list[str]:
    [mutation] = runner.load(root, [name])
    return [r.observed for r in runner.run_mutation(root, mutation, corpus, oracle_libm)]


def _clean(root: Path) -> bool:
    return _git(root, "status", "--porcelain") == "" and (root / "pkg/mod.py").read_text() == "VALUE = 1\n"


_OFF_BY_ONE = _patch("mutation: off by one\ncatcher: CAUGHT pytest t/test_mod.py::test_value\ncatcher: MISSED pytest t/test_mod.py::test_other\n")


def test_run_reads_caught_and_missed_and_restores_the_tree(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"b/1": _OFF_BY_ONE})
    [mutation] = runner.load(root, ["b/1"])
    results = runner.run_mutation(root, mutation, False, False)
    assert [(r.observed, r.catcher.expected) for r in results] == [("CAUGHT", "CAUGHT"), ("MISSED", "MISSED")]
    assert "assert 2 == 1" in results[0].failure
    assert _clean(root)


def test_a_patch_that_does_not_apply_is_stale_and_touches_nothing(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"b/1": _patch("mutation: m\ncatcher: CAUGHT pytest t/test_mod.py::test_value\n", _DIFF.replace("-VALUE = 1", "-VALUE = 7"))})
    assert _verdicts(root, "b/1") == ["STALE"]
    assert _clean(root)


def test_a_catcher_that_names_no_test_is_stale(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"b/1": _patch("mutation: m\ncatcher: CAUGHT pytest t/test_mod.py::test_nope\n")})
    assert _verdicts(root, "b/1") == ["STALE"]
    assert _clean(root)


def test_a_mutation_that_does_not_build_is_invalid_not_caught(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"b/1": _patch("mutation: m\ncatcher: CAUGHT pytest t/test_mod.py::test_value\n", _DIFF.replace("+VALUE = 2", "+VALUE = ("))})
    assert _verdicts(root, "b/1") == ["INVALID"]
    assert _clean(root)


def test_requires_corpus_skips_when_the_corpus_is_absent(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"b/1": _patch("mutation: m\ncatcher: CAUGHT pytest t/test_mod.py::test_value\nrequires: corpus\n")})
    assert _verdicts(root, "b/1", corpus=False) == ["SKIPPED"]
    assert _verdicts(root, "b/1", corpus=True) == ["CAUGHT"]
    assert _clean(root)


def test_requires_oracle_libm_skips_off_the_oracle_host(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"b/1": _patch("mutation: m\ncatcher: CAUGHT pytest t/test_mod.py::test_value\nrequires: oracle-libm\n")})
    assert _verdicts(root, "b/1", oracle_libm=False) == ["SKIPPED"]
    assert _verdicts(root, "b/1", oracle_libm=True) == ["CAUGHT"]
    assert _clean(root)


def _canaries(root: Path, outputs: list[float]) -> None:
    # The fixture's layout: 2 x 18 column-major, [0,10) cos, [10,15) log, [15,18) exp.
    inputs = [0.1 * k + 0.3 for k in range(18)]
    path = root / runner.CANARIES
    path.parent.mkdir(parents=True)
    path.write_bytes(struct.pack("<qq", 2, 18) + struct.pack("<36d", *(v for x, y in zip(inputs, outputs, strict=True) for v in (x, y))))


def test_oracle_libm_recomputes_the_canary_fixture_with_this_libm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPEECH_ORACLE_LIBM", raising=False)
    inputs = [0.1 * k + 0.3 for k in range(18)]
    outputs = [math.cos(x) if k < 10 else math.log(x) if k < 15 else math.exp(x) for k, x in enumerate(inputs)]
    _canaries(tmp_path, outputs)
    assert runner.oracle_libm(tmp_path)
    outputs[17] = math.nextafter(outputs[17], math.inf)
    _canaries(tmp_path / "off", outputs)
    assert not runner.oracle_libm(tmp_path / "off")


def test_oracle_libm_honours_the_forced_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEECH_ORACLE_LIBM", "strict")
    assert runner.oracle_libm(tmp_path)
    monkeypatch.setenv("SPEECH_ORACLE_LIBM", "ulp")
    assert not runner.oracle_libm(tmp_path)


def test_oracle_libm_agrees_with_the_catchers_gate() -> None:
    # One fixture, one override, one verdict: the runner cannot skip what a catcher runs strict.
    assert runner.oracle_libm(runner.ROOT) == oracle_strict()


def test_a_skipped_catcher_is_skipped(tmp_path: Path) -> None:
    root = _repo(tmp_path, {"b/1": _patch("mutation: m\ncatcher: CAUGHT pytest t/test_mod.py::test_skipped\n")})
    assert _verdicts(root, "b/1") == ["SKIPPED"]


def test_load_orders_ids_naturally_and_selects_by_battery_or_id(tmp_path: Path) -> None:
    root = _repo(tmp_path, {f"b/{i}": _OFF_BY_ONE for i in ("10a", "2", "1b", "1a")} | {"c/1": _OFF_BY_ONE})
    assert [m.name for m in runner.load(root, [])] == ["b/1a", "b/1b", "b/2", "b/10a", "c/1"]
    assert [m.name for m in runner.load(root, ["c"])] == ["c/1"]
    assert [m.name for m in runner.load(root, ["b/2", "c/1"])] == ["b/2", "c/1"]
    with pytest.raises(runner.PatchError, match="no mutation matches"):
        runner.load(root, ["b/9"])


def test_check_reports_a_stale_patch_and_an_unresolved_catcher(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        {
            "b/good": _OFF_BY_ONE,
            "b/moved": _patch("mutation: m\ncatcher: CAUGHT pytest t/test_mod.py::test_value\n", _DIFF.replace("-VALUE = 1", "-VALUE = 7")),
            "b/renamed": _patch("mutation: m\ncatcher: CAUGHT pytest t/test_mod.py::test_renamed\n"),
        },
    )
    problems = runner.check(root, runner.load(root, []), ("pytest",))
    assert len(problems) == 2
    assert any(p.startswith("b/moved: STALE") and "does not apply" in p for p in problems)
    assert any(p.startswith("b/renamed: STALE") and "test_renamed" in p for p in problems)


def test_changed_since_selects_by_patched_file_catcher_file_or_new_patch(tmp_path: Path) -> None:
    far = _patch("mutation: m\ncatcher: CAUGHT pytest t/test_far.py::test_far\n", _DIFF.replace("pkg/mod.py", "pkg/other.py"))
    root = _repo(tmp_path, {"b/near": _OFF_BY_ONE, "b/far": far})
    _git(root, "switch", "-q", "-c", "work")
    assert runner.changed_since(root, runner.load(root, []), "main") == []
    (root / "t/test_mod.py").write_text(_TEST_MOD + "\n")
    assert [m.name for m in runner.changed_since(root, runner.load(root, []), "main")] == ["b/near"]
    _git(root, "checkout", "-q", "--", "t/test_mod.py")
    (root / "pkg/other.py").write_text("X = 1\n")
    _git(root, "add", "pkg/other.py")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "other")
    (root / "tests/mutations/b/new.patch").write_text(_OFF_BY_ONE)
    assert [m.name for m in runner.changed_since(root, runner.load(root, []), "main")] == ["b/far", "b/new"]


def test_main_exits_nonzero_when_an_observed_verdict_differs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wrong = _patch("mutation: off by one\ncatcher: MISSED pytest t/test_mod.py::test_value\n")
    root = _repo(tmp_path, {"b/1": _OFF_BY_ONE, "b/2": wrong})
    assert runner.main(["b/1"], root) == 0
    assert runner.main([], root) == 1
    out = capsys.readouterr().out
    assert "| b | 2 | off by one | pytest t/test_mod.py::test_value | MISSED | CAUGHT |" in out
    assert _clean(root)


def test_the_committed_mutations_apply_and_their_pytest_catchers_resolve() -> None:
    assert runner.check(runner.ROOT, runner.load(runner.ROOT, []), ("pytest",)) == []
