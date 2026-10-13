"""The live mutation-battery runner (issue #33, ADR-0011).

A mutation is `tests/mutations/<battery>/<id>.patch`: a header, then a git diff. The header
(the text `git apply` ignores before the first `diff --git`) is

    mutation: <one line>
    catcher: <CAUGHT|MISSED> cargo --lib <module::path::test>
    catcher: <CAUGHT|MISSED> cargo --test <target> <test name>
    catcher: <CAUGHT|MISSED> pytest <node id>
    requires: corpus

`--check` is the cheap half CI runs: every patch applies, every catcher names exactly one test.
A run applies one mutation, builds what its catchers need, runs each catcher, restores the
tree, and compares each observed verdict with the expected one.

Stdlib only, Python >= 3.12: the Rust CI job runs `--check --kind cargo` on the runner image's
`python3`.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ("CAUGHT", "MISSED")
REQUIRES = ("corpus",)
# `tests/conftest.py` and `src/rust/tests/common/mod.rs` both resolve the corpus here.
CORPUS = Path("data/LRE03-LRE07")
CARGO_DIR = Path("src/rust")
MATURIN = ["uv", "run", "maturin", "develop", "--release", "--manifest-path", "src/rust/speech-py/Cargo.toml"]


class PatchError(ValueError):
    pass


@dataclass(frozen=True)
class Catcher:
    expected: str
    kind: str  # "cargo" | "pytest"
    target: str  # "--lib" | "--test <target>" | "" for pytest
    name: str  # the exact test name, or the pytest node id

    def label(self) -> str:
        return " ".join(part for part in (self.kind, self.target, self.name) if part)


@dataclass(frozen=True)
class Mutation:
    name: str  # "<battery>/<id>"
    description: str
    catchers: tuple[Catcher, ...]
    requires: tuple[str, ...]
    files: tuple[str, ...]
    text: str


def _parse_catcher(value: str) -> Catcher:
    parts = value.split()
    if len(parts) < 2 or parts[0] not in EXPECTED:
        raise PatchError(f"catcher verdict must be one of {EXPECTED}: {value!r}")
    expected, kind, rest = parts[0], parts[1], parts[2:]
    if kind == "cargo":
        if len(rest) == 2 and rest[0] == "--lib":
            return Catcher(expected, kind, "--lib", rest[1])
        if len(rest) == 3 and rest[0] == "--test":
            return Catcher(expected, kind, f"--test {rest[1]}", rest[2])
        raise PatchError(f"cargo target must be `--lib <name>` or `--test <target> <name>`: {value!r}")
    if kind == "pytest":
        if len(rest) == 1 and "::" in rest[0]:
            return Catcher(expected, kind, "", rest[0])
        raise PatchError(f"pytest catcher must be one node id `<file>::<test>`: {value!r}")
    raise PatchError(f"catcher kind must be cargo or pytest: {value!r}")


def parse_patch(name: str, text: str) -> Mutation:
    lines = text.splitlines()
    diff_at = next((i for i, line in enumerate(lines) if line.startswith("diff --git ")), None)
    if diff_at is None:
        raise PatchError(f"{name}: no diff")
    descriptions: list[str] = []
    catchers: list[Catcher] = []
    requires: list[str] = []
    for line in lines[:diff_at]:
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        value = value.strip()
        if not sep or key not in ("mutation", "catcher", "requires"):
            raise PatchError(f"{name}: unknown key in header line {line!r}")
        if key == "mutation":
            descriptions.append(value)
        elif key == "catcher":
            catchers.append(_parse_catcher(value))
        elif value not in REQUIRES:
            raise PatchError(f"{name}: requires must be one of {REQUIRES}: {value!r}")
        else:
            requires.append(value)
    if len(descriptions) != 1:
        raise PatchError(f"{name}: needs exactly one mutation: line")
    if not catchers:
        raise PatchError(f"{name}: no catcher: line")
    files = tuple(m.group(1) for line in lines[diff_at:] if (m := re.match(r"diff --git a/\S+ b/(\S+)", line)))
    return Mutation(name, descriptions[0], tuple(catchers), tuple(requires), files, text)


def cargo_verdict(out: str) -> tuple[str, str]:
    m = re.search(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", out)
    if m is None:
        return "STALE", "no test result line"
    passed, failed = int(m.group(1)), int(m.group(2))
    if passed + failed != 1:
        return "STALE", f"{passed} passed, {failed} failed, {m.group(3)} ignored: not exactly one test"
    if failed:
        lines = out.splitlines()
        at = next((i for i, line in enumerate(lines) if " panicked at " in line), None)
        return "CAUGHT", lines[at + 1].strip() if at is not None and at + 1 < len(lines) else ""
    return "MISSED", ""


def pytest_verdict(code: int, out: str) -> tuple[str, str]:
    # A module that no longer imports fails collection; with a node id pytest then exits 4,
    # the code it also uses for a test that does not exist.
    if "ERROR collecting" in out or re.search(r"errors? during collection", out):
        return "INVALID", next((line[1:].strip() for line in reversed(out.splitlines()) if line.startswith("E ")), "collection error")
    if code in (4, 5):
        return "STALE", "no such test"
    last = next((line for line in reversed(out.splitlines()) if line.strip()), "")
    counts = {word.rstrip("s") if word != "passed" else word: int(n) for n, word in re.findall(r"(\d+) (passed|failed|skipped|errors?)", last)}
    failed = counts.get("failed", 0) + counts.get("error", 0)
    passed, skipped = counts.get("passed", 0), counts.get("skipped", 0)
    if failed + passed + skipped != 1:
        return "STALE", f"{last.strip()!r}: not exactly one test"
    if failed:
        return "CAUGHT", next((line[1:].strip() for line in out.splitlines() if line.startswith("E ")), "")
    return ("SKIPPED", "") if skipped else ("MISSED", "")


def catcher_file(root: Path, catcher: Catcher) -> str:
    if catcher.kind == "pytest":
        return catcher.name.split("::")[0]
    if catcher.target.startswith("--test "):
        target = catcher.target.split()[1]
        main = f"src/rust/tests/{target}/main.rs"
        return main if (root / main).exists() else f"src/rust/tests/{target}.rs"
    segs = catcher.name.split("::")
    for k in range(len(segs) - 1, 0, -1):
        base = "src/rust/src/" + "/".join(segs[:k])
        for candidate in (f"{base}.rs", f"{base}/mod.rs"):
            if (root / candidate).exists():
                return candidate
    return "src/rust/src/lib.rs"


@dataclass(frozen=True)
class Result:
    mutation: Mutation
    catcher: Catcher
    observed: str
    line: str


def _natural(text: str) -> list[int | str]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def load(root: Path, selectors: list[str]) -> list[Mutation]:
    paths = sorted(root.glob("tests/mutations/*/*.patch"), key=lambda p: (p.parent.name, _natural(p.stem)))
    names = [f"{p.parent.name}/{p.stem}" for p in paths]
    for sel in selectors:
        if not any(n == sel or n.partition("/")[0] == sel for n in names):
            raise PatchError(f"no mutation matches {sel!r}")
    return [
        parse_patch(n, p.read_text())
        for n, p in zip(names, paths, strict=True)
        if not selectors or any(n == sel or n.partition("/")[0] == sel for sel in selectors)
    ]


def _run(cmd: list[str], cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True)


def _git_apply(root: Path, mutation: Mutation, *flags: str) -> subprocess.CompletedProcess[str]:
    return _run(["git", "apply", *flags, "-"], root, mutation.text)


def _untracked(root: Path) -> set[str]:
    return set(_run(["git", "ls-files", "--others", "--exclude-standard"], root).stdout.splitlines())


def _cargo(root: Path, targets: list[str], extra: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = ["cargo", "test", "--release", "-p", "speech", *(tok for t in targets for tok in t.split()), *extra]
    res = _run(cmd, root / CARGO_DIR)
    return subprocess.CompletedProcess(res.args, res.returncode, res.stdout + res.stderr, "")


def _pytest(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    res = _run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args], root)
    return subprocess.CompletedProcess(res.args, res.returncode, res.stdout + res.stderr, "")


def _run_catcher(root: Path, catcher: Catcher) -> tuple[str, str]:
    if catcher.kind == "cargo":
        return cargo_verdict(_cargo(root, [catcher.target], [catcher.name, "--", "--exact"]).stdout)
    res = _pytest(root, "-q", catcher.name)
    return pytest_verdict(res.returncode, res.stdout)


def run_mutation(root: Path, mutation: Mutation, corpus: bool) -> list[Result]:
    def every(observed: str, line: str) -> list[Result]:
        return [Result(mutation, c, observed, line) for c in mutation.catchers]

    if _git_apply(root, mutation, "--check").returncode:
        return every("STALE", "patch does not apply")
    if "corpus" in mutation.requires and not corpus:
        return every("SKIPPED", f"{CORPUS} absent")
    if _git_apply(root, mutation).returncode:
        raise RuntimeError(f"{mutation.name}: git apply failed after --check passed")
    rebuilt = False
    # A failing catcher can leave files behind (a proptest regression seed under the mutation).
    before = _untracked(root)
    try:
        targets = sorted({c.target for c in mutation.catchers if c.kind == "cargo"})
        if targets:
            build = _cargo(root, targets, ["--no-run"])
            if build.returncode:
                first = next((line for line in build.stdout.splitlines() if line.startswith("error")), "build failed")
                return every("STALE" if "no test target named" in build.stdout else "INVALID", first)
        if any(c.kind == "pytest" for c in mutation.catchers) and any(f.startswith(f"{CARGO_DIR}/") for f in mutation.files):
            rebuilt = True
            if _run(MATURIN, root).returncode:
                return every("INVALID", "maturin develop failed")
        return [Result(mutation, c, *_run_catcher(root, c)) for c in mutation.catchers]
    finally:
        if _git_apply(root, mutation, "-R").returncode:
            raise RuntimeError(f"{mutation.name}: restore failed, revert the patch by hand")
        for left in sorted(_untracked(root) - before):
            path = root / left
            path.unlink()
            for parent in path.parents:
                if parent == root or any(parent.iterdir()):
                    break
                parent.rmdir()
        if rebuilt and _run(MATURIN, root).returncode:
            raise RuntimeError(f"{mutation.name}: the restored speech_rs did not build, run maturin develop by hand")


def _collected(root: Path, ids: list[str]) -> set[str]:
    res = _pytest(root, "--collect-only", "-q", *ids)
    return {line for line in res.stdout.splitlines() if "::" in line} if res.returncode == 0 else set()


def check(root: Path, mutations: list[Mutation], kinds: tuple[str, ...]) -> list[str]:
    problems = [f"{m.name}: STALE: patch does not apply" for m in mutations if _git_apply(root, m, "--check").returncode]
    # One collection for every node id; per id only when a missing one fails the batch. An exact
    # node id names one test, so a parametrized base name is absent from the collected set.
    ids = sorted({c.name for m in mutations for c in m.catchers if c.kind == "pytest"}) if "pytest" in kinds else []
    collected = _collected(root, ids) if ids else set()
    if ids and not collected:
        collected = set().union(*(_collected(root, [i]) for i in ids))
    listed: dict[str, set[str]] = {}
    for m in mutations:
        for c in m.catchers:
            if c.kind not in kinds:
                continue
            if c.kind == "cargo":
                if c.target not in listed:
                    res = _cargo(root, [c.target], ["--", "--list"])
                    listed[c.target] = set() if res.returncode else set(re.findall(r"^(\S.*): test$", res.stdout, re.M))
                ok = c.name in listed[c.target]
            else:
                ok = c.name in collected
            if not ok:
                problems.append(f"{m.name}: STALE: {c.label()} does not name exactly one test")
    return problems


def changed_since(root: Path, mutations: list[Mutation], ref: str) -> list[Mutation]:
    def git(*args: str) -> list[str]:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines()

    [base] = git("merge-base", ref, "HEAD")
    changed = set(git("diff", "--name-only", base)) | set(git("ls-files", "--others", "--exclude-standard"))
    return [
        m for m in mutations if f"tests/mutations/{m.name}.patch" in changed or changed & set(m.files) or changed & {catcher_file(root, c) for c in m.catchers}
    ]


def main(argv: list[str], root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description="Check or run the live mutation batteries (ADR-0011).")
    parser.add_argument("select", nargs="*", help="a battery or battery/id; default every mutation")
    parser.add_argument("--check", action="store_true", help="patches apply and catchers resolve; nothing is built or run")
    parser.add_argument("--kind", choices=("cargo", "pytest"), action="append", help="with --check: resolve only these catchers")
    parser.add_argument("--changed-since", metavar="REF", help="only mutations whose patch, patched file or catcher file differs from REF")
    args = parser.parse_args(argv)
    try:
        mutations = load(root, args.select)
    except PatchError as e:
        print(e, file=sys.stderr)
        return 2
    if args.changed_since:
        mutations = changed_since(root, mutations, args.changed_since)
    if args.check:
        problems = check(root, mutations, tuple(args.kind or ("cargo", "pytest")))
        print("\n".join([*problems, f"{len(mutations)} mutations checked, {len(problems)} problems"]))
        return 1 if problems else 0
    corpus = (root / CORPUS).is_dir()
    print("| battery | # | mutation | catcher | expected | observed |\n|---|---|---|---|---|---|", flush=True)
    results: list[Result] = []
    for m in mutations:
        battery, _, ident = m.name.partition("/")
        for r in run_mutation(root, m, corpus):
            print(f"| {battery} | {ident} | {m.description} | {r.catcher.label()} | {r.catcher.expected} | {r.observed} |", flush=True)
            results.append(r)
    if any(r.line for r in results):
        print("\nfirst failure lines:\n" + "\n".join(f"- {r.mutation.name} {r.catcher.label()}: {r.line}" for r in results if r.line))
    differ = [r for r in results if r.observed != r.catcher.expected]
    print(f"\n{len(results)} verdicts, {len(differ)} differ from expected")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
