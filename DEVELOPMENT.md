# Development

Most commits in this repository carry a `Co-Authored-By: Claude` trailer: the code is written largely with coding agents (Claude Code, briefed by [CLAUDE.md](CLAUDE.md) and the contracts in [docs/agents/](docs/agents/)), one roadmap phase at a time, each phase a dated design spec and implementation plan under [docs/superpowers/](docs/superpowers/README.md). This page says which decisions are the author's, what evidence a change needs before it lands, and how the split is enforced.

## Division of labour

Agents implement. They take a spec or a GitHub issue with stated acceptance criteria and turn it into a branch and a pull request: the port of a legacy module against its golden fixtures, a new cell behind the `CellLayer` seam, a fast-tree twin with its parity leg, test suites, mutation batteries, first drafts of docs, and commissioned reviews of a branch.

The author decides. The roadmap and its phase ordering, which legacy behaviours are fixed and which are kept (the [fix protocol](IMPROVEMENTS.md) adjudications), the acceptance criteria written into each spec, the architecture decisions recorded as [ADRs](docs/adr/), the numbers [RESULTS.md](RESULTS.md) and the README quote, every pin that is re-measured rather than widened, and every merge to `main` stay human-owned. An agent may draft any of these; none becomes true until the author has checked it against the tree or re-run it.

## Evidence rule

A pull request is accepted when its description names the gates it re-ran and they pass. The standing gates are the committed golden suites (`cd src/rust && cargo test`; `uv run pytest tests`; `uv run pytest tests/pyo3` once `speech_rs` is built) and the parity legs that hold the three implementation axes together: the exact-vs-fast legs (`src/rust/tests/phase7_parity_{sad,lid}.rs`, `phase9_fast_parity.rs`, `phase10_bicell_parity.rs`), the streaming bit-equality gates (`phase8_gate.rs`, `phase9_stream_causal.rs`, `stream_incremental_resmooth.rs`), the cell gradient tiers (`phase9_cell_grad.rs`, `tests/pyo3/test_phase9_seam.py`) and the from-scratch subset gates (`tests/pyo3/test_phase*_gates.py`, corpus-gated). Decisions must not move: a boundary, an argmax, a segment count. Numeric pins are set at measured times ten and are never widened to pass; a breach stops the work for adjudication ([ADR-0003](docs/adr/0003-decisions-are-gated-on-identity-numbers-are-pinned.md)).

A numerical or scientific claim needs an independent check on top: a second oracle tier (the C++ strict-IEEE harness, Octave, Perl, the resurrected 2015 binary), a central-difference gradient check, a run-twice bit-identity, or a mutation that breaks the test that claims to cover it. Every phase closes with a mutation battery whose scorecard is recorded in [RESULTS.md](RESULTS.md), gaps included. Claims an agent writes into a PR, a spec or `RESULTS.md` are re-run or checked against the tree before merge; a commissioned review's findings are fact-checked one by one before any becomes an issue.

## Enforcement

Merge authority and `main` are human-only. The agent's permission layer, [.claude/settings.json](.claude/settings.json), allows issue work, feature-branch pushes and PR creation, and denies the merge commands, force pushes and pushes to `main`. It binds the agent's tooling, not GitHub:

- `Bash(gh pr merge:*)`
- `Bash(gh api * --method PUT:*)`
- `Bash(gh api * -X PUT:*)`
- `Bash(gh api --method PUT:*)`
- `Bash(gh api -X PUT:*)`
- `Bash(gh api *merge*)`
- `Bash(gh repo:*)`
- `Bash(git push --force:*)`
- `Bash(git push -f:*)`
- `Bash(git push --force-with-lease:*)`
- `Bash(git push * --force*)`
- `Bash(git push * -f*)`
- `Bash(git push origin main:*)`
- `Bash(git push origin main)`
- `Bash(git push -u origin main:*)`
- `Bash(git push --set-upstream origin main:*)`
- `Bash(git push origin HEAD:main:*)`
- `Bash(git push * main:*)`
- `Bash(git push * main)`
- `Bash(git push *:main)`
- `Bash(git push *:main *)`
- `Bash(git push *:refs/heads/main*)`

`tests/test_development_md.py` fails if this list and the settings file drift apart. `tests/test_license_hygiene.py` does the same for the Phase 6 license rule: it fails if any tracked text file names a file under the corpus root. Branch, issue, PR and label conventions are the machine-facing contracts in [docs/agents/issue-tracker.md](docs/agents/issue-tracker.md) and [docs/agents/triage-labels.md](docs/agents/triage-labels.md); every PR runs the CI described in [CLAUDE.md](CLAUDE.md) (Conventions, CI).

## Worked examples

- [Commit ff025ee](https://github.com/GovitPowerz/Speech/commit/ff025ee): the release seam returned an exactly zero gradient. Found by the Phase 5 exit gate on the project's own earlier claims, invisible to every finiteness and determinism gate (the in-engine gradient check read its own local fold and stayed correct). Every "trained" checkpoint of the two prior phases had been seed weights run through optimizer no-ops; the fix stashes the folded map, and the gate now asserts the seam against the analytic fold.
- [RESULTS.md, Phase 9 Task 9](RESULTS.md): a real retraction in the streaming decision layer, produced only by real corpus data with a trained causal net. With the fix reverted the entire committed streaming suite (38 legs) stays green, which is why the corpus tiers exist. The fix is one clamp; the phase 8 latency claim was rescoped from universal to conditional rather than left standing.
- [Commit c28ec2b](https://github.com/GovitPowerz/Speech/commit/c28ec2b): the 2015 production binary resurrected under Rosetta 2 as the parity oracle, after it turned out to be Mach-O x86_64 rather than the Linux ELF an earlier spec assumed. The parity proof it produced is frozen at tag `legacy-parity-v1` ([ADR-0001](docs/adr/0001-port-truth-after-the-parity-tag.md)).
- [Commit c02509e](https://github.com/GovitPowerz/Speech/commit/c02509e): an over-long weight pack loaded silently under the wrong architecture's name, on every cell. The brief's premise (strict length everywhere) was shown wrong by three committed artifacts that depend on the file-load tolerance; the fix splits the two seams the legacy itself splits.
- [RESULTS.md, Phase 10 mutation battery, item 6](RESULTS.md): a symmetric kernel substitution inside a shared step kernel that no test catches, measured (drift 1.14x against a pin sized for SIMD variance) rather than asserted away, and turned into a design-time rule in the module that would receive the edit.
- [Commit 723656b](https://github.com/GovitPowerz/Speech/commit/723656b): a 77.6x per-push speedup in the streaming re-smooth accepted only against a side-by-side equivalence oracle (9418 emissions and 9490 final entries bit-identical over four config profiles), with a mutation ladder that gives the oracle's resolution in holdback units.
