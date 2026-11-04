"""Run-lifecycle drivers: init, train, retrain, test (the Legacy regime on `RunState`), and
the Launcher (`baseline`: a `BaselineSpec` prepared into a `PreparedRun`, trained by the Modern
loop, scored, recorded; `checkpoint` is the directory of packs both regimes write)."""
