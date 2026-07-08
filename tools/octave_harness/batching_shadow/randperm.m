function p = randperm(n)
  % HARNESS-LOCAL shadow of the builtin randperm, injected via addpath ordering (Octave
  % addpath prepends -- this dir is added AFTER functions_dir in stage_batching.m, so it
  % takes precedence). Real CreateBatches.m calls `X(randperm(length(X)))` to shuffle each
  % per-class index pool; a real PRNG would make the fixture non-reproducible across
  % extractor runs. Substituting a FIXED, deterministic reverse-order permutation
  % (p = n:-1:1) keeps CreateBatches.m itself UNMODIFIED (TIER 1) while making its output
  % fully reproducible -- and, since it is a simple closed-form function of n (not an
  % opaque lookup table), the Python create_batches port's test can reproduce the SAME
  % permutation-per-length with a trivial stand-in `rng.permutation` (see
  % tests/test_phase4c_batching.py), pinning create_batches's own shuffle output, not just
  % GetNewBatch's rotation. See IMPROVEMENTS.md (phase4c batching Octave-compat entry).
  p = n:-1:1;
end
