function h = figure(varargin)
  % HARNESS-LOCAL no-op shadow (see CostFunction.m in this dir for the addpath-precedence
  % rationale). legacy CheckGrad.m's per-genome diagnostic plot (:91-96/:155-159) uses the
  % old-style `subplot 211` call form, which errors on this Octave/graphics-toolkit
  % combination ("invalid axes handle or RCN argument") independent of headlessness.
  % Shadowing figure/subplot/semilogy/hold/grid as no-ops lets the REAL CheckGrad.m body
  % run to completion unmodified; see IMPROVEMENTS.md (phase4c Octave-compat entry).
  h = -1;
end
