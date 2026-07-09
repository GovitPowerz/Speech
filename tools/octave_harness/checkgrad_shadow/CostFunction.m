function [cost,param_out,best_pos,MultiDeriv,MultiWeights,NNCostSeg,MultiDerivLID,MultiWeightsLID,NNCostLID] = CostFunction(param,PS,epoch)
  % HARNESS-LOCAL SHADOW of legacy/Optimizer_V6.2.2/functions/CostFunction.m, injected via
  % addpath ordering (tools/octave_harness/stage_checkgrad.m) so the REAL, UNMODIFIED
  % CheckGrad.m (which calls CostFunction directly, :36 + the per-weight loop) exercises a
  % cheap deterministic quadratic surrogate instead of shelling out to the engine
  % (system('python RunFsp.py ...')). The vendored CostFunction.m is untouched; this file
  % lives only here and is never used outside the checkgrad stage.
  %
  % f(w) = 0.5*sum(c.*(w-target).^2), df/dw = c.*(w-target) -- a pure quadratic form over
  % the REAL flat NN weight vector `w`, re-derived from `param` on every call via the REAL
  % vec2struct + nnet2MatFile (both pure/vendored, unmodified) -- exactly what CheckGrad's
  % own per-weight perturbation loop does (vec2struct(...,PS.VP.mask,...) +
  % nnet2MatFile(configStruct,'',1)). So a weight-kk epsilon nudge in `param` (applied by
  % CheckGrad via the real network2config/weights2nnet/vec2struct round-trip) shows up here
  % as a matching nudge in w(kk), and the analytic gradient c.*(w-target) is EXACT for a
  % quadratic form (no truncation error), so central-diff vs analytic should agree to
  % floating-point rounding alone.
  epoch; %#ok<VUNUS> -- accepted for signature parity with the real CostFunction, unused
  [K,J] = size(param);
  MultiDeriv = [];
  MultiWeights = [];
  NNCostSeg = zeros(1,J);
  for jj = 1:J
    [configStruct,~] = vec2struct(param(:,jj),PS.VP.mask,PS,'',0);
    [~,~,w,~,~] = nnet2MatFile(configStruct,'',1);
    if (jj == 1)
      Kw = length(w);
      MultiDeriv = zeros(Kw,J);
      MultiWeights = zeros(Kw,J);
    end
    idxw = (1:Kw)';
    target = 0.1*sin(0.37*idxw);
    c = 1.0 + 0.05*mod(idxw,5);
    d = w - target;
    NNCostSeg(jj) = 0.5*sum(c.*d.^2);
    MultiDeriv(:,jj) = c.*d;
    MultiWeights(:,jj) = w;
  end
  cost = NNCostSeg;
  param_out = param;
  best_pos = 1;
  MultiDerivLID = zeros(0,J);
  MultiWeightsLID = zeros(0,J);
  NNCostLID = zeros(1,J);
end
