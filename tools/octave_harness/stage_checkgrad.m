function stage_checkgrad(out_dir)
  % Drive the REAL vendored CheckGrad.m (legacy/Optimizer_V6.2.2/functions/CheckGrad.m,
  % 163 LOC) unchanged, over a tiny algo-3 spectral+BLSTM net. TIER 1 (real function), with
  % TWO harness-local Octave-compat accommodations living in checkgrad_shadow/ (neither
  % touches the vendored .m; see that dir's CostFunction.m/figure.m for the rationale):
  %   1. A CostFunction.m SHADOW (the real one shells out to the engine via
  %      system('python RunFsp.py ...')) implementing a pure quadratic surrogate over the
  %      REAL flat NN weight vector (re-derived from param via the REAL vec2struct +
  %      nnet2MatFile on every call).
  %   2. No-op figure/subplot/semilogy/hold/grid shadows (CheckGrad's old-style
  %      `subplot 211` plotting call errors on this Octave/graphics-toolkit combination).
  % Both are added via addpath(shadow_dir) AFTER run_stage.m's addpath(functions_dir) --
  % Octave addpath prepends, so the later call wins and the shadow dir's files resolve
  % before the same-named vendored/builtin ones.
  %
  % PS.NS.LID.BackPropagationActivated = 0 keeps CheckGrad's algo-6 LID branch (:101-161)
  % unexercised -- Task 10 pins the SAD gradient-check path only; algo 6's LID branch is
  % structurally the same central-diff loop mirrored onto the LID net (LID_ prefix), and
  % the vec2struct/config bijection for algo 6 is already bit-pinned (Task 8).
  %
  % PS.Corpora.Train.Batches.WorstCases = [] (the CreateBatches degenerate-gate output when
  % nbOfTargetClasses*nbOfWorstCases+nbOfCasesPerBatch > file_nb, :15-17) takes CheckGrad's
  % "full corpus" branch (:31-34), skipping GetNewBatch entirely -- WriteWeightedListing
  % still runs for real and needs a fabricated Corpora.Train.listing cell array + a
  % writable PS.FS.listing path (both provided here).
  %
  % epsilon = 1e-5 (CheckGrad.m:12, unconditional) drives the central diff
  % (PlusNNCost-MinusNNCost)/(2*epsilon); the surrogate is an EXACT quadratic form, so the
  % central diff carries no truncation error -- analytic and numeric derivatives agree to
  % floating-point rounding alone, making this a STRICT-arithmetic golden.

  harness_dir = fileparts(mfilename('fullpath'));
  shadow_dir = fullfile(harness_dir, 'checkgrad_shadow');
  addpath(shadow_dir);

  PS = base_ps_checkgrad();

  nb_files = 3;
  PS.Corpora.Train.file_nb = nb_files;
  PS.Corpora.Train.filesValues = [1 1.0 2.0; 0 1.0 3.0; 1 1.0 1.0];
  for ii = 1:nb_files
    PS.Corpora.Train.listing{ii,1}.filename = sprintf('f%d.wav', ii);
    PS.Corpora.Train.listing{ii,1}.segfilename = sprintf('f%d.seg', ii);
    PS.Corpora.Train.listing{ii,1}.lang = 'eng';
    PS.Corpora.Train.listing{ii,1}.dial = 'us';
    PS.Corpora.Train.listing{ii,1}.duration = 2.0;
  end
  PS.Corpora.Train.Batches.WorstCases = [];

  PS.FS.listing = fullfile(out_dir, 'checkgrad_listing.flst');
  PS.FS.logFile = fullfile(out_dir, 'checkgrad_log.txt');

  % Size the genome (mask-independent walk, mirrors stage_vec2struct.m's probe pattern).
  probe = zeros(6000, 1);
  [~, ~, cp] = vec2struct(probe, struct(), PS, '', 0);
  n = cp - 1;

  % Deterministic param spanning negatives/[0,adim]/>adim, distinct from stage_vec2struct's
  % own formula so the two stages' fixtures are not accidentally identical.
  idx = (1:n)';
  nnet_in = 10 * (0.35 + 0.5*sin(0.53*idx) + 0.2*cos(0.17*idx));

  [MultiDeriv_BackProp, MultiDerivLID_BackProp, MultiDeriv_Num, MultiDerivLID_Num, MultiWeights, MultiWeightsLID] = CheckGrad(nnet_in, PS);

  Kw = size(MultiDeriv_BackProp, 1);

  save('-v7', fullfile(out_dir, 'checkgrad.mat'), 'nnet_in', 'MultiWeights', 'MultiDeriv_BackProp', 'MultiDeriv_Num');

  printf('OCTAVE_STAGE checkgrad n=%d Kw=%d\n', n, Kw);
end


function PS = base_ps_checkgrad()
  % Minimal-but-faithful PS (mirrors stage_vec2struct.m's base_ps), TINY net (LSTM [1,1],
  % Output [2,1]) to keep CheckGrad's O(length(weights)) central-diff loop fast.
  PS.VP.nbproc.outer = 7;
  PS.VP.nbproc.inner = 1;
  PS.VP.OutputFile = 'MultiConfigResults.mat';
  PS.VP.Display_MillisecondsPerPixel = 64;
  PS.FS.name_dir_fig = '/tmp/fsp/Figures';
  PS.FS.name_dir = '/tmp/fsp';
  PS.VP.offset = 0;
  PS.VP.durmax = 240;
  PS.VP.algo = 3;
  PS.VP.nbworker = 1;
  PS.VP.epoch = 0;
  PS.VP.adim = 10;
  PS.NS.coeff_NN = 10;
  PS.VP.VRCTS_isFast = 1;
  PS.VP.VRCTS_force = 0;
  PS.VP.balance = 6;
  PS.VP.exclude_nontrans = 0;
  PS.VP.useVRCTSFeatures = 0;
  PS.VP.nnmatfile = 'NNweights.mat';
  PS.VP.mask = struct();
  PS.VP.minSegmentLength = 0;
  PS.VP.addNoise = 0;
  PS.Corpora.mappingFile = '/tmp/fsp/languagemapping.csv';
  PS.FS.listing = '/tmp/fsp/fileslisting';

  PS.NS.NNType = 0;
  PS.NS.BackPropagationActivated = 1;
  PS.NS.LSTM_net_size = [1, 1];
  PS.NS.LSTMSubSampling = [1];
  PS.NS.Output_net_size = [2, 1];
  PS.NS.OutputSubSampling = [1];
  PS.NS.LSTM.MaxSat = -10;
  PS.NS.BackPropWER = -1.0;
  PS.NS.BalanceBackProp = 0.0;
  PS.NS.InputNormalizationType = -1;
  PS.NS.IsCellsPeepholesActive = 1;
  PS.NS.IsGatesPeepholesActive = 1;
  PS.NS.IsGatesRecurrentPeepholesActive = 1;

  PS.NS.LID.BackPropagationActivated = 0;
end
