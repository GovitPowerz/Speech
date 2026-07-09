function stage_vec2struct(out_dir)
  % Drive the REAL vendored vec2struct.m (legacy/Optimizer_V6.2.2/functions/vec2struct.m,
  % 1693 LOC) over crafted param vectors + masks for six algo/config cases, dumping the
  % genome<->config bijection goldens the Python port (src/python/speech/genome.py) is
  % bit-pinned against. TIER 1: the real function (+ the real printConfig.m it calls at
  % mode==1) are exercised unchanged.
  %
  % Per case we dump: the INPUT param vector, the returned out_param (the mask/sortrows/
  % clamp re-encoding), count_param (the genome-length walk invariant -- risk R3), and the
  % printConfig-serialized .config text (the engine-facing key->value dict, weights excluded
  % and emitted via the .bin codec instead). Cases: algo0 VRCTS, algo1 TDC, algo2 LTSV
  % (calibration-law decode + clamps in isolation, no NN), algo3 spectral+BLSTM (full NN
  % weight walk), algo6 twin (SAD + LID mirror), and a masked algo3 (Force_Symetry +
  % Force_Identical_Rows tie blocks, plus a scalar-field mask write-back).
  %
  % param is a deterministic per-case formula (sin/cos of the index) -- vec2struct is PURE
  % arithmetic (rem/round/abs/min/max/sortrows, no libm), so every golden is STRICT bits /
  % string-exact on every platform.

  cases = {'algo0', 'tdc', 'calib', 'spectral', 'twin', 'masked'};
  counts = struct();
  R = struct();

  for ci = 1:numel(cases)
    name = cases{ci};
    [PS, mask, overrides] = build_case(name);

    % Pass 1: size the genome (count_param is mask-independent).
    probe = zeros(6000, 1);
    [~, ~, cp] = vec2struct(probe, struct(), PS, '', 0);
    n = cp - 1;

    % Pass 2: the real run over an exact-length param, mode==1 to trigger printConfig.
    param = mk_param(n, ci);
    for kk = 1:size(overrides, 1)
      param(overrides(kk, 1)) = overrides(kk, 2);
    end
    conf_file = fullfile(out_dir, sprintf('vec2struct_%s.config', name));
    [~, out_param, count_param] = vec2struct(param, mask, PS, conf_file, 1);

    R.([name '_param']) = param;
    R.([name '_out_param']) = out_param;
    R.([name '_count']) = count_param;
    counts.(name) = count_param;
  end

  save('-v7', fullfile(out_dir, 'vec2struct.mat'), '-struct', 'R');

  printf('OCTAVE_STAGE vec2struct algo0=%d tdc=%d calib=%d spectral=%d twin=%d masked=%d\n', ...
         counts.algo0, counts.tdc, counts.calib, counts.spectral, counts.twin, counts.masked);
end


function param = mk_param(n, seed)
  % Deterministic, per-case-distinct column vector spanning negatives, [0,adim], and >adim
  % so abs/sign/min-clamp/rem-round decode branches are all exercised. adim is 10 for every
  % case, so the range ~[-3.5, 10.5] straddles the decode/clamp thresholds.
  idx = (1:n)';
  param = 10 * (0.35 + 0.5 * sin(0.7 * seed + 0.37 * idx) + 0.2 * cos(0.11 * idx));
end


function [PS, mask, overrides] = build_case(name)
  % overrides: Kx2 [1-based-index, value] param overrides applied AFTER mk_param (used by
  % the calibration case to pin specific law decodes + the [0,1] clamps).
  overrides = zeros(0, 2);
  mask = struct();

  switch name
    case 'algo0'
      PS = base_ps(0);
    case 'tdc'
      PS = base_ps(1);
    case 'calib'
      PS = base_ps(2);
      % algo2 (LTSV, no NN): front matter (1..21) + spectral common (22..37) put the six
      % calibration-law params at 38..43. Pin CostLawSpeech->sqrt(3), NoSpeech->cubic(4),
      % ParamSpeech clamp-high (2->1), ParamNoSpeech clamp-low (-0.5->0), ThreshSpeech 0.3,
      % ThreshNoSpeech clamp-high (2.5->1).  adim=10, so rem(round(5*|p|/10),5):
      %   p=16 -> round(8)=8 -> 3 (sqrt); p=18 -> round(9)=9 -> 4 (cubic).
      overrides = [38, 16; 39, 18; 40, 20; 41, -5; 42, 3; 43, 25];
    case 'spectral'
      PS = base_ps(3);
    case 'twin'
      PS = base_ps(6);
      PS.VP.mask.nnet = 1;      % -> AlgName_BackPropOutputNetworkOnly (count-neutral)
      PS.VP.mask.nnetLID = 1;   % -> AlgName_LID_BackPropOutputNetworkOnly
    case 'masked'
      PS = base_ps(3);
      mask.Force_Symetry = 1;         % backward LSTM blocks tied to forward
      mask.Force_Identical_Rows = 1;  % block jj>1 tied to block 0; output ii==1 repmat
      mask.AlgName_decision_thresh_rising = 0.7;  % scalar-field mask + write-back
    otherwise
      error('stage_vec2struct: unknown case %s', name);
  end
end


function PS = base_ps(algo)
  % Minimal-but-faithful PS mirroring Train_BLSTM_Seg.m's construction (:126-345) with TINY
  % nets (keeps genomes small and the count walk auditable). Only the fields vec2struct
  % actually reads are populated.
  PS.VP.nbproc.outer = 7;
  PS.VP.nbproc.inner = 1;
  PS.VP.OutputFile = 'MultiConfigResults.mat';
  PS.VP.Display_MillisecondsPerPixel = 64;
  PS.FS.name_dir_fig = '/tmp/fsp/Figures';
  PS.FS.name_dir = '/tmp/fsp';
  PS.VP.offset = 0;
  PS.VP.durmax = 240;
  PS.VP.algo = algo;
  PS.VP.nbworker = 1;   % keep 1: nbworker>1 injects a pwd-dependent LockFilesDir
  PS.VP.epoch = 0;
  PS.VP.adim = 10;
  PS.NS.coeff_NN = 10;
  PS.VP.VRCTS_isFast = 1;
  PS.VP.VRCTS_force = 0;
  PS.VP.balance = 6;    % adds AlgName_CostPonderation (not Pruning: <9)
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
  PS.NS.LSTM_net_size = [3, 2];
  PS.NS.LSTMSubSampling = [2];
  PS.NS.Output_net_size = [4, 2, 1];   % Output(1) = 2*LSTM(end) = 4
  PS.NS.OutputSubSampling = [1, 1];
  PS.NS.LSTM.MaxSat = -10;             % <0 -> no MaxSaturation fields
  PS.NS.BackPropWER = -1.0;
  PS.NS.BalanceBackProp = 0.0;
  PS.NS.InputNormalizationType = -1;
  PS.NS.IsCellsPeepholesActive = 1;
  PS.NS.IsGatesPeepholesActive = 1;
  PS.NS.IsGatesRecurrentPeepholesActive = 1;

  % LID sub-net (read only for algo==6).
  PS.NS.LID.NNType = 0;
  PS.NS.LID.BackPropagationActivated = 0;
  PS.NS.LID.LSTM_net_size = [2, 3];
  PS.NS.LID.LSTMSubSampling = [1];
  PS.NS.LID.Output_net_size = [6, 1];  % Output(1) = 2*LID_LSTM(end) = 6
  PS.NS.LID.OutputSubSampling = [1];
  PS.NS.LID.Mode = 7;
  PS.NS.LID.PostProcessMode = 0;
  PS.NS.LID.TargetEnforcementStep = 0;
  PS.NS.LID.BackPropWER = -1.0;
  PS.NS.LID.classes_ponderations = [];
  PS.NS.LID.InputNormalizationType = -1;
  PS.NS.LID.IsCellsPeepholesActive = 1;
  PS.NS.LID.IsGatesPeepholesActive = 1;
  PS.NS.LID.IsGatesRecurrentPeepholesActive = 1;
  PS.NS.LID.LSTM.MaxSat = -10;
  PS.VP.LID.nnmatfile = 'LIDNNweights.mat';
end
