function stage_masking(out_dir)
  % Drive the REAL vendored MaskingValidation.m (54 LOC) + vec2struct.m unchanged over a
  % PASS and a FAIL case. TIER 1 (real functions); no shadow needed (no CostFunction /
  % plotting calls on this path).
  %
  % PASS: mask a single well-formed scalar field (AlgName_decision_thresh_rising=0.7 --
  % the same case Task 8 already bit-pins as a correct round trip). hasFailed must be 0.
  %
  % FAIL: mask AlgName_min_speech (a _padding_block-family vector field, vec2struct.m
  % :XXX/genome.py `_padding_block`) with a NEGATIVE last component. `_padding_block`'s
  % ENCODE ((fv+0.1)*adim, using the RAW masked fv) is NOT the algebraic inverse of its
  % DECODE (-0.1+abs(p/adim)) for fv<0 -- abs() destroys the sign asymmetrically, so
  % out_param, re-decoded with mask=[], recovers a DIFFERENT value than the mask
  % originally forced (measured: mask -5 encodes to out_param -49, which decodes back to
  % +4.8, not -5 and not the algo<5 floor-at-0 clamp genome.py's first call stores).
  % Discovered by direct execution against the real vendored functions, not guessed: the
  % field is presumably meant to hold a non-negative duration, and MaskingValidation
  % correctly flags the mismatch -- this is the validator's intended job. hasFailed must
  % be 1. See IMPROVEMENTS.md (phase4c MaskingValidation entry).

  cases = {'pass', 'fail'};
  R = struct();
  for ci = 1:numel(cases)
    name = cases{ci};
    PS = base_ps_masking();
    PS.FS.logFile = fullfile(out_dir, sprintf('masking_%s_log.txt', name));
    switch name
      case 'pass'
        PS.VP.mask.AlgName_decision_thresh_rising = 0.7;
      case 'fail'
        PS.VP.mask.AlgName_min_speech = [1.0, 1.0, -5.0];
      otherwise
        error('stage_masking: unknown case %s', name);
    end

    probe = zeros(6000, 1);
    [~, ~, cp] = vec2struct(probe, struct(), PS, '', 0);
    n = cp - 1;
    idx = (1:n)';
    vector = 10 * (0.35 + 0.5*sin(0.41*idx) + 0.2*cos(0.13*idx));

    [hasFailed, count_param] = MaskingValidation(vector, PS, 0);

    R.([name '_hasFailed']) = double(hasFailed);
    R.([name '_count']) = count_param;
    R.([name '_vector']) = vector;
  end

  save('-v7', fullfile(out_dir, 'masking.mat'), '-struct', 'R');
  printf('OCTAVE_STAGE masking pass_failed=%d fail_failed=%d\n', R.pass_hasFailed, R.fail_hasFailed);
end


function PS = base_ps_masking()
  % Mirrors stage_vec2struct.m's base_ps(3) (the validated "spectral" algo-3 case).
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
  PS.NS.LSTM_net_size = [3, 2];
  PS.NS.LSTMSubSampling = [2];
  PS.NS.Output_net_size = [4, 2, 1];
  PS.NS.OutputSubSampling = [1, 1];
  PS.NS.LSTM.MaxSat = -10;
  PS.NS.BackPropWER = -1.0;
  PS.NS.BalanceBackProp = 0.0;
  PS.NS.InputNormalizationType = -1;
  PS.NS.IsCellsPeepholesActive = 1;
  PS.NS.IsGatesPeepholesActive = 1;
  PS.NS.IsGatesRecurrentPeepholesActive = 1;
end
