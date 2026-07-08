function stage_computecost(out_dir)
  % Drive the ComputeCost.m COST-ASSEMBLY block (legacy/Optimizer_V6.2.2/functions/
  % ComputeCost.m) over the COMMITTED phase4a/4b MultiConfigResults fixtures + crafted
  % per-balance variants, dumping the goldens the Python port (src/python/speech/
  % engine.py) is bit-pinned against.
  %
  % TIER: FALLBACK (stage-local transcription, documented in IMPROVEMENTS). ComputeCost.m
  % is NOT called wholesale: its top half (:20-282) shells out to the engine via
  % `system('python RunFsp.py ...')` (:173-191) after a config-write loop (vec2struct +
  % nnet2MatFile, :47-156) and then LOADS the worker .mat/.bin the shell-out wrote
  % (:216-282). Octave cannot run the engine, and the `!`-escape cleaning (:37) deletes any
  % pre-injected worker files before the shell-out, so there is no clean injection point
  % that leaves the vendored .m unmodified. Instead this stage transcribes the pure
  % ASSEMBLY lines (:374 sortrows, :428-652 the balance-law error + cost, plus the
  % :285-300 pooled-stats and :354-364 deriv-averaging/L2 blocks) line-for-line with
  % `% legacy:` provenance, and drives them on injected result matrices. Real MATLAB
  % semantics (sortrows stable-ascending, median, hist center-binning, std ddof=1, cumsum,
  % exp/log) are exercised by Octave; the Python port must match. The engine-shelling
  % top half is out of scope for the pure cost assembly (it is the seam, tested via
  % speech_rs.Engine in tests/pyo3).
  %
  % Dumps per group. STRICT (pure arithmetic, bit-exact everywhere): aggregate (sortrows),
  % average_derivs (col0/max(1,col1)), l2 (weights.^2 penalty), pooled mean, and the
  % crafted integer balance 0/3/4/5. CANARY-gated (libm-bearing): pooled std (sqrt), the
  % balance-10 LID calibration (hist/cumsum/std/exp/log), and the committed-fixture
  % realistic cases.

  R = struct();

  % ---- Committed MultiConfigResults fixtures (io::binary .bin -> matrix) --------------
  harness_dir = fileparts(mfilename('fullpath'));
  repo_root = fullfile(harness_dir, '..', '..');
  tier2_mcr = read_iobin(fullfile(repo_root, 'tests', 'reference_data', 'phase4a', ...
                                  'tier2_spectral_MultiConfigResults.bin'));
  twin_mcr = read_iobin(fullfile(repo_root, 'tests', 'reference_data', 'phase4b', ...
                                 'twin_train_MultiConfigResults.bin'));
  R.tier2_mcr = tier2_mcr;
  R.twin_mcr = twin_mcr;

  % ---- aggregate_workers: sortrows([1 2 3]) over concatenated worker matrices ----------
  % legacy: ComputeCost.m:219 MultiConfigResultsRecomp = [MultiConfigResultsRecomp;MultiConfigResults];
  % legacy: ComputeCost.m:374 MultiConfigResults = sortrows(MultiConfigResultsRecomp,[1 2 3]);
  agg_w1 = [2 1 1 11; 1 1 2 22];
  agg_w2 = [1 1 1 33; 2 1 2 44];
  agg_out = sortrows([agg_w1; agg_w2], [1 2 3]);
  R.agg_w1 = agg_w1;
  R.agg_w2 = agg_w2;
  R.agg_out = agg_out;

  % ---- average_derivs: col0/max(1,col1) ------------------------------------------------
  % legacy: ComputeCost.m:357 MultiDeriv(:,ii) = MultiDeriv(:,ii)./max(1,MultiDerivCount(:,ii));
  avg_in = [6 3; 10 0; -4 2; 0 5];
  avg_out = avg_in(:, 1) ./ max(1, avg_in(:, 2));
  R.avg_in = avg_in;
  R.avg_out = avg_out;

  % ---- l2_penalty: cost += l2*sum((w.*(1-isBias)).^2)/2 ; grad += l2*(w.*(1-isBias)) ----
  % legacy: ComputeCost.m:554 NNCostLID += PS.NS.L2_regul*sum((MultiWeightsLID.*(1-isBiasLID)).^2)/2;
  % legacy: ComputeCost.m:362 MultiDerivLID += PS.NS.L2_regul*(MultiWeightsLID.*(1-isBiasLID));
  l2_w = [0.5; -1.0; 2.0; 0.25; -0.5];
  l2_isBias = [0; 1; 0; 1; 0];
  l2_regul = 0.1;
  l2_cost = l2_regul * sum((l2_w .* (1 - l2_isBias)) .^ 2) / 2;
  l2_grad = l2_regul * (l2_w .* (1 - l2_isBias));
  R.l2_w = l2_w;
  R.l2_isBias = l2_isBias;
  R.l2_regul = l2_regul;
  R.l2_cost = l2_cost;
  R.l2_grad = l2_grad;

  % ---- pool_input_stats: :285-300 pooled mean/var over per-worker (nb,mean,std) ---------
  pool_nb = [100, 300];
  pool_mean = [1 3; 2 4];   % dim x worker
  pool_std = [0.5 1.5; 1.0 2.0];
  [pnb, pmean, pstd] = pool_stats(pool_nb, pool_mean, pool_std);
  R.pool_nb = pool_nb;
  R.pool_mean = pool_mean;
  R.pool_std = pool_std;
  R.pool_out_nb = pnb;
  R.pool_out_mean = pmean;
  R.pool_out_std = pstd;

  % ---- Crafted integer MultiConfigResults for the STRICT balance 0/3/4/5 pins ----------
  % Error_vad columns (1-based): 1 Pfa, 2 Pmiss, 3 (100-success), 4 cpu, 5 seg-num, 6 seg-denom.
  % Prepend [file conf chan] id columns; conf==1 everywhere so config_idx==1 selects all.
  cb_ev = [20 10 70 5 12 100;
           95 30 40 10 8 100;
           40 95 25 15 20 100];
  cb_id = [1 1 1; 2 1 1; 3 1 1];
  cb_mcr = [cb_id, cb_ev];
  R.cb_mcr = cb_mcr;
  [R.cb0_error, R.cb0_cost, R.cb0_nnseg, R.cb0_cpumean] = balance_cost(cb_mcr, 1, 0, 0, 0.5, 3);
  [R.cb3_error, R.cb3_cost, R.cb3_nnseg, R.cb3_cpumean] = balance_cost(cb_mcr, 1, 3, 0, 0.5, 3);
  [R.cb4_error, R.cb4_cost, R.cb4_nnseg, R.cb4_cpumean] = balance_cost(cb_mcr, 1, 4, 0, 0.5, 3);
  [R.cb5_error, R.cb5_cost, R.cb5_nnseg, R.cb5_cpumean] = balance_cost(cb_mcr, 1, 5, 0, 0.5, 3);

  % ---- Committed tier2 (algo 3) realistic balance goldens (canary-gated) ----------------
  [R.tier2_b0_error, R.tier2_b0_cost, R.tier2_b0_nnseg, R.tier2_b0_cpumean] = balance_cost(tier2_mcr, 1, 0, 0, 0.5, 3);
  [R.tier2_b5_error, R.tier2_b5_cost, R.tier2_b5_nnseg, R.tier2_b5_cpumean] = balance_cost(tier2_mcr, 1, 5, 0, 0.5, 3);

  % ---- Committed twin (algo 6) raw NNCostSeg / NNCostLID pieces (canary-gated) ----------
  % legacy: ComputeCost.m:430 NNcost = sum(Error_vad(:,5))/max(1e-6,sum(Error_vad(:,end)));
  % legacy: ComputeCost.m:552 NNCostLID = sum(Error_vad(:,15))/max(1e-6,sum(Error_vad(:,end-1)));
  twin_ev = twin_mcr(twin_mcr(:, 2) == 1, 4:end);
  R.twin_nnseg = sum(twin_ev(:, 5)) / max(1e-6, sum(twin_ev(:, end)));
  R.twin_nnlid = sum(twin_ev(:, 15)) / max(1e-6, sum(twin_ev(:, end - 1)));

  % ---- Crafted 2-class balance-10 (the cutoff-search branch, :560-593; canary) ----------
  % Error_vad width 20: 1-5 as above, 6-14 filler, 15 LID-num, 16 flag, 17/18 the two
  % per-class in-band scores (>150 -> class fired, value = score-200), 19 LID-denom, 20 seg-denom.
  b10a_ev = zeros(4, 20);
  b10a_ev(:, 1) = [10; 20; 30; 40];    % Pfa (unused by balance 10 error)
  b10a_ev(:, 2) = [5; 6; 7; 8];        % Pmiss
  b10a_ev(:, 3) = [90; 80; 70; 60];    % 100-success
  b10a_ev(:, 4) = [5; 5; 5; 5];        % cpu
  b10a_ev(:, 5) = [12; 8; 20; 4];      % seg-num
  b10a_ev(:, 15) = [50; 40; 30; 20];   % LID-num
  b10a_ev(:, 16) = [0; 0; 0; 0];       % flag (recomputed inside)
  b10a_ev(:, 17) = [250; 280; 40; 25]; % class 1 (files 1,2 fired: 50, 80)
  b10a_ev(:, 18) = [30; 20; 210; 260]; % class 2 (files 3,4 fired: 10, 60)
  b10a_ev(:, 19) = [100; 100; 100; 100]; % LID-denom
  b10a_ev(:, 20) = [150; 150; 150; 150]; % seg-denom
  b10a_id = [(1:4)', ones(4, 1), ones(4, 1)];
  b10a_mcr = [b10a_id, b10a_ev];
  R.b10a_mcr = b10a_mcr;
  [R.b10a_error, R.b10a_cost, R.b10a_nnseg, R.b10a_cpumean, R.b10a_nnlid, R.b10a_cutoff] = balance10_cost(b10a_mcr, 1, 0);

  % ---- Crafted 3-class balance-10 (the else branch, :595-616; canary) -------------------
  % Error_vad width 21: 17/18/19 = three per-class scores, 20 LID-denom, 21 seg-denom.
  b10b_ev = zeros(3, 21);
  b10b_ev(:, 1) = [10; 20; 30];
  b10b_ev(:, 5) = [6; 9; 3];
  b10b_ev(:, 15) = [45; 35; 25];
  b10b_ev(:, 17) = [250; 40; 30];   % class 1 fired file 1 (val 50)
  b10b_ev(:, 18) = [35; 270; 45];   % class 2 fired file 2 (val 70)
  b10b_ev(:, 19) = [20; 30; 220];   % class 3 fired file 3 (val 20)
  b10b_ev(:, 20) = [100; 100; 100]; % LID-denom
  b10b_ev(:, 21) = [150; 150; 150]; % seg-denom
  b10b_id = [(1:3)', ones(3, 1), ones(3, 1)];
  b10b_mcr = [b10b_id, b10b_ev];
  R.b10b_mcr = b10b_mcr;
  [R.b10b_error, R.b10b_cost, R.b10b_nnseg, R.b10b_cpumean, R.b10b_nnlid, R.b10b_cutoff] = balance10_cost(b10b_mcr, 1, 0);

  save('-v7', fullfile(out_dir, 'computecost.mat'), '-struct', 'R');

  printf('OCTAVE_STAGE computecost cb0=%g cb5=%g b10a=%g b10b=%g agg_rows=%d pool_nb=%d\n', ...
         R.cb0_cost, R.cb5_cost, R.b10a_cost, R.b10b_cost, size(agg_out, 1), pnb);
end


function M = read_iobin(path)
  % io::binary: i64 LE rows, i64 LE cols, f64 LE column-major.
  fid = fopen(path, 'r');
  if (fid < 0)
    error('stage_computecost: cannot open %s', path);
  end
  rows = fread(fid, 1, 'int64');
  cols = fread(fid, 1, 'int64');
  data = fread(fid, rows * cols, 'double');
  fclose(fid);
  M = reshape(data, rows, cols);
end


function [nb, meanOut, stdOut] = pool_stats(nbIn, meanIn, stdIn)
  % legacy: ComputeCost.m:285-300 (single-config slice; nbIn/meanIn/stdIn indexed by worker jj).
  dim = size(meanIn, 1);
  nb = 0;
  meanOut = zeros(dim, 1);
  for jj = 1:numel(nbIn)
    nb = nb + nbIn(jj);                                   % legacy: :289
    meanOut = meanOut + nbIn(jj) * meanIn(:, jj);         % legacy: :290
  end
  meanOut = meanOut / nb;                                 % legacy: :292
  stdOut = zeros(dim, 1);
  for jj = 1:numel(nbIn)
    stdOut = stdOut + nbIn(jj) * (stdIn(:, jj) .^ 2 + (meanOut - meanIn(:, jj)) .^ 2); % legacy: :297
  end
  stdOut = sqrt(stdOut / nb);                             % legacy: :299
end


function [errv, cost, nnseg, cpumean] = balance_cost(mcr, ii, balance, mode, balbp, algo)
  % Transcribes the balance 0/3/4/5 error + final cost for one config. mode==4 -> coeffRprop=0.
  % legacy: ComputeCost.m:429 Error_vad = MultiConfigResults(MultiConfigResults(:,2)==ii,4:end);
  Error_vad = mcr(mcr(:, 2) == ii, 4:end);
  % legacy: ComputeCost.m:430 NNcost = sum(Error_vad(:,5))/max(1e-6,sum(Error_vad(:,end)));
  NNcost = sum(Error_vad(:, 5)) / max(1e-6, sum(Error_vad(:, end)));
  nnseg = NNcost;
  % legacy: ComputeCost.m:432-436
  if (mode >= 0)
    cpu_mean = median(Error_vad(:, 4));
  else
    cpu_mean = 0.0;
  end
  cpumean = cpu_mean;
  if (mode == 4)
    coeffRprop = 0;
  else
    coeffRprop = 1;
  end
  if (balance == 0)
    % legacy: ComputeCost.m:440
    errv = Error_vad(:, 1) + Error_vad(:, 2) + 3 * (cpu_mean / 5);
  elseif (balance == 3)
    % legacy: ComputeCost.m:450 (over-90 saturation OFF: the 0* coefficient)
    errv = (100 * (1 - coeffRprop) * NNcost + coeffRprop * (balbp * (Error_vad(:, 1) + 0 * (Error_vad(:, 1) > 90) .* ((Error_vad(:, 1) - 90) .^ 2)) + (1 - balbp) * (Error_vad(:, 2) + 0 * (Error_vad(:, 2) > 90) .* ((Error_vad(:, 2) - 90) .^ 2)) + (algo == 1) * (1 * cpu_mean .* (cpu_mean > 100) + 0.01 * cpu_mean .* (cpu_mean <= 100)))) / 100;
  elseif (balance == 4)
    % legacy: ComputeCost.m:458 (over-90 saturation ON: the 1* coefficient)
    errv = (100 * (1 - coeffRprop) * NNcost + coeffRprop * (balbp * (Error_vad(:, 1) + 1 * (Error_vad(:, 1) > 90) .* ((Error_vad(:, 1) - 90) .^ 2)) + (1 - balbp) * (Error_vad(:, 2) + 1 * (Error_vad(:, 2) > 90) .* ((Error_vad(:, 2) - 90) .^ 2)) + (algo == 1) * (1 * cpu_mean .* (cpu_mean > 100) + 0.01 * cpu_mean .* (cpu_mean <= 100)))) / 100;
  elseif (balance == 5)
    % legacy: ComputeCost.m:466
    errv = 100 * (1 - coeffRprop) * NNcost + coeffRprop * (Error_vad(:, 3) + 0.0 * cpu_mean);
  else
    error('stage_computecost: balance_cost only ports 0/3/4/5, got %d', balance);
  end
  % legacy: ComputeCost.m:646-651 (balance < 6, mode not 4/1: cost = mean(error))
  cost = mean(errv);
end


function [errv, cost, nnseg, cpumean, nnlid, cutoff_out] = balance10_cost(mcr, ii, mode)
  % Transcribes the balance-10 LID calibration error + cost for one config.
  % legacy: ComputeCost.m:429
  Error_vad = mcr(mcr(:, 2) == ii, 4:end);
  % legacy: ComputeCost.m:430
  NNcost = sum(Error_vad(:, 5)) / max(1e-6, sum(Error_vad(:, end)));
  nnseg = NNcost;
  % legacy: ComputeCost.m:432-436
  if (mode >= 0)
    cpumean = median(Error_vad(:, 4));
  else
    cpumean = 0.0;
  end
  % legacy: ComputeCost.m:552 (raw NNCostLID, L2 term omitted -- applied by forward_backward)
  nnlid = sum(Error_vad(:, 15)) / max(1e-6, sum(Error_vad(:, end - 1)));
  scores = Error_vad(:, 17:end - 2);
  if (size(scores, 2) == 2)
    % legacy: ComputeCost.m:561-576 (cutoff search)
    xval = (0:0.01:99.99) + 0.005;
    tmp = Error_vad(Error_vad(:, 18) > 150, 18) - 200;
    [n, t] = hist(tmp, xval);
    n = cumsum(100 * n / sum(n));
    tmp = 300 - Error_vad(Error_vad(:, 17) > 150, 17);
    [n2, t] = hist(tmp, xval);
    n2 = fliplr(cumsum(100 * fliplr(n2) / sum(n2)));
    [val, pos] = min(abs(n - n2));
    if ((n(pos) == 0) && (n2(pos) == 0))
      cutoff = (t(find(n, 1, 'first')) + t(find(n2, 1, 'last'))) / 2;
    else
      cutoff = t(pos);
    end
    cutoff = max(0.1, min(99.9, cutoff));
    cutoff_out = cutoff;
    % legacy: ComputeCost.m:580-586
    relError1 = (Error_vad(:, 17) > 150) .* (Error_vad(:, 17) - 200 - 100 + cutoff);
    relError1 = relError1 .* ((relError1 >= 0) / (cutoff) + (relError1 < 0) / (100 - cutoff));
    relError2 = (Error_vad(:, 18) > 150) .* (Error_vad(:, 18) - 200 - cutoff);
    relError2 = relError2 .* ((relError2 >= 0) / (100 - cutoff) + (relError2 < 0) / (cutoff));
    relError = relError1 + relError2;
    LIDflag = 100 * (relError > 0);
    LIDscore = (100 - LIDflag + 100 * exp(-relError)) / 100;
    % legacy: ComputeCost.m:590
    errv = LIDscore - log(max(1e-24, 1 * min(1, std((Error_vad(:, 17) > 150) .* (Error_vad(:, 17) - 200) + (Error_vad(:, 17) <= 150) .* Error_vad(:, 17))))) + 10 * (cutoff <= 0.1) + 10 * (cutoff >= 99.9);
  else
    % legacy: ComputeCost.m:598-612 (else branch; per-file cutoff vector)
    cutoff_out = -1;   % sentinel: per-file cutoff is a vector in this branch
    relError = zeros(size(scores, 1), 1);
    for mm = 1:size(scores, 2)
      tmp = scores;
      tmp(:, mm) = zeros(size(scores, 1), 1);
      cutoff = max(0.01, min(99.9, (Error_vad(:, 17 + mm - 1) > 150) .* max(tmp, [], 2)));
      relErrortmp = (Error_vad(:, 17 + mm - 1) > 150) .* (Error_vad(:, 17 + mm - 1) - 200 - cutoff);
      relErrortmp = relErrortmp .* ((relErrortmp >= 0) ./ (100 - cutoff) + (relErrortmp < 0) ./ (cutoff));
      relError = relError + relErrortmp;
    end
    LIDflag = 100 * (relError > 0);
    LIDscore = (100 - LIDflag + 100 * exp(-relError)) / 100;
    % legacy: ComputeCost.m:612
    errv = LIDscore;
  end
  % legacy: ComputeCost.m:630 (balance >= 10: cost = mean(error.^2))
  cost = mean(errv .^ 2);
end
