function stage_qpso(out_dir)
  % Phase 4c Task 11: drive the MODIFIED-COPY QuantumPSO (tools/octave_harness/qpso_modified/)
  % over a shared random table and dump the per-epoch trajectory + a standalone Levy sample.
  % The modified copy substitutes rand/randperm/stblrnd with table reads (a stage-global
  % cursor) and CostFunction with the quadratic surrogate; the wall-clock reseed is removed.
  % The Python port (speech.optimizers.quantum_pso) replays the SAME committed table and is
  % pinned against these dumps.
  %
  % The qpso_modified dir is addpath'd with '-begin' so its QuantumPSO shadows the vendored
  % one on the path (no other stage calls QuantumPSO; the vendored normmat/forcecol are still
  % reached for the init positions/velocities).
  harness_dir = fileparts(mfilename('fullpath'));
  addpath(fullfile(harness_dir, 'qpso_modified'), '-begin');
  % normmat/forcecol (used for the init positions/velocities) live in functions/hiddenutils.
  addpath(fullfile(harness_dir, '..', '..', 'legacy', 'Optimizer_V6.2.2', 'functions', 'hiddenutils'));

  table = read_bin_col(fullfile(out_dir, 'qpso_random_table.bin'));

  % ---- Standalone Levy dump: tbl_stblrnd over the first 2*K table entries (V then W) ----
  % Pins levy_stable_cms independently of the full run (V uses table(1:8), W table(9:16)).
  K = 8;
  qpso_reset_cursor(table);
  levy_out = tbl_stblrnd(1.3, 1, 0.5, 0, 1, K);   % 1 x K
  levy_in = table(1:2*K);                          % 2K x 1 (documented input slice)

  % ---- Main run params (small D/ps/me), from Train_BLSTM.m:820-902 shapes ----
  D = 3; ps = 4; me = 3; adim = 10.0;
  shw = 1; ac1 = 2.1; ac2 = 2.1; iw1 = 0.9; iw2 = 0.6; iwe = 300;
  ergrd = 1e-99; ergrdep = 500; errgoal = NaN; trelea = 3; PSOseed = 1;
  PSOparams = [shw me ps ac1 ac2 iw1 iw2 iwe ergrd ergrdep errgoal trelea PSOseed];
  VR = [zeros(D,1), adim*ones(D,1)];               % [minx maxx] per dim, Train_BLSTM.m:853
  mv = (VR(:,2)-VR(:,1))/2;                         % mvden = 2, Train_BLSTM.m:857
  minmax = 0;
  PSOseedValue = [5 4 6; 3 5 7; 6 2 4; 4 6 5];      % fixed ps x D seed (Train_BLSTM PSOseedparam analog)
  center = [3.5 4.5 6.5];                           % surrogate cost minimum (off every seed row -> gbest evolves)

  cost_fn = @(pos, mode) qpso_surrogate(pos, mode, center);

  qpso_reset_cursor(table);
  res = QuantumPSO(cost_fn, PSOparams, D, mv, VR, minmax, PSOseedValue);

  init_pos = res.init_pos;
  init_pbest = res.init_pbest;
  init_pbestval = res.init_pbestval;
  init_gbest = res.init_gbest;
  init_gbestval = res.init_gbestval;
  pos_traj = res.pos_traj;
  pbest_traj = res.pbest_traj;
  pbestval_traj = res.pbestval_traj;
  gbest_traj = res.gbest_traj;
  gbestval_traj = res.gbestval_traj;
  final_out = res.OUT;
  cursor_end = res.cursor_end;

  save('-v7', fullfile(out_dir, 'qpso.mat'), ...
       'levy_in', 'levy_out', ...
       'init_pos', 'init_pbest', 'init_pbestval', 'init_gbest', 'init_gbestval', ...
       'pos_traj', 'pbest_traj', 'pbestval_traj', 'gbest_traj', 'gbestval_traj', ...
       'final_out', 'cursor_end');
  printf('OCTAVE_STAGE qpso D=%d ps=%d me=%d K=%d cursor_end=%d te=%d\n', ...
         D, ps, me, K, cursor_end, res.te);
end
