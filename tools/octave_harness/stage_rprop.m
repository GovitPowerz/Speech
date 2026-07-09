function stage_rprop(out_dir)
  % Drive the REAL vendored Rprop.m (legacy/Optimizer_V6.2.2/functions/Rprop.m, 38 LOC)
  % over a scripted derivative/cost sequence and dump per-step state trajectories.
  % TIER 1: the real function is called unchanged; the state (weights/delta/deltaweight/
  % derivatives) is fed back call-to-call exactly as the training loop would.
  %
  % etap 1.2 / etam 0.5 / delta0 0.01 (the :5 rand is DEAD -- overwritten at :6, its draw
  % discarded; see IMPROVEMENTS) / deltamin 1e-9 / deltamax 1.0. Pure arithmetic (sign,
  % min/max, mul, add -- no libm), so these goldens are STRICT BITS on every platform.
  %
  % The crafted sequence exercises every branch:
  %   >0 (same-sign): delta *= etap grow  |  <0 (sign flip): delta *= etam shrink + the
  %   cost-gated backtrack (prev_cost < cost) + derivative-zeroing  |  ==0: plain step.
  % cost_script has BOTH prev_cost < cost steps (backtrack fires) and prev_cost >= cost
  % steps (shrink + zero, no weight backtrack).
  N = 5;
  K = 6;
  weights0 = [1.0; -1.0; 0.5; 2.0; -0.3];
  deriv_script = [ ...
     2,  1,  1,  1,  1, -1; ...   % el1: mostly same-sign (grow), flips at step 6
    -1, -2, -1, -1, -1,  1; ...   % el2: same-sign (grow), flips at step 6
     1, -1,  1, -1,  1,  1; ...   % el3: alternating (shrink / zero)
     1,  1,  0,  1,  1,  1; ...   % el4: a zero-derivative step (==0 branch)
     1, -1,  1, -1,  1,  1  ...   % el5: alternating (shrink / zero)
  ];
  cost_script = [10, 12, 8, 9, 20, 15];   % steps 2,4: prev<cost (backtrack); step 6: prev>=cost (no backtrack)

  weights_traj = zeros(N, K);
  delta_traj = zeros(N, K);
  deltaweight_traj = zeros(N, K);
  derivout_traj = zeros(N, K);

  weights = weights0;
  delta = zeros(N, 1);          % ignored by the init branch (reset to delta0*ones)
  deltaweight = zeros(N, 1);    % ignored by the init branch (overwritten per element)
  prev_deriv = [];              % empty -> the init branch (Rprop.m:10)
  prev_cost = 0;                % unused by the init branch
  for s = 1:K
    d = deriv_script(:, s);
    c = cost_script(s);
    [weights, delta, deltaweight, deriv_out] = ...
        Rprop(weights, d, prev_deriv, c, prev_cost, delta, deltaweight);
    weights_traj(:, s) = weights;
    delta_traj(:, s) = delta;
    deltaweight_traj(:, s) = deltaweight;
    derivout_traj(:, s) = deriv_out;
    prev_deriv = deriv_out;
    prev_cost = c;
  end

  save('-v7', fullfile(out_dir, 'rprop.mat'), ...
       'deriv_script', 'cost_script', 'weights0', ...
       'weights_traj', 'delta_traj', 'deltaweight_traj', 'derivout_traj');
  printf('OCTAVE_STAGE rprop N=%d K=%d\n', N, K);
end
