function stage_smorms3(out_dir)
  % Drive the REAL vendored SMORMS3 classdef (legacy/Optimizer_V6.2.2/functions/
  % SMORMS3.m) over scripted gradient sequences and dump per-step state trajectories.
  % TIER 1: the real classdef ctor (:151-187) + optimization_step (:324-363) +
  % f_df_wrapper (:306-321) are exercised unchanged. We call the public
  % optimization_step() per step (NOT optimize() :190-212) to capture trajectories and
  % avoid optimize()'s LogAndDisplay/save side effects, which need a populated FS
  % struct. The math (r=1/(delta+1); stepRate/MMS EMA; dtheta; delta; lrate x10 warmup
  % capped at 1e-1) is byte-for-byte the vendored source.

  % ---- Main trajectory: 2-cell theta (the weightsIni contract), M=7, 12 steps ----
  % 12 steps proves the lrate x10 warmup (1e-9 -> 1e-1 by step 8) AND the 1e-1 cap
  % (steps 9-12 plateau at 1e-1).
  theta0 = { [0.1; 0.2; 0.3], [0.4 0.6; 0.5 0.7] };   % 3x1 vector cell + 2x2 matrix cell
  M = 7;
  num_steps = 12;

  % Deterministic, per-dim-distinct gradient script with step-to-step sign flips, so
  % every dimension's MMS/stepRate/delta EMA is non-trivially exercised (delta both
  % grows and shrinks across dims). grad_script(:, s) is the gradient at step s.
  grad_script = zeros(M, num_steps);
  for d = 1:M
    for s = 1:num_steps
      grad_script(d, s) = ((-1) ^ (d + s)) * (0.5 + 0.1 * d + 0.05 * s);
    end
  end
  offset = zeros(M, 1);   % pass-through theta_out (theta_out = theta)

  f_df = @(theta, passthru, ec) smorms3_f_df(theta, passthru, ec, grad_script, offset);
  obj = SMORMS3(f_df, theta0, struct());   % one dummy varargin (Octave lvalue accommodation)

  theta0_flat = obj.theta;   % SMORMS3 flattened it in the ctor (:174)
  theta_traj = zeros(M, num_steps);
  mms_traj = zeros(M, num_steps);
  steprate_traj = zeros(M, num_steps);
  delta_traj = zeros(M, num_steps);
  lrate_traj = zeros(1, num_steps);
  cost_traj = zeros(1, num_steps);
  for s = 1:num_steps
    obj.optimization_step();
    theta_traj(:, s) = obj.theta;
    mms_traj(:, s) = obj.MMS;
    steprate_traj(:, s) = obj.stepRate;
    delta_traj(:, s) = obj.delta;
    lrate_traj(1, s) = obj.lrate;
    cost_traj(1, s) = obj.hist_f_flat(end);
  end
  % Pin the column-major flat -> original (cell) reconstruction (:286-303).
  theta_final = obj.theta_flat_to_original(obj.theta);
  theta_final_cell0 = theta_final{1};
  theta_final_cell1 = theta_final{2};

  % ---- Round-trip variant: f_df returns a MODIFIED theta_out (nonzero offset) ----
  rt_M = 3;
  rt_num_steps = 4;
  rt_theta0 = { [1.0; 2.0; 3.0] };
  rt_grad_script = zeros(rt_M, rt_num_steps);
  for d = 1:rt_M
    for s = 1:rt_num_steps
      rt_grad_script(d, s) = 0.3 * d - 0.1 * s;
    end
  end
  rt_offset = [0.05; -0.10; 0.20];   % nonzero: theta_out = theta + rt_offset every step
  rt_f_df = @(theta, passthru, ec) smorms3_f_df(theta, passthru, ec, rt_grad_script, rt_offset);
  rt_obj = SMORMS3(rt_f_df, rt_theta0, struct());
  rt_theta0_flat = rt_obj.theta;
  rt_theta_traj = zeros(rt_M, rt_num_steps);
  for s = 1:rt_num_steps
    rt_obj.optimization_step();
    rt_theta_traj(:, s) = rt_obj.theta;
  end

  save('-v7', fullfile(out_dir, 'smorms3.mat'), ...
       'grad_script', 'theta0_flat', 'theta_traj', 'mms_traj', 'steprate_traj', ...
       'delta_traj', 'lrate_traj', 'cost_traj', 'theta_final_cell0', 'theta_final_cell1', ...
       'rt_grad_script', 'rt_theta0_flat', 'rt_offset', 'rt_theta_traj');
  printf('OCTAVE_STAGE smorms3 M=%d num_steps=%d rt_M=%d rt_num_steps=%d\n', ...
         M, num_steps, rt_M, rt_num_steps);
end
