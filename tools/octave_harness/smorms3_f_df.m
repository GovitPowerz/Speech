function [f, df, theta_out, passthru] = smorms3_f_df(theta, passthru, ec, grad_script, offset)
  % Injected SMORMS3 objective for the Phase 4c Octave harness.
  %
  % `theta` is a cell array (SMORMS3's original parameter format). We return:
  %   df        - the scripted per-step gradient (column `ec` of grad_script)
  %               reshaped into theta's cell structure (df must have theta's shape,
  %               per SMORMS3.m:21).
  %   theta_out - theta + offset. The theta_out ROUND-TRIP subtlety: SMORMS3.m:355
  %               updates theta = theta_out + dtheta, where theta_out comes FROM this
  %               objective (SMORMS3.m:330,355), NOT the input theta. A nonzero offset
  %               pins that the port adds dtheta to the RETURNED theta.
  %   passthru  - an unused varargin, echoed back verbatim. SMORMS3.m:313 assigns
  %               `[f, df_full, theta_local_out, obj.varargin_stored{:}] = obj.f_df(...)`;
  %               under Octave an EMPTY varargin_stored miscounts the lvalue cs-list
  %               (probe: "function called with too many outputs"), so the harness passes
  %               exactly one dummy varargin. It is value-neutral: the gradient depends
  %               only on `ec`, and the SMORMS3 update math is untouched. See README +
  %               IMPROVEMENTS (Octave-compat adjudication).
  g = grad_script(:, ec);
  df = cell(size(theta));
  theta_out = cell(size(theta));
  i = 1;
  for k = 1:numel(theta)
    n = numel(theta{k});
    df{k} = reshape(g(i:i + n - 1), size(theta{k}));
    theta_out{k} = theta{k} + reshape(offset(i:i + n - 1), size(theta{k}));
    i = i + n;
  end
  f = sum(g .^ 2);
end
