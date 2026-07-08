function [out, pos_out] = qpso_surrogate(pos, mode, center)
  % Quadratic cost surrogate replacing the engine `CostFunction` (feval) in the modified
  % QuantumPSO copy: cost(row) = sum_j (row_j - center_j)^2, positions passed through
  % (pos_out == pos; the real engine may project onto VR bounds -- the surrogate does not,
  % and both the stage and the Python port agree on that, keeping the trajectory
  % self-consistent). The sum is an EXPLICIT left-to-right fold so it matches the Python
  % surrogate's fold order bit-for-bit. `mode` (-2/-1/i/0) is ignored (deterministic cost).
  n = size(pos, 1);
  C = repmat(center(:)', n, 1);
  sq = (pos - C) .^ 2;
  out = sq(:, 1);
  for j = 2:size(sq, 2)
    out = out + sq(:, j);
  end
  pos_out = pos;
end
