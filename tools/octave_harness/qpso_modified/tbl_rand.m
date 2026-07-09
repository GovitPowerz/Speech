function r = tbl_rand(varargin)
  % HARNESS-SUB for MATLAB `rand`: read prod(size) sequential f64s from the shared table and
  % reshape COLUMN-MAJOR (Octave `reshape` default), so element (i,j) = table(cursor+(j-1)*m+i).
  % This reproduces MATLAB rand([m,n])'s column-major stream layout, which the Python
  % TableRng matches with numpy reshape(order='F'). Supported shapes (all that QuantumPSO.m
  % uses): tbl_rand() -> 1x1 scalar; tbl_rand([m,n]) and tbl_rand(m,n) -> m x n.
  global QPSO_TABLE QPSO_CURSOR
  if numel(varargin) == 0
    sz = [1, 1];
  elseif numel(varargin) == 1
    a = varargin{1};
    if isscalar(a)
      sz = [a, a];          % MATLAB rand(n) == n x n (unused by QuantumPSO, kept faithful)
    else
      sz = a(:)';           % tbl_rand([m,n])
    end
  else
    sz = cell2mat(varargin); % tbl_rand(m,n)
  end
  count = prod(sz);
  vals = QPSO_TABLE(QPSO_CURSOR : QPSO_CURSOR + count - 1);
  QPSO_CURSOR = QPSO_CURSOR + count;
  r = reshape(vals, sz);
end
