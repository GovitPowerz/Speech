function p = tbl_randperm(n, k)
  % HARNESS-SUB for MATLAB `randperm(n,k)` (QuantumPSO.m:467 `randperm(size(pos,1),2)`).
  % MATLAB's builtin randperm is NOT table-reproducible, so both sides substitute an
  % IDENTICAL Durstenfeld/Fisher-Yates shuffle over 1..n consuming exactly n-1 table draws:
  %   for i = n:-1:2:  j = floor(rand*i)+1;  swap p(i), p(j)
  % then return the first k. The Python TableRng.randperm implements the same loop, so the
  % selected neighbour indices match bit-for-bit given the shared table.
  p = 1:n;
  for i = n:-1:2
    r = tbl_rand();
    j = floor(r * i) + 1;
    tmp = p(i); p(i) = p(j); p(j) = tmp;
  end
  if nargin >= 2
    p = p(1:k);
  end
end
