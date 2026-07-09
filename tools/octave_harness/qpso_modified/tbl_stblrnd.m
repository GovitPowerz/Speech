function r = tbl_stblrnd(alpha, beta, gamma, delta, m, n)
  % HARNESS-SUB for stblrnd (QuantumPSO.m:475 `stblrnd(1.3,1,0.5,0,1,D)`): the
  % Chambers-Mallows-Stuck alpha-stable transform (stblrnd.m general `alpha ~= 1` branch,
  % :75-82, + the alpha~=1 scale/shift :95-96) with its two internal `rand(sizeOut)` draws
  % substituted by tbl_rand -- V first (:76), then W (:77). alpha=1.3, beta=1 hits exactly
  % this branch (not the Gaussian/Cauchy/Levy/symmetric special cases). The Python
  % levy_stable_cms mirrors this line-for-line.
  sizeOut = [m, n];
  V = pi/2 * (2 * tbl_rand(sizeOut) - 1);   % stblrnd.m:76
  W = - log( tbl_rand(sizeOut) );           % stblrnd.m:77
  const = beta * tan(pi * alpha / 2);       % stblrnd.m:78
  B = atan(const);                          % stblrnd.m:79
  S = (1 + const * const) .^ (1 / (2 * alpha));  % stblrnd.m:80
  r = S * sin( alpha * V + B ) ./ ( cos(V) ) .^ (1 / alpha) .* ...
      ( cos( (1 - alpha) * V - B ) ./ W ) .^ ((1 - alpha) / alpha);  % stblrnd.m:81-82
  r = gamma * r + delta;                    % stblrnd.m:96 (alpha ~= 1)
end
