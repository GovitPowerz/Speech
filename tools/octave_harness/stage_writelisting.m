function stage_writelisting(out_dir)
  % Phase 4d Task 8: drive the REAL vendored WriteListing.m / WriteWeightedListing.m
  % (TIER 1, unmodified) to dump byte-exact listing-file goldens. Both functions write
  % plain text directly -- there is no .mat -> .bin conversion step here, unlike every
  % other stage in this harness; the goldens ARE the files Octave itself wrote.
  %
  % plain (WriteListing): 7 files, PS.VP.nbworker=3 -- 7 is NOT evenly divisible by 3
  % (3+2+2), so the worker-shard fliplr(length(index)-jj+1:-nbworker:1) interleave is
  % genuinely exercised across all 3 shards, by hand:
  %   jj=1: fliplr(7:-3:1) = fliplr([7 4 1]) = [1 4 7]
  %   jj=2: fliplr(6:-3:1) = fliplr([6 3])   = [3 6]
  %   jj=3: fliplr(5:-3:1) = fliplr([5 2])   = [2 5]
  % (1-based POSITIONS into `index`; index=1:7 here, i.e. identity, so these are also
  % the file numbers). Together they partition {1..7} exactly once each.
  %
  % weighted (WriteWeightedListing): 8 files probing the two %g fields (filesValues(:,2)
  % "weight" and .duration) across the C-style %g style-switch boundaries: plain
  % integers, 6-sig-fig rounding (123456.789 -> 123457), the exponent-switch thresholds
  % at 1e5 (still decimal) / 1e6 (switches to exponential) / 1e-4 (still decimal) / 1e-5
  % (switches to exponential), a rounding-carry case that crosses the exponent boundary
  % (999999.5 -> 1e+06), a 7-digit integer needing 6-sig-fig rounding+exponent
  % (1234567 -> 1.23457e+06), and negative zero (-0 -> "-0", not "0"). The worker-shard
  % block (:10-21) is COMMENTED OUT in this function's legacy source -- not exercised.

  n = 7;
  PS = struct();
  for i = 1:n
    PS.VP.files.liste(i).name = sprintf('audio_%02d.wav', i);
    PS.VP.refsegfiles.liste(i).name = sprintf('ref_%02d.xml', i);
    PS.VP.reflangfiles.liste(i).name = sprintf('lang_%02d', i);
    PS.VP.refdialfiles.liste(i).name = sprintf('dial_%02d', i);
  end
  nbworker = 3;
  PS.VP.nbworker = nbworker;

  index = 1:n;
  WriteListing(fullfile(out_dir, 'plain'), index, PS);

  % --- weighted ---
  clear PS;
  wfilenames = {'wfile_01.wav', 'wfile_02.wav', 'wfile_03.wav', 'wfile_04.wav', ...
                'wfile_05.wav', 'wfile_06.wav', 'wfile_07.wav', 'wfile_08.wav'};
  wsegs      = {'wseg_01.xml', 'wseg_02.xml', 'wseg_03.xml', 'wseg_04.xml', ...
                'wseg_05.xml', 'wseg_06.xml', 'wseg_07.xml', 'wseg_08.xml'};
  wlangs = {'eng', 'fra', 'eng', 'fra', 'eng', 'fra', 'eng', 'fra'};
  wdials = {'us',  'ca',  'us',  'ca',  'us',  'ca',  'us',  'ca'};
  weights   = [1.0, 0.5, 2, 3, 4, 5, 1234567, -0.0];
  durations = [30.0, 123456.789, 1000000, 100000, 0.0001, 0.00001, 999999.5, 0.0];
  wn = numel(wfilenames);

  for i = 1:wn
    PS.Corpora.Train.listing{i, 1}.filename = wfilenames{i};
    PS.Corpora.Train.listing{i, 1}.segfilename = wsegs{i};
    PS.Corpora.Train.listing{i, 1}.lang = wlangs{i};
    PS.Corpora.Train.listing{i, 1}.dial = wdials{i};
    PS.Corpora.Train.listing{i, 1}.duration = durations(i);
  end
  PS.Corpora.Train.filesValues = zeros(wn, 2);
  PS.Corpora.Train.filesValues(:, 2) = weights;

  windex = 1:wn;
  WriteWeightedListing(fullfile(out_dir, 'weighted.lst'), windex, PS);

  printf('OCTAVE_STAGE writelisting n=%d nbworker=%d wn=%d\n', n, nbworker, wn);
end
