function stage_scr(out_dir)
  % Phase 4d Task 11: drive Test_BLSTM.m's `.scr` score-file writer block over INJECTED
  % scores, dumping the golden the Python port (src/python/speech/drivers/test.py) is
  % bit-pinned against. No `.scr` file exists anywhere in the legacy tree -- this is the
  % ONLY oracle.
  %
  % TIER: HYBRID.
  %   * `PS.Corpora.Train.langMapConf` / `PS.Corpora.Train.listing` come from a REAL,
  %     unmodified Tier-1 call to `processListing.m` (functions/processListing.m, plain
  %     MATLAB, no shell-out) over a crafted mappingFile + a crafted single-row training
  %     listing written to out_dir. This is the authoritative source of `keys(langMapConf)`
  %     -- the alphabetical class-key order the writer indexes by (Test_BLSTM.m:249) --
  %     so the ALPHABETICAL ORDER ITSELF is never hand-simulated, it is the real
  %     containers.Map's real `keys()` return.
  %   * The writer LOOP body (Test_BLSTM.m:251-269) is FALLBACK-TIER: Test_BLSTM.m is a
  %     top-level SCRIPT (globals, hostname branches, ParamStruct.mat/save_net loads), not
  %     a callable function, and `scores_test` is the return of `CostFunction.m`'s
  %     engine shell-out (`system('python RunFsp.py ...')`) -- the same "no injection
  %     point that leaves the vendored .m unmodified" situation stage_computecost.m
  %     documents. There is therefore no function boundary to call with an injected
  %     `scores_test`; the loop is transcribed line-for-line (`% legacy:` provenance)
  %     with `scores_test` supplied directly instead of coming from CostFunction. Every
  %     OTHER call inside the loop (`textscan`, `cell2str.m`, `sortrows`, `keys`,
  %     `num2str`, `fopen`/`fprintf`/`fclose`) is the REAL builtin or REAL vendored
  %     function, unmodified -- only the enclosing `for jj`/`for ii` control flow and the
  %     `scores_test` source are stage-local.
  %
  % `ComputeCost.m:708` already applies the `>150 -> value-200` in-band LID-score decode
  % BEFORE `CostFunction.m:409` assigns `PS.VP.BP.LIDscoreDet` -- so by the time
  % Test_BLSTM.m:247 reads it, `scores_test` is ALREADY DECODED. The writer loop itself
  % contains no `>150` handling. `raw_scores` below deliberately includes one value of
  % 250.0 (> 150) to make this observable: the golden line for that class must show
  % `exp(250/100)/...`, NOT `exp((250-200)/100)/...` -- if the writer secretly re-applied
  % the sentinel, the golden would differ from what write_scores (which is never handed
  % un-decoded scores either) computes.
  %
  % Non-vacuity design (one crafted file, 5 classes):
  %   zzz;99;0  aaa;11;1  mmm;unk;2  bbb;22;3  nnn;33;4  (mapping.csv, lang;dial;classid)
  %   id order:          zzz_99, aaa_11, mmm_unk, bbb_22, nnn_33
  %   keys() alpha order: aaa_11, bbb_22, mmm_unk, nnn_33, zzz_99   <- VISIBLY different,
  %   proving the class-id-order bug (drivers/test.py's old `_class_keys`) would emit
  %   wrong labels/order. `mmm_unk`'s 3-char dial ('unk') renders clean ("mmm-unk"); the
  %   2-char dials ('99','11','22','33') expose the load-bearing `tmp(end-2:end)` quirk:
  %   for a 6-char "lang_dd" key the last 3 characters are "_dd", not "dd" -- e.g.
  %   "aaa_11" renders as "aaa-_11", NOT "aaa-11". This is a genuine legacy composition
  %   artifact (processListing.m:10 keys on `[lang '_' dial]`; Test_BLSTM.m:264-265 slices
  %   the last 3 chars assuming a 2-char dial swallows the underscore), reproduced here,
  %   not "fixed".
  %   raw_scores: aaa_11=250.0 (>150 passthrough, ties with bbb_22), bbb_22=250.0 (tie ->
  %   pins MATLAB sortrows' STABLE tie-break, original column order preserved),
  %   mmm_unk=-1000.0 (near-zero after softmax, many leading-zero decimals),
  %   nnn_33=0.0 (exp(0)=1, a "clean" numerator), zzz_99=123.456789 (typical many-decimal
  %   value). Together these probe num2str('%15.15f') formatting across magnitudes.
  %   filename: 'corpus/session01/conversation_utt00001_channelA.wav' -- the basename
  %   ('conversation_utt00001_channelA.wav', 34 chars) is > 21 chars, so
  %   Test_BLSTM.m:257's `filename(1:end-21)` genuinely truncates (-> 'conversation_'),
  %   rather than vacuously collapsing to ''.

  mapping_text = ['zzz;99;0' char(10) 'aaa;11;1' char(10) 'mmm;unk;2' char(10) ...
                  'bbb;22;3' char(10) 'nnn;33;4' char(10)];
  mapping_path = fullfile(out_dir, 'mapping.csv');
  fid = fopen(mapping_path, 'w');
  fprintf(fid, '%s', mapping_text);
  fclose(fid);

  full_filename = 'corpus/session01/conversation_utt00001_channelA.wav';
  listing_row = [full_filename ';corpus/session01/conversation_utt00001_channelA.stm;nnn;33;1.0;30.0' char(10)];
  listing_path = fullfile(out_dir, 'listing.csv');
  fid = fopen(listing_path, 'w');
  fprintf(fid, '%s', listing_row);
  fclose(fid);

  % ---- Tier 1: real processListing.m over the crafted mapping/listing -------------------
  Corpora = struct();
  Corpora.mappingFile = mapping_path;
  Corpora.trainListings = {listing_path};
  Corpora.validListings = {};
  Corpora = processListing(Corpora);

  % legacy: Test_BLSTM.m:249 keySet = keys(PS.Corpora.Train.langMapConf);
  keySet = keys(Corpora.Train.langMapConf);

  % ---- Injected raw (already-decoded) LID scores, looked up by key ----------------------
  rawMap = containers.Map('KeyType', 'char', 'ValueType', 'double');
  rawMap('aaa_11') = 250.0;
  rawMap('bbb_22') = 250.0;
  rawMap('mmm_unk') = -1000.0;
  rawMap('nnn_33') = 0.0;
  rawMap('zzz_99') = 123.456789;

  scores_test = zeros(1, numel(keySet));
  for jj = 1:numel(keySet)
    scores_test(1, jj) = rawMap(keySet{jj});
  end
  raw_scores_ordered = scores_test;  % dumped for the Python-side replay (same key order)

  % ---- FALLBACK-TIER transcription of Test_BLSTM.m:251-269 (jj=1 only; one file) --------
  scores_dir = fullfile(out_dir, 'scores');
  mkdir(scores_dir);
  for jj = 1:size(scores_test, 1)
    % legacy: Test_BLSTM.m:252 scores_test(jj,:) = exp(scores_test(jj,:)/100);
    scores_test(jj, :) = exp(scores_test(jj, :) / 100);
    % legacy: Test_BLSTM.m:253 scores_test(jj,:) = scores_test(jj,:)/max(1e-3,sum(scores_test(jj,:)));
    scores_test(jj, :) = scores_test(jj, :) / max(1e-3, sum(scores_test(jj, :)));
    % legacy: Test_BLSTM.m:254 [val,ind] = sortrows(scores_test(jj,:)',-1);
    [val, ind] = sortrows(scores_test(jj, :)', -1);
    % legacy: Test_BLSTM.m:255-257 (textscan split on '/', cell2str, 21-char strip)
    C = textscan(full_filename, '%s', 'delimiter', '/');
    filename = cell2str(C{1}(end));
    filename = filename(1:end - 21);
    % legacy: Test_BLSTM.m:258 fopen([PS.FS.name_dir_fig '/scores/' filename '.scr'], 'w+');
    fileID = fopen(fullfile(scores_dir, [filename '.scr']), 'w+');
    text = '';
    for ii = 1:size(scores_test, 2)
      % legacy: Test_BLSTM.m:261-266
      formatstr = '%15.15f';
      tmp = keySet{ind(ii)};
      lang = tmp(1:3);
      dial = tmp(end - 2:end);
      text = [text sprintf('%s %s-%s %s\n', filename, lang, dial, num2str(val(ii), formatstr))];
    end
    % legacy: Test_BLSTM.m:268-269
    fprintf(fileID, text);
    fclose(fileID);
  end

  % The committed golden basename is fixed ('expected.scr'), independent of the crafted
  % filename's own (legacy-derived) stem -- a harness/fixture naming choice, not part of
  % the pinned law (the pinned law is the FILE'S CONTENT, i.e. what fprintf wrote).
  produced = dir(fullfile(scores_dir, '*.scr'));
  if numel(produced) ~= 1
    error('stage_scr: expected exactly one .scr file, found %d', numel(produced));
  end
  copyfile(fullfile(scores_dir, produced(1).name), fullfile(out_dir, 'expected.scr'));

  % Dump the injected inputs + observed key order for the Python-side replay/extractor.
  keys_path = fullfile(out_dir, 'keys_alpha_order.txt');
  fid = fopen(keys_path, 'w');
  for jj = 1:numel(keySet)
    fprintf(fid, '%s\n', keySet{jj});
  end
  fclose(fid);

  save('-v7', fullfile(out_dir, 'scr_scores.mat'), 'raw_scores_ordered');

  printf('OCTAVE_STAGE scr n_classes=%d produced_name=%s\n', numel(keySet), produced(1).name);
end
