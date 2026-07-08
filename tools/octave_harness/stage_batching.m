function stage_batching(out_dir)
  % Drive the REAL vendored CreateBatches.m (158 LOC) + GetNewBatch.m (94 LOC) unchanged
  % (TIER 1), plus the FALLBACK-TIER getworstandbest_transcribed.m (verbatim transcription
  % of the private getCases.m:116-165 subfunction -- see that file's header). randperm is
  % SHADOWED (batching_shadow/randperm.m, addpath'd after functions_dir so it wins) with a
  % FIXED deterministic reverse-order permutation, keeping CreateBatches.m itself
  % unmodified while making its shuffle reproducible AND trivially replicable in Python
  % (see tests/test_phase4c_batching.py) -- pinning create_batches's own output, not just
  % GetNewBatch's rotation.
  %
  % Cases (class VALUES are deliberately chosen to dodge two MATLAB struct-array
  % auto-expansion traps that are otherwise very easy to trip over, documented in
  % IMPROVEMENTS.md):
  %   single   : nbOfTargetClasses<=1, isMultiLingual<=0 (CreateBatches.m:19-27).
  %   multi_nb : nbOfTargetClasses>1, isMultiLingual<=0 (:43-60), classes {1,2,5} so the
  %              LAST loop index (ii=3=nbOfTargetClasses) lands on the non-target value 5,
  %              NOT a target -- a clean, non-clobbered aggregate (Cases(3) = class-5 only).
  %   clobber  : nbOfTargetClasses>1, isMultiLingual<=0, classes {0,1,2} (the natural
  %              contiguous labeling): possibleValues(3)=2 is ITSELF a valid target
  %              (0<2<3), so `Cases(ii=3)` -- the SAME slot pre-reserved for the aggregate
  %              -- gets silently OVERWRITTEN with class-2's data, and class-0's indices
  %              (assigned to the aggregate at ii=1) are LOST. Structural-only (no
  %              rotation): pins that Cases(3).index ends up as class-2's data.
  %   sub      : nbOfTargetClasses>1, isMultiLingual>0 (:61-86), classes {1,2,5,8} so
  %              ii=1,2 map cleanly to the two target Cases slots and ii=3,4 (values 5,8,
  %              both non-target) become TWO SubCases groups under Cases(3).
  %   degenerate: nbOfTargetClasses*nbOfWorstCases+nbOfCasesPerBatch > file_nb (:15-17) --
  %              WorstCases=[], Cases=(1:file_nb)'; GetNewBatch takes the `~isfield(...,
  %              'currentClass')` branch (:3-5), Batch = Batches.Cases verbatim.
  %
  % For single/multi_nb/sub: dump the CreateBatches-produced index arrays (pins
  % create_batches's shuffle), then rotate GetNewBatch N steps, dumping the returned Batch/
  % Validation/countPass trajectory -- PURE integer cursor arithmetic (no RNG anywhere in
  % GetNewBatch.m), so this is a STRICT-bits golden.

  harness_dir = fileparts(mfilename('fullpath'));
  shadow_dir = fullfile(harness_dir, 'batching_shadow');
  addpath(shadow_dir);

  R = struct();

  % --- single: nbOfTargetClasses<=1, isMultiLingual<=0 ---
  Corpora.file_nb = 6;
  Corpora.filesValues = zeros(6,1);  % class column unused on this branch
  Corpora = CreateBatches(Corpora, 2, 1, 0, 1);
  B = Corpora.Batches;
  R.single_index = B.Cases(1).index;
  R.single_worst_index = B.WorstCases(1).index;
  [R.single_batch_traj, R.single_validation_traj, R.single_countpass_traj] = rotate(B, 8);

  % --- multi_nb: nbOfTargetClasses>1, isMultiLingual<=0, clean (no clobber) aggregate ---
  clear Corpora B;
  Corpora.file_nb = 6;
  Corpora.filesValues = [1;1;2;2;5;5];
  Corpora = CreateBatches(Corpora, 2, 1, 0, 3);
  B = Corpora.Batches;
  R.multi_nb_case1_index = B.Cases(1).index;
  R.multi_nb_case2_index = B.Cases(2).index;
  R.multi_nb_case3_index = B.Cases(3).index;
  R.multi_nb_worst1_index = B.WorstCases(1).index;
  R.multi_nb_worst2_index = B.WorstCases(2).index;
  R.multi_nb_worst3_index = B.WorstCases(3).index;
  [R.multi_nb_batch_traj, R.multi_nb_validation_traj, R.multi_nb_countpass_traj] = rotate(B, 8);

  % --- clobber: nbOfTargetClasses>1, isMultiLingual<=0, contiguous {0,1,2} -- structural only ---
  clear Corpora B;
  Corpora.file_nb = 6;
  Corpora.filesValues = [0;0;1;1;2;2];
  Corpora = CreateBatches(Corpora, 2, 1, 0, 3);
  B = Corpora.Batches;
  R.clobber_case3_index = B.Cases(3).index;  % expect: class-2's indices, NOT class-0's

  % --- sub: nbOfTargetClasses>1, isMultiLingual>0 ---
  clear Corpora B;
  Corpora.file_nb = 12;
  Corpora.filesValues = [1;1;1;2;2;2;5;5;5;8;8;8];
  Corpora = CreateBatches(Corpora, 2, 1, 1, 3);
  B = Corpora.Batches;
  R.sub_case1_index = B.Cases(1).index;
  R.sub_case2_index = B.Cases(2).index;
  R.sub_case3_sub1_index = B.Cases(3).SubCases(1).index;
  R.sub_case3_sub2_index = B.Cases(3).SubCases(2).index;
  R.sub_worst1_index = B.WorstCases(1).index;
  R.sub_worst2_index = B.WorstCases(2).index;
  R.sub_worst3_index = B.WorstCases(3).index;
  [R.sub_batch_traj, R.sub_validation_traj, R.sub_countpass_traj] = rotate(B, 10);

  % --- degenerate: nbOfTargetClasses*nbOfWorstCases+nbOfCasesPerBatch > file_nb ---
  clear Corpora B;
  Corpora.file_nb = 3;
  Corpora.filesValues = [1;2;0];
  Corpora = CreateBatches(Corpora, 2, 5, 0, 3);
  B = Corpora.Batches;
  R.degenerate_cases = B.Cases;
  [Batch, WorstCases, B2] = GetNewBatch(B);
  R.degenerate_batch = Batch;
  R.degenerate_validation = B2.Validation;

  % --- getWorstAndBest (FALLBACK TIER, getworstandbest_transcribed.m) ---
  idx = (1:8)';
  cost = [8;3;6;1;7;2;5;4];
  sortingTable = sortrows([cost idx]);
  [ib, im, iw, sb, sm, sw] = getworstandbest_transcribed(8, sortingTable, 3, 2, 2);
  R.gwb_basic_indBest = ib; R.gwb_basic_indMiddle = im; R.gwb_basic_indWorst = iw;
  R.gwb_basic_scoreBest = sb; R.gwb_basic_scoreMiddle = sm; R.gwb_basic_scoreWorst = sw;

  [ib2, im2, iw2, sb2, sm2, sw2] = getworstandbest_transcribed(8, sortingTable, 0, 0, 0);
  R.gwb_edge_indBest = ib2; R.gwb_edge_indMiddle = im2; R.gwb_edge_indWorst = iw2;

  save('-v7', fullfile(out_dir, 'batching.mat'), '-struct', 'R');

  printf('OCTAVE_STAGE batching single_n=%d multi_nb_n=%d sub_n=%d degenerate_batch_len=%d\n', ...
         numel(R.single_index), numel(R.multi_nb_case1_index)+numel(R.multi_nb_case2_index)+numel(R.multi_nb_case3_index), ...
         numel(R.sub_case1_index)+numel(R.sub_case2_index), numel(R.degenerate_batch));
end


function [batch_traj, validation_traj, countpass_traj] = rotate(Batches, nsteps)
  file_nb = numel(Batches.Validation);
  minibatch = Batches.nbOfCasesPerBatch;
  batch_traj = zeros(minibatch, nsteps);
  validation_traj = zeros(file_nb, nsteps);
  countpass_traj = zeros(1, nsteps);
  for step = 1:nsteps
    [Batch, ~, Batches] = GetNewBatch(Batches);
    batch_traj(:, step) = Batch;
    validation_traj(:, step) = Batches.Validation;
    countpass_traj(step) = Batches.countPass;
  end
end
