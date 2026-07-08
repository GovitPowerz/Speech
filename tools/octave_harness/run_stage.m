function run_stage(stage, out_dir)
  % Phase 4c Octave harness dispatcher. Mirrors the C++ oracle_harness convention: a
  % single entry point that runs a named stage into an output directory. Adds the
  % harness's own dir (for the stage_* / smorms3_f_df helpers) and the VENDORED
  % legacy/Optimizer_V6.2.2/functions dir (the .m sources under test) to the path, then
  % dispatches. The vendored .m are CALLED via addpath, never modified.
  %
  % Usage (from the extractor):
  %   octave-cli --no-gui --quiet --path <harness_dir> --eval "run_stage('smorms3','<out>')"
  harness_dir = fileparts(mfilename('fullpath'));
  functions_dir = fullfile(harness_dir, '..', '..', 'legacy', 'Optimizer_V6.2.2', 'functions');
  if exist(functions_dir, 'dir') ~= 7
    error('run_stage: vendored functions dir not found: %s (legacy/ is local-only)', functions_dir);
  end
  addpath(harness_dir);
  addpath(functions_dir);

  if exist(out_dir, 'dir') ~= 7
    mkdir(out_dir);
  end

  switch stage
    case 'smorms3'
      stage_smorms3(out_dir);
    case 'rprop'
      stage_rprop(out_dir);
    case 'vec2struct'
      stage_vec2struct(out_dir);
    case 'computecost'
      stage_computecost(out_dir);
    otherwise
      error('run_stage: unknown stage %s', stage);
  end
end
