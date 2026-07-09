function qpso_reset_cursor(table)
  % Phase 4c Task 11 harness: install the shared random table and reset the stage-global
  % cursor to 1. Every table-read substitution (tbl_rand / tbl_randperm / tbl_stblrnd) reads
  % from QPSO_TABLE at QPSO_CURSOR and advances it -- the SAME sequential f64 stream the
  % Python TableRng reads from the committed qpso_random_table.bin. This is the mechanism
  % that replaces QuantumPSO.m:91's wall-clock reseed (removed) with a reproducible source.
  global QPSO_TABLE QPSO_CURSOR
  QPSO_TABLE = table(:);
  QPSO_CURSOR = 1;
end
