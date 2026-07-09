function v = read_bin_col(path)
  % Read an io::binary .bin (i64 LE rows, i64 LE cols, f64 LE column-major) as a flat column
  % vector of length rows*cols. Used to load the committed qpso_random_table.bin (N x 1).
  fid = fopen(path, 'rb');
  if fid < 0
    error('read_bin_col: cannot open %s', path);
  end
  rows = fread(fid, 1, 'int64');
  cols = fread(fid, 1, 'int64');
  v = fread(fid, rows * cols, 'double');
  fclose(fid);
end
