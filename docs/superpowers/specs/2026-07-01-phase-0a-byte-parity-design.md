# Phase 0a - Byte-Parity Foundation Design Spec

Date: 2026-07-01
Status: approved (design), pending implementation plan
Branch: feature/phase-0-io-parity

## 1. Goal and scope

Implement the byte-parity I/O foundation of the port: the weight `.bin` codec, the legacy
`.config` parser, the typed config model, the `config <-> nnet` `adim_coeff` seam, and the flat
weight packer/unpacker. Deliver both the Rust implementation (crate `speech`) and a Python
reference port (package `speech`) so the packer is cross-checked from two independent directions
against the same real artifact.

**Single acceptance test:** parsing `1_worker_1.config`, applying the `adim_coeff` scaling, and
packing the flat weight vector reproduces `NNweights_config1.bin` (269,384 bytes) byte-for-byte;
and reading that `.bin` and re-writing it reproduces the input bytes exactly. Both must hold in
Rust and in Python, and the two flat vectors must be identical.

This is Phase 0a only. Phase 0b (cost laws, segmentation, scoring, VRCTS/STM I/O) is a separate
later cycle with its own spec.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Cycle scope | Phase 0a alone (byte-parity). 0b is the next cycle. |
| 2 | Golden fixtures | Vendor the real artifacts into `tests/reference_data/` (committed): `NNweights_config1.bin` (~269 KB), `1_worker_1.config` (~3 KB), and the 16-byte empty `weightsDerivatives_*.mat`. Tests run in CI. |
| 3 | Model scope | NNType 0 (BLSTM) only - the OpenSAD15 artifact is NNType 0. NNType 1/2 ordering is out of scope. |
| 4 | `.mat` I/O | `io::matfile` stays deferred to Phase 1. Weight files travel through the custom `.bin` format, not matio (despite some `.mat` extensions). |
| 5 | TOML schema | `legacy_config` parses the legacy format exhaustively for the load-bearing keys; the native TOML schema models only the ~25 Phase-0 keys and grows later. Native config stays TOML-first with `legacy_config` as a one-way importer. |
| 6 | Duplicate keys | Last-value-wins (confirmed from `ConfigFile.cpp:25`, `_Params[name] = val`). |

## 3. The weight `.bin` codec (exact)

Authoritative source: C++ `Helpers.hpp` (`Matrix2BinaryFile` write, `BinaryFile2Vector` read);
MATLAB `WriteMatrix2Binary.m` / `ReadMatrixFromBinary.m` agree.

```
bytes[0:8]                       = int64 LE  nbrows
bytes[8:16]                      = int64 LE  nbcols
bytes[16 : 16 + 8*nbrows*nbcols] = f64 LE payload, COLUMN-MAJOR
                                   (write order: for jj in 0..nbcols { for kk in 0..nbrows { matrix(kk,jj) } })
```

- Endianness is machine-native `reinterpret_cast`; all shipped artifacts are little-endian
  x86_64. The port hardcodes LE and asserts.
- `BinaryFile2Vector` requires `nbrows*nbcols > 0`, else `exit(1)` - the port returns an error.
- A weight file is written as an `N x 1` column vector (`nbcols == 1`), so its payload is the flat
  weight vector directly.
- Verified against `NNweights_config1.bin`: header bytes `87 83 00 00 00 00 00 00 | 01 00 00 00 00
  00 00 00` = (33671, 1); file size 269384 = 16 + 33671*8.
- GOTCHA: files named `weights_*.mat` / `NNweights_config*.bin` emitted by `nnet2MatFile.m` go
  through `WriteMatrix2Binary`, so despite a `.mat` extension they ARE this `.bin` format. Sniff
  the header; never feed them to a MAT reader.

## 4. The legacy `.config` parser (exact)

Authoritative source: C++ `ConfigFile.cpp`.

- Line format: `KEY value`, whitespace-delimited (`instream >> name >> val`), then `getline` the
  rest of the line.
- `#`-prefixed names are skipped (comments).
- Continuation: if the key contains the continuation char (default `_`) AND the remaining line has
  `> 1` char, the rest of the line is appended to the value (multi-token values).
- Duplicate keys: last-wins (`_Params[name] = val`).
- Vectors are comma-separated (e.g. `LSTMNeuronNb 23,24,24`). The `val*count` repeat syntax exists
  in the typed getters.

**Load-bearing keys for 0a (NNType 0):** `NNType`, `LSTMNeuronNb`, `LSTMSubSampling`,
`OutputNeuronNb`, `OutputSubSampling`, `PeepholesActive`, the per-block families
`AlgName_[Forward|Backward]_Layer_i_LSTMBlock_j_{Input,Forget,Output}GateWeights` and `_CellWeight`
(each value is a comma-separated f64 list = one full matrix ROW of length `ncols`),
`AlgName_Output_Layer_i_Neuron_j_Weights`, `AlgName_NormalizeInputMean`, `AlgName_NormalizeInputStd`,
plus the `AlgName_LID_`-prefixed twins where present. Config stores config-domain (adim-scaled) rows.

## 5. The `config <-> nnet` `adim_coeff` seam (exact, HIGHEST RISK)

`adim_coeff` scaling lives ENTIRELY at the config<->nnet boundary (`config2network` /
`network2config`), NEVER in the flat packer or the `.bin` codec. The `.bin` stores already-scaled
nnet-domain values.

```
adim_coeff(LSTM layer i)   = sqrt( LSTMNeuronNb[i] * LSTMSubSampling[i] + LSTMNeuronNb[i+1] )
adim_coeff(output layer i) = sqrt( OutputNeuronNb[i] * OutputSubSampling[i] )   # NO +next term
```

- `config2network` (config -> nnet, decode): `nnet_row = config_row / adim_coeff`, then restore the
  LAST element (bias): `nnet_row(end) = nnet_row(end) * adim_coeff`. Net effect: every element
  except the final bias column is divided by `adim_coeff`; the bias column is unchanged.
- `network2config` (nnet -> config, encode): `config_row = [ nnet_row(1:end-1)*adim_coeff ,
  nnet_row(end) ]`.
- The division/restore applies to the WHOLE row including recurrent, fan-in, and the 12 peephole
  columns; only the single final bias column is exempt. LID variants are byte-identical logic with
  `AlgName_LID_` prefixes.
- Use `f64::sqrt` (IEEE754, same op as MATLAB double `sqrt`) for bit-reproducibility.
- The output-layer formula has NO added next-layer term (asymmetric vs LSTM) - assert it explicitly
  (`config2network.m:127`, `network2config.m:72`).

## 6. The flat weight vector packer (exact)

Authoritative source: MATLAB `config2weights.m` / `weights2nnet.m` / `nnet2MatFile.m` (all agree;
`nnet2MatFile` also emits parallel `isBias` flags). NNType 0 only.

**Global order:** all `forward.layer(j)` for every LSTM layer, then all `backward.layer(j)`, then
`output.layer(j)`, then `mean`, then `std`.

**Per LSTM layer** (each of `inputweights`/`forgetweights`/`outputweights`/`cellweights` is an
`output_size x ncols` matrix, where `output_size = LSTMNeuronNb[i+1]`, `input_size = ncols -
output_size - 5`, `ncols = input_size + output_size + 5`; columns: `1..output_size` = recurrent,
`output_size+1..output_size+input_size` = fan-in, last 5 = `[3 input-gate-peephole cols][1
recurrent-peephole col][1 bias]`). Each block is flattened ROW-MAJOR (`reshape(tmp',numel,1)` =
C-order over the original matrix). Emission order:

```
 1. inputweights(:, indices_in)     indices_in  = (output_size+1)..(output_size+input_size)
 2. forgetweights(:, indices_in)
 3. outputweights(:, indices_in)
 4. cellweights(:, indices_in)
 5. inputweights(:, indices_rec)    indices_rec = 1..output_size
 6. forgetweights(:, indices_rec)
 7. outputweights(:, indices_rec)
 8. cellweights(:, indices_rec)
 9. PEEPHOLE BUNDLE (12 cols x output_size rows), assembled as:
      [ inputweights(:,end-1)  forgetweights(:,end-1)  outputweights(:,end-1)          # 3 recurrent-peephole cols
        inputweights(:,end-4:end-2)  forgetweights(:,end-4:end-2)  outputweights(:,end-4:end-2) ]  # 3x3 gate-peephole cols
      flattened row-major. cellweights has NO peephole cols.
10. inputweights(:,end)    BIAS input gate    (output_size elems)
11. forgetweights(:,end)   BIAS forget gate
12. outputweights(:,end)   BIAS output gate
13. cellweights(:,end)     BIAS cell
```

**Per output layer** (`weights` is `output_size x (input_size+1)`): emit `weights(:,1:input_size)`
row-major, then `weights(:,end)` (bias) row-major. `OutputNeuronNb[0] = 48 = 2 * last LSTM size`
(bidirectional concat).

**Tail:** append `normalize.mean` (length `LSTMNeuronNb[0]` = input feature dim = 23) then
`normalize.std` (same length). The tail is exactly `flat[-46:]` -> `mean = flat[-46:-23]`, `std =
flat[-23:]`; `std` all strictly positive.

The packer NEVER applies `adim_coeff`.

## 7. Element-count formula (verified == 33671)

```
Per BLSTM layer i (applied to BOTH forward and backward, so x2):
  out = LSTMNeuronNb[i+1];  fin = LSTMNeuronNb[i] * LSTMSubSampling[i]   # fan-in is subsampled
  per-direction elems = 4*out*fin  +  4*out*out  +  12*out  +  4*out
Per output layer i:
  OutputNeuronNb[i+1] * OutputNeuronNb[i]  +  OutputNeuronNb[i+1]
Tail: 2 * LSTMNeuronNb[0]
```

For `LSTMNeuronNb=[23,24,24]`, `LSTMSubSampling=[4,1]`, `OutputNeuronNb=[48,12,1]`,
`OutputSubSampling=[1,1]`:
- forward: layer1 `4*24*92 + 4*24*24 + 12*24 + 4*24 = 11520`; layer2 (`fin=24`) `4*24*24 + 4*24*24 +
  288 + 96 = 4992`; per-direction `16512`, x2 = `33024`.
- output: `48*12 + 12 = 588`, `12*1 + 1 = 13` -> `601`.
- tail: `2*23 = 46`.
- Total `33024 + 601 + 46 = 33671`. EXACT.

## 8. Module boundaries and interfaces

**Rust (crate `speech`):**

- `src/rust/src/io/binary.rs`
  - `read_matrix(path) -> Result<(usize /*rows*/, usize /*cols*/, Vec<f64> /*column-major*/)>`
  - `write_matrix(path, rows, cols, data: &[f64]) -> Result<()>`
  - `read_weight_vector(path) -> Result<Vec<f64>>` (asserts `cols == 1`)
  - Header via `byteorder` LE; `rows*cols > 0` guard.
- `src/rust/src/legacy_config.rs`
  - `parse_legacy_config(text: &str) -> LegacyConfig` (raw `IndexMap<String,String>`, last-wins,
    `_`-continuation, `#`-comments)
  - typed getters: scalar, comma-vector, and the `AlgName_*` weight-row families
- `src/rust/src/config.rs` - typed model + the seam + the packer
  - `Config` / `NnetSpec` (`LSTMNeuronNb`, subsampling, output sizes, peepholes, prefix)
  - `Nnet` (per-layer `output_size x ncols` matrices for forward/backward/output + mean/std)
  - `config_to_nnet(&LegacyConfig, &NnetSpec) -> Nnet` (applies `adim_coeff` decode)
  - `nnet_to_flat(&Nnet) -> Vec<f64>` (the 13-block packer; no scaling)
  - `flat_to_nnet(&[f64], &NnetSpec) -> Nnet` (inverse packer)
  - `nnet_to_config` / helper for the encode direction
  - `element_count(&NnetSpec) -> usize`

**Python (package `speech`):**

- `src/python/speech/weight_bridge.py` - the reference oracle
  - `read_bin(path) -> tuple[int,int,NDArray]` (`struct.unpack('<qq')` + `np.frombuffer('<f8')`)
  - `write_bin(rows, cols, data, path)`
  - `config_to_nnet(cfg, spec)` / `nnet_to_flat(nnet)` (numpy port; row-major via `X.reshape(-1)`
    on `X`, NOT `X.T`)
- `src/python/speech/config_bridge.py`
  - `parse_legacy_config(text) -> dict` (last-wins) + weight-row extraction

Each side independently packs `1_worker_1.config` and compares to `NNweights_config1.bin`.

## 9. Golden fixtures, strategy, and acceptance

**Vendored into `tests/reference_data/` (committed):**
- `NNweights_config1.bin` (269,384 bytes) - primary golden.
- `1_worker_1.config` (~3 KB) - the source config.
- `weightsDerivatives_bestNNWeight_1_..._worker_1.mat` (16 bytes) - the empty-file guard fixture.

Source path (local, git-ignored):
`/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Optimizer_V6.2.2/Executables/15-Oct-2015_BLSTM_OpenSAD15/`.

**Tests (Rust `tests/` + Python `tests/`):**
1. `.bin` header decode: rows==33671, cols==1, size==269384.
2. `.bin` round-trip: `read_matrix` then `write_matrix` == input bytes.
3. Empty-file guard: the 16-byte fixture errors (`rows*cols <= 0`).
4. Element count: `element_count(spec)` == 33671 for the paired config's dims.
5. Whole-pipeline (the acceptance): `parse(1_worker_1.config)` -> `config_to_nnet` -> `nnet_to_flat`
   == `read_weight_vector(NNweights_config1.bin)` bit-for-bit (`==` on `f64`, not approx).
6. Tail: `flat[-46:-23]` (mean) and `flat[-23:]` (std, all `> 0`).
7. Inverse: `flat_to_nnet(read_weight_vector(...), spec)` then `nnet_to_flat` round-trips.
8. Cross-language: Rust flat vector == Python flat vector == `.bin` payload (a Python test loads a
   Rust-emitted vector, or both compare to the fixture).
9. Output-layer adim asymmetry: an explicit assertion that the output layer uses
   `sqrt(out*sub)` with no next-term (a targeted unit test on `adim_coeff`).

## 10. Risks and pitfalls (must be pinned by tests)

- `adim_coeff` in the packer or codec, or scaling the bias column, or the LSTM `+next` term on
  output layers -> the byte-parity test fails but the diff is 269 KB with no localization. Mitigate
  with the independent Python oracle to bisect, plus the element-count and tail assertions.
- MATLAB `reshape(X',n,1)` = ROW-major flatten of `X`. Using `X.T.reshape(-1)` (numpy) or a
  column-major flatten yields a plausible-length wrong vector.
- The 12-column peephole bundle is a synthetic matrix from specific columns (`end-1`, then
  `end-4:end-2`); easy to mis-order.
- `.mat`-named files that are actually `.bin` - sniff the header, never the extension.
- Native-endianness assumption: hardcode LE, assert.

## 11. Definition of done

- `cargo build --release`, `cargo clippy --all-targets -- -D warnings`, `cargo test`, `cargo fmt
  --all -- --check` all green in `src/rust`.
- `uv run ruff check`, `ruff format --check`, `mypy`, `pytest tests` all green.
- The 9 tests above pass, including the byte-exact acceptance in both Rust and Python.
- `io/binary.rs`, `legacy_config.rs`, `config.rs` no longer rely on the crate-root
  `#![allow(dead_code, unused_variables)]` for their own items (narrow or drop it for these
  modules; it stays for the still-stub modules). `weight_bridge.py` / `config_bridge.py` bodies are
  real (no `...` stubs) for the 0a functions.
- CLAUDE.md architecture rows for these modules updated from "stub" to implemented; README roadmap
  marks Phase 0a done.
- Final step (in the plan): invoke the `smart-commit` skill over the whole branch.

## 12. Out of scope

Phase 0b (cost laws, `tasks::segmenter`, `tasks::segmentation_io`, `Segmentation` container,
`compute_errors`); NNType 1/2 weight ordering; `io::matfile` / any MAT5 read or write; the audio
front-end, feature extraction, and the NN forward/backward pass; WER/coverage/delay scoring (needs a
`.csv` fixture not present in the tree).
