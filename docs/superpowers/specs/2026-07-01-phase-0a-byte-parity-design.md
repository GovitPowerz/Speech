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

**Acceptance.** The `.bin` codec and the config parser are validated bit-for-bit / real-value
against real artifacts: (1) reading `NNweights_config1.bin` (269,384 bytes) and re-writing it
reproduces the input bytes exactly; (2) parsing `1_worker_1.config` yields the real `NnetSpec`.
Both hold in Rust and Python.

CORRECTION (discovered during implementation): the packer + `adim` seam CANNOT be validated against
an independent legacy artifact - no `(structured-config, matching-.bin)` pair exists in the tree
(`nnet_best` is the CMA-ES optimizer genome, not a weight-pack; `BestConfigStruct` matches none of
685 candidate `.bin` files). The packer/adim is therefore validated by: line-by-line cross-check
against `config2weights.m` / `config2network.m`; the exact element count (33,671); the mutual-inverse
`flat <-> nnet` bijection; the `adim` invariants (bias exempt, body scaled by `1/adim`); the
normalize `mean`/`std` tail matching a real `.bin`; and - across Tasks 3-4 - Rust-pack == Python-pack
(two independent implementations agreeing on the same structured input). The definitive bit-exact
packer/adim golden is DEFERRED to Phase 2, where inference on a real `.bin` matched against the legacy
VRCTS segmentation output validates the whole load->infer path end-to-end.

This is Phase 0a only. Phase 0b (cost laws, segmentation, scoring, VRCTS/STM I/O) is a separate
later cycle with its own spec.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Cycle scope | Phase 0a alone (byte-parity). 0b is the next cycle. |
| 2 | Golden fixtures | Vendor into `tests/reference_data/phase0/` (committed): the real `NNweights_config1.bin` (~269 KB) and `1_worker_1.config` (~3 KB), PLUS derived binary fixtures extracted from the `save_net` `.mat` by a committed prep script (`scripts/extract_phase0_fixtures.py`, scipy): `best_config_domain.bin` (the structured config-domain weight rows) + `best_config_manifest.json` (row names/shapes) + `best_net_flat.bin` (the paired nnet-domain flat vector `nnet_best[:33671]`). Tests run in CI. |
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

IMPORTANT (verified against the real `1_worker_1.config`, 76 lines / 2.9 KB): the `.config` TEXT
file holds hyperparameters + structural sizes + a `BLSTM_weightsFile` POINTER - it does NOT contain
the weight matrices (33,671 doubles cannot fit in 2.9 KB). The text-file key prefix is the `algName`
value `BLSTM_` (e.g. `BLSTM_LSTMNeuronNb`). The `AlgName_` literal prefix appears only in the
in-memory struct / `.mat` (see below).

**Load-bearing `.config` keys for 0a (NNType 0), all `BLSTM_`-prefixed:** `BLSTM_NNType` (0),
`BLSTM_LSTMNeuronNb` (`23,24,24`), `BLSTM_LSTMSubSampling` (`4,1`), `BLSTM_OutputNeuronNb`
(`48,12,1`), `BLSTM_OutputSubSampling` (`1,1`), `BLSTM_NNetInputSize` (23), the six peephole flags
`BLSTM_[Forward|Backward]_Is{Cells,Gates,GatesRecurrent}PeepholesActive`, `BLSTM_weightsFile`, and
the cost keys `BLSTM_CostLaw{Speech,NoSpeech}` / `..Param..` / `..Thresh..`. The parser reads these
into the typed `NnetSpec` + `Config`; the weight VALUES do not come from here.

**Structured weight source (for the packer/adim golden):** the `save_net` `.mat`
(`LSTM_15-Oct-2015_BLSTM_OpenSAD15.mat`) holds `BestConfigStruct`, a struct whose fields
`AlgName_[Forward|Backward]_Layer_i_LSTMBlock_j_{Input,Forget,Output}GateWeights` / `_CellWeight`
(one per block j = neuron row, value = a config-domain row of length `ncols`),
`AlgName_Output_Layer_i_Neuron_j_Weights`, and `AlgName_NormalizeInput{Mean,Std}` ARE the structured
config-domain weights, and `nnet_best` (padded to 50000, first 33,671 real) is the paired nnet-domain
flat vector. These are loaded with scipy in the Python oracle and extracted into the committed binary
fixtures (Decision 2). `BestConfigStruct` stores config-domain (adim-scaled) rows.

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

**Per LSTM layer** (`output_size = LSTMNeuronNb[i+1]`, `input_size = LSTMNeuronNb[i] *
LSTMSubSampling[i]`). CORRECTION (verified against the real `.mat` + `config2weights.m`): the four
gate matrices do NOT share a width. `inputweights`/`forgetweights`/`outputweights` are
`output_size x (input_size + output_size + 5)` (columns: `1..output_size` recurrent,
`+input_size` fan-in, then `[3 gate-peephole cols][1 recurrent-peephole col][1 bias]`), but
`cellweights` is `output_size x (input_size + output_size + 1)` - it has NO peephole columns, so its
bias is its OWN last column. Each gate's bias is therefore `m[:, -1]` (its own last column), NOT a
shared `ncols-1` index. Each block is flattened ROW-MAJOR (`reshape(tmp',numel,1)` = numpy
`X.reshape(-1)`, C-order). Emission order (`idx_in`/`idx_rec` identical across all four gates):

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
10. inputweights(:,-1)     BIAS input gate    (output_size elems; own last col)
11. forgetweights(:,-1)    BIAS forget gate
12. outputweights(:,-1)    BIAS output gate
13. cellweights(:,-1)      BIAS cell           (cell's last col is input_size+output_size, not ncols-1)
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
  - `NnetSpec` (from the `.config`: `LSTMNeuronNb`, subsampling, output sizes/subsampling, input
    size, peephole flags) + `Config` (adds cost keys, `weightsFile`)
  - `StructuredWeights` (config-domain per-block rows: forward/backward layers x blocks x
    `{input,forget,output,cell}` rows, output-neuron rows, mean, std) - the in-memory form of
    `BestConfigStruct`, populated from the committed fixture
  - `Nnet` (nnet-domain per-layer `output_size x ncols` matrices for forward/backward/output +
    mean/std)
  - `config_to_nnet(&StructuredWeights, &NnetSpec) -> Nnet` (applies `adim_coeff` decode)
  - `nnet_to_flat(&Nnet) -> Vec<f64>` (the 13-block packer; no scaling)
  - `flat_to_nnet(&[f64], &NnetSpec) -> Nnet` (inverse packer)
  - `element_count(&NnetSpec) -> usize`
- `src/rust/src/legacy_config.rs`
  - `parse_legacy_config(text) -> LegacyConfig` + `NnetSpec`/`Config` extraction (sizes, peephole
    flags, cost keys). NO weight rows come from here.

**Python (package `speech`):**

- `src/python/speech/weight_bridge.py` - the numpy reference oracle
  - `read_bin(path) -> tuple[int,int,NDArray]` (`struct.unpack('<qq')` + `np.frombuffer('<f8')`);
    `write_bin(rows, cols, data, path)`
  - `config_to_nnet(structured, spec)` / `nnet_to_flat(nnet)` / `flat_to_nnet(flat, spec)` (numpy;
    row-major via `X.reshape(-1)` on `X`, NOT `X.T`); `element_count(spec)`
- `src/python/speech/config_bridge.py`
  - `parse_legacy_config(text) -> dict` (last-wins) + `NnetSpec` extraction
- `scripts/extract_phase0_fixtures.py` (committed prep + ground-truth oracle, scipy)
  - `scipy.io.loadmat` the `save_net` `.mat`; extract `BestConfigStruct`'s structured config-domain
    rows -> `best_config_domain.bin` + `best_config_manifest.json`; extract `nnet_best[:33671]`
    (assert `nnet_best[33671:]` all zero) -> `best_net_flat.bin`. Before emitting, ASSERT the numpy
    `nnet_to_flat(config_to_nnet(structured, spec)) == nnet_best[:33671]` bit-for-bit - this is the
    authoritative check that the extraction + adim + packer reproduce the real flat net.

## 9. Golden fixtures, strategy, and acceptance

**Vendored into `tests/reference_data/phase0/` (committed):**
- `NNweights_config1.bin` (269,384 bytes) - real `.bin` codec golden.
- `1_worker_1.config` (~3 KB) - real config-parse golden.
- `best_config_domain.bin` + `best_config_manifest.json` - the structured config-domain weight rows
  from `BestConfigStruct` (emitted by the prep script).
- `best_net_flat.bin` (269,384 bytes) - `nnet_to_flat(config_to_nnet(BestConfigStruct))`, i.e. the
  SELF-DERIVED pack (there is no independent nnet-domain golden; see the section 1 correction). It
  serves as the fixed-point that Rust and Python must both reproduce (cross-language agreement).

Source (local, git-ignored): `.bin`/`.config` under
`/Users/govit/Git/Govit/FastSpeechProcessing-legacy/Optimizer_V6.2.2/Executables/15-Oct-2015_BLSTM_OpenSAD15/`;
the `.mat` at `.../Optimizer_V6.2.2/save_net/LSTM_15-Oct-2015_BLSTM_OpenSAD15.mat`. The prep script
regenerates the derived fixtures from these.

**Tests (Rust `tests/` + Python `tests/`):**
1. `.bin` header decode: `read_matrix(NNweights_config1.bin)` -> rows==33671, cols==1; file size
   ==269384.
2. `.bin` round-trip: `read_matrix` then `write_matrix` == input bytes.
3. Short/empty guard: a 16-byte header with `rows*cols == 0` (constructed in-test) errors.
4. Config parse: `parse(1_worker_1.config)` -> `NnetSpec` with `LSTMNeuronNb==[23,24,24]`,
   `LSTMSubSampling==[4,1]`, `OutputNeuronNb==[48,12,1]`, `OutputSubSampling==[1,1]`,
   `NNetInputSize==23`, all six peephole flags `true`, cost laws `"log"`/`"log"`.
5. Element count: `element_count(spec)` == 33671 == `read_matrix(NNweights_config1.bin).rows`.
6. Packer bijection (no adim): `flat_to_nnet(read_weight_vector(NNweights_config1.bin), spec)` then
   `nnet_to_flat` == the input flat, bit-for-bit. Tail: `flat[-46:-23]` (mean), `flat[-23:]` (std,
   all `> 0`).
7. Pipeline integrity + cross-language fixed-point: load `best_config_domain.bin` + manifest,
   `config_to_nnet` (adim), `nnet_to_flat` == `read_weight_vector(best_net_flat.bin)` bit-for-bit.
   This is NOT an independent golden (the fixture is self-derived); it pins pipeline determinism and,
   because Rust and Python run the SAME assertion against the SAME committed fixture, forces the two
   implementations to agree bit-for-bit.
8. adim invariants: on the committed config-domain fixture, assert `nnet[:,-1] == config[:,-1]` (bias
   exempt) and `nnet[:,:-1] * adim == config[:,:-1]` (body rescaled), for LSTM and output layers; and
   a targeted check that the output-layer adim uses `sqrt(out*sub)` with NO next-term.
9. Normalize-tail vs real `.bin`: `best_net_flat.bin[-46:]` (the `mean`/`std` tail) equals
   `NNweights_config1.bin[-46:]` (both nets share the corpus normalization) - a small real-artifact
   anchor on the tail placement; `std` (`[-23:]`) strictly positive.

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
