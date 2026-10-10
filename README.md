# fedscf

`fedscf` loads the Federal Reserve's [Survey of Consumer Finances](https://www.federalreserve.gov/econres/scfindex.htm) (SCF) public microdata in Python, for every wave from 1989 to 2025. It pins the hash of each file the Fed publishes, refuses a download that does not match, reads every wave into one layout, and computes standard errors by the Fed's own method.

## Install

`fedscf` is not on PyPI yet. Install it from this repository:

```bash
uv pip install git+https://github.com/PolicyEngine/fedscf
```

## Quick start

```python
import fedscf

scf = fedscf.load_summary(2025, ["networth", "income"])
weights = fedscf.load_replicate_weights(2025)

result = fedscf.estimate(scf, fedscf.median("networth"), weights)
round(result.estimate), round(result.standard_error)
# (215916, 6589): the report's Table 2 prints 215.9 thousand (6.4)
```

`load_summary` reads the Summary Extract Public Data, the file behind the Fed's triennial report. `load_full` reads the full public data set (about 5,000 variables; pass `columns`), and `join` matches the two row for row:

```python
full = fedscf.load_full(2025, ["x5729", "x3014"])
both = fedscf.join(scf, full)
```

## What you get

**One layout for every wave.** Each reader returns `yy1` (family), `y1` (record), `implicate` (1 to 5) and `wgt`, then the variables you asked for, with lower-case names. `wgt` sums to the number of families over the rows returned. The Fed's files differ from wave to wave in ways that break naive code:

- `yy1` is stored as a 16-bit integer and exceeds 3,276 in every wave, so `y1 - yy1 * 10` overflows unless widened first. Done naively on the 2022 extract, the filter for the first implicate keeps 3,270 of 4,595 families.
- The 1989 files name the identifiers `x1` and `xx1`.
- Older full files and several replicate-weight files use upper-case names.

**Implicates.** Each family appears five times, once per implicate of the multiple imputation. `implicates=` keeps a subset and rescales `wgt` so the population stays the same; `implicates=1` gives each row its full-file weight `x42001`.

**Verified downloads.** `registry.json` lists the zip and Stata member SHA-256 of all 39 current files (summary extract, full data set and replicate weights for 13 waves) plus 14 earlier summary extracts. Every load checks both hashes, on download and on every cache hit, and raises `fedscf.HashMismatchError` on a mismatch. Files are cached under `~/.cache/fedscf` (macOS: `~/Library/Caches/fedscf`), or under `$FEDSCF_CACHE`.

**Standard errors.** `fedscf.estimate` follows the codebook's `MEANIT` macro and the Fed's [standard error note](https://www.federalreserve.gov/econres/files/Standard_Error_Documentation.pdf): the estimate is the mean of the five per-implicate estimates, the sampling variance comes from the 999 bootstrap replicate weights applied to the first implicate, and the total variance is the sampling variance plus 6/5 of the variance across implicates. `fedscf.pool` is the combining step on its own, for regressions or anything else estimated once per implicate.

```python
fedscf.estimate(scf, fedscf.mean("income"), weights)
fedscf.estimate(scf[scf["networth"] > 0], fedscf.median("income"), weights)
fedscf.pool(per_implicate_coefficients, per_implicate_variances)
```

## Dollars and vintages

The Fed reissues files under the same names. Each time it publishes a wave, it restates every earlier wave's summary extract in the new wave's dollars: since 2026-10-09 all extracts are in 2025 dollars, and before that they were in 2022 dollars. A load records what it read in `frame.attrs["fedscf"]`, including `released` and `dollar_year`. To reproduce an earlier analysis, pin the vintage:

```python
fedscf.load_summary(2022, ["networth"], released="2024-04-03")  # 2022 dollars
fedscf.load_summary(2022, ["networth"], sha256="3bb4d890...")  # the same file
```

One variable is wrong in the Fed's restated extracts: for every wave but the newest in a release, `nhnfin` has been converted to the release's dollars twice. The Fed defines it as `nfin - houses`, and computing that from the same file gives the right value. `load_summary` warns when it returns `nhnfin` from an affected file; [methods](https://policyengine.github.io/fedscf/methods.html) has the details.

Superseded vintages download from the Internet Archive's copy of the Fed's URL. A hash this package does not know is checked against the Fed's current file, so a new reissue can be pinned before `fedscf` is updated. `fedscf verify` compares the Fed's current files with the pins, and a weekly CI job runs it.

## Checks against the Fed's numbers

`pytest -m fed` downloads the Fed's files and checks, among other things, that:

- for all 13 waves, median and mean net worth and income match the Fed's tables computed from the public data, to the last printed digit. Those tables take the median of the five implicates stacked, which `fedscf.weighted_quantile(scf["networth"], scf["wgt"], 0.5)` reproduces; `fedscf.estimate` averages the five implicate medians, as `MEANIT` does;
- for all 13 waves, the summary and full files join one to one on `(yy1, y1)`, the extract's `wgt` equals the full file's `x42001 / 5` exactly, and each of the 999 replicate weights sums to the population within 0.1 percent;
- 2022 and 2025 estimates and standard errors fall within 1 and 10 percent of those in the Fed's report. The report uses internal data, which hold families the public files omit (7 in each of 2022 and 2025, those at or above the Forbes 400 threshold) and values the public files blur, so its means run higher.

## Microcosm manifest fields

Registry entries use the field names of the artifact entries in Microcosm's source-stage manifest (`kind`, `format`, `vintage`, `locator`, `member`, `sha256`, `member_sha256`, `expected_rows`, `size_bytes`). `Artifact.manifest_entry()` writes one; `Artifact.from_manifest()` reads one.

## Command line

```bash
fedscf list --wave 2025          # pinned files
fedscf fetch 2025 full           # download, verify, print the Stata path
fedscf verify                    # re-download current files and compare with the pins
fedscf pin 2022 summary --released 2026-10-09  # registry entry for what the Fed serves now
```

## Moving from `scf` 0.1

This repository's earlier package (in `PSLmodels/scf` until 2026-10-09), imported as `scf`, read only the summary extract through functions microdf no longer exports. `fedscf.load(years, columns)` replaces `scf.load`. It returns `year` and the identifier columns for every call, and reads 2022 and 2025 as well. `as_microdataframe` is not carried over.

## Development

```bash
uv sync
uv run pytest            # offline tests, about a minute
uv run pytest -m fed     # against the Fed's files (downloads about 520 MB)
uv run pytest -m archive # a superseded file from the Internet Archive
```

The SCF is produced by the Board of Governors of the Federal Reserve System. This package is not affiliated with the Fed.
