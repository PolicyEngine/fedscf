# Changelog

## 1.0.0 (unreleased)

The package is rebuilt and renamed from `scf` to `fedscf`, so that its import name cannot collide with the unrelated `scf` package on PyPI.

The repository moved from `PSLmodels/scf` to `PolicyEngine/fedscf`; GitHub redirects the old URLs.

### Added

- A registry of the Fed's Stata files for every wave from 1989 to 2025 (summary extract, full public data set, replicate weights), keyed by wave, file and Fed release date, with the SHA-256 of each zip and Stata member, row counts and sizes. It also pins 14 superseded summary extracts, downloadable from the Internet Archive. Entries use the field names of Microcosm's source-stage manifest.
- A fetch that checks both hashes on every download and cache hit, accepts caller-supplied hashes, and raises `HashMismatchError` on a mismatch.
- Readers for the summary extract, the full public data set and the replicate weights that give every wave one layout, decode implicates without the 16-bit overflow, rescale weights when implicates are dropped, and join summary and full files one to one.
- `estimate` and `pool`: the Fed's standard errors from replicate weights and multiple imputation.
- `FedDataWarning` for the Fed's `nhnfin`, which is converted to newer dollars twice in restated extracts.
- A `fedscf` command line: `list`, `fetch`, `verify`, `pin`.
- Tests against the Fed's files and published tables (`pytest -m fed`).

### Changed

- `fedscf.load(years, columns)` replaces `scf.load`. It returns `year`, `yy1`, `y1`, `implicate` and `wgt` with every call and covers 2022 and 2025.

### Removed

- The microdf dependency, and `as_microdataframe`.
