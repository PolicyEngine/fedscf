# Methods and invariants

## The Fed's files

The Fed publishes three Stata files for each wave, each a single `.dta` inside a zip:

| File | Zip | Rows | Contents |
| --- | --- | --- | --- |
| Summary extract | `scfpYYYYs.zip` | 5 per family | about 360 derived variables used in the Fed's report, including `wgt` |
| Full public data set | `scfYYYYs.zip` (`scfYYs.zip` through 2001) | 5 per family | about 5,000 survey variables `x1`... and their imputation flags `j1`... |
| Replicate weights | `scfYYYYrw1s.zip` (`scfYYrw1s.zip` through 1998) | 1 per family | `wt1b1`-`wt1b999` and `mm1`-`mm999` |

The registry records, for each file, the Fed's "Last updated" date and the SHA-256 of the zip and of the Stata member. Pins for current files were computed from downloads on 2026-10-09. Earlier summary extracts were recovered from the Internet Archive's captures of the same URLs; each was checked against the current file, value for value.

## Vintages of the summary extract

| Released | Dollars | Archive copies pinned |
| --- | --- | --- |
| 2023-10-18 | 2022 | 2019, 2022 |
| 2024-04-03 | 2022 | every wave from 1989 to 2022 |
| 2026-10-09 | 2025 | current |

Comparing the 2024 and 2026 releases value by value: the 2026 release multiplies the dollar variables of the 2024 release by 4789/4376, the ratio of the September CPI-U-RS for 2025 and 2022 in the Fed's `bulletin.macro.txt`, leaves weights, identifiers and categorical variables unchanged, and adds five variables (`mpdebt`, `mpsave`, `mpspend`, `totspendnh`, `vacation`). The 2024 release added `emergwork` to the 2019 and 2022 extracts and revised a few of their values.

## A defect in the Fed's `nhnfin`

The Fed's macro defines `NHNFIN=NFIN-HOUSES`. That holds for the newest wave in each release: 2022 in the 2023 and 2024 releases, 2025 in the 2026 release. For every earlier wave a release restates, `nhnfin` is that difference times the wave's CPI factor to the release's dollars, so the conversion has been applied twice: by 4789/1898 = 2.52 for 1989 in the current files, 4789/4376 = 1.09 for 2022. Use `nfin - houses`. `load_summary` issues a `fedscf.FedDataWarning` when it returns `nhnfin` from an affected file, and `tests/test_fed_data.py` records the defect for every wave.

Comparing the 2024 and 2026 releases also turned up a few nonzero values of capital-gains variables (`kgbus` in 1989, `kgstmf` and `kgtotal` in 2022, the two waves compared column by column) that are identical in both releases instead of rescaled. We have not established why.

## Standard errors

For a statistic $\theta$ estimated on implicates $m = 1, \dots, M$ ($M = 5$):

$$\hat\theta = \frac{1}{M}\sum_m \hat\theta_m, \qquad
B = \frac{1}{M-1}\sum_m (\hat\theta_m - \hat\theta)^2, \qquad
U = \frac{1}{R-1}\sum_{r=1}^{R} (\hat\theta_{1,r} - \bar\theta_{1,\cdot})^2,$$

where $\hat\theta_{1,r}$ is the statistic on the first implicate with replicate weight $r$ ($R = 999$), and the variance is $U + (1 + 1/M) B$. Replicate weight $r$ for a family is $\max(0, \text{wt1b}_r) \cdot \max(0, \text{mm}_r)$, with missing values counted as zero; this is the codebook's `MEANIT` macro. `fedscf.pool` applies the same combination to estimates and variances from any source, as the codebook's `MISECOMP` macro does.

## Invariants

These hold for every input and are tested with Hypothesis (offline) or against the Fed's files (`pytest -m fed`):

- **Implicate decoding.** `implicate == y1 - yy1 * 10` lies in 1 to 5 for every row, whatever integer width the file stores the identifiers in.
- **Weight totals.** Keeping $k$ of $n$ implicates multiplies `wgt` by $n/k$; the total is then the mean population of the kept implicates, which equals the full total when implicates carry equal totals (the Fed's do, to one part in a million). Selections compose.
- **Join.** `join` returns a bijection on `(yy1, y1)` with matching weights, or raises. Its result does not depend on row order.
- **Verified fetch.** A path `fetch` returns holds bytes with the pinned SHA-256; any other hash raises `HashMismatchError` and caches nothing under that pin. Vintages with one file name never share a cache path.
- **Pooling.** With identical implicate estimates the pooled variance equals the sampling variance exactly; otherwise it exceeds it. The estimate is the mean of the implicate estimates and does not depend on their order.
- **Quantiles.** `weighted_quantile` returns one of the values, is non-decreasing in `q`, and with integer weights equals the quantile of the sample in which each row is repeated by its weight.
- **Determinism.** The same inputs give the same frames and estimates.

Differential tests compare `estimate` with a line-by-line transcription of `MEANIT`, `pool` with `MISECOMP`, the replicate variance with microdf's, the quantile with NumPy's, and all 13 waves with the Fed's public-data tables.
