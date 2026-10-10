"""Differential tests against the Federal Reserve's files and published tables.

These download the Fed's files on first use (about 520 MB of zips for all
waves) and are deselected by default. Run them with ``pytest -m fed``, and
the one test that fetches a superseded file from the Internet Archive with
``pytest -m archive``. Set ``FEDSCF_CACHE`` to reuse a cache between runs.

Published values are transcribed from the workbooks the Fed released with the
2025 survey on 2026-10-09, all in thousands of 2025 dollars:

* ``scf2025_tables_public_real_historical.xlsx``: tables computed from the
  public data. Table 4 gives median and mean net worth, Tables 1 ("89-98" and
  "01-25") median and mean income, for every wave.
* ``2025_tables1_2_stderrs.xlsx``: the report's Tables 1 and 2 with standard
  errors, computed from the Fed's internal data (also printed, rounded, in
  Table 2 of https://www.federalreserve.gov/publications/files/scf26.pdf).
"""

from __future__ import annotations

import math
import warnings
from functools import cache, lru_cache

import numpy as np
import pandas as pd
import pytest
from pandas.io.stata import StataReader

import fedscf

pytestmark = pytest.mark.fed

#: Public-data Table 4: (median, mean) family net worth.
PUBLIC_NET_WORTH = {
    1989: (118.74, 477.81),
    1992: (112.70, 424.18),
    1995: (122.43, 451.41),
    1998: (143.07, 565.98),
    2001: (158.91, 728.35),
    2004: (160.09, 772.32),
    2007: (189.49, 871.95),
    2010: (115.09, 739.50),
    2013: (113.39, 736.07),
    2016: (131.32, 930.68),
    2019: (154.47, 947.41),
    2022: (210.89, 1159.45),
    2025: (216.10, 1232.00),
}

#: Public-data Table 1: (median, mean) before-tax family income.
PUBLIC_INCOME = {
    1989: (65.805, 103.5),
    1992: (60.424, 90.798),
    1995: (65.076, 94.613),
    1998: (66.742, 105.847),
    2001: (73.488, 126.709),
    2004: (74.149, 121.477),
    2007: (74.085, 131.788),
    2010: (68.351, 117.048),
    2013: (65.046, 120.697),
    2016: (71.095, 138.056),
    2019: (74.915, 134.796),
    2022: (76.89, 154.734),
    2025: (82.162, 144.63),
}

#: Report Tables 1 and 2 (internal data): estimate and standard error.
BULLETIN = {
    ("networth", "median", 2022): (211.14, 8.1683939889241),
    ("networth", "median", 2025): (215.92, 6.35626687280876),
    ("networth", "mean", 2022): (1164.11, 25.363146104854735),
    ("networth", "mean", 2025): (1241.5, 30.954794096297046),
    ("income", "median", 2022): (76.89, 1.2997644167806004),
    ("income", "median", 2025): (82.162, 1.0974083770253549),
    ("income", "mean", 2022): (155.293, 4.632061215989897),
    ("income", "mean", 2025): (145.235, 2.4358322192954067),
}

#: CPI-U-RS for September of each survey year, times 100 (``CPIADJ`` in the
#: Fed's ``bulletin.macro.txt``). The 2026 release restated 2022-dollar
#: extracts in 2025 dollars by this ratio.
CPI_SEPTEMBER = {2022: 4376, 2025: 4789}

#: ``CPIADJ`` denominators for every wave, from the same macro.
CPI_SEPTEMBER_BY_WAVE = {
    1989: 1898,
    1992: 2112,
    1995: 2261,
    1998: 2400,
    2001: 2614,
    2004: 2785,
    2007: 3058,
    2010: 3204,
    2013: 3438,
    2016: 3548,
    2019: 3775,
    2022: 4376,
    2025: 4789,
}

WAVES = fedscf.WAVES


@cache
def summary(wave: int) -> pd.DataFrame:
    return fedscf.load_summary(wave, ["networth", "income"])


@lru_cache(maxsize=2)
def replicate_weights(wave: int) -> pd.DataFrame:
    return fedscf.load_replicate_weights(wave)


def thousands(value: float) -> float:
    return value / 1000


def test_registry_pins_reach_every_current_file():
    """Every current file downloads, matches both hashes and has its rows."""
    for item in fedscf.artifacts(include_superseded=False):
        path = fedscf.fetch_artifact(item)  # raises on any hash mismatch
        with StataReader(path) as reader:
            first = next(iter(reader.variable_labels()))
        rows = len(pd.read_stata(path, columns=[first]))
        assert rows == item.expected_rows, item.key


@pytest.mark.parametrize("wave", WAVES)
def test_net_worth_reproduces_the_public_table(wave):
    """Pooled median and mean match Table 4 to the printed hundredth."""
    data = summary(wave)
    median = fedscf.weighted_quantile(data["networth"], data["wgt"], 0.5)
    mean = fedscf.weighted_mean(data["networth"], data["wgt"])
    published_median, published_mean = PUBLIC_NET_WORTH[wave]
    assert thousands(median) == pytest.approx(published_median, abs=0.0051)
    assert thousands(mean) == pytest.approx(published_mean, abs=0.0051)


@pytest.mark.parametrize("wave", WAVES)
def test_income_reproduces_the_public_table(wave):
    """Pooled median and mean match Table 1 to the printed thousandth."""
    data = summary(wave)
    median = fedscf.weighted_quantile(data["income"], data["wgt"], 0.5)
    mean = fedscf.weighted_mean(data["income"], data["wgt"])
    published_median, published_mean = PUBLIC_INCOME[wave]
    assert thousands(median) == pytest.approx(published_median, abs=0.00051)
    assert thousands(mean) == pytest.approx(published_mean, abs=0.00051)


@pytest.mark.parametrize("wave", WAVES)
def test_wave_structure(wave):
    """Summary and full files join one to one; weights and replicates agree."""
    data = fedscf.load_summary(wave, [])
    full = fedscf.load_full(wave, ["x42001"])
    joined = fedscf.join(data, full)
    families = data["yy1"].nunique()
    assert len(data) == len(full) == len(joined) == 5 * families
    # The extract's weight is the full file's X42001 / 5, exactly.
    np.testing.assert_array_equal(joined["wgt"], joined["x42001"] / 5)
    # Each implicate's X42001 sums to the same number of families.
    by_implicate = full.groupby("implicate")["x42001"].sum()
    assert by_implicate.max() / by_implicate.min() - 1 < 2e-6
    assert data["wgt"].sum() == pytest.approx(by_implicate.mean(), rel=2e-6)
    # One implicate kept carries the whole population.
    first = fedscf.select_implicates(data, 1)
    assert first["wgt"].sum() == pytest.approx(by_implicate.loc[1], rel=1e-12)
    # Replicate weights: one row per family, each replicate a full population.
    replicates = fedscf.load_replicate_weights(wave)
    assert sorted(replicates.index) == sorted(data["yy1"].unique())
    totals = replicates.sum()
    assert ((totals / data["wgt"].sum() - 1).abs() < 1e-3).all()
    assert (replicates.to_numpy() >= 0).all()


def test_2022_facts():
    """The 2022 numbers in this package's design brief, computed 2026-10-09."""
    data = summary(2022)
    full = fedscf.load_full(2022, ["x42001"])
    assert len(data) == len(full) == 22_975
    np.testing.assert_array_equal(data["wgt"], full["x42001"] / 5)
    assert round(data["wgt"].sum() / 1e6, 2) == 131.31
    assert len(fedscf.join(data, full)) == 22_975
    # The brief's medians and means are in 2022 dollars, from the extract the
    # Fed served until 2026-10-09; the current one is in 2025 dollars.
    to_2022 = CPI_SEPTEMBER[2022] / CPI_SEPTEMBER[2025]
    median = fedscf.weighted_quantile(data["networth"], data["wgt"], 0.5)
    mean = fedscf.weighted_mean(data["networth"], data["wgt"])
    assert median * to_2022 == pytest.approx(192_700, abs=0.01)
    assert round(mean * to_2022) == 1_059_457


def test_the_int16_overflow_this_package_avoids():
    """Multiplying yy1 in its stored width loses 29 percent of 2022 families."""
    path = fedscf.fetch(2022, "summary")
    raw = pd.read_stata(path, columns=["y1", "yy1"], convert_categoricals=False)
    assert raw["yy1"].dtype == np.int16 and raw["yy1"].max() > 3276
    naive = raw["y1"] - raw["yy1"] * 10
    assert (naive == 1).sum() == 3_270
    assert (summary(2022)["implicate"] == 1).sum() == 4_595
    assert round(1 - 3_270 / 4_595, 2) == 0.29


@pytest.mark.parametrize(
    "key", sorted(BULLETIN), ids=lambda key: "-".join(map(str, key))
)
def test_standard_errors_are_close_to_the_report(key):
    """The Fed's method on public data lands near the report's internal numbers.

    The report uses internal data, which hold families the public files omit
    (7 of 4,602 in 2022 and of 4,367 in 2025, the ones at or above the Forbes
    400 threshold) and values the public files blur or round. Observed gaps:
    point estimates within 0.8 percent (the means fall short, as removing the
    wealthiest families implies) and standard errors within 9 percent.
    """
    column, kind, wave = key
    published, published_se = BULLETIN[key]
    statistic = fedscf.median(column) if kind == "median" else fedscf.mean(column)
    result = fedscf.estimate(summary(wave), statistic, replicate_weights(wave))
    assert thousands(result.estimate) == pytest.approx(published, rel=0.01)
    assert thousands(result.standard_error) == pytest.approx(published_se, rel=0.10)
    if kind == "mean" and column == "networth":
        assert thousands(result.estimate) < published


def test_2025_median_net_worth_matches_the_report():
    """The mean of the five implicate medians, as the codebook's MEANIT takes it."""
    result = fedscf.estimate(
        summary(2025), fedscf.median("networth"), replicate_weights(2025)
    )
    assert round(thousands(result.estimate), 1) == 215.9  # report Table 2
    assert round(thousands(result.standard_error), 1) == pytest.approx(6.4, abs=0.25)


def test_sampling_variance_agrees_with_microdf_on_real_data():
    mdf = pytest.importorskip("microdf")
    data = summary(2025)
    replicates = replicate_weights(2025)
    first = data.loc[data["implicate"] == 1]
    matrix = replicates.loc[first["yy1"]].to_numpy()
    n = matrix.shape[1]
    series = mdf.MicroSeries(first["networth"].to_numpy(), weights=first["wgt"] * 5)
    theirs = mdf.replicate_variance(
        series,
        lambda s: s.mean(),
        matrix,
        method="bootstrap",
        centering="replicate-mean",
    ) * (n / (n - 1))
    ours = fedscf.estimate(data, fedscf.mean("networth"), replicates)
    assert ours.sampling_variance == pytest.approx(theirs, rel=1e-9)


def test_load_stacks_waves_in_one_dollar_year():
    stacked = fedscf.load([2022, 2025], ["networth"])
    assert list(stacked.columns) == [
        "year",
        "yy1",
        "y1",
        "implicate",
        "wgt",
        "networth",
    ]
    assert stacked.groupby("year").size().to_dict() == {2022: 22_975, 2025: 21_800}
    assert {record["dollar_year"] for record in stacked.attrs["fedscf"]} == {2025}
    single = fedscf.load(2025, "networth", implicates=1)
    assert len(single) == 4_360 and math.isclose(
        single["wgt"].sum(), summary(2025)["wgt"].sum(), rel_tol=1e-6
    )


def test_load_marks_its_provenance():
    record = summary(2025).attrs["fedscf"]
    item = fedscf.artifact(2025, "summary")
    assert record["sha256"] == item.sha256 and record["verified"] is True
    assert record["dollar_year"] == 2025 and record["implicates"] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("wave", WAVES)
def test_nhnfin_defect_in_the_2026_release(wave):
    """Records a defect in the Fed's files, and that the loader flags it.

    The Fed defines NHNFIN = NFIN - HOUSES (bulletin.macro.txt). The 2026-10-09
    extracts for 1989-2022 multiply it by the wave's factor to 2025 dollars a
    second time; only the 2025 extract is right. The earlier releases have the
    same defect in 2022 dollars, for every wave before 2022.
    """
    columns = ["nhnfin", "nfin", "houses", "income"]
    if wave < 2025:
        with pytest.warns(fedscf.FedDataWarning, match="nhnfin"):
            data = fedscf.load_summary(wave, columns)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", fedscf.FedDataWarning)
            data = fedscf.load_summary(wave, columns)
    definition = data["nfin"] - data["houses"]
    holding = np.isclose(data["nhnfin"], definition, rtol=1e-9, atol=0.5)
    if wave == 2025:
        assert holding.all()
    else:
        nonzero = definition != 0
        ratio = (data["nhnfin"][nonzero] / definition[nonzero]).to_numpy()
        # The factor is the wave's September CPI-U-RS to 2025's. Rounding of
        # the public values spreads the ratio by up to about one part in 10^4.
        factor = CPI_SEPTEMBER[2025] / CPI_SEPTEMBER_BY_WAVE[wave]
        assert np.median(ratio) == pytest.approx(factor, rel=1e-6)
        assert (np.abs(ratio / factor - 1) < 1e-3).mean() > 0.999
        assert holding[~nonzero].all() and not holding[nonzero].any()


@pytest.mark.archive
def test_superseded_2022_extract_from_the_internet_archive():
    """The 2024-04-03 extract: Microcosm's pin, in 2022 dollars.

    It is the current extract divided by the CPI ratio, value for value.
    """
    earlier = fedscf.artifact(2022, "summary", "2024-04-03")
    # The pins in Microcosm's source_stages.json, stage scf_wealth.
    assert earlier.sha256 == (
        "3bb4d890ae2463ff6039ec7692e375f544dd98a55a37ca2cb2340354b9cc9d80"
    )
    assert earlier.member_sha256 == (
        "6b8dd2d935a76ed225ddebc80fb2db22a467f0c80d9a1acaa67b4584aa4bafd1"
    )
    old = fedscf.load_summary(2022, ["networth", "income"], released="2024-04-03")
    assert old.attrs["fedscf"]["dollar_year"] == 2022
    median = fedscf.weighted_quantile(old["networth"], old["wgt"], 0.5)
    assert median == 192_700
    assert round(fedscf.weighted_mean(old["networth"], old["wgt"])) == 1_059_457
    new = summary(2022)
    np.testing.assert_array_equal(old[["yy1", "y1", "wgt"]], new[["yy1", "y1", "wgt"]])
    ratio = CPI_SEPTEMBER[2025] / CPI_SEPTEMBER[2022]
    for column in ("networth", "income"):
        np.testing.assert_allclose(new[column], old[column] * ratio, rtol=1e-12)
    # A caller holding Microcosm's pin resolves to the same vintage.
    same = fedscf.load_summary(2022, ["networth"], sha256=earlier.sha256)
    assert same.attrs["fedscf"]["released"] == "2024-04-03"
