"""Pooling, weighted statistics and the Fed's standard-error method.

Invariants:

* ``pool``: the pooled variance equals the sampling variance when the
  implicates agree, and never falls below it; the estimate is the mean of the
  implicate estimates and does not depend on their order;
* ``weighted_quantile``: the result is one of the values, is non-decreasing in
  ``q``, and ignores the scale of the weights and the order of the rows;
* ``estimate``: variances are non-negative, the result does not depend on row
  order, and a second call returns the same numbers.

Differential tests check :func:`fedscf.estimate` against a line-by-line
transcription of the codebook's ``MEANIT`` macro, :func:`fedscf.pool` against
``MISECOMP``, the quantile against NumPy's, and the replicate variance against
microdf's.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import fedscf
from conftest import replicate_table, summary_table, synthetic_families, to_dta_bytes

finite = st.floats(-1e9, 1e9, allow_nan=False)
variance = st.floats(0, 1e12, allow_nan=False)


# --- pool --------------------------------------------------------------------


@settings(max_examples=300)
@given(value=finite, within=variance, m=st.integers(2, 10))
def test_pooled_variance_is_the_within_variance_when_implicates_agree(value, within, m):
    pooled = fedscf.pool([value] * m, [within] * m)
    assert pooled.imputation_variance == 0.0
    assert pooled.variance == within  # exactly, not approximately
    assert pooled.estimate == pytest.approx(value, rel=1e-12, abs=1e-12)
    assert fedscf.pool([value] * m, within).variance == within


@settings(max_examples=300)
@given(st.lists(st.tuples(finite, variance), min_size=2, max_size=8))
def test_pool_matches_rubins_formula_and_its_bounds(pairs):
    estimates = [pair[0] for pair in pairs]
    variances = [pair[1] for pair in pairs]
    m = len(pairs)
    pooled = fedscf.pool(estimates, variances)
    between = np.var(estimates, ddof=1)
    assert pooled.estimate == pytest.approx(np.mean(estimates), rel=1e-12, abs=1e-9)
    assert pooled.sampling_variance == pytest.approx(np.mean(variances), rel=1e-12)
    assert pooled.imputation_variance == pytest.approx(between, rel=1e-9, abs=1e-9)
    assert pooled.variance == pytest.approx(
        np.mean(variances) + (1 + 1 / m) * between, rel=1e-9, abs=1e-9
    )
    assert pooled.variance >= pooled.sampling_variance >= 0
    assert pooled.variance >= pooled.imputation_variance >= 0
    assert min(estimates) - 1e-6 <= pooled.estimate <= max(estimates) + 1e-6
    assert pooled.standard_error == math.sqrt(pooled.variance)
    assert pooled.n_implicates == m
    assert pooled.implicate_estimates == tuple(estimates)


@settings(max_examples=200)
@given(st.lists(st.tuples(finite, variance), min_size=2, max_size=8), st.randoms())
def test_pool_does_not_depend_on_implicate_order(pairs, random):
    shuffled = list(pairs)
    random.shuffle(shuffled)
    a = fedscf.pool(*zip(*pairs, strict=True))
    b = fedscf.pool(*zip(*shuffled, strict=True))
    assert a.estimate == pytest.approx(b.estimate, rel=1e-9, abs=1e-6)
    assert a.variance == pytest.approx(b.variance, rel=1e-9, abs=1e-6)


def misecomp(
    estimates: list[float], standard_errors: list[float]
) -> tuple[float, float]:
    """MACRO MISECOMP from the SCF codebook, transcribed for five implicates."""
    nimp = 5
    beta = 0.0
    sigma = 0.0
    st_ = 0.0
    for j in range(nimp):
        beta = estimates[j] + beta
        sigma = standard_errors[j] ** 2 + sigma
    beta = beta / nimp
    sigma = sigma / nimp
    for i in range(nimp):
        st_ = st_ + (beta - estimates[i]) ** 2
    sigma = math.sqrt(sigma + (1 + 1 / 5) * st_ / (5 - 1))
    return beta, sigma


@settings(max_examples=300)
@given(st.lists(st.tuples(finite, st.floats(0, 1e6)), min_size=5, max_size=5))
def test_pool_agrees_with_misecomp(pairs):
    estimates = [pair[0] for pair in pairs]
    errors = [pair[1] for pair in pairs]
    beta, sigma = misecomp(estimates, errors)
    pooled = fedscf.pool(estimates, [error**2 for error in errors])
    assert pooled.estimate == pytest.approx(beta, rel=1e-12, abs=1e-9)
    assert pooled.standard_error == pytest.approx(sigma, rel=1e-9, abs=1e-6)


def test_pool_refuses_malformed_input():
    with pytest.raises(ValueError, match="two or more"):
        fedscf.pool([1.0], [1.0])
    with pytest.raises(ValueError, match="one variance per estimate"):
        fedscf.pool([1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="non-negative"):
        fedscf.pool([1.0, 2.0], [1.0, -1.0])
    with pytest.raises(ValueError, match="non-negative"):
        fedscf.pool([1.0, 2.0], math.nan)
    with pytest.raises(ValueError, match="finite"):
        fedscf.pool([1.0, math.inf], 1.0)


def test_replicate_variance_uses_the_meanit_divisor():
    values = [1.0, 2.0, 4.0, 9.0]
    mean = sum(values) / 4
    assert fedscf.replicate_variance(values) == pytest.approx(
        sum((v - mean) ** 2 for v in values) / 3
    )
    assert fedscf.replicate_variance([3.3] * 7) == 0.0
    with pytest.raises(ValueError):
        fedscf.replicate_variance([1.0])
    with pytest.raises(ValueError):
        fedscf.replicate_variance([1.0, math.nan])


# --- weighted statistics -----------------------------------------------------

values_and_weights = st.integers(1, 12).flatmap(
    lambda n: st.tuples(
        st.lists(finite, min_size=n, max_size=n),
        st.lists(st.integers(1, 20), min_size=n, max_size=n),
    )
)
quantiles = st.sampled_from([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])


@settings(max_examples=500)
@given(values_and_weights, quantiles)
def test_weighted_quantile_equals_the_quantile_of_the_expanded_sample(drawn, q):
    """A row of integer weight k counts as k rows."""
    values, weights = drawn
    expanded = np.repeat(values, weights)
    expected = np.quantile(expanded, q, method="inverted_cdf")
    assert fedscf.weighted_quantile(values, weights, q) == expected


@pytest.mark.skipif(
    np.lib.NumpyVersion(np.__version__) < "2.0.0",
    reason="np.quantile takes weights from NumPy 2.0",
)
@settings(max_examples=500)
@given(values_and_weights, quantiles)
def test_weighted_quantile_agrees_with_numpy(drawn, q):
    values, weights = drawn
    expected = np.quantile(values, q, weights=weights, method="inverted_cdf")
    assert fedscf.weighted_quantile(values, weights, q) == expected


@settings(max_examples=300)
@given(values_and_weights, st.randoms())
def test_weighted_quantile_properties(drawn, random):
    values, weights = drawn
    results = [fedscf.weighted_quantile(values, weights, q) for q in (0.1, 0.5, 0.9, 1)]
    assert all(result in values for result in results)
    assert results == sorted(results)
    assert results[-1] == max(values)
    # Weights matter only in proportion (powers of two rescale exactly).
    scaled = [weight * 8 for weight in weights]
    assert fedscf.weighted_quantile(values, scaled, 0.5) == results[1]
    order = list(range(len(values)))
    random.shuffle(order)
    shuffled = fedscf.weighted_quantile(
        [values[i] for i in order], [weights[i] for i in order], 0.5
    )
    assert shuffled == results[1]
    # A row with no weight changes nothing.
    padded = fedscf.weighted_quantile([*values, -1e12], [*weights, 0], 0.5)
    assert padded == results[1]


def test_weighted_quantile_edge_cases():
    assert fedscf.weighted_quantile([1.0, 2.0, 3.0], [1, 1, 1], 0.5) == 2.0
    # The cumulative share reaches one half exactly at the second value.
    assert fedscf.weighted_quantile([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1], 0.5) == 2.0
    assert math.isnan(fedscf.weighted_quantile([1.0, 2.0], [0, 0], 0.5))
    assert math.isnan(fedscf.weighted_quantile([1.0, math.nan], [1, 1], 0.5))
    assert fedscf.weighted_quantile([1.0, math.nan], [1, 0], 0.5) == 1.0
    for q in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            fedscf.weighted_quantile([1.0], [1.0], q)
    with pytest.raises(ValueError, match="non-negative"):
        fedscf.weighted_quantile([1.0, 2.0], [1.0, -1.0], 0.5)
    with pytest.raises(ValueError, match="aligned"):
        fedscf.weighted_quantile([1.0, 2.0], [1.0], 0.5)


@settings(max_examples=300)
@given(values_and_weights)
def test_weighted_mean_and_total(drawn):
    values, weights = drawn
    mean = fedscf.weighted_mean(values, weights)
    assert mean == pytest.approx(
        np.average(values, weights=weights), rel=1e-9, abs=1e-6
    )
    assert min(values) - 1e-6 <= mean <= max(values) + 1e-6
    total = fedscf.weighted_total(values, weights)
    assert total == pytest.approx(mean * sum(weights), rel=1e-9, abs=1e-3)
    assert fedscf.weighted_total([1.0] * len(weights), weights) == sum(weights)


def test_weighted_mean_of_no_weight_is_nan():
    assert math.isnan(fedscf.weighted_mean([1.0, 2.0], [0.0, 0.0]))


# --- estimate ----------------------------------------------------------------


@pytest.fixture(scope="module")
def scf(tmp_path_factory):
    """A synthetic wave read through the real readers."""
    directory = tmp_path_factory.mktemp("wave")
    families = synthetic_families(n_families=60, seed=7)
    (directory / "summary.dta").write_bytes(to_dta_bytes(summary_table(families)))
    (directory / "rw.dta").write_bytes(to_dta_bytes(replicate_table(families, seed=8)))
    data = fedscf.read_summary(directory / "summary.dta")
    data["owner"] = (data["networth"] > 100_000).astype(float)
    return data, fedscf.read_replicate_weights(directory / "rw.dta")


def meanit(data: pd.DataFrame, replicates: pd.DataFrame, column: str):
    """MACRO MEANIT from the SCF codebook, transcribed.

    This is the branch the macro takes with a WHERE clause, in which each
    replicate's mean and median are normalised by that replicate's own weight
    total (``POP=MDATA[+,2:&LEN]``). Returns the mean and median over
    implicates and their combined standard deviations (``TVM``, ``TVD``).
    """
    wgt0 = data["wgt"].to_numpy() * 5  # WGT0 = X42001
    values = data[column].to_numpy()
    implic = data["implicate"].to_numpy()
    im = np.zeros(5)
    id_ = np.zeros(5)
    for i in range(1, 6):
        imp_values = values[implic == i]
        imp_weights = wgt0[implic == i]
        order = np.argsort(imp_values, kind="stable")  # PROC SORT; BY &VAR
        imp_values, imp_weights = imp_values[order], imp_weights[order]
        pop = imp_weights.sum()
        im[i - 1] = (imp_values * imp_weights).sum() / pop
        share = np.cumsum(imp_weights) / pop
        id_[i - 1] = imp_values[np.min(np.nonzero(share >= 0.5))]
    first = data.loc[implic == 1].sort_values(column, kind="stable")
    mdata = replicates.loc[first["yy1"]].to_numpy()
    x = first[column].to_numpy()
    rm = np.zeros(mdata.shape[1])
    rd = np.zeros(mdata.shape[1])
    for r in range(mdata.shape[1]):
        pop = mdata[:, r].sum()
        rm[r] = (x * mdata[:, r]).sum() / pop
        share = np.cumsum(mdata[:, r]) / pop
        rd[r] = x[np.min(np.nonzero(share >= 0.5))]
    ivm = ((im - im.sum() / 5) ** 2).sum() / 4
    ivd = ((id_ - id_.sum() / 5) ** 2).sum() / 4
    n = rm.size
    rvm = ((rm - rm.sum() / n) ** 2).sum() / (n - 1)
    rvd = ((rd - rd.sum() / n) ** 2).sum() / (n - 1)
    tvm = math.sqrt((6 / 5) * ivm + rvm)
    tvd = math.sqrt((6 / 5) * ivd + rvd)
    return im.sum() / 5, id_.sum() / 5, tvm, tvd


@pytest.mark.parametrize("column", ["networth", "income"])
def test_estimate_agrees_with_meanit(scf, column):
    data, replicates = scf
    imean, imedian, tvm, tvd = meanit(data, replicates, column)
    mean = fedscf.estimate(data, fedscf.mean(column), replicates)
    median = fedscf.estimate(data, fedscf.median(column), replicates)
    assert mean.estimate == pytest.approx(imean, rel=1e-12)
    assert mean.standard_error == pytest.approx(tvm, rel=1e-10)
    assert median.estimate == pytest.approx(imedian, rel=1e-12)
    assert median.standard_error == pytest.approx(tvd, rel=1e-10)
    assert mean.standard_error > 0 and median.standard_error > 0


def test_estimate_variance_parts(scf):
    data, replicates = scf
    result = fedscf.estimate(data, fedscf.mean("networth"), replicates)
    assert result.n_implicates == 5 and len(result.implicate_estimates) == 5
    assert result.estimate == pytest.approx(np.mean(result.implicate_estimates))
    assert result.imputation_variance == pytest.approx(
        np.var(result.implicate_estimates, ddof=1)
    )
    assert result.variance == pytest.approx(
        result.sampling_variance + 1.2 * result.imputation_variance
    )
    assert result.variance >= result.sampling_variance > 0
    assert result.standard_error == math.sqrt(result.variance)


def test_estimate_is_deterministic_and_ignores_row_order(scf):
    data, replicates = scf
    for statistic in (
        fedscf.mean("networth"),
        fedscf.median("networth"),
        fedscf.total("income"),
        fedscf.ratio("income", "owner"),
    ):
        first = fedscf.estimate(data, statistic, replicates)
        assert fedscf.estimate(data, statistic, replicates) == first
        shuffled = data.sample(frac=1, random_state=0)
        reordered = fedscf.estimate(
            shuffled, statistic, replicates.sample(frac=1, random_state=1)
        )
        assert reordered.estimate == pytest.approx(first.estimate, rel=1e-12)
        assert reordered.variance == pytest.approx(first.variance, rel=1e-9)


def test_identical_implicates_have_no_imputation_variance(scf):
    data, replicates = scf
    first = data.loc[data["implicate"] == 1]
    copies = []
    for implicate in range(1, 6):
        copy = first.copy()
        copy["implicate"] = implicate
        copy["y1"] = copy["yy1"] * 10 + implicate
        copies.append(copy)
    same = pd.concat(copies, ignore_index=True)
    for statistic in (fedscf.mean("networth"), fedscf.median("networth")):
        result = fedscf.estimate(same, statistic, replicates)
        assert result.imputation_variance == 0.0
        assert result.variance == result.sampling_variance


@settings(max_examples=50, deadline=None)
@given(factor=st.floats(-1000, 1000).filter(lambda value: abs(value) > 1e-3))
def test_rescaling_a_variable_rescales_estimate_and_error(scf, factor):
    data, replicates = scf
    scaled = data.assign(networth=data["networth"] * factor)
    for make in (fedscf.mean, fedscf.total):
        base = fedscf.estimate(data, make("networth"), replicates)
        other = fedscf.estimate(scaled, make("networth"), replicates)
        assert other.estimate == pytest.approx(base.estimate * factor, rel=1e-9)
        assert other.standard_error == pytest.approx(
            base.standard_error * abs(factor), rel=1e-9
        )


def test_total_of_one_is_the_population_in_every_implicate(scf):
    data, replicates = scf
    ones = data.assign(one=1.0)
    result = fedscf.estimate(ones, fedscf.total("one"), replicates)
    # The fixture gives each implicate a weight total of one million.
    assert result.implicate_estimates == pytest.approx([1_000_000.0] * 5, rel=1e-9)
    assert result.estimate == pytest.approx(ones["wgt"].sum(), rel=1e-9)
    assert result.imputation_variance == pytest.approx(0.0, abs=1e-6)


def test_ratio_is_a_subgroup_mean_and_a_share(scf):
    data, replicates = scf
    owners = data.loc[data["owner"] == 1]
    data = data.assign(owner_income=data["income"] * data["owner"], one=1.0)
    by_ratio = fedscf.estimate(data, fedscf.ratio("owner_income", "owner"), replicates)
    by_filter = fedscf.estimate(owners, fedscf.mean("income"), replicates)
    assert by_ratio.estimate == pytest.approx(by_filter.estimate, rel=1e-12)
    assert by_ratio.standard_error == pytest.approx(by_filter.standard_error, rel=1e-9)
    share = fedscf.estimate(data, fedscf.ratio("owner", "one"), replicates)
    assert 0 < share.estimate < 1
    assert share.estimate == pytest.approx(
        fedscf.estimate(data, fedscf.mean("owner"), replicates).estimate, rel=1e-12
    )


def test_estimate_on_a_subset_of_implicates(scf):
    data, replicates = scf
    three = fedscf.select_implicates(data, [1, 2, 3])
    result = fedscf.estimate(three, fedscf.total("income"), replicates)
    full = fedscf.estimate(data, fedscf.total("income"), replicates)
    assert result.n_implicates == 3
    assert result.implicate_estimates == pytest.approx(
        full.implicate_estimates[:3], rel=1e-12
    )
    assert result.sampling_variance == pytest.approx(full.sampling_variance, rel=1e-12)
    assert result.variance == pytest.approx(
        result.sampling_variance + (1 + 1 / 3) * result.imputation_variance
    )


def test_without_replicate_weights_the_standard_error_is_unknown(scf):
    data, _ = scf
    result = fedscf.estimate(data, fedscf.mean("networth"))
    assert math.isfinite(result.estimate) and result.imputation_variance > 0
    assert math.isnan(result.sampling_variance) and math.isnan(result.standard_error)


def test_estimate_refuses_what_it_cannot_do(scf):
    data, replicates = scf
    mean = fedscf.mean("networth")
    with pytest.raises(ValueError, match="two or more implicates"):
        fedscf.estimate(data.loc[data["implicate"] == 1], mean, replicates)
    with pytest.raises(ValueError, match="first implicate"):
        fedscf.estimate(data.loc[data["implicate"] > 1], mean, replicates)
    with pytest.raises(KeyError, match="no replicate weights"):
        fedscf.estimate(data, mean, replicates.iloc[1:])
    with pytest.raises(ValueError, match="lacks column"):
        fedscf.estimate(data.drop(columns="implicate"), mean, replicates)
    with pytest.raises(ValueError, match="not finite for implicate"):
        fedscf.estimate(data.assign(networth=math.nan), mean, replicates)
    # A subpopulation no replicate is sure to draw: one family.
    one_family = data.loc[data["yy1"] == data["yy1"].iloc[0]]
    with pytest.raises(ValueError, match="undefined in"):
        fedscf.estimate(one_family, mean, replicates)


def test_custom_statistics_are_plain_functions(scf):
    data, replicates = scf

    def gap(frame: pd.DataFrame, weights: np.ndarray) -> float:
        high = fedscf.weighted_quantile(frame["networth"], weights, 0.9)
        return high - fedscf.weighted_quantile(frame["networth"], weights, 0.5)

    result = fedscf.estimate(data, gap, replicates)
    assert result.estimate > 0 and result.standard_error > 0


def test_sampling_variance_agrees_with_microdf(scf):
    """A second implementation of replicate-weight variance."""
    mdf = pytest.importorskip("microdf")
    data, replicates = scf
    first = data.loc[data["implicate"] == 1]
    matrix = replicates.loc[first["yy1"]].to_numpy()
    n = matrix.shape[1]
    for name, statistic, ours in (
        ("mean", lambda s: s.mean(), fedscf.mean("networth")),
        ("total", lambda s: s.sum(), fedscf.total("networth")),
    ):
        series = mdf.MicroSeries(first["networth"].to_numpy(), weights=first["wgt"])
        # microdf's bootstrap divides by R; MEANIT divides by R - 1.
        theirs = mdf.replicate_variance(
            series, statistic, matrix, method="bootstrap", centering="replicate-mean"
        ) * (n / (n - 1))
        result = fedscf.estimate(data, ours, replicates)
        assert result.sampling_variance == pytest.approx(theirs, rel=1e-9), name


small_waves = st.integers(2, 5).flatmap(
    lambda n: st.tuples(
        st.just(n),
        st.lists(finite, min_size=5 * n, max_size=5 * n),
        st.lists(st.floats(0.5, 100), min_size=5 * n, max_size=5 * n),
        st.lists(st.floats(0.5, 100), min_size=4 * n, max_size=4 * n),
    )
)


@settings(max_examples=200, deadline=None)
@given(small_waves)
def test_estimate_invariants_on_arbitrary_small_waves(drawn):
    n, values, weights, replicate_values = drawn
    family = np.repeat(np.arange(1, n + 1), 5)
    implicate = np.tile(np.arange(1, 6), n)
    data = pd.DataFrame(
        {
            "yy1": family,
            "y1": family * 10 + implicate,
            "implicate": implicate,
            "wgt": weights,
            "x": values,
        }
    )
    replicates = pd.DataFrame(
        np.array(replicate_values).reshape(n, 4),
        index=pd.Index(np.arange(1, n + 1), name="yy1"),
    )
    for statistic in (fedscf.mean("x"), fedscf.median("x"), fedscf.total("x")):
        result = fedscf.estimate(data, statistic, replicates)
        assert result.sampling_variance >= 0 and result.imputation_variance >= 0
        assert result.variance >= result.sampling_variance
        assert math.isfinite(result.standard_error)
        assert fedscf.estimate(data, statistic, replicates) == result
    mean = fedscf.estimate(data, fedscf.mean("x"), replicates)
    assume(max(values) - min(values) < 1e9)
    assert min(values) - 1e-3 <= mean.estimate <= max(values) + 1e-3
    median = fedscf.estimate(data, fedscf.median("x"), replicates)
    assert min(values) <= median.estimate <= max(values)
