"""Point estimates and standard errors for the SCF, by the Fed's method.

The method is the one in the SCF codebook (``codebk2025.txt``, section
"SAMPLING ERROR" and its ``MEANIT`` macro, and ``MACRO MISECOMP``) and in the
Fed's `standard error note
<https://www.federalreserve.gov/econres/files/Standard_Error_Documentation.pdf>`_:

* point estimate: the mean of the per-implicate estimates;
* imputation variance: the variance of those estimates, with divisor
  ``m - 1``;
* sampling variance: the variance of the statistic across the 999 replicate
  weights, computed on the first implicate, with divisor ``R - 1`` around the
  mean of the replicate estimates;
* standard error: ``sqrt(sampling variance + (1 + 1/m) * imputation variance)``,
  which is the codebook's ``SQRT((6/5)*imputation variance + sampling
  variance)`` for the SCF's five implicates.

:func:`pool` is the combining step on its own, for estimates and variances
computed any other way (a regression run once per implicate, say).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fedscf.read import FAMILY_ID

__all__ = [
    "Estimate",
    "Statistic",
    "estimate",
    "mean",
    "median",
    "pool",
    "quantile",
    "ratio",
    "replicate_variance",
    "total",
    "weighted_mean",
    "weighted_quantile",
    "weighted_total",
]

#: A statistic maps a frame's rows and one weight per row to a number.
Statistic = Callable[[pd.DataFrame, np.ndarray], float]

ArrayLike = Sequence[float] | np.ndarray | pd.Series


def _arrays(values: ArrayLike, weights: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype="float64")
    w = np.asarray(weights, dtype="float64")
    if x.ndim != 1 or x.shape != w.shape:
        raise ValueError("values and weights must be one-dimensional and aligned")
    if np.isnan(w).any() or (w < 0).any():
        raise ValueError("weights must be non-negative numbers")
    return x, w


def weighted_total(values: ArrayLike, weights: ArrayLike) -> float:
    """``sum(values * weights)``."""
    x, w = _arrays(values, weights)
    return float(np.dot(x, w))


def weighted_mean(values: ArrayLike, weights: ArrayLike) -> float:
    """``sum(values * weights) / sum(weights)``; NaN when the weights sum to zero."""
    x, w = _arrays(values, weights)
    denominator = w.sum()
    return float(np.dot(x, w) / denominator) if denominator > 0 else math.nan


def weighted_quantile(values: ArrayLike, weights: ArrayLike, q: float) -> float:
    """The smallest value whose cumulative weight share reaches ``q``.

    This is the median rule in the codebook's ``MEANIT`` macro
    (``IMP[MIN(LOC(IMP[,2]>=.5)),1]``), extended to any ``q`` in ``(0, 1]``.
    The result is always one of ``values``. Rows with zero weight are ignored;
    the result is NaN when no row has positive weight.
    """
    if not 0 < q <= 1:
        raise ValueError("q must be in (0, 1]")
    x, w = _arrays(values, weights)
    positive = w > 0
    x, w = x[positive], w[positive]
    if x.size == 0:
        return math.nan
    if np.isnan(x).any():
        return math.nan
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(w[order])
    share = cumulative / cumulative[-1]
    return float(x[order][int(np.searchsorted(share, q, side="left"))])


def total(column: str) -> Statistic:
    """Statistic: the weighted total of ``column``."""
    return lambda frame, weights: weighted_total(frame[column], weights)


def mean(column: str) -> Statistic:
    """Statistic: the weighted mean of ``column``."""
    return lambda frame, weights: weighted_mean(frame[column], weights)


def quantile(column: str, q: float) -> Statistic:
    """Statistic: the weighted ``q`` quantile of ``column``."""
    return lambda frame, weights: weighted_quantile(frame[column], weights, q)


def median(column: str) -> Statistic:
    """Statistic: the weighted median of ``column``."""
    return quantile(column, 0.5)


def ratio(numerator: str, denominator: str) -> Statistic:
    """Statistic: one column's weighted total over another's.

    With a 0/1 ``denominator`` this is the mean of ``numerator`` among the
    flagged rows; with a 0/1 ``numerator`` as well it is a share.
    """

    def statistic(frame: pd.DataFrame, weights: np.ndarray) -> float:
        bottom = weighted_total(frame[denominator], weights)
        if bottom == 0:
            return math.nan
        return weighted_total(frame[numerator], weights) / bottom

    return statistic


@dataclass(frozen=True)
class Estimate:
    """A multiply-imputed estimate and the two parts of its variance."""

    #: Mean of the per-implicate estimates.
    estimate: float
    #: Variance from sampling: the replicate-weight variance, or for
    #: :func:`pool` the mean of the per-implicate variances.
    sampling_variance: float
    #: Variance across implicates, with divisor ``m - 1``.
    imputation_variance: float
    #: The per-implicate estimates.
    implicate_estimates: tuple[float, ...]

    @property
    def n_implicates(self) -> int:
        return len(self.implicate_estimates)

    @property
    def variance(self) -> float:
        """``sampling_variance + (1 + 1/m) * imputation_variance``."""
        inflation = 1.0 + 1.0 / self.n_implicates
        return self.sampling_variance + inflation * self.imputation_variance

    @property
    def standard_error(self) -> float:
        return math.sqrt(self.variance)


def _between(estimates: np.ndarray) -> float:
    # Identical estimates carry no imputation variance. Returning exactly zero,
    # not the rounding error of a mean, makes the pooled variance equal the
    # sampling variance bit for bit.
    if (estimates == estimates[0]).all():
        return 0.0
    return float(np.var(estimates, ddof=1))


def pool(
    estimates: Sequence[float] | np.ndarray,
    variances: Sequence[float] | np.ndarray | float,
) -> Estimate:
    """Combine per-implicate estimates and variances (Rubin's rules).

    Args:
        estimates: One estimate per implicate; at least two.
        variances: The sampling variance of each estimate, or a single number
            that applies to all of them. The codebook's ``MISECOMP`` passes one
            per implicate (squared standard errors); ``MEANIT`` passes the one
            replicate-weight variance computed on the first implicate.

    Returns:
        The pooled estimate, with ``variance = mean(variances) +
        (1 + 1/m) * var(estimates, ddof=1)``.
    """
    values = np.asarray(estimates, dtype="float64")
    if values.ndim != 1 or values.size < 2:
        raise ValueError(
            "pooling needs one estimate from each of two or more implicates"
        )
    within = np.asarray(variances, dtype="float64")
    if within.ndim == 0:
        within = np.full(values.shape, float(within))
    if within.shape != values.shape:
        raise ValueError("give one variance per estimate, or a single variance")
    if not np.isfinite(values).all():
        raise ValueError("estimates must be finite")
    if np.isnan(within).any() or (within < 0).any():
        raise ValueError("variances must be non-negative numbers")
    # A single shared variance is its own mean; do not round-trip it.
    shared = (within == within[0]).all()
    return _combine(values, float(within[0] if shared else within.mean()))


def _combine(values: np.ndarray, sampling_variance: float) -> Estimate:
    return Estimate(
        estimate=float(values.mean()),
        sampling_variance=sampling_variance,
        imputation_variance=_between(values),
        implicate_estimates=tuple(float(value) for value in values),
    )


def replicate_variance(replicate_estimates: Sequence[float] | np.ndarray) -> float:
    """Variance across replicate estimates, as in ``MEANIT``.

    ``sum((r - mean(r)) ** 2) / (R - 1)``.
    """
    values = np.asarray(replicate_estimates, dtype="float64")
    if values.ndim != 1 or values.size < 2:
        raise ValueError("need two or more replicate estimates")
    if not np.isfinite(values).all():
        raise ValueError("replicate estimates must be finite")
    if (values == values[0]).all():
        return 0.0
    return float(np.var(values, ddof=1))


def estimate(
    data: pd.DataFrame,
    statistic: Statistic,
    replicate_weights: pd.DataFrame | None = None,
    *,
    weight: str = "wgt",
) -> Estimate:
    """Estimate a statistic from SCF data with its standard error.

    Args:
        data: Rows from :func:`fedscf.load_summary`, :func:`fedscf.load_full`
            or :func:`fedscf.join`, holding two or more implicates. Filter the
            rows first to estimate for a subpopulation, but drop implicates
            only with :func:`fedscf.select_implicates`, which rescales ``wgt``:
            each implicate's weights are taken as ``wgt`` times the number of
            implicates present.
        statistic: What to estimate, for example ``fedscf.mean("networth")``,
            or any function of ``(frame, weights)``.
        replicate_weights: The wave's :func:`fedscf.load_replicate_weights`.
            Without it the sampling variance, and so the standard error, is
            NaN: an imputation-only standard error would understate the
            uncertainty.
        weight: The weight column.

    The statistic is evaluated once per implicate with that implicate's
    weights scaled to the population, and once per replicate on the first
    implicate's rows. A statistic that is undefined in some replicate (a
    subpopulation with no drawn family) raises, as ``MEANIT`` does.
    """
    for column in (FAMILY_ID, "implicate", weight):
        if column not in data.columns:
            raise ValueError(f"data lacks column {column!r}")
    implicates = sorted(int(i) for i in data["implicate"].unique())
    if len(implicates) < 2:
        raise ValueError("estimate needs two or more implicates to pool")
    membership = data["implicate"].to_numpy()
    weights = data[weight].to_numpy(dtype="float64") * len(implicates)
    per_implicate = []
    for implicate in implicates:
        rows = membership == implicate
        value = float(statistic(data.loc[rows], weights[rows]))
        if not math.isfinite(value):
            raise ValueError(f"statistic is not finite for implicate {implicate}")
        per_implicate.append(value)
    if replicate_weights is None:
        return _combine(np.array(per_implicate), math.nan)
    if 1 not in implicates:
        raise ValueError(
            "replicate weights belong to the first implicate, which data lacks"
        )
    first = data.loc[membership == 1]
    families = first[FAMILY_ID].to_numpy()
    positions = replicate_weights.index.get_indexer(families)
    if (positions < 0).any():
        raise KeyError(
            f"{int((positions < 0).sum())} families have no replicate weights"
        )
    matrix = replicate_weights.to_numpy(dtype="float64")[positions]
    replicates = np.array(
        [float(statistic(first, matrix[:, i])) for i in range(matrix.shape[1])]
    )
    undefined = int((~np.isfinite(replicates)).sum())
    if undefined:
        raise ValueError(
            f"statistic is undefined in {undefined} of {replicates.size} replicates"
        )
    return pool(per_implicate, replicate_variance(replicates))
