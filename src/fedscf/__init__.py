"""fedscf: the Federal Reserve's Survey of Consumer Finances public microdata.

A registry of the Fed's Stata files with pinned hashes, a verified fetch and
cache, readers that give every wave from 1989 to 2025 the same layout, and the
Fed's method for standard errors.

>>> import fedscf
>>> scf = fedscf.load_summary(2025, ["networth", "income"])  # doctest: +SKIP
>>> weights = fedscf.load_replicate_weights(2025)  # doctest: +SKIP
>>> fedscf.estimate(scf, fedscf.median("networth"), weights)  # doctest: +SKIP
"""

from importlib import metadata

from fedscf.fetch import (
    CACHE_ENV_VAR,
    HashMismatchError,
    UnverifiedError,
    cache_dir,
    fetch,
    fetch_artifact,
    resolve,
)
from fedscf.read import (
    N_IMPLICATES,
    N_REPLICATES,
    FedDataWarning,
    join,
    load,
    load_full,
    load_replicate_weights,
    load_summary,
    read_full,
    read_replicate_weights,
    read_summary,
    select_implicates,
)
from fedscf.registry import (
    FILES,
    MANIFEST_FIELDS,
    WAVES,
    Artifact,
    artifact,
    artifacts,
)
from fedscf.stats import (
    Estimate,
    Statistic,
    estimate,
    mean,
    median,
    pool,
    quantile,
    ratio,
    replicate_variance,
    total,
    weighted_mean,
    weighted_quantile,
    weighted_total,
)

try:
    __version__ = metadata.version("fedscf")
except metadata.PackageNotFoundError:  # running from a source tree
    __version__ = "0"

__all__ = [
    "CACHE_ENV_VAR",
    "FILES",
    "MANIFEST_FIELDS",
    "N_IMPLICATES",
    "N_REPLICATES",
    "WAVES",
    "Artifact",
    "Estimate",
    "FedDataWarning",
    "HashMismatchError",
    "Statistic",
    "UnverifiedError",
    "artifact",
    "artifacts",
    "cache_dir",
    "estimate",
    "fetch",
    "fetch_artifact",
    "join",
    "load",
    "load_full",
    "load_replicate_weights",
    "load_summary",
    "mean",
    "median",
    "pool",
    "quantile",
    "ratio",
    "read_full",
    "read_replicate_weights",
    "read_summary",
    "replicate_variance",
    "resolve",
    "select_implicates",
    "total",
    "weighted_mean",
    "weighted_quantile",
    "weighted_total",
]
