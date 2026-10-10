"""Read SCF Stata files into DataFrames with one layout for every wave.

What the Fed's files look like, as read from all 39 current files on
2026-10-09 and from the codebook (``codebk2025.txt``, sections "CASE ID
NUMBERS", "ANALYSIS WEIGHTS" and "SAMPLING ERROR"):

* Each family appears five times, once per *implicate* of the multiple
  imputation. ``y1 = yy1 * 10 + implicate``, where ``yy1`` identifies the
  family. The 1989 files call these ``x1`` and ``xx1``.
* ``yy1`` is stored as a 16-bit integer and exceeds 3,276 in every wave, so
  ``yy1 * 10`` overflows unless it is widened first. The naive product keeps
  only 3,270 of the 4,595 first implicates in 2022.
* Variable names are upper case in the full files through 2010 and in some
  replicate-weight files, and lower case elsewhere.
* The summary extract's ``wgt`` equals the full file's ``x42001 / 5`` exactly.
  ``x42001`` differs across a family's five implicates, while each implicate's
  weights sum to the same number of families (to within one part in a
  million).
* The replicate-weight file has one row per family, with weights
  ``wt1b1``-``wt1b999`` and multiplicities ``mm1``-``mm999`` computed for the
  first implicate. Both are missing where a replicate did not draw the family.

Every reader returns lower-case column names, ``yy1``, ``y1`` and
``implicate`` as 64-bit integers, and a weight ``wgt`` that sums to the number
of families over the rows returned.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.io.stata import StataReader

from fedscf.fetch import fetch_artifact, resolve
from fedscf.registry import WAVES, Artifact

__all__ = [
    "FAMILY_ID",
    "FedDataWarning",
    "N_IMPLICATES",
    "N_REPLICATES",
    "RECORD_ID",
    "join",
    "load",
    "load_full",
    "load_replicate_weights",
    "load_summary",
    "read_full",
    "read_replicate_weights",
    "read_summary",
    "select_implicates",
]

#: Implicates per family.
N_IMPLICATES = 5
#: Bootstrap replicates in a replicate-weight file.
N_REPLICATES = 999
#: Family identifier, constant across a family's implicates.
FAMILY_ID = "yy1"
#: Record identifier: ``yy1 * 10 + implicate``.
RECORD_ID = "y1"

# The 1989 files name the identifiers x1 and xx1.
_ID_ALIASES = {"x1": RECORD_ID, "xx1": FAMILY_ID}
_KEYS = [FAMILY_ID, RECORD_ID]
_STANDARD = [FAMILY_ID, RECORD_ID, "implicate", "wgt"]

PathLike = str | os.PathLike[str]


def _canonical(name: str) -> str:
    lowered = name.lower()
    return _ID_ALIASES.get(lowered, lowered)


def _as_list(columns: str | Iterable[str]) -> list[str]:
    return [columns] if isinstance(columns, str) else list(columns)


def _stored_names(path: PathLike) -> dict[str, str]:
    """Canonical (lower-case, ``y1``/``yy1``) name of each variable -> stored name."""
    with StataReader(path) as reader:
        names = list(reader.variable_labels())
    stored: dict[str, str] = {}
    for name in names:
        canonical = _canonical(name)
        if canonical in stored:
            raise ValueError(
                f"{path}: variables {stored[canonical]!r} and {name!r} collide"
            )
        stored[canonical] = name
    return stored


def _read(
    path: PathLike,
    required: Sequence[str],
    columns: str | Iterable[str] | None,
    on_missing: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Read ``required`` plus the requested columns under canonical names.

    Returns the raw frame and the requested columns, in order and without
    duplicates. Requested columns the file lacks come back as NaN when
    ``on_missing="nan"``.
    """
    if on_missing not in ("raise", "nan"):
        raise ValueError("on_missing must be 'raise' or 'nan'")
    stored = _stored_names(path)
    absent = [name for name in required if name not in stored]
    if absent:
        raise ValueError(f"{path} lacks required variable(s) {absent}")
    if columns is None:
        wanted = list(stored)
    else:
        wanted = list(dict.fromkeys(_canonical(name) for name in _as_list(columns)))
    # "implicate" is derived, not stored; asking for it by name is harmless.
    missing = [name for name in wanted if name not in stored and name != "implicate"]
    if missing and on_missing == "raise":
        raise KeyError(f"{path} has no variable(s) {missing}")
    to_read = list(
        dict.fromkeys([*required, *(name for name in wanted if name in stored)])
    )
    raw = pd.read_stata(
        path,
        columns=[stored[name] for name in to_read],
        convert_categoricals=False,
    )
    raw.columns = to_read
    for name in missing:
        raw[name] = np.nan
    return raw, wanted


def _with_implicate(
    raw: pd.DataFrame, weight: pd.Series, wanted: list[str]
) -> pd.DataFrame:
    """Standard columns first, then the requested ones."""
    family = raw[FAMILY_ID].astype("int64")
    record = raw[RECORD_ID].astype("int64")
    # Widen before multiplying: yy1 is int16 on disk and yy1 * 10 overflows.
    implicate = record - family * 10
    if not implicate.between(1, N_IMPLICATES).all():
        raise ValueError(
            "y1 - yy1 * 10 is not an implicate number from 1 to "
            f"{N_IMPLICATES} in every row"
        )
    if not record.is_unique:
        raise ValueError("record identifier y1 repeats")
    if len(raw) != N_IMPLICATES * family.nunique():
        raise ValueError(f"not every family has {N_IMPLICATES} implicates")
    frame = pd.DataFrame(
        {
            FAMILY_ID: family,
            RECORD_ID: record,
            "implicate": implicate,
            "wgt": weight.astype("float64"),
        }
    )
    extra = [name for name in wanted if name not in _STANDARD]
    return pd.concat([frame, raw[extra]], axis=1)


def select_implicates(
    frame: pd.DataFrame, implicates: int | Iterable[int] | None
) -> pd.DataFrame:
    """Keep some implicates and rescale ``wgt`` to the same population.

    Each implicate is a complete data set whose ``x42001`` sums to the number
    of families, and ``wgt`` divides that by the number of implicates present.
    Keeping ``k`` of ``n`` implicates multiplies ``wgt`` by ``n / k``, so the
    weights still sum to the population (exactly so when the implicates carry
    equal weight totals, which the Fed's files do to one part in a million).
    With one implicate kept, ``wgt`` is that implicate's ``x42001``, up to
    floating-point rounding.

    ``None`` keeps every implicate and returns the frame unchanged.
    """
    if implicates is None:
        return frame
    keep = sorted(
        {int(i) for i in ([implicates] if np.isscalar(implicates) else implicates)}
    )
    present = sorted(int(i) for i in frame["implicate"].unique())
    if not keep:
        raise ValueError("implicates is empty")
    unknown = [i for i in keep if i not in present]
    if unknown:
        raise ValueError(f"implicate(s) {unknown} not in the data; have {present}")
    kept = frame.loc[frame["implicate"].isin(keep)].reset_index(drop=True)
    kept["wgt"] = kept["wgt"] * (len(present) / len(keep))
    kept.attrs = dict(frame.attrs)
    return kept


def read_summary(
    path: PathLike,
    columns: str | Iterable[str] | None = None,
    *,
    implicates: int | Iterable[int] | None = None,
    on_missing: str = "raise",
) -> pd.DataFrame:
    """Read a Summary Extract Public Data file (``rscfpYYYY.dta``).

    Args:
        path: The Stata file.
        columns: Variables to read, in any case; all of them by default.
        implicates: Implicates to keep; see :func:`select_implicates`.
        on_missing: ``"raise"`` (default) or ``"nan"`` for variables the file
            does not have. Variable lists change from wave to wave.

    Returns:
        ``yy1``, ``y1``, ``implicate``, ``wgt``, then the requested variables.
    """
    raw, wanted = _read(path, [*_KEYS, "wgt"], columns, on_missing)
    frame = _with_implicate(raw, raw["wgt"], wanted)
    return select_implicates(frame, implicates)


def read_full(
    path: PathLike,
    columns: str | Iterable[str] | None = None,
    *,
    implicates: int | Iterable[int] | None = None,
    on_missing: str = "raise",
) -> pd.DataFrame:
    """Read a full public data set (``pYYiN.dta``).

    The files hold about 5,000 variables, so pass ``columns``. ``wgt`` is
    ``x42001 / 5``, the summary extract's weight. Arguments and return are as
    for :func:`read_summary`.
    """
    raw, wanted = _read(path, [*_KEYS, "x42001"], columns, on_missing)
    weight = raw["x42001"].astype("float64") / N_IMPLICATES
    frame = _with_implicate(raw, weight, wanted)
    return select_implicates(frame, implicates)


def read_replicate_weights(path: PathLike, *, raw: bool = False) -> pd.DataFrame:
    """Read a replicate-weight file into one row per family.

    Returns a frame indexed by ``yy1`` with columns ``rw1``-``rw999``, where
    ``rw_i = max(0, wt1b_i) * max(0, mm_i)`` and a missing value counts as
    zero. This is the ``WGT1-WGT999`` the codebook's ``MEANIT`` macro builds: a
    family's weight in replicate ``i`` times the number of times the
    replicate drew it. Each column sums to about the number of families.

    With ``raw=True`` the file's own ``wt1b*`` and ``mm*`` columns are
    returned instead, missing values included.
    """
    stored = _stored_names(path)
    absent = [name for name in _KEYS if name not in stored]
    if absent:
        raise ValueError(f"{path} lacks required variable(s) {absent}")
    table = pd.read_stata(path, convert_categoricals=False)
    table.columns = [_canonical(name) for name in table.columns]
    family = table[FAMILY_ID].astype("int64")
    record = table[RECORD_ID].astype("int64")
    if not (record == family * 10 + 1).all():
        raise ValueError(f"{path}: rows are not the first implicate of each family")
    if not family.is_unique:
        raise ValueError(f"{path}: a family appears more than once")
    index = pd.Index(family.to_numpy(), name=FAMILY_ID)
    numbers = range(1, N_REPLICATES + 1)
    weight_names = [f"wt1b{i}" for i in numbers]
    count_names = [f"mm{i}" for i in numbers]
    absent = [name for name in (*weight_names, *count_names) if name not in table]
    if absent:
        raise ValueError(f"{path} lacks {len(absent)} replicate column(s)")
    if raw:
        out = table[[*weight_names, *count_names]].astype("float64")
        out.index = index
        return out
    weights = table[weight_names].to_numpy(dtype="float64")
    counts = table[count_names].to_numpy(dtype="float64")
    # MEANIT: WGTS = MAX(0, MULT) * MAX(0, RWGT), where SAS MAX skips missing.
    combined = np.nan_to_num(np.fmax(weights, 0.0)) * np.nan_to_num(
        np.fmax(counts, 0.0)
    )
    return pd.DataFrame(combined, index=index, columns=[f"rw{i}" for i in numbers])


def join(summary: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    """Join summary-extract rows to full-file rows, one to one on ``(yy1, y1)``.

    Every row of each frame must match exactly one row of the other, and the
    two weights must agree. Rows keep the order of ``summary``; the full
    file's variables follow the summary's.
    """
    for name, frame in (("summary", summary), ("full", full)):
        absent = [column for column in _STANDARD if column not in frame.columns]
        if absent:
            raise ValueError(f"{name} frame lacks column(s) {absent}")
        if frame.duplicated(_KEYS).any():
            raise ValueError(f"{name} frame repeats a (yy1, y1) key")
    clash = (set(summary.columns) & set(full.columns)) - set(_STANDARD)
    if clash:
        raise ValueError(f"both frames have column(s) {sorted(clash)}")
    right = full.drop(columns=["implicate"]).rename(columns={"wgt": "_full_wgt"})
    # Keys are unique on both sides, so a left join in which every summary row
    # finds a partner and the row counts agree is a bijection.
    merged = summary.merge(
        right, on=_KEYS, how="left", validate="one_to_one", indicator=True
    )
    matched = int((merged["_merge"] == "both").sum())
    if matched != len(summary) or matched != len(full):
        raise ValueError(
            "(yy1, y1) keys do not match one to one: "
            f"{len(summary) - matched} summary row(s) and "
            f"{len(full) - matched} full-file row(s) have no partner"
        )
    if not np.allclose(merged["wgt"], merged["_full_wgt"], rtol=1e-9, atol=0.0):
        raise ValueError("summary wgt and full-file x42001 / 5 disagree")
    merged = merged.drop(columns=["_merge", "_full_wgt"])
    merged.attrs = {"fedscf": [summary.attrs.get("fedscf"), full.attrs.get("fedscf")]}
    return merged


def _provenance(item: Artifact, verified: bool, frame: pd.DataFrame) -> dict[str, Any]:
    record = item.manifest_entry()
    record["dollar_year"] = item.dollar_year
    record["verified"] = verified
    if "implicate" in frame.columns:
        record["implicates"] = sorted(int(i) for i in frame["implicate"].unique())
    return record


def _check_rows(frame: pd.DataFrame, item: Artifact, rows: int) -> None:
    if item.expected_rows is not None and rows != item.expected_rows:
        raise ValueError(
            f"SCF {item.wave} {item.file} file has {rows} rows; the registry "
            f"expects {item.expected_rows}"
        )


def _load(
    wave: int,
    file: str,
    reader: Any,
    reader_options: dict[str, Any],
    implicates: int | Iterable[int] | None,
    fetch_options: dict[str, Any],
) -> pd.DataFrame:
    selection = {
        name: fetch_options.pop(name, None)
        for name in ("released", "sha256", "member_sha256")
    }
    item = resolve(wave, file, **selection)
    path = fetch_artifact(item, **fetch_options)
    frame = reader(path, **reader_options)
    _check_rows(frame, item, len(frame))
    verified = not fetch_options.get("allow_unverified", False)
    if implicates is not None:
        frame = select_implicates(frame, implicates)
    frame.attrs["fedscf"] = _provenance(item, verified, frame)
    _warn_known_issues(item, frame)
    return frame


class FedDataWarning(UserWarning):
    """A variable the Fed's file is known to get wrong."""


def _warn_known_issues(item: Artifact, frame: pd.DataFrame) -> None:
    # Found 2026-10-09 by checking every pinned summary extract: the Fed's
    # bulletin.macro.txt defines NHNFIN=NFIN-HOUSES, which holds for the newest
    # wave of each release. For every earlier wave the release restates in
    # newer dollars, nhnfin is (nfin - houses) times that wave's CPI factor,
    # so it carries the dollar adjustment twice.
    if (
        item.file == "summary"
        and item.dollar_year is not None
        and item.wave < item.dollar_year
        and "nhnfin" in frame.columns
    ):
        warnings.warn(
            f"nhnfin in the Fed's {item.wave} summary extract released "
            f"{item.released} is converted to {item.dollar_year} dollars twice; "
            "the Fed defines it as nfin - houses, so use that instead.",
            FedDataWarning,
            stacklevel=4,
        )


def load_summary(
    wave: int,
    columns: str | Iterable[str] | None = None,
    *,
    implicates: int | Iterable[int] | None = None,
    on_missing: str = "raise",
    **fetch_options: Any,
) -> pd.DataFrame:
    """Fetch, verify and read a wave's Summary Extract Public Data.

    Dollar variables are in the dollars of ``frame.attrs["fedscf"]["dollar_year"]``.
    The Fed restates every wave's extract in the newest wave's dollars each
    time it publishes a wave, so the same call returns different dollars
    before and after a release unless ``released`` or ``sha256`` pins it.

    ``fetch_options`` are passed to :func:`fedscf.fetch`: ``released``,
    ``sha256``, ``member_sha256``, ``cache_dir``, ``allow_unverified``,
    ``refresh``. Other arguments are as for :func:`read_summary`.
    """
    options = {"columns": columns, "on_missing": on_missing}
    return _load(wave, "summary", read_summary, options, implicates, fetch_options)


def load_full(
    wave: int,
    columns: str | Iterable[str] | None = None,
    *,
    implicates: int | Iterable[int] | None = None,
    on_missing: str = "raise",
    **fetch_options: Any,
) -> pd.DataFrame:
    """Fetch, verify and read a wave's full public data set; see :func:`read_full`."""
    options = {"columns": columns, "on_missing": on_missing}
    return _load(wave, "full", read_full, options, implicates, fetch_options)


def load_replicate_weights(
    wave: int, *, raw: bool = False, **fetch_options: Any
) -> pd.DataFrame:
    """Fetch, verify and read a wave's replicate weights.

    See :func:`read_replicate_weights` for the layout.
    """
    options = {"raw": raw}
    return _load(
        wave, "replicate_weights", read_replicate_weights, options, None, fetch_options
    )


def load(
    years: int | Iterable[int] | None = None,
    columns: str | Iterable[str] | None = None,
    *,
    implicates: int | Iterable[int] | None = None,
    on_missing: str = "raise",
    cache_dir: PathLike | None = None,
) -> pd.DataFrame:
    """Load summary-extract variables for one or more waves, stacked.

    This is ``scf.load`` from this repository's earlier package, rebuilt. It
    returns ``year``, ``yy1``, ``y1``, ``implicate``, ``wgt`` and the requested
    variables, for every wave by default. All waves come from the Fed's
    current release, so their dollar variables share one dollar year.
    """
    if years is None:
        waves = list(WAVES)
    elif isinstance(years, (int, np.integer)):
        waves = [int(years)]
    else:
        waves = [int(year) for year in years]
    frames = []
    for wave in waves:
        frame = load_summary(
            wave,
            columns,
            implicates=implicates,
            on_missing=on_missing,
            cache_dir=cache_dir,
        )
        frame.insert(0, "year", wave)
        frames.append(frame)
    stacked = pd.concat(frames, ignore_index=True)
    stacked.attrs = {"fedscf": [frame.attrs["fedscf"] for frame in frames]}
    return stacked
