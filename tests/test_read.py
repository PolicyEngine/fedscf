"""Readers, implicate handling and the summary/full join, on synthetic files.

Invariants:

* ``implicate == y1 - yy1 * 10`` is in 1..5 for every row, whatever integer
  width the identifiers are stored in;
* selecting ``k`` of ``n`` implicates multiplies ``wgt`` by ``n / k``, which
  keeps the weight total when implicates carry equal totals, and selections
  compose;
* the join is a bijection on ``(yy1, y1)`` or it raises;
* reading the same file twice gives equal frames.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import fedscf
from conftest import (
    full_table,
    replicate_table,
    summary_table,
    synthetic_families,
    to_dta_bytes,
)
from fedscf.read import N_IMPLICATES, N_REPLICATES

STANDARD = ["yy1", "y1", "implicate", "wgt"]


def write(tmp_path, table: pd.DataFrame, name: str = "file.dta"):
    path = tmp_path / name
    path.write_bytes(to_dta_bytes(table))
    return path


def test_summary_has_the_standard_layout(summary_path, families):
    frame = fedscf.read_summary(summary_path)
    assert list(frame.columns) == [*STANDARD, "networth", "income"]
    assert [str(frame[name].dtype) for name in STANDARD] == [
        "int64",
        "int64",
        "int64",
        "float64",
    ]
    assert len(frame) == len(families)
    assert (frame["implicate"] == frame["y1"] - frame["yy1"] * 10).all()
    assert sorted(frame["implicate"].unique()) == [1, 2, 3, 4, 5]
    assert isinstance(frame.index, pd.RangeIndex)
    np.testing.assert_array_equal(frame["networth"], families["networth"])


def test_identifiers_survive_16_bit_storage(summary_path, families):
    """The fixture's yy1 is int16 and exceeds 3,276, as the Fed's does.

    Multiplying it by ten in its own width wraps around, which is the bug the
    readers avoid by widening first.
    """
    assert families["yy1"].dtype == np.int16 and families["yy1"].max() > 3276
    narrow = families["yy1"].to_numpy()
    with np.errstate(over="ignore"):
        naive = families["y1"].to_numpy() - narrow * np.int16(10)
    assert (naive == 1).sum() < len(families) // N_IMPLICATES
    frame = fedscf.read_summary(summary_path)
    assert (frame["implicate"] == 1).sum() == len(families) // N_IMPLICATES


@settings(max_examples=200, deadline=None)
@given(
    family=st.integers(1, 32_767),
    implicate=st.integers(1, 5),
    family_type=st.sampled_from([np.int16, np.int32, np.int64]),
)
def test_implicate_is_recovered_for_any_family_id(family, implicate, family_type):
    raw = pd.DataFrame(
        {
            "yy1": np.array([family] * N_IMPLICATES, dtype=family_type),
            "y1": np.array(
                [family * 10 + i for i in range(1, N_IMPLICATES + 1)], dtype=np.int32
            ),
        }
    )
    from fedscf.read import _with_implicate

    frame = _with_implicate(raw, pd.Series(np.ones(N_IMPLICATES)), [])
    assert frame["implicate"].tolist() == [1, 2, 3, 4, 5]
    assert frame.loc[implicate - 1, "y1"] == family * 10 + implicate


def test_column_subset_and_case(summary_path):
    frame = fedscf.read_summary(summary_path, ["INCOME", "networth", "income"])
    assert list(frame.columns) == [*STANDARD, "income", "networth"]
    single = fedscf.read_summary(summary_path, "networth")
    assert list(single.columns) == [*STANDARD, "networth"]
    # Standard columns are not repeated when asked for by name.
    again = fedscf.read_summary(summary_path, ["wgt", "y1", "implicate", "income"])
    assert list(again.columns) == [*STANDARD, "income"]


def test_missing_columns_raise_or_come_back_empty(summary_path):
    with pytest.raises(KeyError, match="mpsave"):
        fedscf.read_summary(summary_path, ["networth", "mpsave"])
    frame = fedscf.read_summary(summary_path, ["networth", "mpsave"], on_missing="nan")
    assert frame["mpsave"].isna().all() and frame["networth"].notna().all()
    with pytest.raises(ValueError, match="on_missing"):
        fedscf.read_summary(summary_path, on_missing="skip")


def test_1989_identifier_names_are_translated(tmp_path, families):
    table = summary_table(families).rename(columns={"y1": "x1", "yy1": "xx1"})
    frame = fedscf.read_summary(write(tmp_path, table))
    assert list(frame.columns)[:4] == STANDARD
    np.testing.assert_array_equal(frame["yy1"], families["yy1"])
    upper = full_table(families, upper=True).rename(columns={"Y1": "X1", "YY1": "XX1"})
    full = fedscf.read_full(write(tmp_path, upper, "full.dta"), ["x5729"])
    np.testing.assert_array_equal(full["y1"], families["y1"])


def test_full_file_weight_is_x42001_over_five(full_path, families):
    frame = fedscf.read_full(full_path, ["X5729"])
    assert list(frame.columns) == [*STANDARD, "x5729"]
    np.testing.assert_array_equal(frame["wgt"], families["x42001"] / N_IMPLICATES)
    everything = fedscf.read_full(full_path)
    assert list(everything.columns) == [*STANDARD, "x42001", "x5729", "x3014"]


def test_summary_and_full_weights_are_the_same(summary_path, full_path):
    summary = fedscf.read_summary(summary_path, [])
    full = fedscf.read_full(full_path, [])
    np.testing.assert_array_equal(summary["wgt"], full["wgt"])


@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda t: t.assign(y1=t["y1"] + 5), "implicate number"),
        (lambda t: t.iloc[:-1], "implicates"),
        (lambda t: pd.concat([t, t.iloc[[0]]]), "repeats"),
    ],
)
def test_malformed_identifiers_are_refused(tmp_path, families, damage, message):
    table = damage(summary_table(families))
    with pytest.raises(ValueError, match=message):
        fedscf.read_summary(write(tmp_path, table))


def test_files_without_identifiers_or_weight_are_refused(tmp_path, families):
    table = summary_table(families)
    with pytest.raises(ValueError, match="wgt"):
        fedscf.read_summary(write(tmp_path, table.drop(columns="wgt")))
    with pytest.raises(ValueError, match="yy1"):
        fedscf.read_summary(write(tmp_path, table.drop(columns="yy1")))
    with pytest.raises(ValueError, match="x42001"):
        fedscf.read_full(write(tmp_path, table))


def test_names_that_collide_after_case_folding_are_refused(tmp_path, families):
    table = summary_table(families).assign(x1=1)
    with pytest.raises(ValueError, match="collide"):
        fedscf.read_summary(write(tmp_path, table))


def test_reading_is_deterministic(summary_path, full_path, replicate_path):
    for reader, path in (
        (fedscf.read_summary, summary_path),
        (fedscf.read_full, full_path),
        (fedscf.read_replicate_weights, replicate_path),
    ):
        pd.testing.assert_frame_equal(reader(path), reader(path))


# --- implicate selection -----------------------------------------------------


def test_reader_and_select_implicates_agree(summary_path):
    everything = fedscf.read_summary(summary_path)
    for keep in (1, [2], [1, 3], range(1, 6)):
        pd.testing.assert_frame_equal(
            fedscf.read_summary(summary_path, implicates=keep),
            fedscf.select_implicates(everything, keep),
        )


def test_selecting_implicates_keeps_the_population(summary_path):
    everything = fedscf.read_summary(summary_path)
    total = everything["wgt"].sum()
    for keep in ([1], [5], [2, 4], [1, 2, 3], [1, 2, 3, 4, 5]):
        kept = fedscf.read_summary(summary_path, implicates=keep)
        assert sorted(kept["implicate"].unique()) == keep
        assert kept["wgt"].sum() == pytest.approx(total, rel=1e-12)


def test_one_implicate_carries_its_x42001(full_path, families):
    first = fedscf.read_full(full_path, ["x42001"], implicates=1)
    np.testing.assert_allclose(first["wgt"], first["x42001"], rtol=1e-15)


def test_select_implicates_validates_its_arguments(summary_path):
    frame = fedscf.read_summary(summary_path)
    assert fedscf.select_implicates(frame, None) is frame
    with pytest.raises(ValueError, match="empty"):
        fedscf.select_implicates(frame, [])
    with pytest.raises(ValueError, match="not in the data"):
        fedscf.select_implicates(frame, [0, 1])
    with pytest.raises(ValueError, match="not in the data"):
        fedscf.select_implicates(frame, 6)


@st.composite
def stacked_frames(draw, equal_totals: bool):
    """A stacked frame with 2-5 implicates and arbitrary positive weights."""
    n_families = draw(st.integers(1, 6))
    n_implicates = draw(st.integers(2, 5))
    weights = draw(
        st.lists(
            st.floats(0.01, 1e6, allow_nan=False),
            min_size=n_families * n_implicates,
            max_size=n_families * n_implicates,
        )
    )
    implicate = np.tile(np.arange(1, n_implicates + 1), n_families)
    wgt = np.array(weights)
    if equal_totals:
        for i in range(1, n_implicates + 1):
            wgt[implicate == i] /= wgt[implicate == i].sum()
    family = np.repeat(np.arange(1, n_families + 1), n_implicates)
    frame = pd.DataFrame(
        {"yy1": family, "y1": family * 10 + implicate, "implicate": implicate}
    )
    frame["wgt"] = wgt / n_implicates
    keep = draw(
        st.lists(
            st.integers(1, n_implicates), min_size=1, max_size=n_implicates, unique=True
        )
    )
    return frame, keep


@settings(max_examples=200, deadline=None)
@given(stacked_frames(equal_totals=True))
def test_weight_total_is_preserved_when_implicates_carry_equal_totals(drawn):
    frame, keep = drawn
    kept = fedscf.select_implicates(frame, keep)
    assert kept["wgt"].sum() == pytest.approx(frame["wgt"].sum(), rel=1e-9)


@settings(max_examples=200, deadline=None)
@given(stacked_frames(equal_totals=False))
def test_rescaled_total_is_the_mean_population_of_the_kept_implicates(drawn):
    """The exact identity, with no assumption about the weights."""
    frame, keep = drawn
    n = frame["implicate"].nunique()
    kept = fedscf.select_implicates(frame, keep)
    per_implicate = frame.groupby("implicate")["wgt"].sum() * n
    assert kept["wgt"].sum() == pytest.approx(per_implicate.loc[keep].mean(), rel=1e-9)
    assert len(kept) == frame["implicate"].isin(keep).sum()


@settings(max_examples=200, deadline=None)
@given(stacked_frames(equal_totals=False), st.data())
def test_selections_compose(drawn, data):
    """Selecting B from a selection of A is selecting B, for B within A."""
    frame, keep = drawn
    inner = data.draw(
        st.lists(st.sampled_from(keep), min_size=1, max_size=len(keep), unique=True)
    )
    twice = fedscf.select_implicates(fedscf.select_implicates(frame, keep), inner)
    once = fedscf.select_implicates(frame, inner)
    pd.testing.assert_frame_equal(twice, once, rtol=1e-12)


# --- replicate weights -------------------------------------------------------


def test_replicate_weights_are_weight_times_multiplicity(replicate_path, families):
    table = replicate_table(families)
    combined = fedscf.read_replicate_weights(replicate_path)
    assert combined.shape == (len(families) // N_IMPLICATES, N_REPLICATES)
    assert combined.index.name == "yy1" and combined.index.is_unique
    assert list(combined.columns[:2]) == ["rw1", "rw2"]
    weights = table[[f"wt1b{i}" for i in range(1, 1000)]].to_numpy()
    counts = table[[f"mm{i}" for i in range(1, 1000)]].to_numpy()
    drawn = ~np.isnan(counts)
    values = combined.to_numpy()
    np.testing.assert_array_equal(values[drawn], weights[drawn] * counts[drawn])
    assert (values[~drawn] == 0).all() and (values >= 0).all()
    assert not np.isnan(values).any()


def test_raw_replicate_weights_keep_missing_values(replicate_path, families):
    raw = fedscf.read_replicate_weights(replicate_path, raw=True)
    assert raw.shape == (len(families) // N_IMPLICATES, 2 * N_REPLICATES)
    assert raw["wt1b1"].isna().equals(raw["mm1"].isna())
    assert raw.isna().any().any()


def test_replicate_file_layout_variants(tmp_path, families):
    table = replicate_table(families)
    expected = fedscf.read_replicate_weights(write(tmp_path, table))
    upper = table.rename(columns=str.upper)
    old = upper.rename(columns={"Y1": "X1", "YY1": "XX1"})
    for variant in (upper, old, table[table.columns[::-1]]):
        pd.testing.assert_frame_equal(
            fedscf.read_replicate_weights(write(tmp_path, variant, "v.dta")), expected
        )


@pytest.mark.parametrize(
    "damage, message",
    [
        (lambda t: t.assign(y1=t["y1"] + 1), "first implicate"),
        (lambda t: t.drop(columns="mm999"), "replicate column"),
        (lambda t: t.drop(columns="yy1"), "yy1"),
    ],
)
def test_malformed_replicate_files_are_refused(tmp_path, families, damage, message):
    with pytest.raises(ValueError, match=message):
        fedscf.read_replicate_weights(
            write(tmp_path, damage(replicate_table(families)))
        )


# --- join --------------------------------------------------------------------


def test_join_is_one_to_one_and_keeps_summary_order(summary_path, full_path):
    summary = fedscf.read_summary(summary_path)
    full = fedscf.read_full(full_path, ["x5729", "x3014"])
    joined = fedscf.join(summary, full)
    assert list(joined.columns) == [*summary.columns, "x5729", "x3014"]
    pd.testing.assert_frame_equal(joined[summary.columns], summary)
    np.testing.assert_array_equal(joined["x5729"], full["x5729"])


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_join_does_not_depend_on_row_order(seed):
    families = synthetic_families(n_families=6, seed=3)
    summary = fedscf.select_implicates(_frame(summary_table(families)), None)
    full = _frame(full_table(families).assign(wgt=families["x42001"] / 5))
    shuffled = full.sample(frac=1, random_state=seed % (2**32)).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        fedscf.join(summary, shuffled), fedscf.join(summary, full)
    )


def _frame(table: pd.DataFrame) -> pd.DataFrame:
    """A reader-shaped frame built in memory."""
    frame = table.copy()
    frame["yy1"] = frame["yy1"].astype("int64")
    frame["y1"] = frame["y1"].astype("int64")
    frame.insert(2, "implicate", frame["y1"] - frame["yy1"] * 10)
    return frame


def test_join_refuses_anything_but_a_bijection(summary_path, full_path):
    summary = fedscf.read_summary(summary_path)
    full = fedscf.read_full(full_path, ["x5729"])
    with pytest.raises(ValueError, match="one to one"):
        fedscf.join(summary, full.iloc[:-1])
    with pytest.raises(ValueError, match="one to one"):
        fedscf.join(summary.iloc[:-1], full)
    with pytest.raises(ValueError, match="one to one"):
        fedscf.join(
            fedscf.select_implicates(summary, [1, 2]),
            fedscf.select_implicates(full, [2, 3]),
        )
    with pytest.raises(ValueError, match="repeats"):
        fedscf.join(summary, pd.concat([full, full.iloc[[0]]]))
    with pytest.raises(ValueError, match="disagree"):
        fedscf.join(summary, full.assign(wgt=full["wgt"] * 1.01))
    with pytest.raises(ValueError, match="both frames"):
        fedscf.join(summary, full.assign(networth=0.0))
    with pytest.raises(ValueError, match="lacks"):
        fedscf.join(summary.drop(columns="wgt"), full)


def test_join_of_matching_implicate_subsets(summary_path, full_path):
    summary = fedscf.read_summary(summary_path, implicates=[1, 4])
    full = fedscf.read_full(full_path, ["x5729"], implicates=[1, 4])
    joined = fedscf.join(summary, full)
    assert len(joined) == len(summary) and sorted(joined["implicate"].unique()) == [
        1,
        4,
    ]


def test_known_fed_defect_warns_only_where_it_applies():
    from fedscf.read import _warn_known_issues

    frame = pd.DataFrame({"nhnfin": [1.0], "nfin": [1.0]})
    for wave, released, columns, expected in (
        (2022, "2026-10-09", ["nhnfin"], True),
        (1989, "2026-10-09", ["nhnfin", "nfin"], True),
        (2025, "2026-10-09", ["nhnfin"], False),
        (2022, "2024-04-03", ["nhnfin"], False),
        (2019, "2024-04-03", ["nhnfin"], True),
        (2019, "2023-10-18", ["nhnfin"], True),
        (2022, "2026-10-09", ["nfin"], False),
    ):
        item = fedscf.artifact(wave, "summary", released)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _warn_known_issues(item, frame[columns])
        warned = [w for w in caught if issubclass(w.category, fedscf.FedDataWarning)]
        assert bool(warned) == expected, (wave, released, columns)
    full = fedscf.artifact(2022, "full")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_known_issues(full, frame)
