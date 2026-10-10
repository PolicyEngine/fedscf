"""The shipped registry: every entry is well formed and the set is consistent."""

from __future__ import annotations

import dataclasses
import datetime
import json
import re
from importlib import resources

import pytest

import fedscf
from fedscf.registry import FILES, MANIFEST_FIELDS, Artifact

ALL = fedscf.artifacts()
CURRENT = fedscf.artifacts(include_superseded=False)


def test_waves_are_the_triennial_surveys_from_1989_to_2025():
    assert fedscf.WAVES == tuple(range(1989, 2026, 3))


def test_every_wave_has_exactly_one_current_vintage_of_each_file():
    assert sorted((item.wave, item.file) for item in CURRENT) == sorted(
        (wave, file) for wave in fedscf.WAVES for file in FILES
    )


def test_keys_are_unique():
    keys = [item.key for item in ALL]
    assert len(set(keys)) == len(keys)


def test_hashes_are_unique():
    for name in ("sha256", "member_sha256"):
        values = [getattr(item, name) for item in ALL]
        assert len(set(values)) == len(values)


@pytest.mark.parametrize("item", ALL, ids=lambda item: "-".join(map(str, item.key)))
def test_entry_is_complete_and_well_formed(item: Artifact):
    assert item.locator.startswith("https://www.federalreserve.gov/econres/files/")
    assert item.locator.endswith(".zip")
    assert item.member.endswith(".dta")
    assert re.fullmatch(r"[0-9a-f]{64}", item.sha256)
    assert re.fullmatch(r"[0-9a-f]{64}", item.member_sha256)
    assert item.kind == "public_microdata"
    assert item.format == "stata_in_zip"
    assert item.vintage == str(item.wave)
    assert item.size_bytes > 0 and item.member_size_bytes > item.size_bytes
    assert datetime.date.fromisoformat(item.released) <= datetime.date(2026, 10, 9)
    # A file cannot be released before its survey year.
    assert int(item.released[:4]) >= item.wave
    assert (item.dollar_year is not None) == (item.file == "summary")
    assert (item.archive is not None) == item.superseded
    if item.superseded:
        assert item.url == item.archive
        assert item.archive.startswith("https://web.archive.org/web/")
        assert item.archive.endswith(item.locator)
    else:
        assert item.url == item.locator


@pytest.mark.parametrize("wave", fedscf.WAVES)
def test_row_counts_agree_across_a_waves_files(wave: int):
    summary = fedscf.artifact(wave, "summary")
    full = fedscf.artifact(wave, "full")
    replicates = fedscf.artifact(wave, "replicate_weights")
    # Five implicates per family in the data files, one row per family in the
    # replicate weights.
    assert summary.expected_rows == full.expected_rows
    assert summary.expected_rows == fedscf.N_IMPLICATES * replicates.expected_rows


def test_current_summary_extracts_share_one_dollar_year():
    assert {
        item.dollar_year
        for item in fedscf.artifacts(file="summary", include_superseded=False)
    } == {2025}


def test_a_superseded_vintage_is_older_than_the_current_one():
    for item in ALL:
        if item.superseded:
            assert item.released < fedscf.artifact(item.wave, item.file).released


def test_artifact_defaults_to_the_current_vintage_and_accepts_a_release_date():
    current = fedscf.artifact(2022, "summary")
    assert not current.superseded and current.released == "2026-10-09"
    earlier = fedscf.artifact(2022, "summary", "2023-10-18")
    assert earlier.superseded and earlier.dollar_year == 2022
    with pytest.raises(KeyError):
        fedscf.artifact(2022, "summary", "1999-01-01")
    with pytest.raises(KeyError):
        fedscf.artifact(2023, "summary")
    with pytest.raises(ValueError):
        fedscf.artifacts(file="codebook")


@pytest.mark.parametrize("item", ALL, ids=lambda item: "-".join(map(str, item.key)))
def test_manifest_entry_round_trips(item: Artifact):
    entry = item.manifest_entry()
    assert list(entry)[: len(MANIFEST_FIELDS)] == list(MANIFEST_FIELDS)
    json.dumps(entry)  # plain JSON types only
    rebuilt = Artifact.from_manifest(entry)
    # The manifest carries what Microcosm needs; the three fedscf-only fields
    # about dollars and supersession are not part of it.
    assert rebuilt == dataclasses.replace(
        item, dollar_year=None, superseded=False, archive=None
    )


def test_from_manifest_reads_a_microcosm_entry():
    # Copied from Microcosm's source_stages.json, stage scf_wealth.
    entry = {
        "kind": "public_microdata",
        "format": "stata_in_zip",
        "vintage": "2022",
        "locator": "https://www.federalreserve.gov/econres/files/scf2022s.zip",
        "member": "p22i6.dta",
        "expected_rows": 22975,
        "integrity_note": "ignored",
    }
    built = Artifact.from_manifest(entry, wave=2022, file="full")
    assert built.expected_rows == 22975 and built.sha256 is None
    assert built.member == fedscf.artifact(2022, "full").member
    with pytest.raises(ValueError):
        Artifact.from_manifest(entry)


def test_artifact_rejects_malformed_values():
    base = dict(wave=2022, locator="https://example.org/a.zip", member="a.dta")
    with pytest.raises(ValueError):
        Artifact(file="codebook", **base)
    with pytest.raises(ValueError):
        Artifact(file="summary", sha256="abc", **base)
    upper = "A" * 64
    assert Artifact(file="summary", sha256=upper, **base).sha256 == upper.lower()


def test_registry_json_is_sorted_and_names_only_known_fields():
    text = resources.files("fedscf").joinpath("registry.json").read_text("utf-8")
    entries = json.loads(text)["artifacts"]
    names = {field.name for field in dataclasses.fields(Artifact)}
    assert all(set(entry) <= names for entry in entries)
    order = [(e["wave"], FILES.index(e["file"]), e["released"]) for e in entries]
    assert order == sorted(order)
