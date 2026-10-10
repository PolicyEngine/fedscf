"""The ``fedscf`` command line."""

from __future__ import annotations

import json

import pytest

import fedscf
from conftest import (
    sha256,
    summary_table,
    synthetic_families,
    to_dta_bytes,
    to_zip_bytes,
)
from fedscf import __main__ as cli
from fedscf.registry import Artifact


def test_list_prints_the_registry(capsys):
    assert cli.main(["list", "--wave", "2022", "--file", "summary"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:3] == ["wave", "file", "released"]
    rows = [line.split() for line in lines[1:]]
    assert [row[2] for row in rows] == ["2023-10-18", "2024-04-03", "2026-10-09"]
    assert rows[0][-1] == "superseded" and rows[-1][-1] != "superseded"


def test_list_everything(capsys):
    assert cli.main(["list"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == len(fedscf.artifacts()) + 1


@pytest.fixture
def served_summary(file_server):
    payload = to_dta_bytes(summary_table(synthetic_families(n_families=5)))
    zipped = to_zip_bytes("rscfp2099.dta", payload)
    url = file_server.put("scfp2099s.zip", zipped)
    return url, zipped, payload


def test_pin_prints_a_registry_entry(capsys, served_summary):
    url, zipped, payload = served_summary
    code = cli.main(
        ["pin", "2025", "summary", "--released", "2099-01-01", "--url", url]
    )
    assert code == 0
    entry = json.loads(capsys.readouterr().out)
    assert entry["sha256"] == sha256(zipped)
    assert entry["member_sha256"] == sha256(payload)
    assert entry["member"] == "rscfp2099.dta"
    assert entry["expected_rows"] == 25
    assert entry["released"] == "2099-01-01" and entry["locator"] == url
    assert Artifact.from_manifest(entry).sha256 == sha256(zipped)


def test_pin_requires_a_release_date():
    with pytest.raises(SystemExit):
        cli.main(["pin", "2025", "summary"])


def test_verify_reports_matches_and_mismatches(capsys, monkeypatch, served_summary):
    url, zipped, payload = served_summary
    good = Artifact(
        wave=2099,
        file="summary",
        locator=url,
        member="rscfp2099.dta",
        sha256=sha256(zipped),
        member_sha256=sha256(payload),
    )
    monkeypatch.setattr(cli, "artifact", lambda wave, file: good)
    assert cli.main(["verify", "2099", "--file", "summary"]) == 0
    assert capsys.readouterr().out.startswith("ok")
    stale = Artifact(**{**good.__dict__, "sha256": "0" * 64})
    monkeypatch.setattr(cli, "artifact", lambda wave, file: stale)
    assert cli.main(["verify", "2099", "--file", "summary"]) == 1
    assert capsys.readouterr().out.startswith("MISMATCH")


def test_fetch_prints_the_verified_path(capsys, monkeypatch, tmp_path, served_summary):
    url, zipped, payload = served_summary
    monkeypatch.setenv(fedscf.CACHE_ENV_VAR, str(tmp_path))
    item = Artifact(
        wave=2099,
        file="summary",
        locator=url,
        member="rscfp2099.dta",
        sha256=sha256(zipped),
        member_sha256=sha256(payload),
    )
    monkeypatch.setattr(
        cli, "fetch", lambda wave, file, released: fedscf.fetch_artifact(item)
    )
    assert cli.main(["fetch", "2099"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed.endswith("rscfp2099.dta") and printed.startswith(str(tmp_path))
