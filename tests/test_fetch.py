"""Verified fetch and cache, against a local server standing in for the Fed's.

Invariants:

* a path :func:`fedscf.fetch_artifact` returns holds bytes with the expected
  SHA-256, whether they were downloaded or found in the cache;
* a hash mismatch always raises :class:`fedscf.HashMismatchError`, and nothing
  is left in the cache under that pin;
* a fetch with no hash to check raises unless the caller opts out.
"""

from __future__ import annotations

import dataclasses
import itertools
import tempfile
from pathlib import Path

import pytest
import requests
from hypothesis import given, settings
from hypothesis import strategies as st

import fedscf
from conftest import FileServer, sha256, to_zip_bytes
from fedscf.fetch import USER_AGENT, sha256_of
from fedscf.registry import Artifact

_names = itertools.count()


def serve(server: FileServer, payload: bytes, member: str = "data.dta") -> Artifact:
    """Serve ``payload`` as a zipped Stata member; return its pinned artifact."""
    name = f"file{next(_names)}.zip"
    zipped = to_zip_bytes(member, payload)
    return Artifact(
        wave=2099,
        file="summary",
        released="2099-01-01",
        locator=server.put(name, zipped),
        member=member,
        sha256=sha256(zipped),
        member_sha256=sha256(payload),
    )


def cached_files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.rglob("*") if path.is_file())


def test_fetch_returns_the_verified_member(file_server, tmp_path):
    item = serve(file_server, b"stata bytes")
    path = fedscf.fetch_artifact(item, cache_dir=tmp_path)
    assert path.read_bytes() == b"stata bytes"
    assert path.name == "data.dta"
    assert path.parent.name == item.sha256[:16]


def test_fetch_names_itself_to_the_server(file_server, tmp_path):
    item = serve(file_server, b"x")
    fedscf.fetch_artifact(item, cache_dir=tmp_path)
    assert file_server.requests[-1][1] == USER_AGENT
    assert USER_AGENT.startswith("fedscf/")


def test_second_fetch_uses_the_cache(file_server, tmp_path):
    item = serve(file_server, b"cache me")
    first = fedscf.fetch_artifact(item, cache_dir=tmp_path)
    file_server.remove(Path(item.locator).name)  # a download would now 404
    assert fedscf.fetch_artifact(item, cache_dir=tmp_path) == first


def test_cache_dir_defaults_to_the_environment_variable(
    file_server, tmp_path, monkeypatch
):
    monkeypatch.setenv(fedscf.CACHE_ENV_VAR, str(tmp_path / "from-env"))
    assert fedscf.cache_dir() == tmp_path / "from-env"
    path = fedscf.fetch_artifact(serve(file_server, b"x"))
    assert tmp_path / "from-env" in path.parents


def test_zip_hash_mismatch_raises_and_caches_nothing(file_server, tmp_path):
    item = dataclasses.replace(serve(file_server, b"payload"), sha256="0" * 64)
    with pytest.raises(fedscf.HashMismatchError, match="reissues"):
        fedscf.fetch_artifact(item, cache_dir=tmp_path)
    assert cached_files(tmp_path) == []


def test_member_hash_mismatch_raises_and_caches_no_member(file_server, tmp_path):
    item = dataclasses.replace(serve(file_server, b"payload"), member_sha256="0" * 64)
    with pytest.raises(fedscf.HashMismatchError, match="member"):
        fedscf.fetch_artifact(item, cache_dir=tmp_path)
    # The zip matched its own pin and stays; the member was never written.
    assert cached_files(tmp_path) == [Path(item.locator).name]


def test_hash_mismatch_error_is_a_value_error():
    assert issubclass(fedscf.HashMismatchError, ValueError)


def test_tampered_cached_member_is_replaced(file_server, tmp_path):
    item = serve(file_server, b"genuine")
    path = fedscf.fetch_artifact(item, cache_dir=tmp_path)
    path.write_bytes(b"tampered")
    assert fedscf.fetch_artifact(item, cache_dir=tmp_path).read_bytes() == b"genuine"


def test_tampered_cached_zip_is_downloaded_again(file_server, tmp_path):
    item = serve(file_server, b"genuine")
    path = fedscf.fetch_artifact(item, cache_dir=tmp_path)
    (path.parent / Path(item.locator).name).write_bytes(b"not the zip")
    path.unlink()
    assert fedscf.fetch_artifact(item, cache_dir=tmp_path).read_bytes() == b"genuine"


def test_unpinned_fetch_needs_an_explicit_opt_out(file_server, tmp_path):
    item = dataclasses.replace(
        serve(file_server, b"payload"), sha256=None, member_sha256=None
    )
    with pytest.raises(fedscf.UnverifiedError):
        fedscf.fetch_artifact(item, cache_dir=tmp_path)
    path = fedscf.fetch_artifact(item, cache_dir=tmp_path, allow_unverified=True)
    assert path.read_bytes() == b"payload"
    assert path.parent.name == "unverified"


def test_allow_unverified_skips_a_wrong_pin(file_server, tmp_path):
    item = dataclasses.replace(serve(file_server, b"payload"), sha256="0" * 64)
    path = fedscf.fetch_artifact(item, cache_dir=tmp_path, allow_unverified=True)
    assert path.read_bytes() == b"payload"


def test_a_member_pin_alone_is_enough(file_server, tmp_path):
    item = dataclasses.replace(serve(file_server, b"payload"), sha256=None)
    path = fedscf.fetch_artifact(item, cache_dir=tmp_path)
    assert path.parent.name == f"member-{item.member_sha256[:16]}"


def test_vintages_with_one_name_do_not_share_a_cache_path(file_server, tmp_path):
    old = serve(file_server, b"old vintage")
    old_path = fedscf.fetch_artifact(old, cache_dir=tmp_path)
    # The Fed reissues the file under the same name.
    name = Path(old.locator).name
    zipped = to_zip_bytes("data.dta", b"new vintage")
    file_server.put(name, zipped)
    new = dataclasses.replace(
        old, sha256=sha256(zipped), member_sha256=sha256(b"new vintage")
    )
    new_path = fedscf.fetch_artifact(new, cache_dir=tmp_path)
    assert new_path != old_path
    assert old_path.read_bytes() == b"old vintage"
    assert new_path.read_bytes() == b"new vintage"
    # The old pin still resolves from the cache, and fails on a fresh download.
    assert fedscf.fetch_artifact(old, cache_dir=tmp_path) == old_path
    with pytest.raises(fedscf.HashMismatchError):
        fedscf.fetch_artifact(old, cache_dir=tmp_path / "fresh")


def test_refresh_downloads_again(file_server, tmp_path):
    item = dataclasses.replace(
        serve(file_server, b"first"), sha256=None, member_sha256=None
    )
    options = dict(cache_dir=tmp_path, allow_unverified=True)
    assert fedscf.fetch_artifact(item, **options).read_bytes() == b"first"
    file_server.put(Path(item.locator).name, to_zip_bytes("data.dta", b"second"))
    assert fedscf.fetch_artifact(item, **options).read_bytes() == b"first"
    refreshed = fedscf.fetch_artifact(item, refresh=True, **options)
    assert refreshed.read_bytes() == b"second"


def test_superseded_vintage_is_fetched_from_its_archive(file_server, tmp_path):
    item = serve(file_server, b"archived vintage")
    moved = dataclasses.replace(
        item,
        locator=f"{file_server.url}/gone/{Path(item.locator).name}",
        archive=item.locator,
        superseded=True,
    )
    path = fedscf.fetch_artifact(moved, cache_dir=tmp_path)
    assert path.read_bytes() == b"archived vintage"


def test_missing_file_raises_an_http_error(file_server, tmp_path):
    item = dataclasses.replace(
        serve(file_server, b"x"), locator=f"{file_server.url}/absent.zip"
    )
    with pytest.raises(requests.HTTPError):
        fedscf.fetch_artifact(item, cache_dir=tmp_path)


@pytest.mark.parametrize("member", ["../escape.dta", "nested/data.dta", "", ".."])
def test_unsafe_member_names_are_refused(file_server, tmp_path, member):
    item = dataclasses.replace(serve(file_server, b"x"), member=member)
    with pytest.raises(ValueError, match="unsafe"):
        fedscf.fetch_artifact(item, cache_dir=tmp_path)


def test_a_zip_without_the_member_is_refused(file_server, tmp_path):
    item = dataclasses.replace(serve(file_server, b"x"), member="other.dta")
    with pytest.raises(ValueError, match="no member"):
        fedscf.fetch_artifact(item, cache_dir=tmp_path)


def test_only_stata_in_zip_is_supported(file_server, tmp_path):
    item = dataclasses.replace(serve(file_server, b"x"), format="csv")
    with pytest.raises(ValueError, match="format"):
        fedscf.fetch_artifact(item, cache_dir=tmp_path)


def flip_one_character(digest: str, position: int) -> str:
    replacement = "0" if digest[position] != "0" else "1"
    return digest[:position] + replacement + digest[position + 1 :]


wrong_hashes = st.one_of(
    st.text(alphabet="0123456789abcdef", min_size=64, max_size=64),
    st.integers(0, 63),  # a near miss: the right hash with one character changed
)


def wrong(strategy_value: str | int, right: str) -> str | None:
    if isinstance(strategy_value, int):
        return flip_one_character(right, strategy_value)
    return None if strategy_value == right else strategy_value


@settings(max_examples=40, deadline=None)
@given(payload=st.binary(max_size=512), guess=wrong_hashes)
def test_any_wrong_zip_hash_raises(file_server, payload, guess):
    """Whatever the server sends, a zip hash that is not its own is refused."""
    url = file_server.put(f"file{next(_names)}.zip", payload)  # need not be a zip
    expected = wrong(guess, sha256(payload))
    if expected is None:
        return
    item = Artifact(
        wave=2099, file="summary", locator=url, member="data.dta", sha256=expected
    )
    with tempfile.TemporaryDirectory() as cache:
        with pytest.raises(fedscf.HashMismatchError):
            fedscf.fetch_artifact(item, cache_dir=cache)
        assert cached_files(Path(cache)) == []


@settings(max_examples=40, deadline=None)
@given(payload=st.binary(max_size=512), guess=wrong_hashes)
def test_any_wrong_member_hash_raises(file_server, payload, guess):
    """A member hash that is not the member's own is refused, zip pin or not."""
    item = serve(file_server, payload)
    expected = wrong(guess, item.member_sha256)
    if expected is None:
        return
    for pinned_zip in (item.sha256, None):
        bad = dataclasses.replace(item, sha256=pinned_zip, member_sha256=expected)
        with tempfile.TemporaryDirectory() as cache:
            with pytest.raises(fedscf.HashMismatchError):
                fedscf.fetch_artifact(bad, cache_dir=cache)
            assert "data.dta" not in cached_files(Path(cache))


@settings(max_examples=40, deadline=None)
@given(payload=st.binary(max_size=512))
def test_the_right_hashes_always_return_the_payload(file_server, payload):
    """The other half of the property: matching pins never raise."""
    item = serve(file_server, payload)
    with tempfile.TemporaryDirectory() as cache:
        path = fedscf.fetch_artifact(item, cache_dir=cache)
        assert sha256_of(path) == item.member_sha256 == sha256(payload)
        # Determinism: the cached path and bytes are the same on a second call.
        again = fedscf.fetch_artifact(item, cache_dir=cache)
        assert again == path and again.read_bytes() == payload


def test_resolve_without_hashes_is_the_registry_entry():
    assert fedscf.resolve(2022, "summary") == fedscf.artifact(2022, "summary")
    earlier = fedscf.resolve(2022, "summary", released="2023-10-18")
    assert earlier == fedscf.artifact(2022, "summary", "2023-10-18")


def test_resolve_finds_the_vintage_a_caller_hash_belongs_to():
    earlier = fedscf.artifact(2022, "summary", "2023-10-18")
    assert fedscf.resolve(2022, "summary", sha256=earlier.sha256) == earlier
    assert (
        fedscf.resolve(2022, "summary", member_sha256=earlier.member_sha256) == earlier
    )
    assert fedscf.resolve(2022, "summary", sha256=earlier.sha256.upper()) == earlier


def test_resolve_trusts_only_the_caller_for_a_hash_it_does_not_know():
    chosen = fedscf.resolve(2022, "summary", sha256="a" * 64)
    current = fedscf.artifact(2022, "summary")
    assert chosen.sha256 == "a" * 64
    assert (chosen.locator, chosen.member) == (current.locator, current.member)
    # Nothing about the registry's file is asserted of the caller's.
    assert chosen.released is None and chosen.member_sha256 is None
    assert chosen.expected_rows is None and chosen.dollar_year is None
    member_only = fedscf.resolve(2022, "summary", member_sha256="b" * 64)
    assert member_only.sha256 is None and member_only.member_sha256 == "b" * 64


def test_resolve_refuses_a_release_date_that_contradicts_the_hash():
    with pytest.raises(KeyError):
        fedscf.resolve(2022, "summary", released="2023-10-18", sha256="a" * 64)
