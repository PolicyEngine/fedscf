"""Download a Fed SCF file, check it against its pinned hashes and cache it.

Every path :func:`fetch` returns points at a Stata file whose SHA-256 was
checked in that call, on the download path and on the cache path alike. The
zip is checked against ``sha256`` and the Stata member against
``member_sha256``. A mismatch raises :class:`HashMismatchError` and leaves
nothing in the cache under that pin.

`pooch <https://www.fatiando.org/pooch/>`_ does the download, the zip hash
check and the cache lookup. It writes to a temporary file and moves it into
place only once the hash matches.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import tempfile
import zipfile
from importlib import metadata
from pathlib import Path, PurePosixPath

import pooch

from fedscf.registry import Artifact, artifact, artifacts

__all__ = [
    "CACHE_ENV_VAR",
    "HashMismatchError",
    "UnverifiedError",
    "cache_dir",
    "fetch",
    "fetch_artifact",
    "resolve",
    "sha256_of",
]

#: Environment variable that moves the cache.
CACHE_ENV_VAR = "FEDSCF_CACHE"

try:
    _VERSION = metadata.version("fedscf")
except metadata.PackageNotFoundError:  # running from a source tree
    _VERSION = "0"

#: The Fed's server answers 403 to urllib's default agent, so name the caller.
USER_AGENT = f"fedscf/{_VERSION} (+https://github.com/PolicyEngine/fedscf)"

_CHUNK_BYTES = 1 << 20


class HashMismatchError(ValueError):
    """A file's SHA-256 is not the expected one."""


class UnverifiedError(ValueError):
    """A fetch was asked for with no hash to check and without opting out."""


def cache_dir() -> Path:
    """Where downloads are kept: ``$FEDSCF_CACHE`` or the OS cache directory."""
    override = os.environ.get(CACHE_ENV_VAR)
    return Path(override).expanduser() if override else Path(pooch.os_cache("fedscf"))


def sha256_of(path: str | os.PathLike[str]) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(
    wave: int,
    file: str = "summary",
    *,
    released: str | None = None,
    sha256: str | None = None,
    member_sha256: str | None = None,
) -> Artifact:
    """Choose the artifact a fetch should check against.

    With no hashes this is the registry entry for ``(wave, file, released)``.
    Hashes the caller supplies take precedence over the registry's:

    * if they match a registry vintage, that vintage is used, so an earlier
      pin keeps resolving after the Fed reissues the file;
    * otherwise the Fed's current URL is checked against the caller's hashes
      alone, which is how to pin a reissue this package does not know yet.
    """
    if sha256 is None and member_sha256 is None:
        return artifact(wave, file, released)
    wanted_zip = sha256.lower() if sha256 else None
    wanted_member = member_sha256.lower() if member_sha256 else None
    matches = [
        item
        for item in artifacts(wave, file)
        if (wanted_zip is None or item.sha256 == wanted_zip)
        and (wanted_member is None or item.member_sha256 == wanted_member)
        and (released is None or item.released == released)
    ]
    if matches:
        current = [item for item in matches if not item.superseded]
        return (current or matches)[-1]
    if released is not None:
        raise KeyError(
            f"the {wave} {file!r} file released {released} is not pinned to "
            "the hashes given"
        )
    return dataclasses.replace(
        artifact(wave, file),
        released=None,
        sha256=wanted_zip,
        member_sha256=wanted_member,
        expected_rows=None,
        size_bytes=None,
        member_size_bytes=None,
        dollar_year=None,
        superseded=False,
        archive=None,
    )


def fetch(
    wave: int,
    file: str = "summary",
    *,
    released: str | None = None,
    sha256: str | None = None,
    member_sha256: str | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    allow_unverified: bool = False,
    refresh: bool = False,
) -> Path:
    """Return the path of a wave's verified Stata file, downloading if needed.

    Args:
        wave: Survey year.
        file: ``"summary"``, ``"full"`` or ``"replicate_weights"``.
        released: Fed release date of the vintage wanted; the current one by
            default.
        sha256: Expected SHA-256 of the zip, in place of the registry's.
        member_sha256: Expected SHA-256 of the Stata file inside the zip.
        cache_dir: Cache directory; see :func:`cache_dir` for the default.
        allow_unverified: Skip hash checks. The file is whatever the Fed
            serves at that moment.
        refresh: Download again even if a cached copy exists.

    Raises:
        HashMismatchError: If the zip or the Stata file does not match.
    """
    chosen = resolve(
        wave, file, released=released, sha256=sha256, member_sha256=member_sha256
    )
    return fetch_artifact(
        chosen, cache_dir=cache_dir, allow_unverified=allow_unverified, refresh=refresh
    )


def fetch_artifact(
    item: Artifact,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    allow_unverified: bool = False,
    refresh: bool = False,
    timeout: float = 120.0,
) -> Path:
    """Fetch and verify one :class:`~fedscf.registry.Artifact`; see :func:`fetch`."""
    if item.format != "stata_in_zip":
        raise ValueError(f"cannot fetch format {item.format!r}")
    pinned = item.sha256 is not None or item.member_sha256 is not None
    if allow_unverified:
        item = dataclasses.replace(item, sha256=None, member_sha256=None)
    elif not pinned:
        raise UnverifiedError(
            f"no hash to check the {item.wave} {item.file!r} file against; pass "
            "sha256= or member_sha256=, or allow_unverified=True"
        )
    directory = _directory(item, cache_dir)
    zip_name = PurePosixPath(item.locator).name
    if refresh:
        for stale in (directory / zip_name, directory / item.member):
            stale.unlink(missing_ok=True)
    try:
        zip_path = pooch.retrieve(
            url=item.url,
            known_hash=f"sha256:{item.sha256}" if item.sha256 else None,
            fname=zip_name,
            path=directory,
            downloader=pooch.HTTPDownloader(
                headers={"User-Agent": USER_AGENT}, timeout=timeout
            ),
        )
    except ValueError as error:
        if "does not match the known hash" not in str(error):
            raise
        raise HashMismatchError(f"{_describe(item)}: {error}{_HINT}") from error
    return _extract_member(Path(zip_path), item, directory)


_HINT = (
    " The Fed reissues SCF files under the same name. If this is a new release, "
    "pass its hash as sha256= once you have checked it, or upgrade fedscf."
)


def _describe(item: Artifact) -> str:
    return f"SCF {item.wave} {item.file} file ({item.url})"


def _directory(item: Artifact, root: str | os.PathLike[str] | None) -> Path:
    """One directory per pinned content, so vintages never share a cache path."""
    base = Path(root).expanduser() if root is not None else cache_dir()
    stem = PurePosixPath(item.locator).name.removesuffix(".zip")
    if item.sha256:
        pin = item.sha256[:16]
    elif item.member_sha256:
        pin = f"member-{item.member_sha256[:16]}"
    else:
        pin = "unverified"
    return base / stem / pin


def _extract_member(zip_path: Path, item: Artifact, directory: Path) -> Path:
    """Write the zip's Stata member to ``directory`` once its hash is confirmed."""
    name = item.member
    if PurePosixPath(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"unsafe member name {name!r}")
    target = directory / name
    expected = item.member_sha256
    if target.exists() and expected is not None and sha256_of(target) == expected:
        return target
    with zipfile.ZipFile(zip_path) as archive:
        if name not in archive.namelist():
            raise ValueError(
                f"{zip_path.name} has no member {name!r}; it holds {archive.namelist()}"
            )
        digest = hashlib.sha256()
        handle, temporary = tempfile.mkstemp(dir=directory, suffix=".part")
        try:
            with os.fdopen(handle, "wb") as out, archive.open(name) as member:
                for chunk in iter(lambda: member.read(_CHUNK_BYTES), b""):
                    digest.update(chunk)
                    out.write(chunk)
            actual = digest.hexdigest()
            if expected is not None and actual != expected:
                raise HashMismatchError(
                    f"{_describe(item)}: SHA256 hash of member {name!r} does not "
                    f"match the known hash: expected {expected} but got {actual}."
                    f"{_HINT}"
                )
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return target
