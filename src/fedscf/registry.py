"""The Federal Reserve's SCF Stata files: where each one lives and what it hashes to.

An :class:`Artifact` is one downloadable file, identified by three things:

* ``wave``: the survey year (1989, 1992, ..., 2025);
* ``file``: ``"summary"`` (the Summary Extract Public Data), ``"full"`` (the
  full public data set) or ``"replicate_weights"``;
* ``released``: the "Last updated" date the Fed's page for that wave shows for
  the file. The Fed reissues files under the same name, so the date is part of
  the key. The 2022 summary extract, for example, has had three vintages.

The remaining fields use the names and meanings of the artifact entries in
Microcosm's source-stage manifest (``kind``, ``format``, ``vintage``,
``locator``, ``member``, ``sha256``, ``member_sha256``, ``expected_rows``,
``size_bytes``), so :meth:`Artifact.manifest_entry` can be dropped into that
manifest as is.

The pins ship in ``registry.json``. Every pinned file was downloaded and read
on 2026-10-09; ``tests/test_fed_data.py`` re-checks the pins against the files
and against the Fed's published tables.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

__all__ = [
    "FILES",
    "MANIFEST_FIELDS",
    "WAVES",
    "Artifact",
    "artifact",
    "artifacts",
]

#: The three Stata files the Fed publishes for each wave.
FILES: tuple[str, ...] = ("summary", "full", "replicate_weights")

#: Field names shared with Microcosm's source-stage manifest.
MANIFEST_FIELDS: tuple[str, ...] = (
    "kind",
    "format",
    "vintage",
    "locator",
    "member",
    "sha256",
    "member_sha256",
    "expected_rows",
    "size_bytes",
)

_SHA256_LENGTH = 64


@dataclass(frozen=True)
class Artifact:
    """One Fed SCF file: its identity, its location and its pinned contents."""

    #: Survey year.
    wave: int
    #: Which of the wave's files this is; one of :data:`FILES`.
    file: str
    #: URL the Fed serves the zip from.
    locator: str
    #: Name of the Stata file inside the zip.
    member: str
    #: The Fed's "Last updated" date for this vintage (ISO ``YYYY-MM-DD``), or
    #: ``None`` for a file the caller pinned that is not in the registry.
    released: str | None = None
    #: Microcosm's artifact kind.
    kind: str = "public_microdata"
    #: Microcosm's artifact format: one Stata ``.dta`` inside a zip.
    format: str = "stata_in_zip"
    #: Microcosm's data vintage; the survey year as a string.
    vintage: str = ""
    #: SHA-256 of the zip.
    sha256: str | None = None
    #: SHA-256 of the Stata member.
    member_sha256: str | None = None
    #: Rows in the Stata member.
    expected_rows: int | None = None
    #: Size of the zip in bytes.
    size_bytes: int | None = None
    #: Size of the Stata member in bytes.
    member_size_bytes: int | None = None
    #: For summary extracts, the year whose dollars the file is stated in.
    dollar_year: int | None = None
    #: True when the Fed's ``locator`` now serves a later vintage.
    superseded: bool = False
    #: Where a superseded vintage can still be downloaded.
    archive: str | None = None

    def __post_init__(self) -> None:
        if self.file not in FILES:
            raise ValueError(f"file must be one of {FILES}, got {self.file!r}")
        if not self.vintage:
            object.__setattr__(self, "vintage", str(self.wave))
        for name in ("sha256", "member_sha256"):
            value = getattr(self, name)
            if value is None:
                continue
            value = value.lower()
            if len(value) != _SHA256_LENGTH or set(value) - set("0123456789abcdef"):
                raise ValueError(f"{name} must be 64 hexadecimal characters")
            object.__setattr__(self, name, value)

    @property
    def key(self) -> tuple[int, str, str | None]:
        """``(wave, file, released)``: what identifies an artifact."""
        return (self.wave, self.file, self.released)

    @property
    def url(self) -> str:
        """Where to download these bytes today."""
        if self.superseded and self.archive:
            return self.archive
        return self.locator

    def manifest_entry(self) -> dict[str, Any]:
        """This artifact as an entry for Microcosm's source-stage manifest.

        The nine :data:`MANIFEST_FIELDS` come first, with unset ones left out,
        followed by ``member_size_bytes`` and the three fields that identify
        the artifact here (``wave``, ``file``, ``released``).
        """
        entry = {name: getattr(self, name) for name in MANIFEST_FIELDS}
        entry["member_size_bytes"] = self.member_size_bytes
        entry.update(wave=self.wave, file=self.file, released=self.released)
        return {name: value for name, value in entry.items() if value is not None}

    @classmethod
    def from_manifest(
        cls,
        entry: Mapping[str, Any],
        *,
        wave: int | None = None,
        file: str | None = None,
        released: str | None = None,
    ) -> Artifact:
        """Build an artifact from a manifest entry.

        Keys this class has no field for are ignored. ``wave``, ``file`` and
        ``released`` are read from the entry when it has them, as entries
        written by :meth:`manifest_entry` do; otherwise pass them.
        """
        names = {field.name for field in dataclasses.fields(cls)}
        values = {name: entry[name] for name in names if name in entry}
        for name, value in (("wave", wave), ("file", file), ("released", released)):
            if value is not None:
                values[name] = value
        missing = [name for name in ("wave", "file") if name not in values]
        if missing:
            raise ValueError(f"manifest entry does not give {missing}; pass them")
        return cls(**values)


def _sort_key(item: Artifact) -> tuple[int, int, str]:
    return (item.wave, FILES.index(item.file), item.released or "")


@lru_cache(maxsize=1)
def _load() -> tuple[Artifact, ...]:
    text = resources.files("fedscf").joinpath("registry.json").read_text("utf-8")
    entries = json.loads(text)["artifacts"]
    loaded = tuple(sorted((Artifact(**entry) for entry in entries), key=_sort_key))
    keys = [item.key for item in loaded]
    if len(set(keys)) != len(keys):
        raise ValueError("registry.json repeats a (wave, file, released) key")
    return loaded


def artifacts(
    wave: int | None = None,
    file: str | None = None,
    *,
    include_superseded: bool = True,
) -> tuple[Artifact, ...]:
    """Registry entries, ordered by wave, file, then release date."""
    if file is not None and file not in FILES:
        raise ValueError(f"file must be one of {FILES}, got {file!r}")
    return tuple(
        item
        for item in _load()
        if (wave is None or item.wave == wave)
        and (file is None or item.file == file)
        and (include_superseded or not item.superseded)
    )


#: Survey years with pinned files.
WAVES: tuple[int, ...] = tuple(sorted({item.wave for item in _load()}))


def artifact(wave: int, file: str = "summary", released: str | None = None) -> Artifact:
    """The registry entry for one file.

    With ``released=None`` this is the vintage the Fed serves today. Pass a
    release date to select an earlier one.
    """
    candidates = artifacts(wave, file)
    if not candidates:
        raise KeyError(f"no pinned {file!r} file for the {wave} SCF; waves are {WAVES}")
    if released is None:
        current = [item for item in candidates if not item.superseded]
        if len(current) != 1:
            raise ValueError(
                f"registry has {len(current)} current {file!r} files for {wave}"
            )
        return current[0]
    for item in candidates:
        if item.released == released:
            return item
    dates = [item.released for item in candidates]
    raise KeyError(f"no {wave} {file!r} file released {released}; have {dates}")
