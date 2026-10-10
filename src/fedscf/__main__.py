"""Command line: list the registry, fetch a file, re-check pins, pin a new file.

``fedscf verify`` downloads each current file afresh and compares it with its
pins, which is how to find out that the Fed has reissued one. ``fedscf pin``
prints the registry entry for whatever a URL serves now.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

import pandas as pd
import pooch

from fedscf.fetch import USER_AGENT, HashMismatchError, fetch, fetch_artifact
from fedscf.registry import FILES, WAVES, Artifact, artifact, artifacts

_BASE_URL = "https://www.federalreserve.gov/econres/files/"


def _list(arguments: argparse.Namespace) -> int:
    print(f"{'wave':<5} {'file':<18} {'released':<11} {'rows':>6}  sha256")
    for item in artifacts(arguments.wave, arguments.file):
        note = "  superseded" if item.superseded else ""
        print(
            f"{item.wave:<5} {item.file:<18} {item.released or '':<11} "
            f"{item.expected_rows or 0:>6}  {(item.sha256 or '')[:16]}{note}"
        )
    return 0


def _fetch(arguments: argparse.Namespace) -> int:
    print(fetch(arguments.wave, arguments.file, released=arguments.released))
    return 0


def _verify(arguments: argparse.Namespace) -> int:
    waves = arguments.waves or list(WAVES)
    files = [arguments.file] if arguments.file else list(FILES)
    failures = 0
    for wave in waves:
        for file in files:
            item = artifact(wave, file)
            # A fresh directory per file forces a download: a cache hit would
            # say nothing about what the Fed serves today.
            with tempfile.TemporaryDirectory() as scratch:
                try:
                    fetch_artifact(item, cache_dir=scratch)
                except HashMismatchError as error:
                    failures += 1
                    print(f"MISMATCH {wave} {file}: {error}", flush=True)
                else:
                    print(f"ok       {wave} {file} {item.sha256}", flush=True)
    return 1 if failures else 0


def _pin(arguments: argparse.Namespace) -> int:
    url = arguments.url
    name = PurePosixPath(url).name
    with tempfile.TemporaryDirectory() as scratch:
        zip_path = Path(scratch) / name
        pooch.HTTPDownloader(headers={"User-Agent": USER_AGENT})(
            url, str(zip_path), None
        )
        zip_bytes = zip_path.read_bytes()
        with zipfile.ZipFile(zip_path) as archive:
            members = [m for m in archive.namelist() if m.lower().endswith(".dta")]
            if len(members) != 1:
                raise SystemExit(f"expected one .dta in {name}, found {members}")
            member_path = Path(archive.extract(members[0], scratch))
        member_bytes = member_path.read_bytes()
        with pd.read_stata(member_path, iterator=True) as reader:
            first_column = next(iter(reader.variable_labels()))
        rows = len(pd.read_stata(member_path, columns=[first_column]))
    entry = Artifact(
        wave=arguments.wave,
        file=arguments.file,
        released=arguments.released,
        locator=url,
        member=members[0],
        sha256=hashlib.sha256(zip_bytes).hexdigest(),
        member_sha256=hashlib.sha256(member_bytes).hexdigest(),
        expected_rows=rows,
        size_bytes=len(zip_bytes),
        member_size_bytes=len(member_bytes),
    )
    print(json.dumps(entry.manifest_entry(), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fedscf", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="show the registry")
    listing.add_argument("--wave", type=int)
    listing.add_argument("--file", choices=FILES)
    listing.set_defaults(run=_list)

    fetching = commands.add_parser("fetch", help="download and verify one file")
    fetching.add_argument("wave", type=int)
    fetching.add_argument("file", choices=FILES, nargs="?", default="summary")
    fetching.add_argument("--released")
    fetching.set_defaults(run=_fetch)

    verifying = commands.add_parser(
        "verify", help="download current files afresh and compare them with the pins"
    )
    verifying.add_argument("waves", type=int, nargs="*")
    verifying.add_argument("--file", choices=FILES)
    verifying.set_defaults(run=_verify)

    pinning = commands.add_parser(
        "pin", help="print a registry entry for what a URL serves now"
    )
    pinning.add_argument("wave", type=int)
    pinning.add_argument("file", choices=FILES)
    pinning.add_argument("--released", required=True, help="Fed 'Last updated' date")
    pinning.add_argument("--url", help=f"defaults to the registry's, under {_BASE_URL}")
    pinning.set_defaults(run=_pin)

    arguments = parser.parse_args(argv)
    if arguments.command == "pin" and arguments.url is None:
        arguments.url = artifact(arguments.wave, arguments.file).locator
    return int(arguments.run(arguments))


if __name__ == "__main__":
    sys.exit(main())
