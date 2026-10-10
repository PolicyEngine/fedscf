"""Shared fixtures: synthetic Fed-shaped Stata files and a local file server.

The synthetic files copy the layout quirks of the real ones (read on
2026-10-09): 16-bit ``yy1`` with values above 3,276, ``x1``/``xx1`` identifiers
in 1989, upper-case names in older files, and replicate weights that are
missing where a replicate did not draw the family.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import http.server
import io
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fedscf.read import N_IMPLICATES, N_REPLICATES


def synthetic_families(n_families: int = 40, seed: int = 0) -> pd.DataFrame:
    """Stacked implicates: ids, an implicate-varying weight and two variables."""
    rng = np.random.default_rng(seed)
    # Straddle the int16 overflow threshold: yy1 * 10 > 32,767 above 3,276.
    family = np.sort(rng.choice(np.arange(3200, 3400), n_families, replace=False))
    yy1 = np.repeat(family, N_IMPLICATES)
    implicate = np.tile(np.arange(1, N_IMPLICATES + 1), n_families)
    base = np.repeat(rng.uniform(500.0, 50_000.0, n_families), N_IMPLICATES)
    x42001 = base * rng.uniform(0.9, 1.1, yy1.size)
    # Give every implicate the same total, as the Fed's weights have.
    for i in range(1, N_IMPLICATES + 1):
        rows = implicate == i
        x42001[rows] *= 1_000_000.0 / x42001[rows].sum()
    return pd.DataFrame(
        {
            "yy1": yy1.astype(np.int16),
            "y1": (yy1.astype(np.int64) * 10 + implicate).astype(np.int32),
            "x42001": x42001,
            "networth": rng.normal(200_000.0, 400_000.0, yy1.size).round(2),
            "income": rng.lognormal(11.0, 1.0, yy1.size).round(2),
        }
    )


def summary_table(families: pd.DataFrame) -> pd.DataFrame:
    table = families[["yy1", "y1", "networth", "income"]].copy()
    table.insert(2, "wgt", families["x42001"] / N_IMPLICATES)
    return table


def full_table(families: pd.DataFrame, upper: bool = False) -> pd.DataFrame:
    table = families[["yy1", "y1", "x42001"]].copy()
    table["x5729"] = families["income"].round().astype(np.int32)
    table["x3014"] = (families["networth"] > 0).astype(np.int8)
    if upper:
        table.columns = [name.upper() for name in table.columns]
    return table


def replicate_table(families: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """One row per family; missing weight and multiplicity where not drawn."""
    rng = np.random.default_rng(seed)
    first = families.loc[families["y1"] % 10 == 1]
    n = len(first)
    counts = rng.poisson(1.0, (n, N_REPLICATES)).astype(float)
    weights = first["x42001"].to_numpy()[:, None] * rng.uniform(
        0.8, 1.2, (n, N_REPLICATES)
    )
    weights[counts == 0] = np.nan
    counts[counts == 0] = np.nan
    numbers = range(1, N_REPLICATES + 1)
    table = pd.DataFrame(
        np.hstack([weights, counts]),
        columns=[f"wt1b{i}" for i in numbers] + [f"mm{i}" for i in numbers],
    )
    table.insert(0, "yy1", first["yy1"].to_numpy())
    table.insert(0, "y1", first["y1"].to_numpy())
    return table


def to_dta_bytes(table: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    table.to_stata(buffer, write_index=False)
    return buffer.getvalue()


def to_zip_bytes(member: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        # A fixed timestamp keeps the zip bytes, and so its hash, reproducible.
        archive.writestr(zipfile.ZipInfo(member, (2026, 10, 9, 0, 0, 0)), payload)
    return buffer.getvalue()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(scope="session")
def families() -> pd.DataFrame:
    return synthetic_families()


@pytest.fixture
def summary_path(tmp_path: Path, families: pd.DataFrame) -> Path:
    path = tmp_path / "rscfp2099.dta"
    path.write_bytes(to_dta_bytes(summary_table(families)))
    return path


@pytest.fixture
def full_path(tmp_path: Path, families: pd.DataFrame) -> Path:
    path = tmp_path / "p99i6.dta"
    path.write_bytes(to_dta_bytes(full_table(families, upper=True)))
    return path


@pytest.fixture
def replicate_path(tmp_path: Path, families: pd.DataFrame) -> Path:
    path = tmp_path / "p99_rw1.dta"
    path.write_bytes(to_dta_bytes(replicate_table(families)))
    return path


@dataclasses.dataclass
class FileServer:
    """A directory served over HTTP on localhost, with a log of requests."""

    directory: Path
    url: str
    requests: list[tuple[str, str]]

    def put(self, name: str, payload: bytes) -> str:
        """Serve ``payload`` at ``name`` and return its URL."""
        (self.directory / name).write_bytes(payload)
        return f"{self.url}/{name}"

    def remove(self, name: str) -> None:
        (self.directory / name).unlink()


@pytest.fixture(scope="session")
def file_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FileServer]:
    directory = tmp_path_factory.mktemp("served")
    log: list[tuple[str, str]] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            log.append((self.path, self.headers.get("User-Agent", "")))
            super().do_GET()

        def log_message(self, *arguments: object) -> None:  # keep output clean
            pass

    handler = functools.partial(Handler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield FileServer(directory, f"http://127.0.0.1:{server.server_address[1]}", log)
    finally:
        server.shutdown()
        server.server_close()
