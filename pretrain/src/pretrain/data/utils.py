"""File system and dataset utility helpers."""

from __future__ import annotations

import hashlib
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def ensure_directory(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, checksum: Optional[str] = None, chunk_size: int = 8192) -> Path:
    """Download file if it does not exist, optionally validating checksum."""

    destination = Path(destination)
    ensure_directory(destination.parent)
    if destination.exists():
        LOGGER.info("File %s already exists; skipping download", destination)
        if checksum:
            _validate_checksum(destination, checksum)
        return destination

    LOGGER.info("Downloading %s to %s", url, destination)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(destination, "wb") as fp, tqdm(total=int(response.headers.get("content-length", 0)), unit="B", unit_scale=True, desc=destination.name) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fp.write(chunk)
                pbar.update(len(chunk))
    if checksum:
        _validate_checksum(destination, checksum)
    return destination


def extract_archive(archive_path: Path, destination: Path) -> Path:
    """Extract tar or zip archives."""

    archive_path = Path(archive_path)
    destination = Path(destination)
    ensure_directory(destination)
    LOGGER.info("Extracting %s to %s", archive_path, destination)
    if archive_path.suffix in {".zip"}:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(destination)
    elif archive_path.suffix in {".gz", ".tgz", ".bz2"} or archive_path.suffixes[-2:] in [(".tar", ".gz"), (".tar", ".bz2")]:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(destination)
    else:
        raise ValueError(f"Unsupported archive type: {archive_path}")
    return destination


def _validate_checksum(path: Path, checksum: str) -> None:
    sha256 = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    if digest != checksum:
        raise ValueError(f"Checksum mismatch for {path}. Expected {checksum}, got {digest}")


__all__ = [
    "ensure_directory",
    "download_file",
    "extract_archive",
]
