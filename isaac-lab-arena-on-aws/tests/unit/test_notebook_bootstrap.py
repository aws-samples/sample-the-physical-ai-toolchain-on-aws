"""Check the notebook's storage boundary and release integrity before cloud work."""
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import socket
import tempfile
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/setup_notebook.py"
spec = importlib.util.spec_from_file_location("setup_notebook", SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


@pytest.fixture(autouse=True)
def isolated_storage_environment(monkeypatch):
    for key in ("VLA_STATE_DIR", "PIP_CACHE_DIR", "TMPDIR"):
        monkeypatch.delenv(key, raising=False)
    yield
    temporary = Path(os.environ.get("TMPDIR", ""))
    if temporary.is_symlink() and temporary.parent.name.startswith("pai-tmp-"):
        temporary.unlink()
        temporary.parent.rmdir()
    tempfile.tempdir = None


def test_state_cache_and_temporary_files_use_notebook_volume(tmp_path, monkeypatch):
    for key in ("VLA_STATE_DIR", "PIP_CACHE_DIR", "TMPDIR"):
        monkeypatch.delenv(key, raising=False)
    setup.prepare_storage(tmp_path)
    for key in ("VLA_STATE_DIR", "PIP_CACHE_DIR", "TMPDIR"):
        path = Path(os.environ[key])
        assert path.resolve().is_relative_to(tmp_path.resolve())
        assert path.is_dir()


def test_long_checkout_can_bind_provider_socket_and_keep_payload_on_notebook_volume(tmp_path):
    component = tmp_path / ("long-checkout-" * 8) / "isaac-lab-arena-on-aws"
    setup.prepare_storage(component)
    temporary = Path(os.environ["TMPDIR"])
    socket_path = temporary / "plugin3909839755"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        assert socket_path.resolve().is_relative_to(component.resolve())
    socket_path.unlink()
    with tempfile.NamedTemporaryFile() as payload:
        assert Path(payload.name).resolve().is_relative_to(component.resolve())
    # The notebook calls prepare_storage again when installing tools.
    setup.prepare_storage(component)
    assert Path(os.environ["TMPDIR"]) == temporary


def test_explicit_state_directory_is_preserved(tmp_path, monkeypatch):
    supplied = tmp_path / "supplied-state"
    monkeypatch.setenv("VLA_STATE_DIR", str(supplied))
    monkeypatch.setenv("PIP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    setup.prepare_storage(tmp_path / "clone")
    assert Path(os.environ["VLA_STATE_DIR"]) == supplied
    assert supplied.is_dir()


def test_release_is_not_installed_when_hash_is_wrong(tmp_path):
    destination = tmp_path / "release.zip"
    with patch.object(setup.urllib.request, "urlopen", return_value=io.BytesIO(b"changed")):
        with pytest.raises(ValueError, match="checksum mismatch"):
            setup.download("https://example.invalid/release", "0" * 64, destination)
    assert not destination.exists()
    assert not destination.with_suffix(".partial").exists()


def test_verified_release_bytes_are_preserved(tmp_path):
    data = b"verified release bytes"
    destination = tmp_path / "release.zip"
    with patch.object(setup.urllib.request, "urlopen", return_value=io.BytesIO(data)):
        setup.download("https://example.invalid/release", hashlib.sha256(data).hexdigest(), destination)
    assert destination.read_bytes() == data


@pytest.mark.parametrize(("version", "ready"), [("1.8.5", False), ("1.9.8", True), ("1.14.0", True)])
def test_terraform_minimum_version_is_enforced(version, ready):
    with patch.object(setup, "available", return_value=True), patch.object(
        setup.subprocess, "check_output", return_value='{"terraform_version":"' + version + '"}'
    ):
        assert setup.terraform_ready() is ready
