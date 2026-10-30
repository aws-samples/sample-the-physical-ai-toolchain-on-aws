"""Install missing notebook client tools on Linux x86_64, without sudo.

Installed tools and temporary payloads stay on the data volume beside the clone.
Long temporary paths use a short private alias for Terraform's Unix sockets.
Training images and model dependencies are built separately by CodeBuild.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile

COMPONENT = Path(__file__).resolve().parents[1]
CLIENT = COMPONENT / "local-dev/notebook-client"
TERRAFORM = (
    "https://releases.hashicorp.com/terraform/1.9.8/terraform_1.9.8_linux_amd64.zip",
    "186e0145f5e5f2eb97cbd785bc78f21bae4ef15119349f6ad4fa535b83b10df8",
)
GIT_LFS = (
    "https://github.com/git-lfs/git-lfs/releases/download/v3.6.1/git-lfs-linux-amd64-v3.6.1.tar.gz",
    "2138d2e405a12f1a088272e06790b76699b79cb90d0317b77aafaf35de908d76",
)
AWS_CLI = (
    "https://awscli.amazonaws.com/awscli-exe-linux-x86_64-2.31.17.zip",
    "1222943e395cabb2a6b987cd75d0f59d566aa40bbdbab6e55232ee7272900d7f",
)


def run(*args, **kwargs):
    """Run a setup command, stopping immediately if it fails."""
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def available(*args, prefix=None):
    """Check an existing tool without changing its installation."""
    if not shutil.which(args[0]):
        return False
    result = subprocess.run(args, capture_output=True, text=True)
    return result.returncode == 0 and (prefix is None or result.stdout.startswith(prefix))


def terraform_ready():
    """Accept an existing Terraform only if it meets the application's minimum."""
    if not available("terraform", "version", "-json"):
        return False
    version = json.loads(subprocess.check_output(["terraform", "version", "-json"], text=True))
    return tuple(map(int, version["terraform_version"].split(".")[:2])) >= (1, 9)


def download(url, checksum, destination):
    """Write an official release archive only after its pinned SHA256 matches."""
    digest = hashlib.sha256()
    partial = destination.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as source, partial.open("wb") as target:
            while block := source.read(1024 * 1024):
                digest.update(block)
                target.write(block)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Release checksum mismatch: {url}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def executable(path, content):
    """Install one verified executable in the notebook client's own bin folder."""
    path.write_bytes(content)
    path.chmod(0o755)


def socket_safe_temporary_directory(directory):
    """Return a short path to temporary storage, keeping payloads on the data volume."""
    directory.mkdir(parents=True, exist_ok=True)
    # Leave room for Terraform's plugin socket name on Linux and macOS.
    if len(os.fsencode(directory)) <= 80:
        return directory
    current = Path(os.environ.get("TMPDIR", ""))
    if (current.is_symlink() and len(os.fsencode(current)) <= 80
            and current.resolve() == directory.resolve()):
        return current
    alias = Path(tempfile.mkdtemp(prefix="pai-tmp-", dir="/tmp")) / "data"
    alias.symlink_to(directory.resolve(), target_is_directory=True)
    return alias


def prepare_storage(component=COMPONENT):
    """Put notebook state, caches and temporary files on its persistent data volume."""
    client = component / "local-dev/notebook-client"
    for key, path in {
        "VLA_STATE_DIR": component / "local-dev/arena-state",
        "PIP_CACHE_DIR": client / "pip-cache",
        "TMPDIR": socket_safe_temporary_directory(client / "tmp"),
    }.items():
        if key == "VLA_STATE_DIR":
            os.environ.setdefault(key, str(path))
        else:
            os.environ[key] = str(path)
        Path(os.environ[key]).expanduser().mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = None
    return client


def setup():
    """Install missing client tools; leave existing system installations unchanged."""
    client = prepare_storage()
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        print("Automatic setup supports Linux x86_64 AWS notebooks. Other clients need the documented tools installed.")
        return
    binary = client / "bin"
    binary.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = str(binary) + os.pathsep + os.environ["PATH"]
    if not available("git", "--version") or not available("bash", "--version"):
        raise RuntimeError("Use the standard SageMaker notebook environment with Git and Bash installed.")

    if not available("python3.11", "--version", prefix="Python 3.11."):
        import sys

        requirements = client / "bootstrap-requirements.txt"
        requirements.write_text(
            "uv==0.9.5 --hash=sha256:6507bbbcd788553ec4ad5a96fa19364dc0f58b023e31d79868773559a83ec181\n"
        )
        run(sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--no-deps",
            "--require-hashes", "--target", client / "uv", "-r", requirements)
        uv = client / "uv/bin/uv"
        os.environ["UV_PYTHON_INSTALL_DIR"] = str(client / "python")
        os.environ["UV_PYTHON_BIN_DIR"] = str(binary)
        os.environ["UV_CACHE_DIR"] = str(client / "uv-cache")
        run(uv, "python", "install", "3.11.14")
        interpreter = subprocess.check_output(
            [str(uv), "python", "find", "--managed-python", "3.11.14"], text=True).strip()
        if not (binary / "python3.11").exists():
            (binary / "python3.11").symlink_to(interpreter)

    # CPython's venv records the invoked executable's directory as its home.
    # Invoking a relocated symlink would make the new venv lose its stdlib.
    # Put the actual installation's bin directory first before `python -m venv`.
    python_bin = Path(shutil.which("python3.11")).resolve().parent
    os.environ["PATH"] = str(python_bin) + os.pathsep + os.environ["PATH"]

    with tempfile.TemporaryDirectory(dir=client) as temporary:
        temporary = Path(temporary)
        if not terraform_ready():
            print("Installing Terraform 1.9.8 on the notebook volume.", flush=True)
            archive = temporary / "terraform.zip"
            download(*TERRAFORM, archive)
            with zipfile.ZipFile(archive) as release:
                executable(binary / "terraform", release.read("terraform"))
        if not available("git", "lfs", "version"):
            print("Installing Git LFS 3.6.1 on the notebook volume.", flush=True)
            archive = temporary / "lfs.tar.gz"
            download(*GIT_LFS, archive)
            with tarfile.open(archive) as release:
                members = [member for member in release if member.isfile()
                           and Path(member.name).name == "git-lfs"]
                if len(members) != 1:
                    raise ValueError("Expected one Git LFS executable in the official release")
                executable(binary / "git-lfs", release.extractfile(members[0]).read())
        if not available("aws", "--version", prefix="aws-cli/2."):
            print("Installing AWS CLI v2 on the notebook volume.", flush=True)
            archive = temporary / "aws.zip"
            download(*AWS_CLI, archive)
            with zipfile.ZipFile(archive) as release:
                for name in release.namelist():
                    if not (temporary / name).resolve().is_relative_to(temporary.resolve()):
                        raise ValueError("Unsafe path in AWS CLI release archive")
                release.extractall(temporary)
            # Zip extraction does not preserve the POSIX executable bits.
            for path in (temporary / "aws").rglob("*"):
                if path.is_file():
                    path.chmod(path.stat().st_mode | 0o100)
            run("bash", temporary / "aws/install", "--install-dir", client / "aws-cli",
                "--bin-dir", binary, "--update")
    print("Notebook client tools ready:", binary)
    print("Operation records and downloaded evidence:", os.environ["VLA_STATE_DIR"])


if __name__ == "__main__":
    setup()
