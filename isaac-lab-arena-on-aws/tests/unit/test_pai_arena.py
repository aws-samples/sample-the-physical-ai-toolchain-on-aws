"""The shared CLI forwards Arena's interface without changing argument semantics."""
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys

from click.testing import CliRunner
import pytest

from pai.commands.arena import arena


@pytest.mark.parametrize("arguments", [
    ["--help"], ["run", "--help"], ["deploy", "--help"], ["destroy", "--help"],
])
def test_shared_help_uses_the_command_the_user_ran(arguments):
    result = CliRunner().invoke(arena, arguments)
    assert result.exit_code == 0, result.output
    assert "usage: pai arena" in result.output
    assert "usage: vla " not in result.output


def test_catalog_is_identical_through_both_installed_commands(tmp_path):
    bin_dir = Path(sys.executable).parent
    env = dict(os.environ, AWS_EC2_METADATA_DISABLED="true")
    values = []
    for command in ([str(bin_dir / "pai"), "arena"], [str(bin_dir / "vla")]):
        output = subprocess.run(
            [*command, "--state-dir", str(tmp_path), "--json", "cells", "--details"],
            capture_output=True, text=True, env=env, check=True,
        )
        values.append(json.loads(output.stdout))
    assert values[0] == values[1]
    assert "gr00t-n16-arena" in json.dumps(values[0])


def test_missing_arena_package_reports_the_install_command(monkeypatch):
    original = builtins.__import__

    def missing_arena(name, *args, **kwargs):
        if name.startswith("vla_pipeline"):
            raise ModuleNotFoundError("Arena is not installed", name="vla_pipeline")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_arena)
    result = CliRunner().invoke(arena, ["cells"])
    assert result.exit_code == 1
    assert "python -m pip install -e . -e ./isaac-lab-arena-on-aws" in result.output


def test_unknown_arena_option_remains_an_error():
    result = CliRunner().invoke(arena, ["run", "--mode", "local", "--made-up-option"])
    assert result.exit_code == 2
    assert "unrecognized arguments: --made-up-option" in result.output
