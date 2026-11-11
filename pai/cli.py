"""pai CLI — root command group and subcommand registration.

Entry point: main() (wired in pyproject.toml as `pai = "pai.cli:main"`).

Command registration contract:
    Each pai/commands/*.py module must expose a register(cli: click.Group) function.
    The registration is wrapped in try/except ImportError so the CLI group loads
    even if a command module is not yet implemented.

Keep `pai --help` fast: no torch/isaacsim imports at module top. Subcommands
import heavy modules inside their function bodies (lazy-import).
"""

import click

from pai import __version__


@click.group()
@click.version_option(version=__version__, prog_name="pai")
def cli():
    """pai — AWS Physical AI Toolchain CLI.

    Train and deploy robot manipulation policies on AWS + NVIDIA:
    GR00T imitation learning, Isaac Lab RL refinement, Cosmos enhancement,
    and edge deployment (Jetson/Greengrass).
    """
    pass


def main():
    """Entry point for setuptools console_scripts."""
    # Register all command modules. Each module must export register(cli).
    # Wrapped in try/except ImportError so the group loads if a module isn't built yet.

    try:
        from pai.commands import doctor
        doctor.register(cli)
    except ImportError:
        pass

    try:
        from pai.commands import config_cmd
        config_cmd.register(cli)
    except ImportError:
        pass

    try:
        from pai.commands import deploy
        deploy.register(cli)
    except ImportError:
        pass

    try:
        from pai.commands import workstation
        workstation.register(cli)
    except ImportError:
        pass

    try:
        from pai.commands import groot
        groot.register(cli)
    except ImportError:
        pass

    try:
        from pai.commands import env
        env.register(cli)
    except ImportError:
        pass

    try:
        from pai.commands import rl
        rl.register(cli)
    except ImportError:
        pass

    # NOTE: there is intentionally no `pai eval` command. Policy evaluation
    # (open-loop and closed-loop) runs IN-PROCESS with Isaac Lab on the GPU
    # workstation, which does not have `pai` installed (the CLI is the laptop-side
    # control plane). Eval runs via training/scripts/eval_*.py directly there.

    try:
        from pai.commands import export
        export.register(cli)
    except ImportError:
        pass

    try:
        from pai.commands import agent
        agent.register(cli)
    except ImportError:
        pass

    cli()


if __name__ == "__main__":
    main()
