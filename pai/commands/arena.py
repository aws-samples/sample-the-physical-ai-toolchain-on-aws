"""Expose Arena's existing parser and execution backend under the shared CLI."""

from __future__ import annotations

import click


@click.command(
    name="arena",
    add_help_option=False,
    context_settings={"ignore_unknown_options": True},
    help="Train, evaluate and verify VLA policies locally or through SageMaker.",
)
@click.argument("arguments", nargs=-1, type=click.UNPROCESSED)
@click.pass_context
def arena(ctx: click.Context, arguments: tuple[str, ...]) -> None:
    """Let Arena own argument parsing, including help and validation errors."""
    try:
        from vla_pipeline.cli import main
    except ModuleNotFoundError as exc:
        if exc.name != "vla_pipeline":
            raise click.ClickException(
                f"Arena dependency {exc.name!r} is missing. From the complete toolchain "
                "checkout, install both packages with: "
                "python -m pip install -e . -e ./isaac-lab-arena-on-aws"
            ) from exc
        raise click.ClickException(
            "Arena is not installed. From the complete toolchain checkout, run: "
            "python -m pip install -e . -e ./isaac-lab-arena-on-aws"
        ) from exc
    ctx.exit(main(list(arguments), prog="pai arena"))


def register(cli: click.Group) -> None:
    cli.add_command(arena)
