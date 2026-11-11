"""pai env — Print shell exports for the lab's raw commands.

The `pai` commands resolve the bucket, role, and image from the Foundation stack
on every call, but the lab's raw `aws ...` commands read shell variables that are
lost whenever a new terminal is opened. Run this once per terminal:

    eval "$(pai env)"

Exports AWS_REGION/AWS_DEFAULT_REGION (the toolchain region, so raw `aws`
commands hit the same region as `pai`), BUCKET, ROLE_ARN, ECR_URI, and — when a
GR00T training job exists — JOB_NAME and MODEL_S3 for the most recent one.
Never prints secrets.
"""

import shlex

import boto3
import click
from botocore.exceptions import BotoCoreError, ClientError

from pai import cfn, config

STACK_NAME = "PhysicalAi-dev-Foundation"


def _latest_groot_job(region: str) -> tuple[str, str]:
    """Return (job name, model S3 URI) of the newest GR00T training job, or ("", "")."""
    sm = boto3.client("sagemaker", region_name=region)
    jobs = sm.list_training_jobs(
        NameContains="TrainGR00T", SortBy="CreationTime", SortOrder="Descending", MaxResults=1,
    ).get("TrainingJobSummaries", [])
    if not jobs:
        return "", ""
    name = jobs[0]["TrainingJobName"]
    desc = sm.describe_training_job(TrainingJobName=name)
    return name, desc.get("ModelArtifacts", {}).get("S3ModelArtifacts", "")


@click.command(name="env")
def env():
    """Print `export` lines for the lab's raw commands: eval "$(pai env)"."""
    region = config.resolve_region()
    values = {
        "AWS_REGION": region,
        "AWS_DEFAULT_REGION": region,
        "BUCKET": cfn.bucket(STACK_NAME, region),
        "ROLE_ARN": cfn.role_arn(STACK_NAME, region),
        "ECR_URI": cfn.ecr_uri(STACK_NAME, region, repo="gr00t-training"),
    }
    try:
        values["JOB_NAME"], values["MODEL_S3"] = _latest_groot_job(region)
    except (BotoCoreError, ClientError):
        values["JOB_NAME"], values["MODEL_S3"] = "", ""

    for key, value in values.items():
        if value:
            click.echo(f"export {key}={shlex.quote(value)}")
        else:
            # Comments keep `eval` safe while telling the user what's missing.
            click.echo(f"# {key}: not found (Foundation stack deployed? training job run yet?)")


def register(cli: click.Group):
    """Register env command with the CLI."""
    cli.add_command(env)
