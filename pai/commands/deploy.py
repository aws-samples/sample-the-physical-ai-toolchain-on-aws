"""Deploy and destroy lifecycle commands for CDK stacks.

Wraps `npx cdk deploy/destroy` for Foundation, Workstation, and Batch stacks.
Shells out to cdk and deploy-workstation.sh — does not duplicate logic.

The --dry-run flag prints the exact command without executing (for safe testing).
"""

from __future__ import annotations

import os
import subprocess

import click

from pai import config, helpers, cfn


@click.group()
def deploy():
    """Deploy infrastructure stacks (Foundation, Workstation, Batch)."""
    pass


@click.group()
def destroy():
    """Destroy infrastructure stacks."""
    pass


def register(cli: click.Group) -> None:
    """Register deploy and destroy commands."""
    cli.add_command(deploy)
    cli.add_command(destroy)


# ============================================================================
# Deploy subcommands
# ============================================================================


@deploy.command(name="foundation")
@click.option("--no-approval", is_flag=True, help="Pass --require-approval never: deploy IAM/security-group changes WITHOUT showing them for review (for non-interactive shells)")
@click.option("--dry-run", is_flag=True, help="Print command without executing")
def deploy_foundation(no_approval: bool, dry_run: bool):
    """Deploy the Foundation stack (ECR, S3, SageMaker role, Batch infra)."""
    env = config.get("environment", "dev")
    stack_name = f"PhysicalAi-{env}-Foundation"
    cdk_dir = str(config.REPO_ROOT / "cdk")

    cmd = ["npx", "cdk", "deploy", stack_name, "--context", "mode=simple"]
    if no_approval:
        cmd += ["--require-approval", "never"]

    if dry_run:
        helpers.info(f"[dry-run] Would execute in {cdk_dir}:")
        helpers.info(f"  {' '.join(cmd)}")
        return

    helpers.heading(f"Deploying {stack_name}")
    try:
        helpers.run(cmd, cwd=cdk_dir, check=True)
        helpers.success(f"\n{stack_name} deployed successfully.")
    except subprocess.CalledProcessError as e:
        helpers.error(f"Deploy failed with exit code {e.returncode}")
        raise click.Abort()


@deploy.command(name="workstation")
@click.option("--instance-type", help="EC2 instance type (e.g., g6e.4xlarge)")
@click.option("--allowed-cidr", help="CIDR allowed to connect via DCV (e.g., 1.2.3.4/32)")
@click.option("--dry-run", is_flag=True, help="Print command without executing")
def deploy_workstation(instance_type: str | None, allowed_cidr: str | None, dry_run: bool):
    """Deploy the Workstation stack (Isaac Sim GPU instance with DCV).

    This wraps cdk/deploy-workstation.sh, which auto-detects your IP and retries
    across AZs on InsufficientInstanceCapacity errors.
    """
    script_path = str(config.REPO_ROOT / "cdk" / "deploy-workstation.sh")

    # Build env dict from os.environ (full parent env) + overrides
    env = os.environ.copy()
    if instance_type:
        env["INSTANCE_TYPE"] = instance_type
    if allowed_cidr:
        env["ALLOWED_CIDR"] = allowed_cidr

    cmd = ["bash", script_path]

    if dry_run:
        helpers.info("[dry-run] Would execute:")
        helpers.info(f"  {' '.join(cmd)}")
        if instance_type:
            helpers.info(f"    INSTANCE_TYPE={instance_type}")
        if allowed_cidr:
            helpers.info(f"    ALLOWED_CIDR={allowed_cidr}")
        return

    helpers.heading("Deploying Workstation")
    try:
        helpers.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        helpers.error(f"Workstation deploy failed with exit code {e.returncode}")
        raise click.Abort()


@deploy.command(name="batch")
@click.option("--no-approval", is_flag=True, help="Pass --require-approval never: deploy IAM/security-group changes WITHOUT showing them for review (for non-interactive shells)")
@click.option("--dry-run", is_flag=True, help="Print command without executing")
def deploy_batch(no_approval: bool, dry_run: bool):
    """Deploy the Batch stack (AWS Batch compute environment for multi-node RL).

    Pre-checks:
      - Foundation stack must exist
      - isaac-lab ECR image must exist (built by Foundation)
    """
    env_name = config.get("environment", "dev")
    stack_name = f"PhysicalAi-{env_name}-Batch"
    foundation_stack = f"PhysicalAi-{env_name}-Foundation"
    cdk_dir = str(config.REPO_ROOT / "cdk")
    region = config.resolve_region()

    # Pre-check: Foundation stack
    foundation_status = cfn.stack_status(foundation_stack, region)
    if not foundation_status or "COMPLETE" not in foundation_status:
        helpers.warn(f"Foundation stack {foundation_stack} not found or incomplete.")
        helpers.warn("Deploy Foundation first: pai deploy foundation")
        if not dry_run:
            raise click.Abort()

    # Pre-check: isaac-lab image (best-effort warning, not a blocker)
    # The image is built by CodeBuild in the Foundation stack. Check if ECR repo exists.
    # (We don't query ECR for the actual image — that requires additional perms. Trust
    # that if the Foundation stack deployed, the CodeBuild ran. This is a best-effort hint.)
    if foundation_status:
        helpers.info("Foundation stack exists. Assuming isaac-lab image is built.")

    cmd = ["npx", "cdk", "deploy", stack_name, "--context", "batch=true"]
    if no_approval:
        cmd += ["--require-approval", "never"]

    if dry_run:
        helpers.info(f"[dry-run] Would execute in {cdk_dir}:")
        helpers.info(f"  {' '.join(cmd)}")
        return

    helpers.heading(f"Deploying {stack_name}")
    try:
        helpers.run(cmd, cwd=cdk_dir, check=True)
        helpers.success(f"\n{stack_name} deployed successfully.")
    except subprocess.CalledProcessError as e:
        helpers.error(f"Deploy failed with exit code {e.returncode}")
        raise click.Abort()


# ============================================================================
# Destroy subcommands
# ============================================================================


@destroy.command(name="foundation")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("--dry-run", is_flag=True, help="Print command without executing")
def destroy_foundation(yes: bool, dry_run: bool):
    """Destroy the Foundation stack."""
    env = config.get("environment", "dev")
    stack_name = f"PhysicalAi-{env}-Foundation"
    cdk_dir = str(config.REPO_ROOT / "cdk")

    cmd = ["npx", "cdk", "destroy", stack_name]

    if dry_run:
        helpers.info(f"[dry-run] Would execute in {cdk_dir}:")
        helpers.info(f"  {' '.join(cmd)}")
        helpers.info("  (with confirmation prompt unless --yes)")
        return

    if not yes:
        if not click.confirm(f"Destroy {stack_name}? This will delete all resources."):
            helpers.info("Aborted.")
            return

    helpers.heading(f"Destroying {stack_name}")
    try:
        helpers.run(cmd, cwd=cdk_dir, check=True)
        helpers.success(f"\n{stack_name} destroyed.")
    except subprocess.CalledProcessError as e:
        helpers.error(f"Destroy failed with exit code {e.returncode}")
        raise click.Abort()


@destroy.command(name="workstation")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("--dry-run", is_flag=True, help="Print command without executing")
def destroy_workstation(yes: bool, dry_run: bool):
    """Destroy the Workstation stack."""
    env = config.get("environment", "dev")
    stack_name = f"PhysicalAi-{env}-Workstation"
    cdk_dir = str(config.REPO_ROOT / "cdk")

    cmd = ["npx", "cdk", "destroy", stack_name]

    if dry_run:
        helpers.info(f"[dry-run] Would execute in {cdk_dir}:")
        helpers.info(f"  {' '.join(cmd)}")
        helpers.info("  (with confirmation prompt unless --yes)")
        return

    if not yes:
        if not click.confirm(f"Destroy {stack_name}? This will terminate the workstation instance."):
            helpers.info("Aborted.")
            return

    helpers.heading(f"Destroying {stack_name}")
    try:
        helpers.run(cmd, cwd=cdk_dir, check=True)
        helpers.success(f"\n{stack_name} destroyed.")
    except subprocess.CalledProcessError as e:
        helpers.error(f"Destroy failed with exit code {e.returncode}")
        raise click.Abort()


@destroy.command(name="batch")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
@click.option("--dry-run", is_flag=True, help="Print command without executing")
def destroy_batch(yes: bool, dry_run: bool):
    """Destroy the Batch stack."""
    env = config.get("environment", "dev")
    stack_name = f"PhysicalAi-{env}-Batch"
    cdk_dir = str(config.REPO_ROOT / "cdk")

    cmd = ["npx", "cdk", "destroy", stack_name]

    if dry_run:
        helpers.info(f"[dry-run] Would execute in {cdk_dir}:")
        helpers.info(f"  {' '.join(cmd)}")
        helpers.info("  (with confirmation prompt unless --yes)")
        return

    if not yes:
        if not click.confirm(f"Destroy {stack_name}? This will delete the Batch compute environment."):
            helpers.info("Aborted.")
            return

    helpers.heading(f"Destroying {stack_name}")
    try:
        helpers.run(cmd, cwd=cdk_dir, check=True)
        helpers.success(f"\n{stack_name} destroyed.")
    except subprocess.CalledProcessError as e:
        helpers.error(f"Destroy failed with exit code {e.returncode}")
        raise click.Abort()
