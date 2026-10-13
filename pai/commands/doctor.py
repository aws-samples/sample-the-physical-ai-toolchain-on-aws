"""pai doctor — Preflight checks for AWS credentials, region, permissions, and infrastructure."""

import sys

import boto3
import click
from botocore.exceptions import BotoCoreError, ClientError

from pai import config, helpers


@click.command()
def doctor():
    """Run preflight checks for credentials, region, permissions, and infrastructure.

    Checks:
    1. AWS credentials are resolvable (sts.get_caller_identity)
    2. Region is configured (warn if not us-west-2)
    3. PassRole permission likely available (heuristic, no actual call)
    4. isaac-lab image exists in ECR
    5. Foundation stack exists
    """
    all_pass = True
    region = config.resolve_region()

    helpers.heading("AWS Physical AI Toolchain — Preflight Check")

    # Check 1: AWS credentials
    helpers.info("\n[1/5] Checking AWS credentials...")
    try:
        sts = boto3.client("sts", region_name=region)
        identity = sts.get_caller_identity()
        account = identity["Account"]
        arn = identity["Arn"]
        helpers.pass_msg(f"AWS credentials valid (Account: {account})")
        helpers.info(f"        Identity: {arn}")
    except (BotoCoreError, ClientError) as e:
        all_pass = False
        helpers.fail_msg("AWS credentials not configured or expired")
        helpers.info("        Remediation: Run 'aws configure' or refresh your session")
        account = None  # will need later for PassRole check

    # Check 2: Region
    helpers.info("\n[2/5] Checking AWS region...")
    if region:
        helpers.pass_msg(f"Region resolved: {region}")
        if region != "us-west-2":
            helpers.warn("        WARNING: The validated region for this toolchain is us-west-2.")
            helpers.warn("        ECR images and Secrets Manager keys are in us-west-2.")
            helpers.warn("        You may encounter missing resources in other regions.")
    else:
        all_pass = False
        helpers.fail_msg("Region not configured")
        helpers.info("        Remediation: Set AWS_DEFAULT_REGION or run 'aws configure'")

    # Check 3: PassRole permission (heuristic — do NOT call PassRole)
    helpers.info("\n[3/5] Checking iam:PassRole permission (best-effort)...")
    if account:
        try:
            iam = boto3.client("iam", region_name=region)
            caller_user = arn.split("/")[-1] if "/" in arn else None

            # Heuristic: if caller is the workstation role, warn
            # The minimal workstation role typically has SSM/EC2/S3 but NOT iam:PassRole
            if "physical-ai" in arn.lower() and "workstation" in arn.lower():
                helpers.warn("        WARNING: Your identity looks like the minimal workstation role.")
                helpers.warn("        The workstation role typically lacks iam:PassRole for SageMaker.")
                helpers.warn("        Launch jobs from a context with PassRole (e.g., your laptop admin creds or CloudShell).")
                helpers.info(f"\n        Required policy for your IAM user/role:")
                helpers.info("        {")
                helpers.info('          "Effect": "Allow",')
                helpers.info('          "Action": "iam:PassRole",')
                helpers.info(f'          "Resource": "arn:aws:iam::{account}:role/physical-ai-dev-sagemaker-role"')
                helpers.info("        }")
            else:
                # We can't actually verify PassRole without calling it (and PassRole on wrong resources fails).
                # Best we can do: assume it's okay if not the workstation role.
                helpers.pass_msg("Identity does not appear to be the restricted workstation role")
                helpers.info("        Note: PassRole is required to launch SageMaker/Batch jobs.")
                helpers.info("        If launches fail with AccessDenied, attach the PassRole policy above.")
        except Exception:
            # IAM calls may fail (credentials issues, etc); don't fail the whole check
            helpers.warn("        Could not verify PassRole heuristic (non-fatal)")

    # Check 4: isaac-lab image in ECR
    helpers.info("\n[4/5] Checking isaac-lab image in ECR...")
    if account and region:
        try:
            ecr = boto3.client("ecr", region_name=region)
            repo_name = "physical-ai/isaac-lab"
            response = ecr.describe_images(
                repositoryName=repo_name,
                imageIds=[{"imageTag": "latest"}],
            )
            if response["imageDetails"]:
                helpers.pass_msg(f"Image found: {account}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest")
            else:
                all_pass = False
                helpers.fail_msg(f"Image not found: {repo_name}:latest")
                helpers.info("        Remediation: Deploy Foundation stack and wait for CodeBuild to complete")
                helpers.info("        Check build status: aws codebuild list-builds --region {region}")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "RepositoryNotFoundException":
                all_pass = False
                helpers.fail_msg(f"ECR repository not found: {repo_name}")
                helpers.info("        Remediation: Deploy Foundation stack: pai deploy foundation")
            else:
                all_pass = False
                helpers.fail_msg(f"Failed to check ECR image: {e.response.get('Error', {}).get('Message', str(e))}")
    else:
        helpers.warn("        Skipping (credentials/region check failed)")

    # Check 5: Foundation infrastructure exists (SSM params or CFN stack)
    helpers.info("\n[5/5] Checking Foundation infrastructure...")
    if region:
        try:
            ssm = boto3.client("ssm", region_name=region)
            ssm.get_parameter(Name="/physical-ai/datasets-bucket")
            helpers.pass_msg("Foundation infrastructure detected (SSM parameters present)")
        except Exception:
            # Fall back to checking CFN stacks: the CDK stack (pai deploy foundation),
            # then the CloudFormation bootstrap stack (workshop/bootstrap.cfn.yaml).
            cfn_client = boto3.client("cloudformation", region_name=region)
            env = config.get("environment", "dev")
            found = None
            for stack_name in (f"PhysicalAi-{env}-Foundation", "physical-ai-foundation"):
                try:
                    stacks = cfn_client.describe_stacks(StackName=stack_name).get("Stacks", [])
                except ClientError:
                    continue
                if stacks and "COMPLETE" in stacks[0].get("StackStatus", ""):
                    found = stack_name
                    break
            if found:
                helpers.pass_msg(f"Foundation stack exists: {found}")
            else:
                all_pass = False
                helpers.fail_msg("Foundation infrastructure not found")
                helpers.info("        Remediation: Deploy foundation: pai deploy foundation")
                helpers.info("          (or the CloudFormation path: aws cloudformation deploy --template-file workshop/bootstrap.cfn.yaml \\")
                helpers.info("            --stack-name physical-ai-foundation --capabilities CAPABILITY_NAMED_IAM)")
    else:
        helpers.warn("        Skipping (region check failed)")

    # Summary
    helpers.heading("\nPreflight Check Summary")
    if all_pass:
        helpers.success("✅ All checks passed — ready to launch jobs!")
    else:
        helpers.error("❌ Some checks failed — fix issues above before launching jobs")
        sys.exit(1)


def register(cli: click.Group):
    """Register doctor command with the CLI."""
    cli.add_command(doctor)
