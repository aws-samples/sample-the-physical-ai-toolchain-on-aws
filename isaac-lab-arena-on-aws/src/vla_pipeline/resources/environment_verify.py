"""Independent removal checks for the remaining Terraform-owned resource kinds."""
from __future__ import annotations

from botocore.exceptions import ClientError

from .environment_cleanup import resources


def missing(function, arguments, codes):
    try:
        function(**arguments)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in codes:
            raise
    else:
        raise ValueError(f"Owned resource still exists for {function.__name__}: {arguments}")


def check_network_resources_removed(env, owned):
    """AWS no longer describes the owned VPC, subnets, security groups or route tables."""
    ec2, checked = env.client("ec2"), []
    for kind, function, argument, code in [
        ("aws_vpc", ec2.describe_vpcs, "VpcIds", "InvalidVpcID.NotFound"),
        ("aws_subnet", ec2.describe_subnets, "SubnetIds", "InvalidSubnetID.NotFound"),
        ("aws_security_group", ec2.describe_security_groups, "GroupIds", "InvalidGroup.NotFound"),
        ("aws_route_table", ec2.describe_route_tables, "RouteTableIds", "InvalidRouteTableID.NotFound"),
        ("aws_internet_gateway", ec2.describe_internet_gateways, "InternetGatewayIds", "InvalidInternetGatewayID.NotFound"),
    ]:
        for item in resources(owned, kind):
            missing(function, {argument: [item["id"]]}, {code})
            checked.append(item["id"])
    return checked


def check_instance_profiles_and_policies_removed(env, owned):
    """The owned IAM instance profiles and managed policies are gone."""
    iam, checked = env.client("iam"), []
    for item in resources(owned, "aws_iam_instance_profile"):
        missing(iam.get_instance_profile, {"InstanceProfileName": item["name"]}, {"NoSuchEntity"})
        checked.append(item["name"])
    for item in resources(owned, "aws_iam_policy"):
        missing(iam.get_policy, {"PolicyArn": item["arn"]}, {"NoSuchEntity"})
        checked.append(item["arn"])
    return checked


def check_owned_image_repositories_removed(env, owned):
    """Newly created ECR repositories are gone; supplied shared repositories are outside this state."""
    ecr, checked = env.client("ecr"), []
    for item in resources(owned, "aws_ecr_repository"):
        missing(ecr.describe_repositories, {"repositoryNames": [item["name"]]}, {"RepositoryNotFoundException"})
        checked.append(item["name"])
    return checked


def check_build_projects_removed(env, owned):
    """CodeBuild no longer finds the owned image-build projects."""
    names = [item["name"] for item in resources(owned, "aws_codebuild_project")]
    for name in names:
        reply = env.client("codebuild").batch_get_projects(names=[name])
        if reply["projects"] or reply.get("projectsNotFound") != [name]:
            raise ValueError("Owned CodeBuild project remains: " + name)
    return names


def check_ssm_configuration_removed(env, owned):
    """The owned SSM configuration parameters and documents are gone."""
    ssm, checked = env.client("ssm"), []
    for item in resources(owned, "aws_ssm_parameter"):
        missing(ssm.get_parameter, {"Name": item["name"]}, {"ParameterNotFound"})
        checked.append(item["name"])
    for item in resources(owned, "aws_ssm_document"):
        missing(ssm.describe_document, {"Name": item["name"]}, {"InvalidDocument"})
        checked.append(item["name"])
    return checked


def verify(env, owned):
    # Resource attachments and bucket configuration disappear with their parent.
    # A new resource kind needs a deliberate check, rather than a silent pass.
    supported = {
        "aws_codebuild_project", "aws_ebs_volume", "aws_ecr_repository", "aws_eip",
        "aws_iam_instance_profile", "aws_iam_policy", "aws_iam_role", "aws_iam_role_policy",
        "aws_iam_role_policy_attachment", "aws_instance", "aws_internet_gateway",
        "aws_nat_gateway", "aws_route_table", "aws_route_table_association",
        "aws_s3_bucket", "aws_s3_bucket_lifecycle_configuration", "aws_s3_bucket_policy",
        "aws_s3_bucket_public_access_block", "aws_s3_bucket_server_side_encryption_configuration",
        "aws_s3_bucket_versioning", "aws_security_group", "aws_ssm_document",
        "aws_ssm_parameter", "aws_subnet", "aws_volume_attachment", "aws_vpc",
        "aws_secretsmanager_secret",
    }
    actual = {r["type"] for state in owned.values() for r in state["resources"]}
    if actual - supported:
        raise ValueError("Add an independent removal check for: " + ", ".join(sorted(actual - supported)))
    return {function.__name__: function(env, owned) for function in (
        check_network_resources_removed, check_instance_profiles_and_policies_removed,
        check_owned_image_repositories_removed, check_build_projects_removed,
        check_ssm_configuration_removed,
    )}
