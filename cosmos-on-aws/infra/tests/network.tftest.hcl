# Network-input and image-build-gate tests. Run with `terraform test` from
# cosmos-on-aws/infra. Every provider is mocked, so no AWS credentials are needed and
# no AWS API is called. These prove which data sources and resources the configuration
# instantiates and what the pod policy contains; they do not prove that EKS accepts
# the subnets or that capacity exists.

# mock_data defaults (rather than override_data) so the preconditions on
# data.aws_caller_identity.current are evaluated on the mock read.
mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "123456789012"
      arn        = "arn:aws:iam::123456789012:role/test"
      user_id    = "AIDTEST"
    }
  }

  mock_data "aws_subnet" {
    defaults = {
      id                = "subnet-0aaaaaaaaaaaaaaa1"
      availability_zone = "us-east-2b"
    }
  }

  mock_data "aws_subnets" {
    defaults = {
      ids = ["subnet-0aaaaaaaaaaaaaaa1"]
    }
  }
}

mock_provider "helm" {}

mock_provider "kubernetes" {}

override_module {
  target = module.cosmos3_eks
  outputs = {
    cluster_name                       = "physical-ai-test-cosmos3"
    cluster_endpoint                   = "https://example.eks.amazonaws.com"
    cluster_certificate_authority_data = base64encode("dummy")
    oidc_provider_arn                  = "arn:aws:iam::123456789012:oidc-provider/oidc.eks.us-east-2.amazonaws.com/id/TEST"
  }
}

override_module {
  target = module.ebs_csi_irsa
  outputs = {
    iam_role_arn = "arn:aws:iam::123456789012:role/test-ebs-csi"
  }
}

override_module {
  target = module.cosmos3_pod_irsa
  outputs = {
    iam_role_arn = "arn:aws:iam::123456789012:role/test-pod"
  }
}

override_module {
  target = module.cluster_autoscaler_irsa
  outputs = {
    iam_role_arn = "arn:aws:iam::123456789012:role/test-autoscaler"
  }
}

variables {
  aws_region = "us-east-2"
}

# Standalone defaults: every SSM lookup and image-build resource exists, exactly as
# before the network variables were added, and the pod policy keeps the Foundation bucket.
run "defaults_standalone" {
  command = plan

  variables {
    enable_eks_cluster = true
  }

  assert {
    condition     = length(data.aws_ssm_parameter.eks_vpc_id) == 1
    error_message = "eks_vpc_id SSM lookup must exist under defaults."
  }
  assert {
    condition     = length(data.aws_ssm_parameter.eks_private_subnet_ids) == 1
    error_message = "eks_private_subnet_ids SSM lookup must exist under defaults."
  }
  assert {
    condition     = length(data.aws_ssm_parameter.cosmos3_ecr) == 1
    error_message = "cosmos3_ecr SSM lookup must exist under defaults."
  }
  assert {
    condition     = length(data.aws_ssm_parameter.cosmos_transfer_ecr) == 1
    error_message = "cosmos_transfer_ecr SSM lookup must exist under defaults."
  }
  assert {
    condition     = length(data.aws_ssm_parameter.cosmos_instance_profile) == 1
    error_message = "cosmos_instance_profile SSM lookup must exist under defaults."
  }
  assert {
    condition     = length(aws_codebuild_project.cosmos3) == 1
    error_message = "cosmos3 CodeBuild project must exist under defaults."
  }
  assert {
    condition     = length(aws_codebuild_project.cosmos_transfer) == 1
    error_message = "cosmos_transfer CodeBuild project must exist under defaults."
  }
  assert {
    condition     = length(aws_iam_role.codebuild) == 1
    error_message = "CodeBuild role must exist under defaults."
  }
  assert {
    condition     = strcontains(aws_iam_policy.cosmos3_pod_s3[0].policy, "physical-ai-dev-datasets-123456789012")
    error_message = "Pod S3 policy must grant the Foundation datasets bucket under defaults."
  }
}

# The workshop tfvars written by event-provision.sh: explicit network, gate off,
# Foundation grant off. No SSM lookup and no image-build resource may exist.
run "workshop" {
  command = plan

  variables {
    enable_eks_cluster                         = true
    environment                                = "evt"
    vpc_id                                     = "vpc-0aaaaaaaaaaaaaaa0"
    private_subnet_ids                         = ["subnet-0aaaaaaaaaaaaaaa1"]
    eks_control_plane_subnet_ids               = ["subnet-0aaaaaaaaaaaaaaa1", "subnet-0aaaaaaaaaaaaaaa2"]
    eks_cluster_api_ingress_security_group_ids = ["sg-0aaaaaaaaaaaaaaa3"]
    eks_gpu_availability_zone                  = "us-east-2b"
    enable_image_builds                        = false
    eks_pod_grant_foundation_datasets_bucket   = false
    eks_pod_extra_s3_buckets                   = ["physical-ai-data-123456789012"]
  }

  assert {
    condition = (
      length(data.aws_ssm_parameter.eks_vpc_id) == 0 &&
      length(data.aws_ssm_parameter.eks_private_subnet_ids) == 0 &&
      length(data.aws_ssm_parameter.cosmos_transfer_ecr) == 0 &&
      length(data.aws_ssm_parameter.cosmos3_ecr) == 0 &&
      length(data.aws_ssm_parameter.cosmos_instance_profile) == 0 &&
      length(data.aws_ssm_parameter.vpc_id) == 0 &&
      length(data.aws_ssm_parameter.private_subnet_ids) == 0
    )
    error_message = "No aws_ssm_parameter data source may be read with the workshop variables."
  }
  assert {
    condition = (
      length(aws_codebuild_project.cosmos_transfer) == 0 &&
      length(aws_codebuild_project.cosmos3) == 0 &&
      length(aws_iam_role.codebuild) == 0 &&
      length(aws_iam_role_policy.codebuild) == 0
    )
    error_message = "No image-build resource may exist with enable_image_builds = false."
  }
  assert {
    condition     = local.eks_vpc_id == "vpc-0aaaaaaaaaaaaaaa0"
    error_message = "The cluster VPC must be the explicit vpc_id."
  }
  assert {
    condition     = local.eks_private_subnet_ids == tolist(["subnet-0aaaaaaaaaaaaaaa1"])
    error_message = "Node subnets must be the explicit private_subnet_ids."
  }
  assert {
    condition     = local.eks_gpu_subnet_ids == tolist(["subnet-0aaaaaaaaaaaaaaa1"])
    error_message = "GPU node subnet must be the node subnet in the GPU AZ."
  }
  assert {
    condition     = jsondecode(aws_iam_policy.cosmos3_pod_s3[0].policy).Statement[0].Resource == ["arn:aws:s3:::physical-ai-data-123456789012"]
    error_message = "Pod ListBucket must grant exactly the workshop data bucket."
  }
  assert {
    condition     = jsondecode(aws_iam_policy.cosmos3_pod_s3[0].policy).Statement[1].Resource == ["arn:aws:s3:::physical-ai-data-123456789012/*"]
    error_message = "Pod Get/PutObject must grant exactly the workshop data bucket objects."
  }
  assert {
    condition     = output.codebuild_cosmos3_project == null
    error_message = "codebuild_cosmos3_project output must be null with the gate off."
  }
}

run "partial_network_rejected" {
  command = plan

  variables {
    enable_eks_cluster = true
    vpc_id             = "vpc-0aaaaaaaaaaaaaaa0"
    private_subnet_ids = []
  }

  expect_failures = [data.aws_caller_identity.current]
}

run "partial_network_rejected_ec2_only" {
  command = plan

  variables {
    enable_eks_cluster = false
    enable_ec2_server  = true
    vpc_id             = "vpc-0aaaaaaaaaaaaaaa0"
    private_subnet_ids = []
  }

  expect_failures = [data.aws_caller_identity.current]
}

run "control_plane_without_vpc_rejected" {
  command = plan

  variables {
    enable_eks_cluster           = true
    eks_control_plane_subnet_ids = ["subnet-0aaaaaaaaaaaaaaa1", "subnet-0aaaaaaaaaaaaaaa2"]
    vpc_id                       = ""
    private_subnet_ids           = []
  }

  expect_failures = [data.aws_caller_identity.current]
}

run "empty_pod_buckets_rejected" {
  command = plan

  variables {
    enable_eks_cluster                         = true
    environment                                = "evt"
    vpc_id                                     = "vpc-0aaaaaaaaaaaaaaa0"
    private_subnet_ids                         = ["subnet-0aaaaaaaaaaaaaaa1"]
    eks_control_plane_subnet_ids               = ["subnet-0aaaaaaaaaaaaaaa1", "subnet-0aaaaaaaaaaaaaaa2"]
    eks_cluster_api_ingress_security_group_ids = ["sg-0aaaaaaaaaaaaaaa3"]
    eks_gpu_availability_zone                  = "us-east-2b"
    enable_image_builds                        = false
    eks_pod_grant_foundation_datasets_bucket   = false
    eks_pod_extra_s3_buckets                   = []
  }

  expect_failures = [aws_iam_policy.cosmos3_pod_s3]
}

run "bad_ids_rejected" {
  command = plan

  variables {
    vpc_id             = "vpc-NOTHEX"
    private_subnet_ids = ["subnet-0aaaaaaaaaaaaaaa1"]
  }

  expect_failures = [var.vpc_id]
}
