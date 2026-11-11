# Uncounted, so it is read on every plan (EKS, EC2-only and both-off). The network
# preconditions live here for that reason: they hold on every path.
data "aws_caller_identity" "current" {
  lifecycle {
    # All-or-nothing: mixing an explicit VPC with SSM subnets (or vice versa) would put
    # the cluster/server in one VPC and its nodes/subnet in another.
    precondition {
      condition     = (var.vpc_id == "") == (length(var.private_subnet_ids) == 0)
      error_message = "Set vpc_id and private_subnet_ids together, or leave both empty to read them from SSM."
    }
    precondition {
      condition     = length(var.eks_control_plane_subnet_ids) == 0 || var.vpc_id != ""
      error_message = "eks_control_plane_subnet_ids requires vpc_id and private_subnet_ids to be set."
    }
  }
}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = var.aws_region
  prefix     = "${var.project_name}-${var.environment}"
}

# Foundation-created ECR repositories and instance profile, read only when needed
data "aws_ssm_parameter" "cosmos_transfer_ecr" {
  count = var.enable_image_builds ? 1 : 0
  name  = "/${var.project_name}/ecr/cosmos-transfer"
}

data "aws_ssm_parameter" "cosmos3_ecr" {
  count = var.enable_image_builds ? 1 : 0
  name  = "/${var.project_name}/ecr/cosmos3"
}

# Read whenever today's code would need it: image builds on (default, keeps the standalone
# data-source set identical) or the EC2 server on without an explicit profile.
data "aws_ssm_parameter" "cosmos_instance_profile" {
  count = (var.enable_image_builds || var.enable_ec2_server) && var.server_instance_profile == "" ? 1 : 0
  name  = "/${var.project_name}/cosmos-instance-profile"
}

# The image-build resources below became count-gated on enable_image_builds. These let
# existing standalone state move to the [0] addresses without a destroy and recreate.
moved {
  from = aws_codebuild_project.cosmos_transfer
  to   = aws_codebuild_project.cosmos_transfer[0]
}

moved {
  from = aws_codebuild_project.cosmos3
  to   = aws_codebuild_project.cosmos3[0]
}

moved {
  from = aws_iam_role.codebuild
  to   = aws_iam_role.codebuild[0]
}

moved {
  from = aws_iam_role_policy.codebuild
  to   = aws_iam_role_policy.codebuild[0]
}

# =============================================================================
# CODEBUILD: COSMOS TRANSFER 2.5
# =============================================================================

resource "aws_codebuild_project" "cosmos_transfer" {
  count        = var.enable_image_builds ? 1 : 0
  name         = "${local.prefix}-cosmos-transfer-build"
  description  = "Build Cosmos Transfer 2.5 container from source"
  service_role = aws_iam_role.codebuild[0].arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_2XLARGE"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true

    environment_variable {
      name  = "ECR_REPO_URI"
      value = data.aws_ssm_parameter.cosmos_transfer_ecr[0].value
    }

    environment_variable {
      name  = "COSMOS_REPO"
      value = var.cosmos_transfer_repo
    }

    environment_variable {
      name  = "COSMOS_REF"
      value = var.cosmos_transfer_ref
    }

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = local.account_id
    }

    environment_variable {
      name  = "AWS_DEFAULT_REGION"
      value = local.region
    }
  }

  source {
    type      = "NO_SOURCE"
    buildspec = file("${path.module}/../../containers/cosmos/buildspec.yml")
  }

  build_timeout = 120

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos-transfer"
  }
}

# =============================================================================
# CODEBUILD: COSMOS 3
# =============================================================================

resource "aws_codebuild_project" "cosmos3" {
  count        = var.enable_image_builds ? 1 : 0
  name         = "${local.prefix}-cosmos3-build"
  description  = "Build Cosmos 3 (cosmos-framework) container from source"
  service_role = aws_iam_role.codebuild[0].arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_2XLARGE"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true

    environment_variable {
      name  = "ECR_REPO_URI"
      value = data.aws_ssm_parameter.cosmos3_ecr[0].value
    }

    environment_variable {
      name  = "COSMOS3_REPO"
      value = var.cosmos3_repo
    }

    environment_variable {
      name  = "COSMOS3_REF"
      value = var.cosmos3_ref
    }

    environment_variable {
      name  = "AWS_ACCOUNT_ID"
      value = local.account_id
    }

    environment_variable {
      name  = "AWS_DEFAULT_REGION"
      value = local.region
    }
  }

  source {
    type      = "NO_SOURCE"
    buildspec = file("${path.module}/../../containers/cosmos3/buildspec.yml")
  }

  build_timeout = 120

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3"
  }
}

# =============================================================================
# IAM: CODEBUILD ROLE
# =============================================================================

resource "aws_iam_role" "codebuild" {
  count = var.enable_image_builds ? 1 : 0
  name  = "${local.prefix}-cosmos-codebuild-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codebuild.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "codebuild" {
  count = var.enable_image_builds ? 1 : 0
  name  = "${local.prefix}-cosmos-codebuild-policy"
  role  = aws_iam_role.codebuild[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Returns a registry-wide token and does not support resource-level permissions:
        # IAM rejects any Resource other than "*". Kept in its own statement so the push
        # actions below stay scoped to this project's two repositories.
        Sid      = "EcrAuthTokenRegistryWide"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = ["*"]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = [
          "arn:aws:ecr:${local.region}:${local.account_id}:repository/${var.project_name}/cosmos-transfer",
          "arn:aws:ecr:${local.region}:${local.account_id}:repository/${var.project_name}/cosmos3",
        ]
      },
      {
        # Scoped to this project's CodeBuild groups rather than account-wide. CreateLogGroup
        # is included because the group does not exist before the first build.
        Sid    = "WriteOwnBuildLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/codebuild/${local.prefix}-*",
          "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/codebuild/${local.prefix}-*:*",
        ]
      }
    ]
  })
}
