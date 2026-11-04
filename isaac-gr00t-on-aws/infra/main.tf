data "aws_caller_identity" "current" {}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = var.aws_region
  prefix     = "${var.project_name}-${var.environment}"
}

# Read shared resources from foundation (via SSM)
data "aws_ssm_parameter" "datasets_bucket" {
  name = "/${var.project_name}/datasets-bucket"
}

data "aws_ssm_parameter" "sagemaker_role_arn" {
  name = "/${var.project_name}/sagemaker-role-arn"
}

data "aws_ssm_parameter" "groot_training_ecr" {
  name = "/${var.project_name}/ecr/gr00t-training"
}

data "aws_ssm_parameter" "groot_inference_ecr" {
  name = "/${var.project_name}/ecr/gr00t-inference"
}

# =============================================================================
# CODEBUILD: GROOT TRAINING CONTAINER
# =============================================================================

resource "aws_codebuild_project" "groot_training" {
  name         = "${local.prefix}-gr00t-training-build"
  description  = "Build GR00T fine-tuning container from PyTorch DLC base + Isaac-GR00T"
  service_role = aws_iam_role.codebuild.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_LARGE"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true

    environment_variable {
      name  = "ECR_REPO_URI"
      value = data.aws_ssm_parameter.groot_training_ecr.value
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
    buildspec = file("${path.module}/../../containers/gr00t-training/buildspec.yml")
  }

  build_timeout = 60

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "groot-training"
  }
}

# =============================================================================
# CODEBUILD: GROOT INFERENCE CONTAINER
# =============================================================================

resource "aws_codebuild_project" "groot_inference" {
  name         = "${local.prefix}-gr00t-inference-build"
  description  = "Build GR00T inference/serving container"
  service_role = aws_iam_role.codebuild.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_LARGE"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true

    environment_variable {
      name  = "ECR_REPO_URI"
      value = data.aws_ssm_parameter.groot_inference_ecr.value
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
    buildspec = file("${path.module}/../../containers/gr00t-inference/buildspec.yml")
  }

  build_timeout = 60

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "groot-inference"
  }
}

# =============================================================================
# IAM: CODEBUILD ROLE
# =============================================================================

resource "aws_iam_role" "codebuild" {
  name = "${local.prefix}-gr00t-codebuild-role"

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
  name = "${local.prefix}-gr00t-codebuild-policy"
  role = aws_iam_role.codebuild.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Returns a registry-wide token and does not support resource-level permissions:
        # IAM rejects any Resource other than "*". Kept in its own statement so the push and
        # pull actions below stay scoped.
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
          "arn:aws:ecr:${local.region}:${local.account_id}:repository/${var.project_name}/gr00t-training",
          "arn:aws:ecr:${local.region}:${local.account_id}:repository/${var.project_name}/gr00t-inference",
          "arn:aws:ecr:${local.region}:763104351884:repository/*",
        ]
      },
      {
        # Scoped to this component's own build log groups rather than account-wide.
        # CreateLogGroup is required because the group does not exist before the first build.
        Sid    = "WriteOwnBuildLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/codebuild/${local.prefix}-gr00t-*",
          "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/codebuild/${local.prefix}-gr00t-*:*",
        ]
      },
      {
        # These projects are declared NO_SOURCE and are started with
        # --source-type-override S3 --source-location-override pointing at a packaged
        # source.zip in the Foundation datasets bucket (see batch-training-guide.md step 3).
        # CodeBuild fetches that archive under this role, which is the only S3 read the build
        # performs -- the buildspec itself touches no other bucket. Scoped to that one bucket
        # rather than every bucket in the account.
        Sid      = "ReadBuildSourceArchive"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:GetObjectVersion"]
        Resource = ["arn:aws:s3:::${data.aws_ssm_parameter.datasets_bucket.value}/*"]
      },
      {
        Sid      = "ListBuildSourceBucket"
        Effect   = "Allow"
        Action   = ["s3:GetBucketLocation", "s3:ListBucket"]
        Resource = ["arn:aws:s3:::${data.aws_ssm_parameter.datasets_bucket.value}"]
      },
      {
        # Optional NGC key. buildspec-n17.yml logs in to nvcr.io only when this
        # secret exists and is readable, and otherwise pulls the base anonymously,
        # so granting read on a secret that does not exist is harmless.
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = ["arn:aws:secretsmanager:*:*:secret:${var.project_name}/ngc-api-key*"]
      }
    ]
  })
}
