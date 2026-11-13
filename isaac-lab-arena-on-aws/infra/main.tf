# =============================================================================
# VLA Model-Evaluation component -- Foundation-consuming IaC (Terraform).
#
# Consumes the AWS Physical AI Toolchain **Foundation** for shared infra (the
# SageMaker execution role + the output bucket, via SSM /<project_name>/*) and
# provisions ONLY what the Foundation does not provide:
#   * the component-local ECR repos for our train/eval images (Foundation has
#     no eval-container ECR), and
#   * the image-build CodeBuild project + its own service role, and
#   * the HF-token grant, attached to the component's OWN workload role.
#
# It no longer attaches anything to the shared Foundation role. It used to add a
# "detachable supplementary grant" so that role could read the HF token, which expanded
# the secret access of every unrelated workload using it -- a component modifying an
# identity it does not own (review finding I19).
#
# It deliberately does NOT create a SageMaker execution role for ORCHESTRATION or
# model-package groups:
#   * output bucket        -> Foundation /<project_name>/models-bucket (versioned),
#   * orchestration role   -> Foundation /<project_name>/sagemaker-role-arn,
#   * model-package groups -> created on demand by the pipeline's RegisterModel.
#
# It DOES create, in trust_boundary.tf, a component-owned trust bucket and two runtime
# roles (workload, validation). That reverses an earlier decision to provision no bucket
# and no roles, and the reason is review findings C4 and C5: with every stage running
# under the single Foundation role -- which holds Put/DeleteObject across the models
# bucket that also holds the staged validation code -- a worker could overwrite the
# validator that judges it and the model bytes after validation, so validated and
# registered bytes could differ. The fix moves trusted artifacts into a bucket the
# Foundation role has no grant on, and gives the workers an identity that cannot write
# there. See trust_boundary.tf for the full rationale and the residual limitation.
#
# If image repositories have another state owner, reference them through
# existing_ecr_repos and remove those names from ecr_repos. This grants build
# access without importing them or taking over their lifecycle.
# =============================================================================

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# --- Foundation contract (read-only, via SSM) ---
data "aws_ssm_parameter" "models_bucket" {
  name = "/${var.project_name}/models-bucket"
}

data "aws_ssm_parameter" "sagemaker_role_arn" {
  name = "/${var.project_name}/sagemaker-role-arn"
}

data "aws_ssm_parameter" "arena_ecr" {
  name = "/${var.project_name}/ecr/isaac-lab-arena"
}

data "aws_ecr_repository" "arena" {
  name = join("/", slice(split("/", nonsensitive(data.aws_ssm_parameter.arena_ecr.value)), 1, length(split("/", nonsensitive(data.aws_ssm_parameter.arena_ecr.value)))))
}

data "aws_ecr_repository" "existing" {
  for_each = var.existing_ecr_repos
  name     = each.value
}

locals {
  account_id = data.aws_caller_identity.current.account_id
  region     = data.aws_region.current.region
  prefix     = "${var.project_name}-${var.environment}"

  # Bucket name + role are not secrets; unwrap the SSM sensitivity for use in
  # resource arguments/outputs.
  models_bucket       = nonsensitive(data.aws_ssm_parameter.models_bucket.value)
  foundation_role_arn = nonsensitive(data.aws_ssm_parameter.sagemaker_role_arn.value)
  # `foundation_role_name` used to be derived here so a component policy could be attached
  # to the shared Foundation role. Nothing does that any more (I19), so it is gone rather
  # than left as dead code inviting the pattern back.

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "model-evaluation"
  }
}

# =============================================================================
# ECR: component-local train/eval image repositories
# =============================================================================
resource "aws_ecr_repository" "repos" {
  for_each = toset(var.ecr_repos)
  name     = each.value
  #checkov:skip=CKV_AWS_136:AES256 (ECR default) is sufficient for this sample; a customer-managed KMS key is a Foundation/account-level decision, not this component's.

  # IMMUTABLE by default (supply-chain: a pushed tag can't be silently replaced).
  # Some accounts' existing repos may be MUTABLE — import there with
  # -var ecr_image_tag_mutability=MUTABLE to avoid a surprising in-place flip.
  image_tag_mutability = var.ecr_image_tag_mutability

  image_scanning_configuration {
    scan_on_push = true
  }

  # Live images are expensive to rebuild; never let `terraform destroy` drop a
  # repo full of pushed images. Teardown is deliberate + documented (README.md, Teardown).
  lifecycle {
    # `pai arena destroy` may override this in its saved deployment directory
    # only after archiving evidence and checking newly owned repositories are
    # empty. Supplied repositories remain data sources and are never deleted.
    prevent_destroy = true
  }

  tags = local.tags
}

# Keep-last-20 as a SEPARATE resource (not an inline lifecycle_rule) so the
# retention policy can evolve without forcing repo replacement.
resource "aws_ecr_lifecycle_policy" "repos" {
  for_each   = aws_ecr_repository.repos
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged images beyond 20 (tagged releases are retained)"
      selection = {
        tagStatus   = "untagged"
        countType   = "imageCountMoreThan"
        countNumber = 20
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# =============================================================================
# HF-secret grant on the component's training and workload roles
#
# FineTune and SimEval read the token through their separate component roles.
# The managed policy and attachments below grant access to the configured secret.
# They do not attach secret permissions to the shared Foundation role.
# =============================================================================
resource "aws_ssm_parameter" "hf_secret_name" {
  name  = "/${var.project_name}/isaac-lab-arena/hf-secret-name"
  type  = "String"
  value = var.hf_secret_name
  tags  = local.tags
}

# Resolve the EXACT secret so the grant names one resource. `${var.hf_secret_name}*` also
# matched every other secret sharing that prefix, which is more than the policy's own
# description claimed.
data "aws_secretsmanager_secret" "hf_token" {
  name = var.hf_secret_name
}

# The NGC token, needed by the image build to `docker login nvcr.io`. Named by a variable
# rather than the second hardcoded convention this used to carry.
data "aws_secretsmanager_secret" "ngc_token" {
  name = var.ngc_secret_name
}

resource "aws_iam_policy" "hf_secret_read" {
  name        = "${local.prefix}-hf-secret-read"
  description = "Component-added: allow the component's WORKLOAD role to read the HF token secret (exact ARN)."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = [data.aws_secretsmanager_secret.hf_token.arn]
    }]
  })

  tags = local.tags
}

# Attached to the COMPONENT's workload role, not the shared Foundation role.
# (Labelled I19 in an earlier cycle; that ID now denotes an unrelated finding.)
# Attaching it to the shared role expanded the secret access of every unrelated workload
# using that role -- a component adding a grant to an identity it does not own. The
# The two WORKER roles need the token: FineTune and SimEval pull gated models, while Validate
# does not. Both are attached explicitly -- when FineTune moved onto its own identity this
# grant did not follow it, so a fresh deployment's FineTune could not resolve the token.
resource "aws_iam_role_policy_attachment" "hf_secret_attach" {
  role       = aws_iam_role.workload.name
  policy_arn = aws_iam_policy.hf_secret_read.arn
}

resource "aws_iam_role_policy_attachment" "hf_secret_attach_training" {
  role       = aws_iam_role.training.name
  policy_arn = aws_iam_policy.hf_secret_read.arn
}

# =============================================================================
# CodeBuild service role (component-local)
#
# The SageMaker EXECUTION role comes from the Foundation; CodeBuild still needs
# its own role to push images + read the build-source zip from the output bucket.
# =============================================================================
resource "aws_iam_role" "codebuild" {
  name = "${local.prefix}-vla-codebuild-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codebuild.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "codebuild" {
  name = "${local.prefix}-vla-codebuild-policy"
  role = aws_iam_role.codebuild.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # Account-wide ECR auth token (required for `docker login`).
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = ["*"]
      },
      # Push/pull layers on our component repos only.
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = concat(
          [for repo in aws_ecr_repository.repos : repo.arn],
          [for repo in data.aws_ecr_repository.existing : repo.arn],
          [data.aws_ecr_repository.arena.arn],
        )
      },
      # Pull read-only from the AWS Deep Learning Container registry (763104351884):
      # the gr00t train image FROMs pytorch-training DLC. GetAuthorizationToken above
      # lets `docker login` succeed, but the layer/manifest pull is authorized per
      # resource -- without this the cross-account HEAD returns 403 Forbidden. Scoped
      # to this region + the exact DLC repo we FROM (least-privilege).
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
        ]
        Resource = ["arn:aws:ecr:${local.region}:763104351884:repository/pytorch-training"]
      },
      # Build logs.
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = ["arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/codebuild/*"]
      },
      # Read the build-source zip.
      #
      # C3 (cycle 8): this granted GetObject across the WHOLE Foundation models bucket, and
      # build_arena_connector.py uploaded the source zip there. The Foundation role holds
      # PutObject and DeleteObject across that bucket and peer components submit jobs under it,
      # so a peer could replace the zip between upload and build -- and the replacement would
      # then execute under this builder role, which can push to the component's ECR
      # repositories. That is the same code-substitution class the validator move closed, left
      # open for the build path.
      #
      # The source now lands in the trust bucket's code/v1/* namespace, where publication
      # belongs to the deploying identity and NO runtime role may write. The grant is narrowed
      # to that namespace: bucket-wide read was more than the builder ever needed.
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:GetObjectVersion", "s3:GetBucketLocation", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.trust.arn,
          "${aws_s3_bucket.trust.arn}/code/v1/*",
        ]
      },
      # Read the NGC + HF tokens: the Arena base build must `docker login nvcr.io`
      # to pull the Isaac Sim base, and gated HF pulls need the HF token.
      # Exact ARNs, not prefix wildcards: `<name>*` also matched every other secret
      # sharing that prefix, granting more than the two named secrets (I19). The NGC
      # secret name is a variable rather than a second hardcoded naming convention.
      {
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = [
          data.aws_secretsmanager_secret.ngc_token.arn,
          data.aws_secretsmanager_secret.hf_token.arn,
        ]
      },
    ]
  })
}

# =============================================================================
# CodeBuild project (image builds)
#
# scripts/build_arena_connector.py and scripts/build_images.py start builds
# against this project (via --project-name) and override
# the S3 source location + buildspec per build (sourceLocationOverride /
# buildspecOverride). The declared S3 source below is just the default.
# =============================================================================
resource "aws_codebuild_project" "image_build" {
  name = var.codebuild_project_name
  #checkov:skip=CKV_AWS_316:privileged_mode is required for docker-in-docker image builds (see environment block).
  #checkov:skip=CKV_AWS_314:CloudWatch logging is left at CodeBuild defaults for this sample; enable a logs_config for shared/production use.
  description  = "Build and push VLA eval/train Docker images (component-local)."
  service_role = aws_iam_role.codebuild.arn

  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_LARGE"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true # required for docker-in-docker

    # N3: the buildspec defaults NGC_SECRET_ID to a fixed name, and the IAM grant below is
    # scoped to the ARN of var.ngc_secret_name. Without delivering the configured name here,
    # setting that variable moved the PERMISSION but not the CONSUMER, so a non-default
    # secret name produced a build that was allowed to read a secret it never asked for and
    # denied the one it did. Buildspec `variables:` are defaults, so this overrides them.
    environment_variable {
      name  = "NGC_SECRET_ID"
      value = var.ngc_secret_name
    }
  }

  source {
    type     = "S3"
    location = "${local.models_bucket}/codebuild/"
  }

  build_timeout = 60

  tags = local.tags
}
