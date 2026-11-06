variable "aws_region" {
  description = "This sample supports only us-east-1; Foundation must use the same region."
  type        = string
  default     = "us-east-1"

  validation {
    condition     = var.aws_region == "us-east-1"
    error_message = "This sample supports only us-east-1. Deploy Foundation and this component in us-east-1."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name prefix; MUST match the Foundation (drives the /<project_name>/* SSM lookups)."
  type        = string
  default     = "physical-ai"
}

variable "codebuild_project_name" {
  description = "Name of this component's image builder. Component-scoped by default so it cannot collide with a project another deployment owns; published to SSM so the build scripts discover the deployed value rather than defaulting to it."
  type        = string
  default     = "physical-ai-vla-image-build"
}

variable "ecr_repos" {
  description = "Component-owned training and Arena base repositories. IMPORTANT: the build scripts and family manifests target these exact names; changing them here creates the repositories but does NOT redirect builds or launches. Use existing_ecr_repos to reference repositories owned by another deployment without renaming."
  type        = list(string)
  default = [
    "vla/gr00t", "vla/openvla", "vla/molmoact2", "vla/isaac-arena",
  ]
}

variable "existing_ecr_repos" {
  description = "Repositories owned by another deployment; grant build access without managing their lifecycle."
  type        = set(string)
  default     = []

  validation {
    condition     = length(setintersection(toset(var.ecr_repos), var.existing_ecr_repos)) == 0
    error_message = "A repository cannot be both component-owned and externally owned."
  }
}

variable "hf_secret_name" {
  description = "Secrets Manager secret name holding the HuggingFace token the FineTune step reads."
  type        = string
  default     = "vla-pipeline/hf-token"
}

variable "ngc_secret_name" {
  description = "Secrets Manager secret name holding the NGC token the Arena base image build uses to `docker login nvcr.io`. A variable rather than a hardcoded second naming convention."
  type        = string
  default     = "vla-pipeline/ngc-token"
}

variable "ecr_image_tag_mutability" {
  description = "ECR tag mutability. IMMUTABLE (default) prevents silently replacing a pushed tag. Set MUTABLE when importing an account's existing repos."
  type        = string
  default     = "IMMUTABLE"
  validation {
    condition     = contains(["IMMUTABLE", "MUTABLE"], var.ecr_image_tag_mutability)
    error_message = "ecr_image_tag_mutability must be IMMUTABLE or MUTABLE."
  }
}

variable "pipeline_prefix" {
  description = "S3 key prefix the pipeline writes under in the Foundation models bucket. Must match PipelineConfig.prefix; the trust boundary scopes write access per sub-prefix, so a mismatch would deny legitimate job output rather than fail silently."
  type        = string
  default     = "vla-pipeline"
}

# I10: the role split narrowed input reads to the Foundation models bucket, but OpenVLA and
# MolmoAct2 both support arbitrary S3 dataset inputs (TRAIN_DATASET_S3URI is a pipeline
# parameter, so it can name any location) and eval-only supports an external checkpoint URI.
# Those advertised paths were denied unless an out-of-band grant existed.
#
# Declared EXPLICITLY rather than widening to s3://*: a worker that can read any bucket in the
# account is a different thing from one that can read the inputs this deployment uses. Empty by
# default, so a deployment that only uses the models bucket grants nothing extra.
variable "additional_input_s3_arns" {
  description = <<-EOT
    Extra read-only S3 ARNs the training and evaluation roles may read, for BYO datasets and
    external eval-only checkpoints. Supply both the bucket ARN and its object ARN pattern, for
    example ["arn:aws:s3:::my-datasets", "arn:aws:s3:::my-datasets/prefix/*"].
  EOT
  type        = list(string)
  default     = []
}

variable "dlc_account_id" {
  description = "AWS account that hosts the Deep Learning Container images these SageMaker jobs start from. Exposed as a variable rather than hardcoded because the DLC registry account differs by Region - a deployment outside the default Region needs the matching account or ECR pulls fail with AccessDenied."
  type        = string
  default     = "763104351884"
}

variable "sklearn_account_id" {
  description = "AWS account hosting the SageMaker scikit-learn image used by Validate and the plumbing jobs in us-east-1. This registry differs from the PyTorch DLC registry."
  type        = string
  default     = "683313688378"
}
