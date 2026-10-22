# =============================================================================
# AWS BATCH: GR00T Fine-Tuning (optional — mirrors Isaac Lab Batch pattern)
# =============================================================================

variable "enable_batch" {
  description = "Enable AWS Batch compute environment for GR00T training"
  type        = bool
  default     = false
}

variable "batch_instance_type" {
  description = "Instance type for Batch compute. Default matches the single-GPU job shape below (g6e.4xlarge = 1x L40S 48 GB). Pass g6e.12xlarge together with batch_job_vcpus=48 / batch_job_memory_mib=180000 / batch_job_gpus=4"
  type        = string
  default     = "g6e.4xlarge"
}

variable "batch_max_vcpus" {
  description = "Maximum vCPUs for the Batch compute environment"
  type        = number
  default     = 96
}

# --- Job-definition resource shape ---
#
# Defaults describe a SINGLE-GPU g6e.4xlarge (1x L40S 48 GB, 16 vCPU, 128 GiB),
# which is the primary target: g6e.12xlarge capacity could not be obtained in
# us-east-2 (30/30 launch attempts failed across all three AZs over 2 h 38 m).
# A job whose resourceRequirements exceed what the compute environment can place
# stays in RUNNABLE forever, so these three must match the instance type in use.
# For the 4-GPU g6e.12xlarge shape pass: 48 / 180000 / 4 AND
# batch_instance_type = g6e.12xlarge. The job definition carries a precondition
# that rejects a GPU count the chosen instance type does not have, because the
# failure mode of a mismatch is silent: too many GPUs parks the job in RUNNABLE,
# and too few per-GPU memory (e.g. the 24 GB cards of g6.12xlarge against the
# 22,023 MiB static footprint plus activations) lets it launch and then OOM.

variable "batch_job_vcpus" {
  description = "vCPUs requested by the GR00T N1.7 job definition (16 = g6e.4xlarge, 48 = g6e.12xlarge)"
  type        = number
  default     = 16
}

variable "batch_job_memory_mib" {
  description = "Memory (MiB) requested by the GR00T N1.7 job definition. Leaves ~8 GiB of a 128 GiB g6e.4xlarge to the ECS agent and kernel"
  type        = number
  default     = 122880
}

variable "batch_job_gpus" {
  description = "GPUs requested by the GR00T N1.7 job definition. 1 requires TUNE_VISUAL=0 (frozen vision tower); 4 is needed before DeepSpeed ZeRO engages"
  type        = number
  default     = 1
}

# --- Read shared VPC from Foundation ---

data "aws_ssm_parameter" "vpc_id" {
  count = var.enable_batch ? 1 : 0
  name  = "/${var.project_name}/vpc-id"
}

data "aws_ssm_parameter" "private_subnet_ids" {
  count = var.enable_batch ? 1 : 0
  name  = "/${var.project_name}/private-subnet-ids"
}

data "aws_ssm_parameter" "checkpoints_bucket" {
  count = var.enable_batch ? 1 : 0
  name  = "/${var.project_name}/checkpoints-bucket"
}

# --- HuggingFace token secret (pre-existing; created out of band) ---
#
# Looked up, never created and never read: only `.arn` is referenced, so the
# token value never enters the Terraform state, a plan, or this repo. The secret
# itself is managed outside this stack (see batch-training-guide.md step 5).
data "aws_secretsmanager_secret" "hf_token" {
  count = var.enable_batch ? 1 : 0
  name  = "${var.project_name}/hf-token"
}

locals {
  batch_vpc_id     = var.enable_batch ? data.aws_ssm_parameter.vpc_id[0].value : ""
  batch_subnet_ids = var.enable_batch ? split(",", data.aws_ssm_parameter.private_subnet_ids[0].value) : []

  # Physical GPU count and per-GPU VRAM for the instance types this guide
  # reasons about, used only by the job definition's precondition below. An
  # instance type that is absent from these maps is not rejected — it simply
  # cannot be checked, so the precondition lets it through.
  batch_gpus_per_instance = {
    "g6e.4xlarge"  = 1
    "g6e.12xlarge" = 4
    "g6.4xlarge"   = 1
    "g6.12xlarge"  = 4
  }
  batch_gpu_mib_per_instance = {
    "g6e.4xlarge"  = 46068 # L40S 48 GB
    "g6e.12xlarge" = 46068 # L40S 48 GB
    "g6.4xlarge"   = 22888 # L4 24 GB
    "g6.12xlarge"  = 22888 # L4 24 GB
  }

  # A 48 GB-class card is required. The derived static footprint at
  # TUNE_VISUAL=0 is 22,023 MiB before activations, so a 24 GB L4 is excluded
  # even on a 4-GPU g6.12xlarge: the job is placeable there, launches, and OOMs.
  batch_min_gpu_mib = 32768

  batch_instance_gpus    = lookup(local.batch_gpus_per_instance, var.batch_instance_type, var.batch_job_gpus)
  batch_instance_gpu_mib = lookup(local.batch_gpu_mib_per_instance, var.batch_instance_type, local.batch_min_gpu_mib)
}

# --- Security Group ---

resource "aws_security_group" "batch" {
  count       = var.enable_batch ? 1 : 0
  name        = "${local.prefix}-gr00t-batch-sg"
  description = "GR00T Batch compute: outbound access for ECR/S3/HF"
  vpc_id      = local.batch_vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = {
    Name        = "${local.prefix}-gr00t-batch-sg"
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- Launch Template (512 GiB disk for large model + container) ---

resource "aws_launch_template" "batch" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-gr00t-batch-lt"

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = 512
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  tags = {
    Name        = "${local.prefix}-gr00t-batch-lt"
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- IAM Role for Batch Instances ---

resource "aws_iam_role" "batch_instance" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-gr00t-batch-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_ecs" {
  count      = var.enable_batch ? 1 : 0
  role       = aws_iam_role.batch_instance[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role_policy" "batch_s3_ecr" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-gr00t-batch-s3-ecr"
  role  = aws_iam_role.batch_instance[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = ["*"]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = [
          "arn:aws:ecr:${local.region}:${local.account_id}:repository/${var.project_name}/gr00t-training"
        ]
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.environment}-*",
          "arn:aws:s3:::${var.project_name}-${var.environment}-*/*"
        ]
      },
      {
        # The container fetches the HuggingFace token at runtime under this
        # instance role (HF_TOKEN_SECRET_ID in the job definition below), which
        # is what removes the plaintext HF_TOKEN from the job definition.
        # Scoped to the single hf-token secret ARN rather than the project
        # prefix: the Batch instances have no reason to read ngc-api-key or any
        # other secret in the account. Read-only -- no Describe, no Put.
        Sid      = "ReadHuggingFaceTokenSecretOnly"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = [data.aws_secretsmanager_secret.hf_token[0].arn]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "batch" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-gr00t-batch-profile"
  role  = aws_iam_role.batch_instance[0].name
}

# --- Batch Service Role ---

resource "aws_iam_role" "batch_service" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-gr00t-batch-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "batch.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_service" {
  count      = var.enable_batch ? 1 : 0
  role       = aws_iam_role.batch_service[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role_policy_attachment" "batch_service_ecs" {
  count      = var.enable_batch ? 1 : 0
  role       = aws_iam_role.batch_service[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonECS_FullAccess"
}

# --- Compute Environment ---

resource "aws_batch_compute_environment" "groot" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-gr00t-compute"
  type  = "MANAGED"

  service_role = aws_iam_role.batch_service[0].arn

  compute_resources {
    type               = "EC2"
    instance_role      = aws_iam_instance_profile.batch[0].arn
    instance_type      = [var.batch_instance_type]
    max_vcpus          = var.batch_max_vcpus
    min_vcpus          = 0
    desired_vcpus      = 0
    security_group_ids = [aws_security_group.batch[0].id]
    subnets            = local.batch_subnet_ids

    launch_template {
      launch_template_id = aws_launch_template.batch[0].id
      version            = "$Latest"
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- Job Queue ---

resource "aws_batch_job_queue" "groot" {
  count    = var.enable_batch ? 1 : 0
  name     = "${local.prefix}-gr00t-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.groot[0].arn
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- Job Definition (GR00T N1.7) ---
#
# Codifies the job definition that was previously registered by hand with
# `aws batch register-job-definition`, so it stops being CLI-managed drift.
# Environment changes belong in `submit-job --container-overrides`, not in a new
# revision of this resource.
#
# HF_TOKEN is deliberately absent. The token is fetched at runtime by
# /opt/ml/code/batch-train from the secret named in HF_TOKEN_SECRET_ID, using the
# Batch container-instance role granted above.
#
# ALTERNATIVE, NOT USED: AWS Batch can inject the secret itself via
# containerProperties.secrets ([{name = "HF_TOKEN", valueFrom = <secret arn>}]).
# That shape additionally requires an `executionRoleArn` on this job definition
# plus the same scoped secretsmanager:GetSecretValue on THAT role, because the
# ECS agent resolves secrets with the execution role and not with the
# container-instance role:
# https://docs.aws.amazon.com/batch/latest/userguide/specifying-sensitive-data-secrets.html
# The runtime-fetch shape was chosen because it needs no new execution role and
# the container was being rebuilt anyway for the embodiment-selectable
# entrypoint, so reading the secret in batch-train cost nothing extra.

resource "aws_batch_job_definition" "groot_n17" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-gr00t-n17-training"
  type  = "container"

  container_properties = jsonencode({
    image = "${local.account_id}.dkr.ecr.${local.region}.amazonaws.com/${var.project_name}/gr00t-training:n17"
    # Selects the Batch branch of the image's ENTRYPOINT. The image dispatches on
    # the basename of $1, so "/opt/ml/code/batch-train" (what live revision 1
    # carries) also works; the bare token is used here because it is what the
    # dispatch literally compares against.
    command = ["batch-train"]

    # Must be placeable on the compute environment's instance type, or the job
    # waits in RUNNABLE indefinitely with no statusReason. Defaults are the
    # single-GPU g6e.4xlarge shape; see the variable block at the top of this file.
    resourceRequirements = [
      { type = "VCPU", value = tostring(var.batch_job_vcpus) },
      { type = "MEMORY", value = tostring(var.batch_job_memory_mib) },
      { type = "GPU", value = tostring(var.batch_job_gpus) }
    ]

    environment = [
      { name = "BASE_MODEL", value = "nvidia/GR00T-N1.7-3B" },
      # Global (pre-accumulation) batch; the trainer derives
      # per_device = BATCH_SIZE / num_gpus, so at 1 GPU this IS the per-device
      # batch. 2 x 8 keeps the effective optimizer batch at 16. DERIVED, not
      # measured: raise only after a real run reports peak VRAM.
      { name = "BATCH_SIZE", value = "2" },
      { name = "GRAD_ACCUM_STEPS", value = "8" },
      # Frozen vision tower (upstream default). This is what makes one 48 GB L40S
      # viable: 877,747,328 trainable parameters instead of ~2.03B. Set to "1"
      # only together with batch_job_gpus = 4.
      { name = "TUNE_VISUAL", value = "0" },
      { name = "MAX_STEPS", value = "100" },
      # MANDATORY and must stay consistent with DATASET_S3_URI below. The image
      # defaults to "ur3" so the validated UR3 flow is unchanged when nothing is
      # set, which means a job definition pointing at 28-DOF G1 joint data has to
      # say so explicitly: the UR3 config is end-effector (XYZ_ROTVEC) and raises
      # inside scipy's Rotation.from_rotvec on a 7-element joint group.
      { name = "EMBODIMENT_CONFIG", value = "g1_dex3" },
      { name = "AWS_DEFAULT_REGION", value = local.region },
      { name = "CHECKPOINT_BUCKET", value = data.aws_ssm_parameter.checkpoints_bucket[0].value },
      { name = "DATASET_S3_URI", value = "s3://${data.aws_ssm_parameter.datasets_bucket.value}/groot-data/pick_place_data/" },
      # Secret id only -- the value is resolved in-container at runtime.
      { name = "HF_TOKEN_SECRET_ID", value = "${var.project_name}/hf-token" }
    ]
  })

  # Both mismatches fail silently at runtime, hours after apply, so they are
  # caught here instead: more GPUs than the instance has parks the job in
  # RUNNABLE with no statusReason, and a card too small to hold the model lets
  # the job launch and then OOM mid-load.
  lifecycle {
    precondition {
      condition     = var.batch_job_gpus == local.batch_instance_gpus
      error_message = "batch_job_gpus (${var.batch_job_gpus}) does not match the ${local.batch_instance_gpus} GPUs on batch_instance_type ${var.batch_instance_type}. Use g6e.4xlarge with 16 / 122880 / 1, or g6e.12xlarge with 48 / 180000 / 4."
    }
    precondition {
      condition     = local.batch_instance_gpu_mib >= local.batch_min_gpu_mib
      error_message = "batch_instance_type ${var.batch_instance_type} has ${local.batch_instance_gpu_mib} MiB per GPU, below the ${local.batch_min_gpu_mib} MiB this training job needs. GR00T N1.7-3B occupies 22,023 MiB static at TUNE_VISUAL=0 before activations; use a 48 GB-class card such as g6e.4xlarge."
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# --- Outputs ---

output "batch_compute_environment" {
  description = "Batch compute environment name for GR00T training"
  value       = var.enable_batch ? aws_batch_compute_environment.groot[0].name : null
}

output "batch_job_queue" {
  description = "Batch job queue name for GR00T training"
  value       = var.enable_batch ? aws_batch_job_queue.groot[0].name : null
}
