# =============================================================================
# AWS BATCH: Multi-Node RL Training (optional — for distributed training path)
# =============================================================================
# Provisions a Batch compute environment with GPU instances, a job queue,
# and a multi-node job definition for distributed Isaac Lab RL training.
#
# Enable by setting var.enable_batch = true and providing a VPC.
# =============================================================================

variable "enable_batch" {
  description = "Enable AWS Batch compute environment for distributed RL training"
  type        = bool
  default     = false
}

variable "vpc_id" {
  description = "VPC ID for Batch compute instances (uses the isaac-lab VPC if empty)"
  type        = string
  default     = ""
}

variable "subnet_ids" {
  description = "Subnet IDs for Batch compute instances (uses private subnets from isaac-lab VPC if empty)"
  type        = list(string)
  default     = []
}

locals {
  batch_vpc_id     = var.vpc_id != "" ? var.vpc_id : local.foundation_vpc_id
  batch_subnet_ids = length(var.subnet_ids) > 0 ? var.subnet_ids : local.foundation_subnet_ids
}

variable "batch_instance_type" {
  description = "Instance type for Batch compute (GPU required)"
  type        = string
  default     = "g6e.4xlarge"
}

variable "batch_max_vcpus" {
  description = "Maximum vCPUs for the Batch compute environment"
  type        = number
  default     = 96
}

# Security group for Batch compute nodes (NCCL inter-node + EFS)
resource "aws_security_group" "batch" {
  count       = var.enable_batch ? 1 : 0
  name        = "${local.prefix}-batch-rl-sg"
  description = "Batch RL compute: self-referencing for NCCL + EFS"
  vpc_id      = local.batch_vpc_id

  # Self-referencing: allow all traffic between compute nodes (NCCL)
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    self        = true
    description = "NCCL inter-node communication"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = {
    Name        = "${local.prefix}-batch-rl-sg"
    Project     = var.project_name
    Environment = var.environment
  }
}

# IAM role for Batch compute instances
resource "aws_iam_role" "batch_instance" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-batch-rl-instance-role"

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

resource "aws_iam_role_policy_attachment" "batch_ssm" {
  count      = var.enable_batch ? 1 : 0
  role       = aws_iam_role.batch_instance[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "batch_s3_ecr" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-batch-rl-s3-ecr"
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
        Resource = ["arn:aws:ecr:${local.region}:${local.account_id}:repository/${var.project_name}/isaac-lab"]
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.environment}-*",
          "arn:aws:s3:::${var.project_name}-${var.environment}-*/*"
        ]
      }
    ]
  })
}

resource "aws_iam_instance_profile" "batch" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-batch-rl-profile"
  role  = aws_iam_role.batch_instance[0].name
}

# Batch service role
resource "aws_iam_role" "batch_service" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-batch-service-role"

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

# Launch template with 512 GiB root volume (Isaac Lab image is ~16 GB)
resource "aws_launch_template" "batch" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-batch-rl-lt"

  block_device_mappings {
    device_name = "/dev/xvda"

    ebs {
      volume_size           = 512
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  tags = {
    Name        = "${local.prefix}-batch-rl-lt"
    Project     = var.project_name
    Environment = var.environment
  }
}

# Compute environment
resource "aws_batch_compute_environment" "rl" {
  count = var.enable_batch ? 1 : 0
  name  = "${local.prefix}-rl-compute"
  type  = "MANAGED"

  service_role = aws_iam_role.batch_service[0].arn

  compute_resources {
    type          = "EC2"
    instance_role = aws_iam_instance_profile.batch[0].arn
    instance_type = [var.batch_instance_type]
    max_vcpus     = var.batch_max_vcpus
    # Scale from zero. A non-zero floor here keeps a GPU instance alive even with an
    # empty queue: this environment held an idle g6e.4xlarge continuously from
    # 2026-09-23 to 2026-10-05 ($3.00/hr in us-east-2, ~$920) with no jobs ever
    # submitted, and it also consumed a slot of the g6e.4xlarge capacity reservation
    # that the GR00T training lab depends on. Batch scales this environment up on
    # demand, so a warm floor buys nothing that justifies that cost.
    # desired_vcpus is deliberately NOT set. It is the one field in this block that Batch
    # itself writes - it raises it to place a job and drops it back when the queue drains -
    # so pinning it here would show drift on every plan and a mid-job apply would push it
    # back to 0 and pull capacity out from under the running job. Omitted, the provider
    # leaves it computed, and with min_vcpus at 0 the environment still starts at 0.
    min_vcpus          = 0
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

# Job queue
resource "aws_batch_job_queue" "rl" {
  count    = var.enable_batch ? 1 : 0
  name     = "${local.prefix}-rl-queue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.rl[0].arn
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
  }
}

# Outputs (conditional)
output "batch_job_queue" {
  description = "Batch job queue name for RL training"
  value       = var.enable_batch ? aws_batch_job_queue.rl[0].name : null
}

output "batch_compute_environment" {
  description = "Batch compute environment name"
  value       = var.enable_batch ? aws_batch_compute_environment.rl[0].name : null
}
