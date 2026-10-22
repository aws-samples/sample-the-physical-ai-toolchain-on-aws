# =============================================================================
# EC2: COSMOS 3 GENERATION SERVER (p5.48xlarge/8x H100 for Super, or any
# single-GPU type like g6e.4xlarge for Nano — see cosmos3-job.yaml/README.md
# "Choosing Super vs Nano")
# =============================================================================
#
# Two launch modes, both through the same aws_instance resource:
#   - Capacity Block (default, required for Super/p5.48xlarge — P5 capacity
#     is scarce): set capacity_reservation_id + availability_zone. Capacity
#     Block purchase itself is NOT managed by Terraform — it's an async
#     external reservation (minutes to days to become active) that doesn't
#     fit a `terraform apply` lifecycle. Purchase it first (see
#     ec2-deployment-guide.md Step 1), then set capacity_reservation_id and
#     apply.
#   - On-demand (for Nano/g6e.* — commonly available without a reservation):
#     leave capacity_reservation_id empty. availability_zone is then optional;
#     Terraform picks any private subnet if unset.
#
# Toggle with var.enable_ec2_server (default false) so `terraform apply` is
# safe to run before you have an active Capacity Block (or when you don't
# need one, for on-demand Nano launches).

variable "enable_ec2_server" {
  description = "Deploy the Cosmos 3 EC2 generation server."
  type        = bool
  default     = false
}

variable "capacity_reservation_id" {
  description = "ID of an ACTIVE EC2 Capacity Block to launch into (e.g. cr-0123456789abcdef0). Leave empty for an on-demand launch (works for commonly-available types like g6e.* used by Nano; P5/Super needs a Capacity Block — see Step 1 of ec2-deployment-guide.md)."
  type        = string
  default     = ""
}

variable "availability_zone" {
  description = "AZ to launch in. Required to match the Capacity Block's reserved AZ when capacity_reservation_id is set; optional otherwise (Terraform picks any private subnet if left empty)."
  type        = string
  default     = ""
}

variable "server_instance_type" {
  description = "Instance type for the Cosmos 3 server. Must match the Capacity Block's reserved type when using one (p5.48xlarge for Super); any available on-demand GPU type works otherwise (e.g. g6e.4xlarge for Nano)."
  type        = string
  default     = "p5.48xlarge"
}

variable "server_ami_id" {
  description = "AMI for the Cosmos 3 server (Deep Learning Base OSS Nvidia Driver GPU AMI recommended)"
  type        = string
  default     = ""
}

variable "server_volume_size_gb" {
  description = "Root EBS volume size in GB (needs room for ~30GB container + ~126GB model weights)"
  type        = number
  default     = 500
}

variable "server_instance_profile" {
  description = "Instance profile NAME for the Cosmos 3 EC2 server. Empty (default) reads /<project_name>/cosmos-instance-profile from SSM (Foundation)."
  type        = string
  default     = ""
}

# Network from SSM (Foundation) only when it was not passed in explicitly.
data "aws_ssm_parameter" "vpc_id" {
  count = var.enable_ec2_server && var.vpc_id == "" ? 1 : 0
  name  = "/${var.project_name}/vpc-id"
}

data "aws_ssm_parameter" "private_subnet_ids" {
  count = var.enable_ec2_server && length(var.private_subnet_ids) == 0 ? 1 : 0
  name  = "/${var.project_name}/private-subnet-ids"
}

locals {
  ec2_vpc_id = var.vpc_id != "" ? var.vpc_id : (var.enable_ec2_server ? data.aws_ssm_parameter.vpc_id[0].value : "")
  ec2_private_subnet_ids = length(var.private_subnet_ids) > 0 ? var.private_subnet_ids : (
    var.enable_ec2_server ? split(",", data.aws_ssm_parameter.private_subnet_ids[0].value) : []
  )
  ec2_instance_profile = var.server_instance_profile != "" ? var.server_instance_profile : (
    length(data.aws_ssm_parameter.cosmos_instance_profile) > 0 ? data.aws_ssm_parameter.cosmos_instance_profile[0].value : ""
  )
}

# Pick a private subnet. Filtered to a specific AZ when one is given (required
# to match a Capacity Block's reserved AZ); otherwise picks the first
# available private subnet (fine for on-demand launches with no AZ pin).
data "aws_subnets" "server_candidates" {
  count = var.enable_ec2_server ? 1 : 0
  filter {
    name   = "subnet-id"
    values = local.ec2_private_subnet_ids
  }
  dynamic "filter" {
    for_each = var.availability_zone != "" ? [var.availability_zone] : []
    content {
      name   = "availability-zone"
      values = [filter.value]
    }
  }
}

locals {
  server_subnet_id = var.enable_ec2_server ? data.aws_subnets.server_candidates[0].ids[0] : ""
}

# Latest Deep Learning Base OSS Nvidia Driver GPU AMI, unless pinned explicitly.
data "aws_ami" "dlami" {
  count       = var.enable_ec2_server && var.server_ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"]
  }
}

locals {
  server_ami_id = var.server_ami_id != "" ? var.server_ami_id : (
    var.enable_ec2_server ? data.aws_ami.dlami[0].image_id : ""
  )
}

resource "aws_security_group" "cosmos_server" {
  count       = var.enable_ec2_server ? 1 : 0
  name        = "${local.prefix}-cosmos-server-sg"
  description = "Cosmos 3 generation server - outbound only (SSM, no inbound needed)"
  vpc_id      = local.ec2_vpc_id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound (ECR, HuggingFace, S3)"
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-server"
  }
}

resource "aws_instance" "cosmos_server" {
  count                  = var.enable_ec2_server ? 1 : 0
  ami                    = local.server_ami_id
  instance_type          = var.server_instance_type
  subnet_id              = local.server_subnet_id
  vpc_security_group_ids = [aws_security_group.cosmos_server[0].id]
  iam_instance_profile   = local.ec2_instance_profile

  # Launch into a Capacity Block only when one is specified. Omitted entirely
  # for on-demand launches (e.g. Nano on g6e.*) — Capacity Blocks require
  # instance_market_options=capacity-block, which is invalid without a target.
  dynamic "capacity_reservation_specification" {
    for_each = var.capacity_reservation_id != "" ? [var.capacity_reservation_id] : []
    content {
      capacity_reservation_target {
        capacity_reservation_id = capacity_reservation_specification.value
      }
    }
  }

  dynamic "instance_market_options" {
    for_each = var.capacity_reservation_id != "" ? [1] : []
    content {
      market_type = "capacity-block"
    }
  }

  root_block_device {
    volume_size = var.server_volume_size_gb
    volume_type = "gp3"
    encrypted   = true
  }

  metadata_options {
    http_tokens = "required" # IMDSv2
  }

  tags = {
    Name        = "${local.prefix}-cosmos3-server"
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-server"
  }
}

output "server_instance_id" {
  description = "Instance ID of the Cosmos 3 generation server (if enabled)"
  value       = var.enable_ec2_server ? aws_instance.cosmos_server[0].id : null
}

output "server_private_ip" {
  description = "Private IP of the Cosmos 3 generation server (if enabled)"
  value       = var.enable_ec2_server ? aws_instance.cosmos_server[0].private_ip : null
}
