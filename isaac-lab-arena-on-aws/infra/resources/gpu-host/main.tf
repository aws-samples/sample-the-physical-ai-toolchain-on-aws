# A disposable host for the existing Arena EC2 path. Application preparation
# stays in `pai arena deploy --prepare-host`; this module owns only its host,
# instance role/profile, security group and development bucket.
terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.64"
    }
  }
}

provider "aws" {
  region              = var.region
  profile             = var.profile
  allowed_account_ids = [var.account_id]
  default_tags {
    tags = {
      Project      = var.project
      DeploymentId = var.name
      ManagedBy    = "pai-arena"
    }
  }
}

variable "account_id" {
  type = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.account_id))
    error_message = "Supply the intended twelve-digit workload account."
  }
}
variable "profile" {
  type    = string
  default = null
}
variable "region" {
  type    = string
  default = "us-east-1"
  validation {
    condition     = var.region == "us-east-1"
    error_message = "This Arena GPU host recipe currently supports us-east-1."
  }
}
variable "name" {
  type = string
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,29}$", var.name))
    error_message = "Use a unique 2-30-character lowercase name."
  }
}
variable "project" {
  type = string
}
variable "subnet_id" {
  type        = string
  default     = null
  description = "A Foundation private subnet with NAT egress."
  validation {
    condition     = (var.subnet_id != null) != (var.extra_subnet != null)
    error_message = "Supply subnet_id or extra_subnet, exclusively."
  }
}
variable "extra_subnet" {
  type = object({
    availability_zone = string
    cidr_block        = string
    route_table_id    = string
  })
  default     = null
  description = "Optional owned private subnet in another zone, using an existing NAT route table."
}
variable "vpc_id" {
  type = string
}
variable "ami_id" {
  type        = string
  description = "Pinned Amazon Ubuntu 22.04 Base OSS NVIDIA GPU DLAMI."
}
variable "instance_type" {
  type        = string
  description = "Explicit GPU choice; the Arena walkthrough uses g6e.8xlarge."
}
variable "root_volume_gib" {
  type    = number
  default = 800
  validation {
    condition     = var.root_volume_gib >= 800
    error_message = "Use at least the walkthrough's 800 GiB root allocation."
  }
}

data "aws_subnet" "selected" {
  id = var.subnet_id != null ? var.subnet_id : aws_subnet.extra[0].id
}
data "aws_ami" "selected" {
  owners = ["amazon"]
  filter {
    name   = "image-id"
    values = [var.ami_id]
  }
}

data "aws_route_table" "egress" {
  count          = var.extra_subnet == null ? 0 : 1
  route_table_id = var.extra_subnet.route_table_id
}
resource "aws_subnet" "extra" {
  count                   = var.extra_subnet == null ? 0 : 1
  vpc_id                  = var.vpc_id
  availability_zone       = var.extra_subnet.availability_zone
  cidr_block              = var.extra_subnet.cidr_block
  map_public_ip_on_launch = false
  tags                    = { Name = "${var.name}-private" }
}
resource "aws_route_table_association" "extra" {
  count          = var.extra_subnet == null ? 0 : 1
  subnet_id      = aws_subnet.extra[0].id
  route_table_id = var.extra_subnet.route_table_id
  lifecycle {
    precondition {
      condition = (
        data.aws_route_table.egress[0].vpc_id == var.vpc_id &&
        anytrue([for route in data.aws_route_table.egress[0].routes :
          route.cidr_block == "0.0.0.0/0" && try(length(route.nat_gateway_id) > 0, false)
        ])
      )
      error_message = "The supplied route table must belong to this VPC and provide NAT egress."
    }
  }
}

resource "aws_security_group" "host" {
  name        = "${var.name}-host"
  description = "Arena GPU host: outbound access, Session Manager, no inbound ports"
  vpc_id      = var.vpc_id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
resource "aws_iam_role" "host" {
  name = "${var.name}-host"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}
resource "aws_iam_role_policy_attachment" "host" {
  for_each   = toset(["AmazonSSMManagedInstanceCore", "AmazonEC2ContainerRegistryReadOnly"])
  role       = aws_iam_role.host.name
  policy_arn = "arn:aws:iam::aws:policy/${each.value}"
}
resource "aws_iam_instance_profile" "host" {
  name = "${var.name}-host"
  role = aws_iam_role.host.name
}
resource "aws_s3_bucket" "development" {
  bucket        = "${var.name}-localdev-${var.account_id}"
  force_destroy = false
}
resource "aws_s3_bucket_versioning" "development" {
  bucket = aws_s3_bucket.development.id
  versioning_configuration {
    status = "Enabled"
  }
}
resource "aws_s3_bucket_public_access_block" "development" {
  bucket                  = aws_s3_bucket.development.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_instance" "host" {
  ami                         = var.ami_id
  instance_type               = var.instance_type
  subnet_id                   = data.aws_subnet.selected.id
  vpc_security_group_ids      = [aws_security_group.host.id]
  iam_instance_profile        = aws_iam_instance_profile.host.name
  associate_public_ip_address = false
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
    instance_metadata_tags      = "disabled"
  }
  root_block_device {
    volume_size           = var.root_volume_gib
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
    iops                  = 3000
    throughput            = 125
  }
  tags       = { Name = var.name }
  depends_on = [aws_iam_role_policy_attachment.host, aws_route_table_association.extra]
  lifecycle {
    precondition {
      condition     = data.aws_subnet.selected.vpc_id == var.vpc_id
      error_message = "The supplied subnet belongs to another VPC."
    }
    precondition {
      condition = (
        data.aws_ami.selected.architecture == "x86_64" &&
        startswith(data.aws_ami.selected.name, "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)")
      )
      error_message = "Select the documented Amazon x86-64 Ubuntu 22.04 GPU DLAMI."
    }
  }
}

output "instance_id" {
  value = aws_instance.host.id
}
output "runtime_role" {
  value = aws_iam_role.host.name
}
output "development_bucket" {
  value = aws_s3_bucket.development.id
}
output "retention_notice" {
  value = "Archive evidence before removing the host. Empty only this owned bucket's recorded objects/versions after preserving evidence. Supplied VPC/subnet/AMI remain."
}
