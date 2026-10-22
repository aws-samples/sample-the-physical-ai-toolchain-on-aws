variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name prefix (must match Foundation when reading its SSM parameters)"
  type        = string
  default     = "physical-ai"
}

variable "vpc_id" {
  description = "VPC to deploy into. Empty (default) reads /<project_name>/vpc-id from SSM (Foundation). Set together with private_subnet_ids."
  type        = string
  default     = ""
  validation {
    condition     = var.vpc_id == "" || can(regex("^vpc-[0-9a-f]{8,17}$", var.vpc_id))
    error_message = "vpc_id must be empty or a VPC ID (vpc-xxxxxxxx)."
  }
}

variable "private_subnet_ids" {
  description = "Private subnets for EKS node groups and the EC2 server. Empty (default) reads /<project_name>/private-subnet-ids from SSM. Must belong to vpc_id."
  type        = list(string)
  default     = []
  validation {
    condition     = alltrue([for s in var.private_subnet_ids : can(regex("^subnet-[0-9a-f]{8,17}$", s))])
    error_message = "private_subnet_ids must contain only subnet IDs (subnet-xxxxxxxx)."
  }
}

variable "enable_image_builds" {
  description = "Create the Cosmos Transfer / Cosmos 3 image-build CodeBuild projects and their role. They push to Foundation-created ECR repositories (read from SSM), so turn this off in accounts without a Foundation deployment. The workshop's event-provision.sh sets it to false."
  type        = bool
  default     = true
}

variable "cosmos_transfer_repo" {
  description = "GitHub repo URL for Cosmos Transfer 2.5 source"
  type        = string
  default     = "https://github.com/nvidia-cosmos/cosmos-transfer2.5.git"
}

variable "cosmos_transfer_ref" {
  description = "Git ref (tag/branch) for Cosmos Transfer build"
  type        = string
  default     = "v1.5.4"
}

variable "cosmos3_repo" {
  description = "GitHub repo URL for Cosmos 3 framework source"
  type        = string
  default     = "https://github.com/NVIDIA/cosmos-framework.git"
}

variable "cosmos3_ref" {
  description = "Git ref (tag/branch) for Cosmos 3 build"
  type        = string
  default     = "main"
}
