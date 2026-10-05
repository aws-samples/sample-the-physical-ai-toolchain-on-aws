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
  description = "Project name prefix"
  type        = string
  default     = "physical-ai"
}

variable "vpc_cidr" {
  description = "CIDR block for the Batch VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "gpu_instance_types" {
  description = "GPU instance types the Batch compute environment may launch (Isaac Sim needs RT Cores — G family only)"
  type        = list(string)
  default     = ["g6e.2xlarge", "g6.2xlarge", "g5.2xlarge", "g5.4xlarge"]
}

variable "max_vcpus" {
  description = "Maximum vCPUs for the GPU compute environment (an array of 4 jobs needs 32)"
  type        = number
  default     = 80
}

variable "image_tag" {
  description = "ECR image tag the job definition runs (immutable; bump to release new code)"
  type        = string
  default     = "v1"
}

variable "alert_email" {
  description = "Email address to subscribe to job-failure alerts (optional — leave empty to skip)"
  type        = string
  default     = ""
}
