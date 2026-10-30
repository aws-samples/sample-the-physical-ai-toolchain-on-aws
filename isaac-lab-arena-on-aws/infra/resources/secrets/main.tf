terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.64"
    }
  }
}

variable "account_id" { type = string }
variable "region" { type = string }
variable "profile" {
  type    = string
  default = null
}
variable "name" { type = string }
variable "secret_names" { type = set(string) }

provider "aws" {
  region              = var.region
  profile             = var.profile
  allowed_account_ids = [var.account_id]
}

# Values are supplied directly through Secrets Manager, never through Terraform
# variables/state, shell arguments or notebook output.
resource "aws_secretsmanager_secret" "token" {
  for_each                = var.secret_names
  name                    = each.value
  recovery_window_in_days = 0
  tags = {
    Project   = var.name
    ManagedBy = "pai-arena"
  }
}

output "secret_arns" {
  value = { for name, secret in aws_secretsmanager_secret.token : name => secret.arn }
}
