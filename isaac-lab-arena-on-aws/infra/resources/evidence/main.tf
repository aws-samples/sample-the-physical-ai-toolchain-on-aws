# Separate from application state: this bucket survives application teardown.
terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 6.64" }
  }
}
variable "account_id" { type = string }
variable "region" { type = string }
variable "name" { type = string }
variable "profile" {
  type    = string
  default = null
}
provider "aws" {
  region              = var.region
  profile             = var.profile
  allowed_account_ids = [var.account_id]
}
resource "aws_s3_bucket" "evidence" {
  bucket        = "${var.name}-evidence-${var.account_id}"
  force_destroy = false
  tags = {
    DeploymentId = var.name
    Purpose      = "Retained runbook evidence"
  }
}
resource "aws_s3_bucket_versioning" "evidence" {
  bucket = aws_s3_bucket.evidence.id
  versioning_configuration { status = "Enabled" }
}
resource "aws_s3_bucket_public_access_block" "evidence" {
  bucket                  = aws_s3_bucket.evidence.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_server_side_encryption_configuration" "evidence" {
  bucket = aws_s3_bucket.evidence.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}
output "bucket" { value = aws_s3_bucket.evidence.id }
