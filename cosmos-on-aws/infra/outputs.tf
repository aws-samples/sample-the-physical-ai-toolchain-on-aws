output "cosmos_transfer_ecr_uri" {
  description = "ECR URI for the Cosmos Transfer 2.5 container"
  value       = var.enable_image_builds ? data.aws_ssm_parameter.cosmos_transfer_ecr[0].value : null
  sensitive   = true
}

output "cosmos3_ecr_uri" {
  description = "ECR URI for the Cosmos 3 container"
  value       = var.enable_image_builds ? data.aws_ssm_parameter.cosmos3_ecr[0].value : null
  sensitive   = true
}

output "codebuild_transfer_project" {
  description = "CodeBuild project for Cosmos Transfer container"
  value       = var.enable_image_builds ? aws_codebuild_project.cosmos_transfer[0].name : null
}

output "codebuild_cosmos3_project" {
  description = "CodeBuild project for Cosmos 3 container"
  value       = var.enable_image_builds ? aws_codebuild_project.cosmos3[0].name : null
}
