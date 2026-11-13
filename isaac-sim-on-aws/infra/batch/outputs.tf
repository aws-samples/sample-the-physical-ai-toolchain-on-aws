output "bucket_name" {
  description = "S3 bucket for configs, assets, and results (use for uploads/downloads and generate_configs.py)"
  value       = aws_s3_bucket.sim.bucket
}

output "repo_uri" {
  description = "ECR repository URI for the Isaac Sim container image"
  value       = aws_ecr_repository.isaac_sim.repository_url
}

output "job_queue" {
  description = "AWS Batch job queue name (submit_job.py / submit_array_job.py target this)"
  value       = aws_batch_job_queue.sim.name
}

output "dashboard_url" {
  description = "CloudWatch dashboard URL"
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards/dashboard/Isaac-Sim"
}
