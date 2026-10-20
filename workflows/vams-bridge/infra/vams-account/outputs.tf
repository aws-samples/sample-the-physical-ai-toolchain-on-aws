output "control_role_arn" {
  description = "Pass to the toolchain-account root as vams_control_role_arn."
  value       = aws_iam_role.control.arn
}

output "data_role_arn" {
  description = "Pass to the toolchain-account root as vams_data_role_arn."
  value       = aws_iam_role.data.arn
}

output "bridge_log_group_name" {
  value = aws_cloudwatch_log_group.bridge.name
}

output "bridge_log_group_arn" {
  value = aws_cloudwatch_log_group.bridge.arn
}

output "vams_orchestration_bus_arn" {
  description = "Read from SSM; pass to the toolchain-account root."
  value       = data.aws_ssm_parameter.orchestration_bus_arn.value
}

output "vams_workflow_execution_role_arn" {
  description = "Read from SSM; pass to the toolchain-account root (direct mode) so the bridge bus admits it."
  value       = data.aws_ssm_parameter.workflow_execution_role_arn.value
}

output "relay_bus_arn" {
  description = "Set only when vams_relay = true."
  value       = var.vams_relay ? aws_cloudwatch_event_bus.relay[0].arn : null
}
