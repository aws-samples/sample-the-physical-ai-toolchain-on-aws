output "control_role_arn" {
  description = "VAMS-account role the runner assumes for callbacks (fixed name; the toolchain root derives the same ARN)."
  value       = aws_iam_role.control.arn
}

output "data_role_arn" {
  description = "VAMS-account role ReadManifest/Collect assume for inputs and outputs (fixed name; derived by the toolchain root)."
  value       = aws_iam_role.data.arn
}

output "bridge_log_group_name" {
  value = aws_cloudwatch_log_group.bridge.name
}

output "vams_orchestration_bus_arn" {
  description = "Read from SSM; pass to the toolchain-account root as vams_orchestration_bus_arn."
  value       = data.aws_ssm_parameter.orchestration_bus_arn.value
}

output "vams_workflow_execution_role_arn" {
  description = "Read from SSM; pass to the toolchain-account root as vams_workflow_execution_role_arn in direct mode (VAMS 2.7+)."
  value       = data.aws_ssm_parameter.workflow_execution_role_arn.value
}

output "relay_role_arn" {
  description = "Relay mode (VAMS 2.6.x) only: pass THIS to the toolchain-account root as vams_workflow_execution_role_arn, because the relay rule's role is what puts events on the bridge bus."
  value       = var.vams_relay ? aws_iam_role.relay[0].arn : null
}

output "relay_bus_arn" {
  description = "Set only when vams_relay = true; the bus the VAMS pipeline is registered against."
  value       = var.vams_relay ? aws_cloudwatch_event_bus.relay[0].arn : null
}

output "registration_results" {
  description = "Import Lambda responses, one per registered component."
  value       = { for k, r in aws_lambda_invocation.register : k => r.result }
}
