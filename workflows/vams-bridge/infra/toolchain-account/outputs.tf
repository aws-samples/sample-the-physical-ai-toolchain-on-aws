output "bridge_bus_arn" {
  description = "Value for the VAMS pipeline's executionConfig.eventBridge.busArn (resourceOverrides.eventBridgeBusArn) and for externalPipelineBridges[].invokeTargets in the VAMS config.json."
  value       = aws_cloudwatch_event_bus.bridge.arn
}

output "runner_role_arn" {
  description = "Runner execution role (fixed name pai-vams-bridge-runner-<environment>). The vams-account root derives this same ARN from toolchain_account_id + environment."
  value       = aws_iam_role.runner.arn
}

output "runner_role_arn_pattern" {
  description = "externalPipelineBridges[].roleArnPattern for the VAMS config.json."
  value       = "arn:${local.partition}:iam::${local.account_id}:role/pai-vams-bridge-runner-*"
}

output "runner_state_machine_arns" {
  value = { for k, sm in aws_sfn_state_machine.runner : k => sm.arn }
}

output "vams_config_snippet" {
  description = "Paste into the VAMS deployment's config.json under app.pipelines.externalPipelineBridges (VAMS 2.7+ direct mode)."
  value = jsonencode({
    accountId      = local.account_id
    roleArnPattern = "arn:${local.partition}:iam::${local.account_id}:role/pai-vams-bridge-runner-*"
    invokeTargets  = [aws_cloudwatch_event_bus.bridge.arn]
  })
}
