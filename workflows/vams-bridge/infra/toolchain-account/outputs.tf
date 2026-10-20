output "bridge_bus_arn" {
  description = "Value for the VAMS pipeline's executionConfig.eventBridge.busArn (resourceOverrides.eventBridgeBusArn) and for externalPipelineBridges[].invokeTargets in the VAMS config.json."
  value       = aws_cloudwatch_event_bus.bridge.arn
}

output "runner_role_arn" {
  description = "Runner execution role. Its name prefix is the externalPipelineBridges[].roleArnPattern the VAMS deployment lists."
  value       = aws_iam_role.runner.arn
}

output "runner_role_arn_pattern" {
  value = "arn:${local.partition}:iam::${local.account_id}:role/${local.name_prefix}-runner-*"
}

output "runner_state_machine_arns" {
  value = { for k, sm in aws_sfn_state_machine.runner : k => sm.arn }
}

output "vams_config_snippet" {
  description = "Paste into the VAMS deployment's config.json (VAMS 2.7+ direct mode)."
  value = jsonencode({
    accountId      = local.account_id
    roleArnPattern = "arn:${local.partition}:iam::${local.account_id}:role/${local.name_prefix}-runner-*"
    invokeTargets  = [aws_cloudwatch_event_bus.bridge.arn]
  })
}
