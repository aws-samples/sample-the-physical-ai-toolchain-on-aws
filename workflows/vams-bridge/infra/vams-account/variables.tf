variable "aws_region" {
  description = "Region of the VAMS deployment"
  type        = string
}

variable "vams_resource_param_prefix" {
  description = "VAMS resource-name SSM prefix, e.g. /<vams name>-<baseStackName>/resourceNames. Everything about the VAMS deployment (bus ARN, event-source prefix, import Lambda, workflow role, artefacts bucket) is read from parameters under it."
  type        = string
}

variable "toolchain_account_id" {
  description = "AWS account id the toolchain and its bridge bus live in"
  type        = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.toolchain_account_id))
    error_message = "toolchain_account_id must be a 12-digit AWS account id."
  }
}

variable "toolchain_runner_role_arn" {
  description = "Output runner_role_arn of the toolchain-account root. The ONLY principal the two VAMS-account roles trust."
  type        = string
}

variable "toolchain_bridge_bus_arn" {
  description = "Output bridge_bus_arn of the toolchain-account root. Target of the VAMS pipeline (direct mode) or of the relay rule (relay mode)."
  type        = string
}

variable "vams_role_external_id" {
  description = "ExternalId both roles require on AssumeRole. Same value as passed to the toolchain-account root."
  type        = string
  sensitive   = true
}

variable "vams_relay" {
  description = "true for VAMS 2.6.x: create a relay bus + rule in this account because the VAMS workflow role cannot target another account. false for VAMS 2.7+ (direct mode via externalPipelineBridges)."
  type        = bool
  default     = false
}

variable "components" {
  description = "vamsSchema bundles to register, by directory name under ../../vamsSchema"
  type        = list(string)
  default     = ["gr00t-finetune"]
}

variable "tags" {
  type    = map(string)
  default = {}
}
