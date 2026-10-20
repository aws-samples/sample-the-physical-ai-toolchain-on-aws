variable "aws_region" {
  description = "Region of the VAMS deployment"
  type        = string
}

variable "vams_resource_param_prefix" {
  description = "VAMS resource-name SSM prefix, e.g. /<vams name>-<baseStackName>/resourceNames. Everything about the VAMS deployment (bus ARN, event-source prefix, import Lambda, workflow role, artefacts bucket) is read from parameters under it."
  type        = string
}

variable "environment" {
  description = "Environment name shared with the toolchain-account root (dev, staging, prod). Both roots derive each other's resource names from it, so the two values must match."
  type        = string
  default     = "dev"
}

variable "toolchain_account_id" {
  description = "AWS account id the toolchain and its bridge bus live in. With `environment` this derives the runner role ARN (the only principal the two VAMS-account roles trust) and the bridge bus ARN."
  type        = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.toolchain_account_id))
    error_message = "toolchain_account_id must be a 12-digit AWS account id."
  }
}

variable "toolchain_region" {
  description = "Region the toolchain-account root is deployed in (derives the bridge bus ARN)."
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

variable "vams_deployment_name" {
  description = "The VAMS deployment's `name` from its config.json. Required when vams_relay = true: the VAMS 2.6.x workflow role may only put events on a bus in its own account whose name carries this value."
  type        = string
  default     = ""
  validation {
    condition     = !var.vams_relay || length(var.vams_deployment_name) > 0
    error_message = "vams_deployment_name is required when vams_relay is true."
  }
}

variable "vams_asset_bucket_names" {
  description = "Names of the VAMS asset buckets whose files may be bridge inputs and which hold the run output prefixes (the buckets listed in the VAMS buckets table). The data role reads objects in these buckets and writes only under their pipelines/*/output/ prefixes."
  type        = list(string)
  validation {
    condition     = length(var.vams_asset_bucket_names) > 0
    error_message = "List at least one VAMS asset bucket."
  }
}

variable "vams_kms_key_arn" {
  description = "ARN of the VAMS customer managed key when the deployment sets useKmsCmkEncryption.enabled. Empty when the deployment uses service-managed keys."
  type        = string
  default     = ""
}

variable "components" {
  description = "vamsSchema bundles to register, by directory name under ../../vamsSchema"
  type        = list(string)
  default     = ["gr00t-finetune", "gr00t-finetune-batch"]
}

variable "tags" {
  type    = map(string)
  default = {}
}
