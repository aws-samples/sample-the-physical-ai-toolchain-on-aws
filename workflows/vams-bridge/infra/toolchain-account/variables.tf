variable "aws_region" {
  description = "Region the toolchain Foundation is deployed in. Read from the Foundation SSM parameters by the caller; no default on purpose (the repo's scripts and config.json disagree on a default)."
  type        = string
}

variable "project_name" {
  description = "Project name prefix (must match the Foundation)"
  type        = string
  default     = "physical-ai"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "vams_account_id" {
  description = "AWS account id of the VAMS deployment. Scopes the bridge bus policy and derives the ARNs of the two VAMS-account roles (fixed names, see locals)."
  type        = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.vams_account_id))
    error_message = "vams_account_id must be a 12-digit AWS account id."
  }
}

variable "vams_region" {
  description = "Region of the VAMS deployment (derives the bridge log group ARN the runner registers with VAMS)."
  type        = string
}

variable "vams_workflow_execution_role_arn" {
  description = "VAMS workflow state-machine role ARN (SSM <vams prefix>/integration/workflowExecutionRoleArn). The only VAMS principal allowed to put events on the bridge bus (direct mode, VAMS 2.7+). In relay mode pass the vams-account root's relay_role_arn output instead."
  type        = string
}

variable "vams_orchestration_bus_arn" {
  description = "VAMS orchestration event bus ARN (SSM <vams prefix>/integration/orchestrationBusArn). The runner registers sub-executions there under the control role."
  type        = string
}

variable "vams_role_external_id" {
  description = "ExternalId both VAMS-account roles require. Generate once (e.g. a UUID), store in this account's Secrets Manager, pass the same value to both roots."
  type        = string
  sensitive   = true
}

variable "sagemaker_pipeline_name" {
  description = "Name of the GR00T fine-tune SageMaker Pipeline (training/gr00t/pipeline.py)"
  type        = string
  default     = "groot-finetune-pipeline"
}

variable "sagemaker_execution_role_arn" {
  description = "Foundation SageMaker execution role ARN (SSM /physical-ai/sagemaker-role-arn), passed to CreateTrainingJob by the training-job runner."
  type        = string
}

variable "batch_job_queue_arn" {
  description = "Batch job queue ARN the N1.7 training job definition runs on (isaac-gr00t-on-aws/infra/batch.tf)."
  type        = string
}

variable "batch_job_definition_name" {
  description = "Batch job definition NAME (not ARN) for GR00T N1.7 training."
  type        = string
}

variable "datasets_bucket" {
  description = "Foundation datasets bucket name (SSM /physical-ai/datasets-bucket)"
  type        = string
}

variable "checkpoints_bucket" {
  description = "Foundation checkpoints bucket name (SSM /physical-ai/checkpoints-bucket)"
  type        = string
}

variable "hf_token_secret_id" {
  description = "Secrets Manager secret id holding the Hugging Face token. Resolved by the job at run time; never a VAMS tag."
  type        = string
  default     = "physical-ai/hf-token"
}

variable "tags" {
  type    = map(string)
  default = {}
}
