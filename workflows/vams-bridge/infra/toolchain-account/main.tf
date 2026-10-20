# workflows/vams-bridge — toolchain-account root
#
# Resources that live in the TOOLCHAIN account: the bridge event bus VAMS targets, one rule per
# runner, the runner execution role, the three runner state machines, and the ReadManifest /
# Collect Lambdas. Nothing here holds a VAMS credential; the runner assumes the VAMS-account roles
# created by ../vams-account at run time.
#
# Status: skeleton. Every resource is declared; bodies marked TODO are incomplete.

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  name_prefix = "pai-vams-bridge"
  partition   = data.aws_partition.current.partition
  account_id  = data.aws_caller_identity.current.account_id
  common_tags = merge(var.tags, { Project = var.project_name, Environment = var.environment, Module = "vams-bridge" })

  # One runner per submit API. detail_type is what the VAMS pipeline.json names in
  # executionConfig.eventBridge.detailType; the rule routes on it.
  runners = {
    "sagemaker-pipeline" = { asl = "runner-sagemaker-pipeline.asl.json", detail_types = ["pai.gr00t-finetune"] }
    "sagemaker-training" = { asl = "runner-sagemaker-training.asl.json", detail_types = ["pai.isaac-lab-rl"] }
    "batch"              = { asl = "runner-batch.asl.json", detail_types = ["pai.gr00t-finetune-batch", "pai.isaac-lab-rl-batch"] }
  }
}

# ---------------------------------------------------------------------------------------------
# Bridge event bus. In direct mode (VAMS 2.7+) the VAMS workflow role puts events here across
# accounts, so the resource policy admits exactly that role and nothing else from the VAMS account.
# In relay mode (VAMS 2.6.x) the VAMS-account relay rule's role is the principal instead; the
# vams-account root passes its ARN in as vams_workflow_execution_role_arn.
# ---------------------------------------------------------------------------------------------
resource "aws_cloudwatch_event_bus" "bridge" {
  name = local.name_prefix
  tags = local.common_tags
}

data "aws_iam_policy_document" "bridge_bus" {
  statement {
    sid     = "VamsWorkflowRolePutEvents"
    effect  = "Allow"
    actions = ["events:PutEvents"]
    principals {
      type        = "AWS"
      identifiers = [var.vams_workflow_execution_role_arn]
    }
    resources = [aws_cloudwatch_event_bus.bridge.arn]
    condition {
      test     = "StringEquals"
      variable = "aws:PrincipalAccount"
      values   = [var.vams_account_id]
    }
  }
}

resource "aws_cloudwatch_event_bus_policy" "bridge" {
  event_bus_name = aws_cloudwatch_event_bus.bridge.name
  policy         = data.aws_iam_policy_document.bridge_bus.json
}

# ---------------------------------------------------------------------------------------------
# Log groups
# ---------------------------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "runner" {
  for_each          = local.runners
  name              = "/aws/vendedlogs/states/${local.name_prefix}-${each.key}"
  retention_in_days = 365
  tags              = local.common_tags
}

resource "aws_cloudwatch_log_group" "lambda" {
  for_each          = toset(["read-manifest", "collect"])
  name              = "/aws/lambda/${local.name_prefix}-${each.key}"
  retention_in_days = 365
  tags              = local.common_tags
}

# ---------------------------------------------------------------------------------------------
# Runner execution role. Specific actions on specific resources (ARCC BSC6 / IAM least privilege):
# the job APIs on the Foundation's pipeline/queue/job prefix, sts:AssumeRole on the two VAMS roles
# only, invoke on the two Lambdas, logging on its own groups.
# ---------------------------------------------------------------------------------------------
data "aws_iam_policy_document" "runner_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["states.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "runner" {
  # Name prefix is what the VAMS deployment lists as externalPipelineBridges[].roleArnPattern
  # ("...:role/pai-vams-bridge-runner-*"), so keep the prefix stable.
  name_prefix        = "${local.name_prefix}-runner-"
  assume_role_policy = data.aws_iam_policy_document.runner_trust.json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "runner" {
  statement {
    sid       = "AssumeVamsRoles"
    actions   = ["sts:AssumeRole"]
    resources = [var.vams_control_role_arn, var.vams_data_role_arn]
  }
  statement {
    sid       = "InvokeRunnerLambdas"
    actions   = ["lambda:InvokeFunction"]
    resources = [aws_lambda_function.read_manifest.arn, aws_lambda_function.collect.arn]
  }
  statement {
    sid = "SageMakerPipeline"
    actions = [
      "sagemaker:StartPipelineExecution",
      "sagemaker:DescribePipelineExecution",
      "sagemaker:StopPipelineExecution",
    ]
    resources = [
      "arn:${local.partition}:sagemaker:${var.aws_region}:${local.account_id}:pipeline/${var.sagemaker_pipeline_name}",
      "arn:${local.partition}:sagemaker:${var.aws_region}:${local.account_id}:pipeline/${var.sagemaker_pipeline_name}/execution/*",
    ]
  }
  statement {
    sid = "SageMakerTrainingJob"
    actions = [
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:AddTags",
    ]
    resources = ["arn:${local.partition}:sagemaker:${var.aws_region}:${local.account_id}:training-job/pai-*"]
  }
  statement {
    sid       = "PassSageMakerExecutionRole"
    actions   = ["iam:PassRole"]
    resources = [var.sagemaker_execution_role_arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
  statement {
    sid       = "BatchSubmit"
    actions   = ["batch:SubmitJob", "batch:TagResource"]
    resources = [
      var.batch_job_queue_arn,
      "arn:${local.partition}:batch:${var.aws_region}:${local.account_id}:job-definition/${var.batch_job_definition_name}:*",
    ]
  }
  statement {
    # DescribeJobs and TerminateJob do not support resource-level scoping.
    sid       = "BatchDescribeTerminate"
    actions   = ["batch:DescribeJobs", "batch:TerminateJob"]
    resources = ["*"]
  }
  statement {
    sid = "StateMachineLogging"
    actions = [
      "logs:CreateLogDelivery", "logs:GetLogDelivery", "logs:UpdateLogDelivery", "logs:DeleteLogDelivery",
      "logs:ListLogDeliveries", "logs:PutResourcePolicy", "logs:DescribeResourcePolicies", "logs:DescribeLogGroups",
    ]
    resources = ["*"] # Step Functions vended-log delivery has no resource-level scoping.
  }
}

resource "aws_iam_role_policy" "runner" {
  role   = aws_iam_role.runner.id
  policy = data.aws_iam_policy_document.runner.json
}

# ---------------------------------------------------------------------------------------------
# Lambdas
# ---------------------------------------------------------------------------------------------
data "archive_file" "runner_code" {
  type        = "zip"
  source_dir  = "${path.module}/../../runner"
  output_path = "${path.module}/.build/runner.zip"
}

data "aws_iam_policy_document" "lambda_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda" {
  name_prefix        = "${local.name_prefix}-lambda-"
  assume_role_policy = data.aws_iam_policy_document.lambda_trust.json
  tags               = local.common_tags
}

data "aws_iam_policy_document" "lambda" {
  statement {
    sid       = "AssumeVamsRoles"
    actions   = ["sts:AssumeRole"]
    resources = [var.vams_control_role_arn, var.vams_data_role_arn]
  }
  statement {
    sid     = "FoundationBuckets"
    actions = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
    resources = [
      "arn:${local.partition}:s3:::${var.datasets_bucket}",
      "arn:${local.partition}:s3:::${var.datasets_bucket}/vams/*",
      "arn:${local.partition}:s3:::${var.checkpoints_bucket}",
      "arn:${local.partition}:s3:::${var.checkpoints_bucket}/vams/*",
    ]
  }
  statement {
    sid       = "DescribeJobsForCollect"
    actions   = ["sagemaker:DescribeTrainingJob", "sagemaker:DescribePipelineExecution", "sagemaker:ListPipelineExecutionSteps"]
    resources = ["*"] # TODO: scope to the pipeline + training-job prefixes once Collect is implemented
  }
  statement {
    sid       = "ReadJobLogTail"
    actions   = ["logs:GetLogEvents", "logs:DescribeLogStreams"]
    resources = [
      "arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:/aws/sagemaker/TrainingJobs:*",
      "arn:${local.partition}:logs:${var.aws_region}:${local.account_id}:log-group:/aws/batch/job:*",
    ]
  }
  statement {
    sid       = "OwnLogs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = [for g in aws_cloudwatch_log_group.lambda : "${g.arn}:*"]
  }
}

resource "aws_iam_role_policy" "lambda" {
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda.json
}

locals {
  lambda_env = {
    VAMS_DATA_ROLE_ARN     = var.vams_data_role_arn
    VAMS_CONTROL_ROLE_ARN  = var.vams_control_role_arn
    VAMS_ROLE_EXTERNAL_ID  = var.vams_role_external_id
    DATASETS_BUCKET        = var.datasets_bucket
    CHECKPOINTS_BUCKET     = var.checkpoints_bucket
    HF_TOKEN_SECRET_ID     = var.hf_token_secret_id
    BATCH_JOB_DEFINITION   = var.batch_job_definition_name
    VAMS_BRIDGE_LOG_GROUP  = var.vams_bridge_log_group_name
  }
}

resource "aws_lambda_function" "read_manifest" {
  function_name    = "${local.name_prefix}-read-manifest"
  role             = aws_iam_role.lambda.arn
  runtime          = "python3.12"
  handler          = "read_manifest.lambda_handler"
  filename         = data.archive_file.runner_code.output_path
  source_code_hash = data.archive_file.runner_code.output_base64sha256
  timeout          = 900
  memory_size      = 1024
  environment { variables = local.lambda_env }
  depends_on       = [aws_cloudwatch_log_group.lambda]
  tags             = local.common_tags
}

resource "aws_lambda_function" "collect" {
  function_name    = "${local.name_prefix}-collect"
  role             = aws_iam_role.lambda.arn
  runtime          = "python3.12"
  handler          = "collect.lambda_handler"
  filename         = data.archive_file.runner_code.output_path
  source_code_hash = data.archive_file.runner_code.output_base64sha256
  timeout          = 900
  memory_size      = 1024
  environment { variables = local.lambda_env }
  depends_on       = [aws_cloudwatch_log_group.lambda]
  tags             = local.common_tags
}

# ---------------------------------------------------------------------------------------------
# Runner state machines, one per submit API, rendered from ../../statemachines/*.asl.json
# ---------------------------------------------------------------------------------------------
resource "aws_sfn_state_machine" "runner" {
  for_each = local.runners
  name     = "${local.name_prefix}-${each.key}"
  role_arn = aws_iam_role.runner.arn
  type     = "STANDARD"

  definition = templatefile("${path.module}/../../statemachines/${each.value.asl}", {
    partition                    = local.partition
    region                       = var.aws_region
    account_id                   = local.account_id
    read_manifest_function_arn   = aws_lambda_function.read_manifest.arn
    collect_function_arn         = aws_lambda_function.collect.arn
    vams_control_role_arn        = var.vams_control_role_arn
    vams_orchestration_bus_arn   = var.vams_orchestration_bus_arn
    vams_bridge_log_group_arn    = var.vams_bridge_log_group_arn
    vams_bridge_log_group_name   = var.vams_bridge_log_group_name
    runner_log_group_arn         = aws_cloudwatch_log_group.runner[each.key].arn
    runner_log_group_name        = aws_cloudwatch_log_group.runner[each.key].name
    sagemaker_pipeline_name      = var.sagemaker_pipeline_name
    sagemaker_execution_role_arn = var.sagemaker_execution_role_arn
    batch_job_queue_arn          = var.batch_job_queue_arn
  })

  logging_configuration {
    log_destination        = "${aws_cloudwatch_log_group.runner[each.key].arn}:*"
    include_execution_data = false # the VAMS task token travels in the execution input
    level                  = "ERROR"
  }

  tags = local.common_tags
}

# ---------------------------------------------------------------------------------------------
# Rules: route each VAMS pipeline's detailType to its runner. EventBridge → Step Functions needs a
# role allowed to StartExecution on exactly that state machine.
# ---------------------------------------------------------------------------------------------
data "aws_iam_policy_document" "events_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "rule_target" {
  name_prefix        = "${local.name_prefix}-rule-"
  assume_role_policy = data.aws_iam_policy_document.events_trust.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy" "rule_target" {
  role = aws_iam_role.rule_target.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "states:StartExecution"
      Resource = [for sm in aws_sfn_state_machine.runner : sm.arn]
    }]
  })
}

resource "aws_cloudwatch_event_rule" "runner" {
  for_each       = local.runners
  name           = "${local.name_prefix}-${each.key}"
  event_bus_name = aws_cloudwatch_event_bus.bridge.name
  event_pattern = jsonencode({
    "detail-type" = each.value.detail_types
    detail        = { body = { TaskToken = [{ exists = true }] } }
  })
  tags = local.common_tags
}

resource "aws_cloudwatch_event_target" "runner" {
  for_each       = local.runners
  rule           = aws_cloudwatch_event_rule.runner[each.key].name
  event_bus_name = aws_cloudwatch_event_bus.bridge.name
  arn            = aws_sfn_state_machine.runner[each.key].arn
  role_arn       = aws_iam_role.rule_target.arn
}
