# workflows/vams-bridge — vams-account root
#
# Resources that live in the VAMS account. Deliberately minimal: two roles the toolchain runner
# assumes, one log group Collect ships job log tails to, the vamsSchema registration, and — only
# for VAMS 2.6.x — a relay bus. No compute, no Lambda of our own.
#
# Everything about the VAMS deployment is read from the SSM parameters VAMS publishes under
# var.vams_resource_param_prefix (VAMS 2.7+, GitHub awslabs/visual-asset-management-system#398).
#
# Status: resources declared and validated (terraform validate); not yet deployed.

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  partition   = data.aws_partition.current.partition
  account_id  = data.aws_caller_identity.current.account_id
  common_tags = merge(var.tags, { Environment = var.environment, Module = "vams-bridge" })

  # Fixed names shared with ../toolchain-account (same `environment`); see the note there.
  name_prefix               = "pai-vams-bridge-${var.environment}"
  control_role_name         = "pai-vams-bridge-control-${var.environment}"
  data_role_name            = "pai-vams-bridge-data-${var.environment}"
  bridge_log_group          = "/aws/vendedlogs/Pipelines/pai-vams-bridge-${var.environment}"
  toolchain_runner_role_arn = "arn:${local.partition}:iam::${var.toolchain_account_id}:role/pai-vams-bridge-runner-${var.environment}"
  toolchain_lambda_role_arn = "arn:${local.partition}:iam::${var.toolchain_account_id}:role/pai-vams-bridge-lambda-${var.environment}"
  toolchain_bridge_bus_arn  = "arn:${local.partition}:events:${var.toolchain_region}:${var.toolchain_account_id}:event-bus/${local.name_prefix}"
}

# ---------------------------------------------------------------------------------------------
# VAMS integration surface (published by VAMS as SSM String parameters)
# ---------------------------------------------------------------------------------------------
data "aws_ssm_parameter" "orchestration_bus_arn" {
  name = "${var.vams_resource_param_prefix}/integration/orchestrationBusArn"
}

data "aws_ssm_parameter" "event_source_prefix" {
  name = "${var.vams_resource_param_prefix}/integration/orchestrationEventSourcePrefix"
}

data "aws_ssm_parameter" "workflow_execution_role_arn" {
  name = "${var.vams_resource_param_prefix}/integration/workflowExecutionRoleArn"
}

data "aws_ssm_parameter" "import_function_name" {
  name = "${var.vams_resource_param_prefix}/lambdaFunctions/importGlobalPipelineWorkflow"
}

data "aws_ssm_parameter" "artefacts_bucket" {
  name = "${var.vams_resource_param_prefix}/s3Buckets/artefacts"
}

# ---------------------------------------------------------------------------------------------
# Trust: both roles trust exactly the toolchain runner role (state machines, no ExternalId possible)
# and the runner Lambdas' role (with ExternalId), and the toolchain
# account as aws:PrincipalAccount. (ARCC BSC6 least privilege; BSC13 no cross-stage trust — deploy
# a prod VAMS against a prod toolchain account only.)
# ---------------------------------------------------------------------------------------------
data "aws_iam_policy_document" "cross_account_trust" {
  # The runner Lambdas assume with boto3 and pass the ExternalId.
  statement {
    sid     = "RunnerLambdas"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = ["arn:${local.partition}:iam::${var.toolchain_account_id}:root"]
    }
    condition {
      test     = "ArnEquals"
      variable = "aws:PrincipalArn"
      values   = [local.toolchain_lambda_role_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "sts:ExternalId"
      values   = [var.vams_role_external_id]
    }
  }
  # The state machines assume through a task state's Credentials.RoleArn, which carries no
  # ExternalId; the caller is pinned to the one runner role ARN instead.
  statement {
    sid     = "RunnerStateMachines"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = ["arn:${local.partition}:iam::${var.toolchain_account_id}:root"]
    }
    condition {
      test     = "ArnEquals"
      variable = "aws:PrincipalArn"
      values   = [local.toolchain_runner_role_arn]
    }
  }
}

# ---------------------------------------------------------------------------------------------
# Control role: VAMS callbacks only. states:SendTask* carry no resource (API design); the bus grant
# is scoped to the orchestration bus; PutLogEvents to the bridge log group only.
# ---------------------------------------------------------------------------------------------
resource "aws_iam_role" "control" {
  name                 = local.control_role_name
  assume_role_policy   = data.aws_iam_policy_document.cross_account_trust.json
  max_session_duration = 3600
  tags                 = local.common_tags
}

data "aws_iam_policy_document" "control" {
  statement {
    sid       = "TaskCallbacks"
    actions   = ["states:SendTaskSuccess", "states:SendTaskFailure", "states:SendTaskHeartbeat"]
    resources = ["*"] # Task-token APIs are not resource-scoped.
  }
  statement {
    sid       = "RegisterSubExecution"
    actions   = ["events:PutEvents"]
    resources = [data.aws_ssm_parameter.orchestration_bus_arn.value]
    # VAMS's own bus policy additionally restricts this role to detail-type
    # pipeline.execution.register on its own execution source; this identity policy mirrors that.
    condition {
      test     = "StringEquals"
      variable = "events:detail-type"
      values   = ["pipeline.execution.register"]
    }
  }
  statement {
    sid       = "ShipLogTail"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"]
    resources = ["${aws_cloudwatch_log_group.bridge.arn}:*"]
  }
}

resource "aws_iam_role_policy" "control" {
  role   = aws_iam_role.control.id
  policy = data.aws_iam_policy_document.control.json
}

# ---------------------------------------------------------------------------------------------
# Data role: read the manifest/config/inputs, write ONLY under the run's output prefixes.
# An input file is any file of the asset, so reads are bucket-wide on the listed asset buckets;
# writes are confined to the VAMS run layout <baseAssetsPrefix>pipelines/<pipeline>/<job>/output/.
# ---------------------------------------------------------------------------------------------
resource "aws_iam_role" "data" {
  name                 = local.data_role_name
  assume_role_policy   = data.aws_iam_policy_document.cross_account_trust.json
  max_session_duration = 3600
  tags                 = local.common_tags
}

locals {
  asset_bucket_arns = [for b in var.vams_asset_bucket_names : "arn:${local.partition}:s3:::${b}"]
  asset_object_arns = [for b in var.vams_asset_bucket_names : "arn:${local.partition}:s3:::${b}/*"]
  run_output_arns = flatten([for b in var.vams_asset_bucket_names : [
    "arn:${local.partition}:s3:::${b}/pipelines/*/output/*",
    "arn:${local.partition}:s3:::${b}/*/pipelines/*/output/*",
  ]])
}

data "aws_iam_policy_document" "data" {
  statement {
    sid       = "ReadInputsManifestAndConfig"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = local.asset_object_arns
  }
  statement {
    # Listing is bucket-wide: staging a whole-asset or folder selection enumerates the asset's
    # files, and the role already reads every object above, so a prefix condition here would hide
    # nothing it cannot fetch.
    sid       = "ListAssetBuckets"
    actions   = ["s3:ListBucket"]
    resources = local.asset_bucket_arns
  }
  statement {
    sid       = "WriteRunOutputs"
    actions   = ["s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"]
    resources = local.run_output_arns
  }
  dynamic "statement" {
    for_each = var.vams_kms_key_arn == "" ? [] : [var.vams_kms_key_arn]
    content {
      sid       = "VamsCmk"
      actions   = ["kms:Decrypt", "kms:GenerateDataKey", "kms:DescribeKey"]
      resources = [statement.value]
      condition {
        test     = "StringEquals"
        variable = "kms:ViaService"
        values   = ["s3.${var.aws_region}.amazonaws.com"]
      }
    }
  }
}

resource "aws_iam_role_policy" "data" {
  role   = aws_iam_role.data.id
  policy = data.aws_iam_policy_document.data.json
}

# ---------------------------------------------------------------------------------------------
# Bridge log group: readable by the VAMS execution-logs API because /aws/vendedlogs/Pipelines/* is
# in VAMS's default executionLogReadPrefixes.
# ---------------------------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "bridge" {
  name              = local.bridge_log_group
  retention_in_days = 365
  tags              = local.common_tags
}

# ---------------------------------------------------------------------------------------------
# Relay (VAMS 2.6.x only). The 2.6.x workflow role may put to a bus in its own account whose name
# carries the VAMS deployment name; this rule forwards everything from the pai.* source to the
# toolchain bus. Not created in direct mode.
# ---------------------------------------------------------------------------------------------
resource "aws_cloudwatch_event_bus" "relay" {
  count = var.vams_relay ? 1 : 0
  name  = "${var.vams_deployment_name}-pai-bridge-relay"
  tags  = local.common_tags
}

data "aws_iam_policy_document" "events_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "relay" {
  count              = var.vams_relay ? 1 : 0
  name_prefix        = "${local.name_prefix}-relay-"
  assume_role_policy = data.aws_iam_policy_document.events_trust.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy" "relay" {
  count = var.vams_relay ? 1 : 0
  role  = aws_iam_role.relay[0].id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "events:PutEvents", Resource = local.toolchain_bridge_bus_arn }]
  })
}

resource "aws_cloudwatch_event_rule" "relay" {
  count          = var.vams_relay ? 1 : 0
  name           = "${local.name_prefix}-relay"
  event_bus_name = aws_cloudwatch_event_bus.relay[0].name
  event_pattern  = jsonencode({ source = [{ prefix = "pai." }] })
}

resource "aws_cloudwatch_event_target" "relay" {
  count          = var.vams_relay ? 1 : 0
  rule           = aws_cloudwatch_event_rule.relay[0].name
  event_bus_name = aws_cloudwatch_event_bus.relay[0].name
  arn            = local.toolchain_bridge_bus_arn
  role_arn       = aws_iam_role.relay[0].arn
}

# ---------------------------------------------------------------------------------------------
# vamsSchema registration: upload each bundle to the VAMS artefacts bucket under vamsSchema/pai-<id>/
# and invoke the VAMS import Lambda directly (no RequestType => register-and-return). The payload is
# the ResourceProperties shape the VAMS CDK construct passes: bundleS3Keys (S3 keys of the uploaded
# files), resourceOverrides and idOverrides as JSON strings. Idempotent: a re-run overwrites the
# definition and clears the archived flag. The invocation fails the apply when the Lambda raises.
# ---------------------------------------------------------------------------------------------
locals {
  bus_arn_for_vams = var.vams_relay ? aws_cloudwatch_event_bus.relay[0].arn : local.toolchain_bridge_bus_arn
  schema_root      = "${path.module}/../../vamsSchema"
  bundle_keys = { for c in var.components : c => merge(
    { pipeline = "vamsSchema/pai-${c}/pipeline.json" },
    fileexists("${local.schema_root}/${c}/workflow.json") ? { workflow = "vamsSchema/pai-${c}/workflow.json" } : {},
    { templates = [for t in fileset("${local.schema_root}/${c}/templates", "*.json") : "vamsSchema/pai-${c}/templates/${t}"] },
  ) }
  schema_hashes = { for c in var.components : c => sha256(join("", [
    for p in sort(fileset("${local.schema_root}/${c}", "**")) : filesha256("${local.schema_root}/${c}/${p}")
  ])) }
}

resource "aws_s3_object" "vams_schema" {
  for_each = { for f in flatten([for c in var.components :
    [for p in fileset("${local.schema_root}/${c}", "**") : { component = c, path = p }]
  ]) : "${f.component}/${f.path}" => f }

  bucket = data.aws_ssm_parameter.artefacts_bucket.value
  key    = "vamsSchema/pai-${each.value.component}/${each.value.path}"
  source = "${local.schema_root}/${each.value.component}/${each.value.path}"
  etag   = filemd5("${local.schema_root}/${each.value.component}/${each.value.path}")
}

resource "aws_lambda_invocation" "register" {
  for_each      = toset(var.components)
  function_name = data.aws_ssm_parameter.import_function_name.value

  input = jsonencode({
    bundleS3Keys      = jsonencode(local.bundle_keys[each.key])
    resourceOverrides = jsonencode({ eventBridgeBusArn = local.bus_arn_for_vams })
    idOverrides       = jsonencode({ pipelineId = "pai-${each.key}", workflowId = "pai-${each.key}" })
  })

  triggers = {
    schema_hash = local.schema_hashes[each.key]
    bus_arn     = local.bus_arn_for_vams
  }

  depends_on = [aws_s3_object.vams_schema]
}
