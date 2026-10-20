# workflows/vams-bridge — vams-account root
#
# Resources that live in the VAMS account. Deliberately minimal: two roles the toolchain runner
# assumes, one log group Collect ships job log tails to, the vamsSchema registration, and — only
# for VAMS 2.6.x — a relay bus. No compute, no Lambda of our own.
#
# Everything about the VAMS deployment is read from the SSM parameters VAMS publishes under
# var.vams_resource_param_prefix (VAMS 2.7+, GitHub awslabs/visual-asset-management-system#398).
#
# Status: skeleton. Registration is declared as a null_resource invoking the VAMS import Lambda via
# the AWS CLI; a TODO notes the alternative of an aws_lambda_invocation data source.

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  name_prefix = "pai-vams-bridge"
  partition   = data.aws_partition.current.partition
  account_id  = data.aws_caller_identity.current.account_id
  common_tags = merge(var.tags, { Module = "vams-bridge" })
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
# Trust: both roles trust exactly the toolchain runner role, with ExternalId, and the toolchain
# account as aws:PrincipalAccount. (ARCC BSC6 least privilege; BSC13 no cross-stage trust — deploy
# a prod VAMS against a prod toolchain account only.)
# ---------------------------------------------------------------------------------------------
data "aws_iam_policy_document" "cross_account_trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "AWS"
      identifiers = [var.toolchain_runner_role_arn]
    }
    condition {
      test     = "StringEquals"
      variable = "sts:ExternalId"
      values   = [var.vams_role_external_id]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:PrincipalAccount"
      values   = [var.toolchain_account_id]
    }
  }
}

# ---------------------------------------------------------------------------------------------
# Control role: VAMS callbacks only. states:SendTask* carry no resource (API design); the bus grant
# is scoped to the orchestration bus; PutLogEvents to the bridge log group only.
# ---------------------------------------------------------------------------------------------
resource "aws_iam_role" "control" {
  name_prefix          = "${local.name_prefix}-control-"
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
# Data role: read the manifest/config/inputs, write ONLY under the run's output prefixes. The exact
# bucket set is deployment-specific; declare it through var-driven statements.
# TODO: take the VAMS asset bucket names from the VAMS buckets SSM parameters / a variable and scope
#       to <bucket>/<baseAssetsPrefix>pipelines/* for writes.
# ---------------------------------------------------------------------------------------------
resource "aws_iam_role" "data" {
  name_prefix          = "${local.name_prefix}-data-"
  assume_role_policy   = data.aws_iam_policy_document.cross_account_trust.json
  max_session_duration = 3600
  tags                 = local.common_tags
}

data "aws_iam_policy_document" "data" {
  statement {
    sid       = "ReadManifestAndConfig"
    actions   = ["s3:GetObject"]
    resources = ["arn:${local.partition}:s3:::*/pipelines/*/input/*"] # TODO: name the asset buckets
  }
  statement {
    sid       = "WriteRunOutputs"
    actions   = ["s3:PutObject", "s3:AbortMultipartUpload"]
    resources = ["arn:${local.partition}:s3:::*/pipelines/*/output/*"] # TODO: name the asset buckets
  }
  # TODO: kms:Decrypt / kms:GenerateDataKey on the VAMS CMK when the deployment uses one
  #       (useKmsCmkEncryption.enabled); key ARN from the VAMS stack outputs.
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
  name              = "/aws/vendedlogs/Pipelines/${local.name_prefix}"
  retention_in_days = 365
  tags              = local.common_tags
}

# ---------------------------------------------------------------------------------------------
# Relay (VAMS 2.6.x only). The VAMS workflow role may put to a bus in its own account whose name
# contains the VAMS deployment name; this rule forwards to the toolchain bus.
# TODO: the bus name must contain the VAMS deployment name — take it as a variable.
# ---------------------------------------------------------------------------------------------
resource "aws_cloudwatch_event_bus" "relay" {
  count = var.vams_relay ? 1 : 0
  name  = "vams-${local.name_prefix}-relay" # TODO: "<vamsName>-pai-bridge"
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
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "events:PutEvents", Resource = var.toolchain_bridge_bus_arn }]
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
  arn            = var.toolchain_bridge_bus_arn
  role_arn       = aws_iam_role.relay[0].arn
}

# ---------------------------------------------------------------------------------------------
# vamsSchema registration: upload each bundle to the VAMS artefacts bucket under vamsSchema/pai-<id>/
# and invoke the VAMS import Lambda directly (no RequestType => register-and-return). The payload
# shape is the same ResourceProperties the VAMS CDK construct passes: bundleS3Keys (S3 keys of the
# uploaded files), resourceOverrides and idOverrides as JSON strings. Idempotent: a re-run
# overwrites the definition and clears the archived flag.
# TODO: replace the local-exec with aws_lambda_invocation once the payload is final; the CLI form is
#       kept here so the exact request shape is visible for review.
# ---------------------------------------------------------------------------------------------
locals {
  bus_arn_for_vams = var.vams_relay ? aws_cloudwatch_event_bus.relay[0].arn : var.toolchain_bridge_bus_arn
  schema_root      = "${path.module}/../../vamsSchema"
  bundle_keys = { for c in var.components : c => {
    pipeline  = "vamsSchema/pai-${c}/pipeline.json"
    workflow  = fileexists("${local.schema_root}/${c}/workflow.json") ? "vamsSchema/pai-${c}/workflow.json" : null
    templates = [for t in fileset("${local.schema_root}/${c}/templates", "*.json") : "vamsSchema/pai-${c}/templates/${t}"]
  } }
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

resource "null_resource" "register" {
  for_each = toset(var.components)

  triggers = {
    schema_hash = sha256(join("", [for p in fileset("${local.schema_root}/${each.key}", "**") :
      filesha256("${local.schema_root}/${each.key}/${p}")]))
    bus_arn = local.bus_arn_for_vams
  }

  provisioner "local-exec" {
    command = <<-EOT
      mkdir -p ${path.module}/.build && \
      aws lambda invoke --region ${var.aws_region} \
        --function-name ${data.aws_ssm_parameter.import_function_name.value} \
        --cli-binary-format raw-in-base64-out \
        --payload '${jsonencode({
          bundleS3Keys      = jsonencode({ for k, v in local.bundle_keys[each.key] : k => v if v != null })
          resourceOverrides = jsonencode({ eventBridgeBusArn = local.bus_arn_for_vams })
          idOverrides       = jsonencode({ pipelineId = "pai-${each.key}", workflowId = "pai-${each.key}" })
        })}' \
        ${path.module}/.build/register-${each.key}.json && \
      python3 -c "import json,sys; r=json.load(open('${path.module}/.build/register-${each.key}.json')); sys.exit(1 if r.get('errorMessage') or r.get('statusCode') != 200 else 0)"
    EOT
  }

  depends_on = [aws_s3_object.vams_schema]
}
