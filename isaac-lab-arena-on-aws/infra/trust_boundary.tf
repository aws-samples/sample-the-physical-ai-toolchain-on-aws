# =============================================================================
# Gated-registration trust boundary (earlier-cycle findings C4 + C5; these are
# C1-C3 in the current report).
# NOTE ON IDS: these labels come from an earlier review cycle. In the current report
# the same concerns are C1-C3. Cross-cycle IDs are not stable -- prefer the described
# defect over the number.
#
# WHY THIS EXISTS
#
# C5: RegisterModel pointed at the FineTune artifact's plain, unversioned S3 URI.
# Validate verified the mounted bytes and recorded their identity but never promoted
# them, so the registered package resolved to "whatever occupies that key now" rather
# than the bytes that passed validation.
#
# C4: FineTune, SimEval and Validate all ran under the single Foundation SageMaker
# execution role, which holds Get/Put/DeleteObject across the models bucket -- the same
# bucket that holds the staged validation code. A training or eval worker could therefore
# overwrite the validator that judges it and the model bytes after validation. Foundation
# additionally attaches AmazonSageMakerFullAccess, whose iam:PassRole to SageMaker means a
# worker retaining that identity could launch work under a newly created validation role,
# so a separate validation role ALONE would not have closed the path.
#
# Together those two meant validated bytes and registered bytes could legitimately differ,
# which makes the gate decorative.
#
# WHAT THIS PROVISIONS
#
# A component-owned trust bucket that the Foundation role has no grant on at all (its
# policy enumerates the datasets/models/checkpoints buckets individually), plus two
# component-owned runtime roles so workers stop holding Foundation credentials:
#
#   * workload  -- FineTune and SimEval. Reads inputs, writes its own job outputs. No
#                  write access to trusted code, evidence or promoted artifacts; no
#                  iam:PassRole; no SageMaker job/pipeline/registry mutation.
#   * validation -- Validate. Reads the checkpoint and evaluation inputs and the trusted
#                  validation code; publishes promoted artifacts and attestations. Cannot
#                  replace the validation code it runs.
#
# The Foundation role remains the pipeline ORCHESTRATION identity. Holders of it are
# effectively administrators of this workflow; that is unchanged and out of our scope.
#
# WHAT THIS DOES NOT FIX
#
# Foundation's grant on the models bucket stays bucket-wide, and Foundation is a separate
# component we do not own. Anything still running under the Foundation role retains that
# access. This boundary works by moving trusted artifacts OUT of that bucket, not by
# narrowing the grant. See README.md for the residual limitation and the ask to
# Foundation's owner.
# =============================================================================

# --- Trust bucket: promoted artifacts and attestations -----------------------------
resource "aws_s3_bucket" "trust" {
  bucket = "${local.prefix}-trust-${local.account_id}"

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Purpose     = "gated-registration-trust-boundary"
  }
}

# Versioning does not by itself prevent a new version becoming current, so it is a
# recovery aid here rather than the immutability mechanism. Immutability comes from the
# conditional create (If-None-Match) plus the deny statements below.
resource "aws_s3_bucket_versioning" "trust" {
  bucket = aws_s3_bucket.trust.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "trust" {
  bucket                  = aws_s3_bucket.trust.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "trust" {
  bucket = aws_s3_bucket.trust.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# The protected namespaces. A promoted artifact or attestation is written ONCE and never
# replaced or deleted; that is what gives the registered ModelDataUrl a stable meaning,
# since the SageMaker model-package API has no VersionId field to pin.
locals {
  trust_artifact_prefix = "artifacts/v1/"
  trust_evidence_prefix = "evidence/v1/"
  trust_protected_arns = [
    "${aws_s3_bucket.trust.arn}/artifacts/v1/*",
    "${aws_s3_bucket.trust.arn}/evidence/v1/*",
  ]
  # C4: executable validation code lives HERE, not in the shared Foundation models bucket.
  # The Foundation SageMaker role holds PutObject and DeleteObject across that bucket, and
  # the documented toolchain has peer components submitting jobs under it -- so any peer
  # worker could replace the validator before Validate downloaded it, and that replacement
  # would then run under the privileged validation role. Scoping this component's own workers
  # away from it was necessary but not sufficient, because the exposure is not this
  # component's workers.
  #
  # NO runtime role may write this namespace. Publication belongs to the deploying identity.
  trust_code_arns = [
    "${aws_s3_bucket.trust.arn}/code/v1/*",
  ]
  # The Foundation models bucket, as an ARN. Its name comes from the Foundation SSM
  # contract read in main.tf.
  models_bucket_arn = "arn:aws:s3:::${local.models_bucket}"
}

data "aws_iam_policy_document" "trust_bucket" {
  # Nobody may DELETE inside the protected namespaces, or change an object's ACL --
  # including the validation role that writes them.
  statement {
    sid       = "DenyDeletionOfPromotedArtifacts"
    effect    = "Deny"
    actions   = ["s3:DeleteObject", "s3:DeleteObjectVersion", "s3:PutObjectAcl"]
    resources = local.trust_protected_arns
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
  }

  # And nobody may OVERWRITE, which the deny above does NOT cover: it lists delete and ACL
  # actions only, so an unconditional PutObject to an existing key was still permitted and
  # the comment claiming create-only publication was not enforced by the bucket at all.
  # Only the application supplied If-None-Match, which is not a control -- a caller with
  # the validation role could simply omit it.
  #
  # This denies object creation in the protected namespaces unless the request carries the
  # If-None-Match precondition. `Null: s3:if-none-match = true` means the header is ABSENT;
  # `Bool: s3:ObjectCreationOperation = true` restricts the deny to the APIs that actually
  # accept the header (PutObject, CompleteMultipartUpload), exempting CreateMultipartUpload,
  # UploadPart and UploadPartCopy which cannot carry it. Both conditions apply together.
  #
  # Consequence, accepted deliberately: CopyObject into these namespaces will fail. The
  # publication path uploads verified bytes rather than copying, which is also why
  # conditional-write enforcement is compatible with it.
  statement {
    sid       = "DenyUnconditionalWritesToProtectedNamespaces"
    effect    = "Deny"
    actions   = ["s3:PutObject"]
    resources = local.trust_protected_arns
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
    condition {
      test     = "Null"
      variable = "s3:if-none-match"
      values   = ["true"]
    }
    condition {
      test     = "Bool"
      variable = "s3:ObjectCreationOperation"
      values   = ["true"]
    }
  }

  # Only the validation role may write into the protected namespaces at all. The workload
  # role is not granted access here and this makes that explicit rather than implicit.
  statement {
    sid       = "OnlyValidationRoleMayPublish"
    effect    = "Deny"
    actions   = ["s3:PutObject"]
    resources = local.trust_protected_arns
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
    condition {
      test     = "StringNotEquals"
      variable = "aws:PrincipalArn"
      values   = [aws_iam_role.validation.arn]
    }
  }

  # C4: no runtime identity may publish or remove executable validation code. Expressed on
  # the bucket so it holds even if an identity policy is later widened, and listing the roles
  # explicitly rather than denying everyone -- the deploying identity has to be able to publish.
  statement {
    sid       = "NoRuntimeRoleMayWriteTrustedCode"
    effect    = "Deny"
    actions   = ["s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion"]
    resources = local.trust_code_arns
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
    condition {
      test     = "ArnEquals"
      variable = "aws:PrincipalArn"
      values = [
        aws_iam_role.workload.arn,
        aws_iam_role.training.arn,
        aws_iam_role.validation.arn,
      ]
    }
  }

  # RegisterModel runs as the Foundation ORCHESTRATION role and must READ the artifact it registers.
  # A real run reached this step with everything else green and failed:
  #
  #   Failed to invoke sagemaker:CreateModelPackage. Error Details: Access denied for bucket:
  #   physical-ai-dev-trust-<acct>, object key: artifacts/v1/sha256/<digest>/model.tar.gz
  #
  # simulate-principal-policy returned implicitDeny with no matching statement: that role has no grant
  # on this bucket at all. It is FOUNDATION-owned, so its identity policy is not ours to edit -- but
  # this bucket is ours, and for a same-account principal a resource-policy Allow is sufficient. So the
  # grant belongs here, which also keeps the permission visible beside the denies that bound it.
  #
  # Deliberately minimal: READ only, and only artifacts/v1/*. Not evidence/v1/* (an approver reads that
  # out of band, and RegisterModel does not), not code/v1/*, and no write of any kind -- the Deny
  # statements above continue to forbid deletion, ACL changes, and unconditional writes for every
  # principal including this one.
  statement {
    sid       = "OrchestrationRoleMayReadPromotedArtifacts"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${aws_s3_bucket.trust.arn}/artifacts/v1/*"]
    principals {
      type        = "AWS"
      identifiers = [local.foundation_role_arn]
    }
  }

  statement {
    sid       = "DenyUnencryptedTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [aws_s3_bucket.trust.arn, "${aws_s3_bucket.trust.arn}/*"]
    principals {
      type        = "AWS"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "trust" {
  bucket = aws_s3_bucket.trust.id
  policy = data.aws_iam_policy_document.trust_bucket.json
  # The bucket policy references the validation role, which must exist first.
  depends_on = [aws_iam_role.validation]
}

# --- Component runtime roles ------------------------------------------------------
data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

# FineTune + SimEval. Deliberately has NO iam:PassRole and no SageMaker mutation actions:
# a worker that can pass a role or create a job can launch work under an identity it was
# not given, which is the indirect path a separate validation role alone would not close.
# Runtime grants every SageMaker job identity needs regardless of what it may write: pull its
# image, emit logs, publish training metrics. Previously each of the three roles below carried
# an identical copy of this as a single statement on Resource "*", which had two problems.
#
# The first is blast radius: one statement mixing ECR, Logs and CloudWatch on "*" grants far
# more than any job needs, and hides which part actually requires a wildcard. Split here into
# one statement per service, each scoped as tightly as the API allows.
#
# The second is drift: three identical copies could diverge silently, and the boundary this
# file establishes depends on the three roles differing ONLY in their S3 grants. Defining the
# shared part once and merging it via source_policy_documents makes that structural rather
# than a thing a reviewer has to diff by eye.
data "aws_iam_policy_document" "job_runtime_common" {
  # ecr:GetAuthorizationToken returns a registry-wide token and does not support
  # resource-level permissions -- IAM rejects any Resource other than "*". Isolated in its own
  # statement so the exception is explicit and auditable, and so the pull actions below stay
  # scoped. This is the only genuinely unconstrainable grant in this document.
  statement {
    sid       = "EcrAuthTokenRegistryWide"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  # Training images use fixed vla/* names; the Foundation Arena image uses the project
  # prefix. Grant both declared sets, including repositories reused from another
  # deployment, without granting access to every repository under either prefix.
  statement {
    sid    = "EcrPullJobImages"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = concat(
      [for repo in aws_ecr_repository.repos : repo.arn],
      [for repo in data.aws_ecr_repository.existing : repo.arn],
      [
        data.aws_ecr_repository.arena.arn,
        "arn:aws:ecr:${local.region}:${var.dlc_account_id}:repository/pytorch-training",
        "arn:aws:ecr:${local.region}:${var.sklearn_account_id}:repository/sagemaker-scikit-learn",
      ],
    )
  }

  # SageMaker writes job logs under /aws/sagemaker/. CreateLogGroup is included because a
  # fresh deployment has no group yet, and it is scoped to the same prefix rather than left
  # account-wide, so a job cannot create or write groups belonging to anything else.
  statement {
    sid    = "WriteJobLogsUnderSageMakerPrefix"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = [
      "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*",
      "arn:aws:logs:${local.region}:${local.account_id}:log-group:/aws/sagemaker/*:log-stream:*",
    ]
  }

  # cloudwatch:PutMetricData has no resource ARN -- it can only be constrained by condition.
  # Restricting the namespace keeps a job from writing metrics into another system's
  # namespace, which matters here because downstream promotion decisions read these metrics:
  # an unconstrained grant would let a training worker publish into whatever namespace a
  # reviewer or alarm happens to watch.
  statement {
    sid       = "PublishMetricsToOwnNamespacesOnly"
    effect    = "Allow"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]

    condition {
      test     = "StringLike"
      variable = "cloudwatch:namespace"
      values = [
        "/aws/sagemaker/*",
        "AWS/SageMaker",
        var.project_name,
        "${var.project_name}/*",
      ]
    }
  }
}

resource "aws_iam_role" "workload" {
  name               = "${local.prefix}-workload"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
  tags = {
    Project = var.project_name
    Purpose = "finetune-and-simeval-worker"
  }
}

data "aws_iam_policy_document" "workload" {
  # Image pull, job logs and metrics, scoped per service. See job_runtime_common above.
  source_policy_documents = [data.aws_iam_policy_document.job_runtime_common.json]

  statement {
    sid    = "ReadInputs"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:ListBucket",
    ]
    resources = concat([
      local.models_bucket_arn,
      "${local.models_bucket_arn}/*",
      # I10: BYO dataset inputs and external eval-only checkpoints live outside the models
      # bucket. Declared explicitly per deployment rather than widening to every bucket in the
      # account -- empty by default, so nothing extra is granted unless configured.
      # C4 (cycle 8): moving sourcedirs into the trust bucket left the workers unable to
      # download their OWN submitted code. runner.py publishes every content-addressed upload
      # to cfg.trust_bucket, and those uploads become sagemaker_submit_directory for FineTune
      # and LIBERO SimEval, so the DLC toolkit fetches them under the job's role. Only
      # Validate had trust-bucket read, so this would have failed every real run. READ ONLY:
      # the bucket-policy Deny on this namespace covers PutObject and DeleteObject, so the
      # publication restriction is unaffected.
      aws_s3_bucket.trust.arn,
      "${aws_s3_bucket.trust.arn}/code/v1/*",
    ], var.additional_input_s3_arns)
  }

  # Writes are scoped to the job-output prefixes. Bucket-wide PutObject let a training or
  # eval worker replace the STAGED VALIDATION CODE, which lives in this same bucket under
  # .../code/ and is executed by the privileged validation role. The role split alone did
  # not establish the boundary: a worker could swap the validator before a later Validate
  # job downloaded it, and that replacement would run with permission to publish promoted
  # artifacts. A hash in the object key does not make the object immutable.
  #
  # This role is the EVALUATION worker, so it writes eval/ and NOT train/. Sharing one role
  # across FineTune and SimEval scoped writes away from the validator but left FineTune able
  # to write under eval/ -- so a training worker could manufacture or replace the raw
  # evaluation evidence Validate then judged. Producing the evidence and producing the
  # weights are different jobs and now have different identities.
  statement {
    sid    = "WriteOnlyEvalOutputs"
    effect = "Allow"
    actions = [
      "s3:PutObject",
    ]
    resources = [
      # C3: raw evaluation evidence now lands in the component-owned handoff bucket, which
      # Foundation has no grant on, instead of the shared models bucket where any peer worker
      # under the Foundation role could replace it before Validate read it.
      "${aws_s3_bucket.handoff.arn}/eval/v1/*",
      "${local.models_bucket_arn}/${var.pipeline_prefix}/eval/*",
      # I9: both worker roles could write the whole plumbing/* namespace, so the
      # rehearsal did not exercise production's DISJOINT write prefixes -- the one
      # property the split exists to establish. The runner already writes
      # plumbing/train and plumbing/eval separately (runner.py:365,385), so each
      # role now gets only its own.
      "${local.models_bucket_arn}/${var.pipeline_prefix}/plumbing/eval/*",
    ]
  }

  # Explicit belt-and-braces deny on the trusted namespaces, so a future broadening of the
  # allow above cannot silently reopen the path.
  statement {
    sid    = "NeverWriteTrustedCodeOrEvidence"
    effect = "Deny"
    actions = [
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:DeleteObjectVersion",
    ]
    resources = [
      "${local.models_bucket_arn}/${var.pipeline_prefix}/code/*",
      "${local.models_bucket_arn}/${var.pipeline_prefix}/validated/*",
      aws_s3_bucket.trust.arn,
      "${aws_s3_bucket.trust.arn}/*",
    ]
  }

  # The trust bucket is NOT granted here. Workers must not be able to write promoted
  # artifacts or attestations, which is the point of the boundary.

}

resource "aws_iam_role_policy" "workload" {
  name   = "${local.prefix}-workload"
  role   = aws_iam_role.workload.id
  policy = data.aws_iam_policy_document.workload.json
}

# FineTune. Separate from the evaluation worker so a training job cannot write, replace or
# manufacture the raw evaluation evidence that Validate judges. Same deliberate omissions as
# the evaluation role: no iam:PassRole and no SageMaker mutation actions, so a worker cannot
# launch work under an identity it was not given.
resource "aws_iam_role" "training" {
  name               = "${local.prefix}-training"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
  tags = {
    Project = var.project_name
    Purpose = "finetune-worker"
  }
}

data "aws_iam_policy_document" "training" {
  # Image pull, job logs and metrics, scoped per service. See job_runtime_common above.
  source_policy_documents = [data.aws_iam_policy_document.job_runtime_common.json]

  statement {
    sid    = "ReadInputs"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:ListBucket",
    ]
    resources = concat([
      local.models_bucket_arn,
      "${local.models_bucket_arn}/*",
      # I10: BYO dataset inputs and external eval-only checkpoints live outside the models
      # bucket. Declared explicitly per deployment rather than widening to every bucket in the
      # account -- empty by default, so nothing extra is granted unless configured.
      # C4 (cycle 8): moving sourcedirs into the trust bucket left the workers unable to
      # download their OWN submitted code. runner.py publishes every content-addressed upload
      # to cfg.trust_bucket, and those uploads become sagemaker_submit_directory for FineTune
      # and LIBERO SimEval, so the DLC toolkit fetches them under the job's role. Only
      # Validate had trust-bucket read, so this would have failed every real run. READ ONLY:
      # the bucket-policy Deny on this namespace covers PutObject and DeleteObject, so the
      # publication restriction is unaffected.
      aws_s3_bucket.trust.arn,
      "${aws_s3_bucket.trust.arn}/code/v1/*",
    ], var.additional_input_s3_arns)
  }

  # train/ only. It has no write access to eval/, which is the whole point of the split.
  statement {
    sid    = "WriteOnlyTrainOutputs"
    effect = "Allow"
    actions = [
      "s3:PutObject",
    ]
    resources = [
      "${local.models_bucket_arn}/${var.pipeline_prefix}/train/*",
      # I9: both worker roles could write the whole plumbing/* namespace, so the
      # rehearsal did not exercise production's DISJOINT write prefixes -- the one
      # property the split exists to establish. The runner already writes
      # plumbing/train and plumbing/eval separately (runner.py:365,385), so each
      # role now gets only its own.
      "${local.models_bucket_arn}/${var.pipeline_prefix}/plumbing/train/*",
    ]
  }

  # I1: the SAME runtime permissions the evaluation role has. Splitting FineTune onto its own
  # identity gave it S3 reads and writes only, so on a FRESH deployment it could not pull its
  # image or write logs -- a broad sandbox grant would have hidden that until someone deployed
  # clean. Every job identity needs these regardless of what it is allowed to write.
}

resource "aws_iam_role_policy" "training" {
  name   = "${local.prefix}-training"
  role   = aws_iam_role.training.id
  policy = data.aws_iam_policy_document.training.json
}

# Validate. Reads what it must judge and the trusted code it runs; publishes promoted
# artifacts and attestations. It cannot replace the validation code, and the bucket policy
# above stops it overwriting or deleting anything it has already published.
resource "aws_iam_role" "validation" {
  name               = "${local.prefix}-validation"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
  tags = {
    Project = var.project_name
    Purpose = "validate-and-promote"
  }
}

data "aws_iam_policy_document" "validation" {
  # Image pull, job logs and metrics, scoped per service. See job_runtime_common above.
  source_policy_documents = [data.aws_iam_policy_document.job_runtime_common.json]

  statement {
    sid    = "ReadCheckpointEvaluationAndTrustedCode"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:ListBucket",
    ]
    resources = concat([
      local.models_bucket_arn,
      "${local.models_bucket_arn}/*",
      # C4: the trusted code namespace it executes now lives in the component's own bucket.
      aws_s3_bucket.trust.arn,
      "${aws_s3_bucket.trust.arn}/*",
      # C3: the raw evidence it judges now comes from the handoff bucket. Read of the whole
      # bucket rather than eval/v1/* alone, because Validate also reads back the receipt it
      # published in validated/v1/* when re-verifying a prior decision.
      aws_s3_bucket.handoff.arn,
      "${aws_s3_bucket.handoff.arn}/*",
      # I3 (cycle 8): additional_input_s3_arns was granted to the training and evaluation roles
      # but NOT to this one, so an external eval-only checkpoint could evaluate successfully on
      # paid GPU capacity and then fail here with an authorization error. Validate both
      # downloads the checkpoint and performs a version-specific GET on it, so it needs the
      # same reads. Fixing one of two consuming roles is the same mistake as fixing one of two
      # evaluators.
    ], var.additional_input_s3_arns)
  }

  statement {
    sid    = "PublishPromotedArtifactsAndAttestations"
    effect = "Allow"
    actions = [
      "s3:PutObject",
      # I11: promotion streams large checkpoints as a multipart upload and aborts the upload
      # when publication fails or is superseded. Without this the abort was denied every time,
      # so the cleanup existed in code and never once succeeded, leaving billable parts.
      "s3:AbortMultipartUpload",
      "s3:ListMultipartUploadParts",
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.trust.arn,
      "${aws_s3_bucket.trust.arn}/*",
      # C3: the gate's own receipt was written to the shared models bucket too, so a peer
      # worker could manufacture a validated result. Only this role may publish here.
      "${aws_s3_bucket.handoff.arn}/validated/v1/*",
    ]
  }


  # SageMaker uploads the ProcessingOutput here. Scoped rather than bucket-wide: an
  # unrestricted PutObject let the validation role replace the very validation code it
  # executes, which lives under .../code/ in this bucket.
  statement {
    sid       = "WriteValidateProcessingOutput"
    effect    = "Allow"
    actions   = ["s3:PutObject"]
    resources = ["${local.models_bucket_arn}/${var.pipeline_prefix}/validated/*"]
  }

  statement {
    sid    = "NeverReplaceTheCodeItRuns"
    effect = "Deny"
    actions = [
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:DeleteObjectVersion",
    ]
    resources = ["${local.models_bucket_arn}/${var.pipeline_prefix}/code/*"]
  }
}

resource "aws_iam_role_policy" "validation" {
  name   = "${local.prefix}-validation"
  role   = aws_iam_role.validation.id
  policy = data.aws_iam_policy_document.validation.json
}

# --- Discovery: the pipeline reads these, mirroring the Foundation SSM contract ------
# The pipeline fails CLOSED when these are absent rather than falling back to the
# Foundation role, so an incomplete deploy cannot silently restore the old boundary.
resource "aws_ssm_parameter" "workload_role_arn" {
  name  = "/${var.project_name}/component/workload-role-arn"
  type  = "String"
  value = aws_iam_role.workload.arn
}

resource "aws_ssm_parameter" "training_role_arn" {
  name  = "/${var.project_name}/component/training-role-arn"
  type  = "String"
  value = aws_iam_role.training.arn
  tags  = local.tags
}

resource "aws_ssm_parameter" "validation_role_arn" {
  name  = "/${var.project_name}/component/validation-role-arn"
  type  = "String"
  value = aws_iam_role.validation.arn
}

resource "aws_ssm_parameter" "trust_bucket" {
  name  = "/${var.project_name}/component/trust-bucket"
  type  = "String"
  value = aws_s3_bucket.trust.id
}

# I11: pipeline_prefix is interpolated into every IAM policy in this file, but the launchers
# defaulted to "vla-pipeline" independently and never discovered the deployed value. A
# legitimate non-default setting therefore deployed policies that DENY the normal launchers'
# outputs, and the variable's note that the two "must match" is not enforcement. Published
# through the same discovery contract as the roles, so the launchers read what was deployed.
# Same defect as pipeline_prefix, one field over: build_images.py defaulted the CodeBuild project
# name to "vla-image-build" and never discovered the deployed value. In this account a
# CloudFormation stack already owns a project by that exact name, running as its OWN role -- so
# every build submitted the source to the trust bucket this component created and then triggered a
# project whose role cannot read it, failing at DOWNLOAD_SOURCE with a 403 that names a role
# Terraform does not manage. Terraform's own codebuild role has held the correct grant all along.
#
# Published so the scripts read what was deployed. A component-scoped default means a fresh account
# cannot collide in the first place, and var.codebuild_project_name still lets a deployment point at
# an externally managed project deliberately.
resource "aws_ssm_parameter" "codebuild_project" {
  name  = "/${var.project_name}/component/codebuild-project"
  type  = "String"
  value = aws_codebuild_project.image_build.name
  tags  = local.tags
}

resource "aws_ssm_parameter" "pipeline_prefix" {
  name  = "/${var.project_name}/component/pipeline-prefix"
  type  = "String"
  value = var.pipeline_prefix
  tags  = local.tags
}

# I11: same reasoning as the handoff bucket. Promoted artifacts stream in as multipart uploads,
# and an aborted or failed publication must not leave billable parts behind when the abort call
# itself fails -- which it always did before the role was granted s3:AbortMultipartUpload.
resource "aws_s3_bucket_lifecycle_configuration" "trust_abort_incomplete" {
  bucket = aws_s3_bucket.trust.id
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}
