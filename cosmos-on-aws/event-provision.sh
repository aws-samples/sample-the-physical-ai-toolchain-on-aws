#!/usr/bin/env bash
# SPDX-License-Identifier: MIT-0
#
# Event pre-provisioning for the "Cosmos 3 on EKS" lab.
#
# Creates the Cosmos 3 EKS cluster with this repo's Terraform, then stages the large
# artifacts, so workshop attendees don't spend their session waiting on downloads.
# Designed to run unattended in CodeBuild (see the workshop CloudFormation template),
# but it works from any shell with credentials.
#
# Two properties matter for a 100-account scale event:
#   * Idempotent - safe to re-run; Terraform reconciles and the staging Job is replaced.
#   * Observable  - fails loudly with a non-zero exit so a per-account health check
#                   (`aws codebuild batch-get-builds`) tells you which accounts are ready.
#
# Environment:
#   AWS_REGION_NAME     (required) Region to deploy into
#   STATE_BUCKET        (required) S3 bucket for Terraform remote state
#   ENVIRONMENT         Name prefix for created resources (default dev). Change it to
#                       isolate this deployment when the account already has a Cosmos 3
#                       cluster: the prefix appears in the cluster name AND in IAM role
#                       names that hardcode "cosmos3", so a different eks_cluster_name
#                       alone is NOT enough to avoid collisions.
#                       NOTE: the IRSA policy grants DATA_BUCKET (and the weights
#                       bucket); with DATA_BUCKET set no Foundation deployment is needed.
#   EKS_CLUSTER_NAME    Cluster name suffix               (default cosmos3)
#   GPU_INSTANCE_TYPE   GPU node instance type            (default g6e.4xlarge)
#   GPU_AZ              AZ to pin the GPU node group to   (default: any private subnet)
#   GPU_CR_ID           Capacity Block ID, if using one   (default: on-demand)
#   GPU_DESIRED_SIZE    GPU nodes to pre-warm             (default 1)
#   WORKSTATION_ROLE_ARN  IAM role granted EKS cluster-admin, so the workshop
#                       workstation can run kubectl and `terraform plan` (the Helm
#                       provider must reach the Kubernetes API to refresh state).
#   CENTRAL_WEIGHTS_S3  Central S3 prefix with unpacked Cosmos 3 Nano weights (optional)
#   CENTRAL_WEIGHTS_URL Presigned HTTPS URL for a Cosmos 3 Nano tar archive (optional;
#                       mutually exclusive with CENTRAL_WEIGHTS_S3)
#   CENTRAL_WEIGHTS_SHA256  Required SHA-256 for CENTRAL_WEIGHTS_URL
#   COSMOS_IMAGE_URI    ECR image for vllm-omni cosmos3; also warms the node's image cache
#   DATA_BUCKET         Workshop data bucket (optional). The pod IRSA policy grants it,
#                       and CENTRAL_WEIGHTS_S3 is relayed into it when the account has no
#                       /physical-ai/datasets-bucket SSM parameter.
#   VPC_ID              VPC to build the cluster in. Empty = read the Foundation VPC
#                       from SSM. Must be set together with EKS_NODE_SUBNET_IDS.
#   EKS_NODE_SUBNET_IDS Comma-separated subnet IDs for the node groups (system + GPU).
#                       Empty = read the Foundation private subnets from SSM. With
#                       GPU_AZ set, at least one must be in GPU_AZ.
#   EKS_CONTROL_PLANE_SUBNET_IDS  Comma-separated subnet IDs for the EKS control plane
#                       ENIs; must span two AZs. Empty = the node subnets. Requires
#                       VPC_ID and EKS_NODE_SUBNET_IDS.
#   EKS_API_INGRESS_SG_IDS  Comma-separated security group IDs allowed to reach the
#                       cluster's private API endpoint on 443 (for example the
#                       workstation's). Empty = no extra rule.
#   ENABLE_IMAGE_BUILDS true or false, default false. true keeps the Cosmos Transfer /
#                       Cosmos 3 image-build CodeBuild projects, which need the Foundation
#                       ECR SSM parameters. Set it when you run this script by hand in a
#                       Foundation account whose state already holds those projects;
#                       otherwise the next run destroys them.
#   POD_GRANT_FOUNDATION_BUCKET  empty (default) = grant the Foundation datasets bucket
#                       only when DATA_BUCKET is unset; true or false = force it on or
#                       off. Set true when you run this by hand in a Foundation account
#                       with DATA_BUCKET set and still need pods to read the Foundation
#                       bucket (for example cosmos3-job.yaml).
#   TF_DIR              Terraform directory               (default: <repo>/cosmos-on-aws/infra)

set -euo pipefail

: "${AWS_REGION_NAME:?set AWS_REGION_NAME}"
: "${STATE_BUCKET:?set STATE_BUCKET}"
ENVIRONMENT="${ENVIRONMENT:-dev}"
EKS_CLUSTER_NAME="${EKS_CLUSTER_NAME:-cosmos3}"
GPU_INSTANCE_TYPE="${GPU_INSTANCE_TYPE:-g6e.4xlarge}"
GPU_AZ="${GPU_AZ:-}"
GPU_CR_ID="${GPU_CR_ID:-}"
GPU_DESIRED_SIZE="${GPU_DESIRED_SIZE:-1}"
CENTRAL_WEIGHTS_S3="${CENTRAL_WEIGHTS_S3:-}"
CENTRAL_WEIGHTS_URL="${CENTRAL_WEIGHTS_URL:-}"
CENTRAL_WEIGHTS_SHA256="${CENTRAL_WEIGHTS_SHA256:-}"
COSMOS_IMAGE_URI="${COSMOS_IMAGE_URI:-}"
WORKSTATION_ROLE_ARN="${WORKSTATION_ROLE_ARN:-}"
VPC_ID="${VPC_ID:-}"
EKS_NODE_SUBNET_IDS="${EKS_NODE_SUBNET_IDS:-}"
EKS_CONTROL_PLANE_SUBNET_IDS="${EKS_CONTROL_PLANE_SUBNET_IDS:-}"
EKS_API_INGRESS_SG_IDS="${EKS_API_INGRESS_SG_IDS:-}"
DATA_BUCKET="${DATA_BUCKET:-}"
# Resolved in section 2 when either weight source is set. Empty keeps the staging Job
# heredoc safe under set -u when only COSMOS_IMAGE_URI is set.
WEIGHTS_BUCKET=""
WEIGHTS_ARCHIVE_KEY=""
ENABLE_IMAGE_BUILDS="${ENABLE_IMAGE_BUILDS:-false}"
POD_GRANT_FOUNDATION_BUCKET="${POD_GRANT_FOUNDATION_BUCKET:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_DIR="${TF_DIR:-$SCRIPT_DIR/infra}"
TERRAFORM_VERSION="${TERRAFORM_VERSION:-1.16.4}"

log() { echo "[$(date -u +%H:%M:%S)] $*"; }
# stderr: survives $( ... ), where stdout would be captured and discarded.
die() { log "ERROR: $*" >&2; exit 1; }
# "a, b" -> ["a", "b"]; validates every element against an ID regex.
# Empty input returns [] BEFORE touching the array: on bash < 4.4 (macOS /bin/bash
# 3.2) expanding an empty array under `set -u` aborts with "unbound variable".
to_hcl_list() {  # $1=name $2=value $3=regex
  local out="" item
  local -a items=()
  [ -n "${2//[[:space:],]/}" ] || { printf '[]'; return 0; }
  IFS=',' read -r -a items <<< "$2"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"; [ -z "$item" ] && continue
    [[ "$item" =~ $3 ]] || die "$1 contains '$item', which is not a valid ID"
    out="${out:+$out, }\"$item\""
  done
  printf '[%s]' "$out"
}

# Validate the switches before anything else runs. Terraform would reject a bad value
# too, but only after the tfvars is published and `terraform init` has run.
case "$ENABLE_IMAGE_BUILDS" in
  true|false) ;;
  *) die "ENABLE_IMAGE_BUILDS must be true or false, got '$ENABLE_IMAGE_BUILDS'" ;;
esac
case "$POD_GRANT_FOUNDATION_BUCKET" in
  ''|true|false) ;;
  *) die "POD_GRANT_FOUNDATION_BUCKET must be empty, true or false, got '$POD_GRANT_FOUNDATION_BUCKET'" ;;
esac
if [ -n "$CENTRAL_WEIGHTS_S3" ] && [ -n "$CENTRAL_WEIGHTS_URL" ]; then
  die "set only one of CENTRAL_WEIGHTS_S3 or CENTRAL_WEIGHTS_URL"
fi
if [ -n "$CENTRAL_WEIGHTS_URL" ] && ! [[ "$CENTRAL_WEIGHTS_SHA256" =~ ^[0-9a-fA-F]{64}$ ]]; then
  die "CENTRAL_WEIGHTS_SHA256 must be 64 hexadecimal characters when CENTRAL_WEIGHTS_URL is set"
fi
if [ -z "$CENTRAL_WEIGHTS_URL" ] && [ -n "$CENTRAL_WEIGHTS_SHA256" ]; then
  die "CENTRAL_WEIGHTS_SHA256 requires CENTRAL_WEIGHTS_URL"
fi

#-----------------------------------------------------------------------------------
# 1. Tooling
#-----------------------------------------------------------------------------------
if ! command -v terraform >/dev/null 2>&1; then
  log "Installing Terraform ${TERRAFORM_VERSION}"
  curl -fsSL -o /tmp/tf.zip \
    "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
  unzip -q -o /tmp/tf.zip -d /usr/local/bin && rm -f /tmp/tf.zip
fi
if ! command -v kubectl >/dev/null 2>&1; then
  log "Installing kubectl"
  curl -fsSL -o /usr/local/bin/kubectl \
    "https://dl.k8s.io/release/$(curl -fsSL https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
  chmod +x /usr/local/bin/kubectl
fi
# NOTE: do not pipe into `head` here. Under `set -o pipefail`, head closing the pipe
# early sends SIGPIPE to terraform and the script dies with exit 141 - which happened
# as soon as terraform printed an extra upgrade-notice line.
terraform version

#-----------------------------------------------------------------------------------
# 1b. Network preflight
#
# When the caller passes a network (the workshop stack passes its own VPC), check it
# here, before Terraform: a wrong ID, a subnet in another VPC, a single-AZ control plane
# or no node subnet in GPU_AZ would otherwise surface much later as an opaque plan or
# CreateCluster error. With no network variables set, Terraform reads the Foundation VPC
# from SSM, as before, and this section is skipped.
#-----------------------------------------------------------------------------------
SUBNET_RE='^subnet-[0-9a-f]{8,17}$'
# "a, b,,c" -> one trimmed, non-empty item per line (empty input prints nothing).
split_csv() {
  local item
  local -a items=()
  [ -n "${1//[[:space:],]/}" ] || return 0
  IFS=',' read -r -a items <<< "$1"
  for item in "${items[@]}"; do
    item="${item//[[:space:]]/}"
    [ -n "$item" ] && printf '%s\n' "$item"
  done
  return 0
}

if [ -n "${VPC_ID}${EKS_NODE_SUBNET_IDS}${EKS_CONTROL_PLANE_SUBNET_IDS}" ]; then
  VPC_RE='^vpc-[0-9a-f]{8,17}$'
  if ! [[ "$VPC_ID" =~ $VPC_RE ]] || [ -z "${EKS_NODE_SUBNET_IDS//[[:space:],]/}" ]; then
    die "VPC_ID and EKS_NODE_SUBNET_IDS must be set together (control-plane subnets also require them)"
  fi
  # Called directly (not in $( )), so a malformed ID exits here with a named cause.
  to_hcl_list EKS_NODE_SUBNET_IDS "$EKS_NODE_SUBNET_IDS" "$SUBNET_RE" >/dev/null
  to_hcl_list EKS_CONTROL_PLANE_SUBNET_IDS "$EKS_CONTROL_PLANE_SUBNET_IDS" "$SUBNET_RE" >/dev/null

  NODE_SUBNETS="$(split_csv "$EKS_NODE_SUBNET_IDS")"
  CP_SUBNETS="$(split_csv "$EKS_CONTROL_PLANE_SUBNET_IDS")"
  # Empty control-plane list = the EKS module falls back to the node subnets.
  [ -n "$CP_SUBNETS" ] || CP_SUBNETS="$NODE_SUBNETS"
  # One describe call for the unique set (the node subnet is usually in both lists).
  SUBNET_ARGS=()
  while IFS= read -r sid; do
    SUBNET_ARGS+=("$sid")
  done <<< "$(printf '%s\n%s\n' "$NODE_SUBNETS" "$CP_SUBNETS" | sort -u | awk 'NF')"
  SUBNET_LIST="${SUBNET_ARGS[*]}"

  if ! SUBNET_ROWS="$(aws ec2 describe-subnets --region "$AWS_REGION_NAME" \
       --subnet-ids "${SUBNET_ARGS[@]}" \
       --query 'Subnets[].[SubnetId,VpcId,AvailabilityZone,AvailabilityZoneId]' \
       --output text)"; then
    die "describe-subnets failed for ${SUBNET_LIST} (wrong ID or Region?)"
  fi
  # $1=subnet ID, $2=column (2 VpcId, 3 AvailabilityZone, 4 AvailabilityZoneId).
  # awk reads all input (no early exit), so the printf never sees SIGPIPE under pipefail.
  subnet_field() {
    printf '%s\n' "$SUBNET_ROWS" \
      | awk -F'\t' -v id="$1" -v col="$2" '$1 == id && !found { print $col; found = 1 }'
  }

  for sid in "${SUBNET_ARGS[@]}"; do
    vpc="$(subnet_field "$sid" 2)"
    [ -n "$vpc" ] || die "describe-subnets returned no row for ${sid}"
    [ "$vpc" = "$VPC_ID" ] || die "subnet ${sid} is in ${vpc}, not VPC_ID ${VPC_ID}"
  done

  # EKS rejects control planes in these AZ IDs (EKS docs). Names map to IDs per account,
  # so the template's derived AZ can land on one in us-east-1.
  EKS_UNSUPPORTED_AZ_IDS="use1-az3 usw1-az2 cac1-az3"
  # Subnet IDs were validated against SUBNET_RE above, so word-splitting them is safe.
  CP_AZS=""
  CP_AZ_COUNT=0
  CP_AZ_LOG=""
  for sid in $CP_SUBNETS; do
    az="$(subnet_field "$sid" 3)"
    az_id="$(subnet_field "$sid" 4)"
    case " $EKS_UNSUPPORTED_AZ_IDS " in *" $az_id "*)
      die "control-plane subnet $sid is in $az ($az_id), which EKS does not support. Delete the stack and recreate it with the parameter EksControlPlaneSecondaryAz set to a supported AZ (or choose a different AZ 2)." ;;
    esac
    case " $CP_AZS " in
      *" $az "*) ;;
      *) CP_AZS="${CP_AZS:+$CP_AZS }$az"; CP_AZ_COUNT=$((CP_AZ_COUNT + 1)) ;;
    esac
    CP_AZ_LOG="${CP_AZ_LOG:+$CP_AZ_LOG, }${sid} ${az} (${az_id})"
  done
  [ "$CP_AZ_COUNT" -ge 2 ] || die "EKS control plane subnets span only ${CP_AZS}; EKS needs two AZs"

  NODE_AZS=""
  NODE_LOG=""
  for sid in $NODE_SUBNETS; do
    az="$(subnet_field "$sid" 3)"
    case " $NODE_AZS " in *" $az "*) ;; *) NODE_AZS="${NODE_AZS:+$NODE_AZS }$az" ;; esac
    NODE_LOG="${NODE_LOG:+$NODE_LOG, }${sid} (${az})"
  done
  if [ -n "$GPU_AZ" ]; then
    case " $NODE_AZS " in
      *" $GPU_AZ "*) ;;
      *) die "no node subnet in GPU_AZ=${GPU_AZ} (node subnets are in ${NODE_AZS})" ;;
    esac
  fi

  log "Network: VPC ${VPC_ID}"
  log "  node subnets: ${NODE_LOG}"
  log "  control-plane subnets: ${CP_AZ_LOG}"
else
  log "Network: no VPC_ID/subnets passed - Terraform reads the Foundation VPC from SSM"
fi

#-----------------------------------------------------------------------------------
# 2. Terraform: EKS cluster with a PRE-WARMED GPU node group
#
# The node group starts at desired=GPU_DESIRED_SIZE rather than 0 so the first job an
# attendee submits schedules immediately, instead of waiting ~5 min for Cluster
# Autoscaler to bring a node up (and another ~4 min to pull a 30 GB image).
#
# Remote state lives in STATE_BUCKET so attendees can later run
# `terraform init && terraform plan` from the workstation and see "No changes" -
# proof that the code in this repo is what built their cluster.
#-----------------------------------------------------------------------------------
cd "$TF_DIR"
cat > backend.tf <<EOF
terraform {
  backend "s3" {
    bucket = "${STATE_BUCKET}"
    key    = "cosmos-on-aws/eks/terraform.tfstate"
    region = "${AWS_REGION_NAME}"
  }
}
EOF
log "Wrote backend.tf -> s3://${STATE_BUCKET}/cosmos-on-aws/eks/terraform.tfstate"
log "Environment prefix: ${ENVIRONMENT}  Cluster suffix: ${EKS_CLUSTER_NAME}"

# Write the variable values to terraform.tfvars instead of passing them as -var flags.
# Terraform auto-loads this file, so an attendee can later run a bare
# `terraform init && terraform plan` and get "No changes" - with -var flags they would
# have to retype every value identically or the plan would show spurious drift.
# The file is also copied to the state bucket so the workstation can fetch the exact
# same values.
# Grant cluster-admin explicitly to (a) the identity running this script, so it can
# install the Helm release and run kubectl, and (b) the workstation role, so attendees
# can use kubectl and run `terraform plan`.
#
# Both are listed explicitly because the EKS module's
# enable_cluster_creator_admin_permissions is disabled: that option derives an access
# entry from the current caller, which makes every plan run by a different principal
# show spurious drift.
SELF_ARN="$(aws sts get-caller-identity --query Arn --output text)"
# Assumed-role session ARNs (arn:aws:sts::<acct>:assumed-role/<role>/<session>) are not
# valid access-entry principals - convert to the underlying role ARN.
case "$SELF_ARN" in
  *:assumed-role/*)
    ACCT="$(aws sts get-caller-identity --query Account --output text)"
    ROLE_NAME="$(echo "$SELF_ARN" | cut -d/ -f2)"
    SELF_ARN="arn:aws:iam::${ACCT}:role/${ROLE_NAME}"
    ;;
esac
ADMIN_ARNS="[\"${SELF_ARN}\"]"
if [ -n "$WORKSTATION_ROLE_ARN" ] && [ "$WORKSTATION_ROLE_ARN" != "$SELF_ARN" ]; then
  ADMIN_ARNS="[\"${SELF_ARN}\", \"${WORKSTATION_ROLE_ARN}\"]"
fi
log "Cluster admins: ${ADMIN_ARNS}"

# Read-only cluster access for the workshop participant role, so the EKS console's
# Resources tab can show Jobs, pods and logs. Resolve the configured name first. Event
# platforms do not use one universal role name, so if that name is absent, accept exactly
# one IAM role whose name contains "participant". Never guess when several match.
VIEW_ARNS="[]"
if [ -n "${PARTICIPANT_ROLE_NAME:-}" ]; then
  PARTICIPANT_ARN=""
  if PARTICIPANT_ARN="$(aws iam get-role --role-name "${PARTICIPANT_ROLE_NAME}" \
       --query 'Role.Arn' --output text 2>/dev/null)" && [ -n "$PARTICIPANT_ARN" ]; then
    log "Resolved configured participant role: ${PARTICIPANT_ARN}"
  else
    PARTICIPANT_ARN=""
    PARTICIPANT_CANDIDATES="$(aws iam list-roles \
      --query 'Roles[?contains(RoleName, `participant`) || contains(RoleName, `Participant`)].Arn' \
      --output text 2>/dev/null || true)"
    set -- $PARTICIPANT_CANDIDATES
    if [ "$#" -eq 1 ] && [ "$1" != None ]; then
      PARTICIPANT_ARN="$1"
      log "Configured participant role '${PARTICIPANT_ROLE_NAME}' not found; auto-detected ${PARTICIPANT_ARN}"
    elif [ "$#" -gt 1 ]; then
      log "Configured participant role '${PARTICIPANT_ROLE_NAME}' not found; multiple participant-role candidates exist, so console view access was not guessed"
    else
      log "Participant role '${PARTICIPANT_ROLE_NAME}' not found - skipping console view access"
    fi
  fi
  if [ -n "$PARTICIPANT_ARN" ]; then
    VIEW_ARNS="[\"${PARTICIPANT_ARN}\"]"
    log "Participant role granted read-only cluster access: ${PARTICIPANT_ARN}"
  fi
fi

# Weights bucket: where section 4 relays either configured weight source and where the
# staging Job reads from. Resolved here, before the tfvars, because the pod IRSA policy
# must grant it. An existing /physical-ai/datasets-bucket parameter (a Foundation account)
# wins; otherwise the stack's DATA_BUCKET. The SSM read sits inside an `if`, so a missing
# parameter - the normal case in a workshop account - cannot abort the script under set -e.
if [ -n "$CENTRAL_WEIGHTS_S3" ] || [ -n "$CENTRAL_WEIGHTS_URL" ]; then
  if SSM_BUCKET="$(aws ssm get-parameter --name /physical-ai/datasets-bucket \
       --region "$AWS_REGION_NAME" --query Parameter.Value --output text 2>/dev/null)" \
     && [ -n "$SSM_BUCKET" ] && [ "$SSM_BUCKET" != None ]; then
    WEIGHTS_BUCKET="$SSM_BUCKET"
    WEIGHTS_SOURCE="SSM /physical-ai/datasets-bucket"
  elif [ -n "$DATA_BUCKET" ]; then
    WEIGHTS_BUCKET="$DATA_BUCKET"
    WEIGHTS_SOURCE="DATA_BUCKET"
  else
    die "a central weight source is set but there is no relay bucket: /physical-ai/datasets-bucket does not exist and DATA_BUCKET is unset"
  fi
  log "Weights bucket: ${WEIGHTS_BUCKET} (source: ${WEIGHTS_SOURCE})"
fi

# The per-participant data bucket (DATA_BUCKET) is created by the workshop CloudFormation
# stack and does not match this module's <project>-<environment>-datasets-<account>
# naming, so the pod IRSA policy grants it explicitly through eks_pod_extra_s3_buckets,
# together with the weights bucket when that differs. Without this the generation pod
# gets AccessDenied reading the teleoperation dataset the lab points at.
EXTRA_BUCKETS_CSV="$DATA_BUCKET"
if [ -n "$WEIGHTS_BUCKET" ] && [ "$WEIGHTS_BUCKET" != "$DATA_BUCKET" ]; then
  EXTRA_BUCKETS_CSV="${EXTRA_BUCKETS_CSV:+$EXTRA_BUCKETS_CSV,}${WEIGHTS_BUCKET}"
fi
EXTRA_BUCKETS="$(to_hcl_list 'DATA_BUCKET/weights bucket' "$EXTRA_BUCKETS_CSV" \
  '^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$')" || exit 1
log "Pod S3 extra buckets: ${EXTRA_BUCKETS}"

# The Foundation datasets bucket (physical-ai-<ENVIRONMENT>-datasets-<account>) is
# granted only when DATA_BUCKET is unset, i.e. a manual run without a workshop stack.
# S3 names are global and that one is predictable, so an account without the bucket
# should not hand the pod write access to whoever creates it.
# POD_GRANT_FOUNDATION_BUCKET=true|false overrides the rule.
if [ -n "$POD_GRANT_FOUNDATION_BUCKET" ]; then
  POD_FOUNDATION="$POD_GRANT_FOUNDATION_BUCKET"
  POD_FOUNDATION_WHY="POD_GRANT_FOUNDATION_BUCKET"
elif [ -n "$DATA_BUCKET" ]; then
  POD_FOUNDATION=false
  POD_FOUNDATION_WHY="DATA_BUCKET is set"
else
  POD_FOUNDATION=true
  POD_FOUNDATION_WHY="DATA_BUCKET is unset"
fi
# Same rule as the Terraform precondition on the pod policy, checked before publishing.
if [ "$POD_FOUNDATION" = false ] && [ "$EXTRA_BUCKETS" = "[]" ]; then
  die "pod S3 policy would grant no bucket: set DATA_BUCKET or POD_GRANT_FOUNDATION_BUCKET=true"
fi
log "Pod S3 Foundation datasets bucket grant: ${POD_FOUNDATION} (${POD_FOUNDATION_WHY})"

# Network lists. Computed here, not inside the heredoc: `die` inside $( ) only exits the
# subshell, so `|| exit 1` stops the script (the cause is already on stderr).
NODE_SUBNETS_HCL="$(to_hcl_list EKS_NODE_SUBNET_IDS "$EKS_NODE_SUBNET_IDS" \
  '^subnet-[0-9a-f]{8,17}$')" || exit 1
CP_SUBNETS_HCL="$(to_hcl_list EKS_CONTROL_PLANE_SUBNET_IDS "$EKS_CONTROL_PLANE_SUBNET_IDS" \
  '^subnet-[0-9a-f]{8,17}$')" || exit 1
API_SG_HCL="$(to_hcl_list EKS_API_INGRESS_SG_IDS "$EKS_API_INGRESS_SG_IDS" \
  '^sg-[0-9a-f]{8,17}$')" || exit 1

cat > terraform.tfvars <<EOF
aws_region                      = "${AWS_REGION_NAME}"
environment                     = "${ENVIRONMENT}"
enable_eks_cluster              = true
enable_ec2_server               = false
eks_cluster_name                = "${EKS_CLUSTER_NAME}"
eks_gpu_node_instance_type      = "${GPU_INSTANCE_TYPE}"
eks_gpu_node_desired_size       = ${GPU_DESIRED_SIZE}
eks_gpu_availability_zone       = "${GPU_AZ}"
eks_gpu_capacity_reservation_id = "${GPU_CR_ID}"
# Grants the workstation IAM role EKS cluster-admin. Without an access entry the
# Helm provider cannot reach the Kubernetes API and even \`terraform plan\` fails with
# "Kubernetes cluster unreachable".
eks_admin_principal_arns        = ${ADMIN_ARNS}
# Read-only access so participants can observe the cluster in the EKS console.
eks_view_principal_arns         = ${VIEW_ARNS}
# Grants the generation pod read/write on the CloudFormation-created data bucket.
eks_pod_extra_s3_buckets        = ${EXTRA_BUCKETS}
# Image-build CodeBuild projects push to Foundation ECR repos; not used by the workshop.
# ENABLE_IMAGE_BUILDS defaults to false (event-provision.sh accepts only true or false).
enable_image_builds                         = ${ENABLE_IMAGE_BUILDS}
eks_pod_grant_foundation_datasets_bucket    = ${POD_FOUNDATION}
# Network: the workshop stack's own VPC. Empty values fall back to Foundation SSM.
vpc_id                                      = "${VPC_ID}"
private_subnet_ids                          = ${NODE_SUBNETS_HCL}
eks_control_plane_subnet_ids                = ${CP_SUBNETS_HCL}
eks_cluster_api_ingress_security_group_ids  = ${API_SG_HCL}
EOF
log "Wrote terraform.tfvars"
cat terraform.tfvars

# Publish BEFORE apply: the cleanup destroy (and the workstation) need these exact values,
# including the network IDs, even if the apply below fails partway and leaves a cluster
# in the stack VPC.
aws s3 cp terraform.tfvars "s3://${STATE_BUCKET}/cosmos-on-aws/eks/terraform.tfvars" \
  --only-show-errors || die "could not publish terraform.tfvars to s3://${STATE_BUCKET}"

terraform init -input=false -upgrade
log "Applying Terraform (EKS cluster + GPU node group)"
terraform apply -input=false -auto-approve

CLUSTER="$(terraform output -raw eks_cluster_name)"
IRSA_ROLE="$(terraform output -raw eks_pod_irsa_role_arn)"
log "Cluster: ${CLUSTER}"

aws eks update-kubeconfig --name "$CLUSTER" --region "$AWS_REGION_NAME"
kubectl get nodes -o wide

#-----------------------------------------------------------------------------------
# 3. Service account bound to the Terraform-created IRSA role.
#
# Without this the staging (and generation) pods fall back to the default
# ServiceAccount, which resolves to the EKS node role - no S3 access at all.
#-----------------------------------------------------------------------------------
kubectl create serviceaccount cosmos3-generator -n default --dry-run=client -o yaml | kubectl apply -f -
kubectl annotate serviceaccount cosmos3-generator -n default \
  "eks.amazonaws.com/role-arn=${IRSA_ROLE}" --overwrite
log "ServiceAccount cosmos3-generator -> ${IRSA_ROLE}"

#-----------------------------------------------------------------------------------
# 3b. Optional hf-token Secret.
#
# The generation Job's secretKeyRef is optional. Create the Secret only when neither
# central weight source is configured and runtime Hugging Face fallback may be needed.
# Central staging deliberately skips both the Secrets Manager read and Kubernetes Secret.
#-----------------------------------------------------------------------------------
HF_TOKEN_VALUE=""
if [ -z "$CENTRAL_WEIGHTS_S3" ] && [ -z "$CENTRAL_WEIGHTS_URL" ]; then
  HF_TOKEN_VALUE="$(aws secretsmanager get-secret-value --secret-id physical-ai/hf-token \
    --region "$AWS_REGION_NAME" --query SecretString --output text 2>/dev/null \
    | python3 -c "
import sys, json
s = sys.stdin.read().strip()
if not s:
    print('')
else:
    try:
        d = json.loads(s)
        print(d.get('token') or d.get('HF_TOKEN') or next(iter(d.values()), ''))
    except Exception:
        print(s)
" 2>/dev/null || echo '')"
fi

#-----------------------------------------------------------------------------------
# 3c. Cluster-scoped read access for the "cosmos3-viewers" group.
#
# eks.tf puts the participant's access entry in this group. The group exists so the
# entry can keep AmazonEKSViewPolicy - which does NOT cover cluster-scoped resources -
# and still read nodes, which the EKS console's Compute tab requires. Without this the
# console reports:
#   nodes is forbidden: User "..." cannot list resource "nodes" at the cluster scope
#
# Deliberately narrow. AmazonEKSAdminViewPolicy would grant the same visibility in one
# line, but it also exposes every Secret in the cluster, including hf-token - which at
# event scale may be a credential shared across participant accounts.
#-----------------------------------------------------------------------------------
cat <<'EOF' | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cosmos3-cluster-viewer
rules:
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["metrics.k8s.io"]
    resources: ["nodes", "pods"]
    verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: cosmos3-cluster-viewer
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cosmos3-cluster-viewer
subjects:
  - apiGroup: rbac.authorization.k8s.io
    kind: Group
    name: cosmos3-viewers
EOF
log "ClusterRole cosmos3-cluster-viewer bound to group cosmos3-viewers (node read for the console)"

if [ -z "$CENTRAL_WEIGHTS_S3" ] && [ -z "$CENTRAL_WEIGHTS_URL" ]; then
  kubectl create secret generic hf-token -n default \
    --from-literal=token="${HF_TOKEN_VALUE}" \
    --dry-run=client -o yaml | kubectl apply -f -
  if [ -n "$HF_TOKEN_VALUE" ]; then
    log "Secret hf-token created (${#HF_TOKEN_VALUE} chars from Secrets Manager)"
  else
    log "Secret hf-token created EMPTY - no physical-ai/hf-token in Secrets Manager. Hugging Face downloads will not work."
  fi
else
  log "Central weights configured - skipping Hugging Face token lookup and Kubernetes Secret creation"
fi

if [ -z "$CENTRAL_WEIGHTS_S3" ] && [ -z "$CENTRAL_WEIGHTS_URL" ] && [ -z "$COSMOS_IMAGE_URI" ]; then
  log "No central artifacts configured (CENTRAL_WEIGHTS_S3 / CENTRAL_WEIGHTS_URL / COSMOS_IMAGE_URI) - skipping staging"
  log "Pre-provisioning complete (cluster only)"
  exit 0
fi

#-----------------------------------------------------------------------------------
# 4. Relay weights into this account's weights bucket.
#
# A direct S3 prefix is synced as unpacked files. A presigned URL is downloaded once by
# CodeBuild, verified before upload, and stored as an archive in participant S3. The URL
# is never logged or passed into Kubernetes.
#-----------------------------------------------------------------------------------
if [ -n "$CENTRAL_WEIGHTS_S3" ]; then
  log "Relaying unpacked weights from ${CENTRAL_WEIGHTS_S3} to s3://${WEIGHTS_BUCKET}/cosmos3-assets/weights/"
  aws s3 sync "$CENTRAL_WEIGHTS_S3" \
    "s3://${WEIGHTS_BUCKET}/cosmos3-assets/weights/" --only-show-errors
elif [ -n "$CENTRAL_WEIGHTS_URL" ]; then
  SHA_PREFIX="$(printf '%s' "$CENTRAL_WEIGHTS_SHA256" | cut -c1-16)"
  WEIGHTS_ARCHIVE_KEY="cosmos3-assets/archives/cosmos3-nano.tar"
  ARCHIVE_TMP="/tmp/cosmos3-nano-${SHA_PREFIX}.tar"
  log "Downloading the presigned Cosmos weights archive (URL redacted)"
  curl --fail --location --silent --show-error --retry 5 --retry-all-errors \
    --connect-timeout 30 --output "$ARCHIVE_TMP" "$CENTRAL_WEIGHTS_URL" \
    || die "presigned Cosmos weights download failed"
  ACTUAL_SHA256="$(sha256sum "$ARCHIVE_TMP" | awk '{print $1}')"
  EXPECTED_SHA256="$(printf '%s' "$CENTRAL_WEIGHTS_SHA256" | tr '[:upper:]' '[:lower:]')"
  [ "$ACTUAL_SHA256" = "$EXPECTED_SHA256" ] \
    || die "Cosmos weights archive checksum mismatch: expected ${EXPECTED_SHA256}, got ${ACTUAL_SHA256}"
  log "Archive checksum passed; relaying it to s3://${WEIGHTS_BUCKET}/${WEIGHTS_ARCHIVE_KEY}"
  aws s3 cp "$ARCHIVE_TMP" "s3://${WEIGHTS_BUCKET}/${WEIGHTS_ARCHIVE_KEY}" \
    --only-show-errors
  rm -f "$ARCHIVE_TMP"
fi

#-----------------------------------------------------------------------------------
# 5. Staging Job: node-local weight cache + image pre-pull, in one step.
#
# hostPath, deliberately not EFS: the cluster runs in the workshop VPC, so the stack's
# EFS is reachable on the network, but efs-csi is not installed (out of scope; the
# cluster installs ebs-csi only). The GPU node is pre-warmed and
# long-lived, so /opt/cosmos3 persists for every pod that lands on it. If the node is
# ever replaced, re-run this script (or just re-apply the Job).
#
# Running the real Cosmos image here is what warms the kubelet image cache, so the
# 30 GB pre-pull and the weight copy are the same operation.
#-----------------------------------------------------------------------------------
# The official aws-cli image supports only direct `aws` execution; its shell and
# archive utilities are not a stable interface. Use AL2023 for the default staging
# path and install the two packages this Job requires before downloading 35 GiB.
STAGE_IMAGE="${COSMOS_IMAGE_URI:-public.ecr.aws/amazonlinux/amazonlinux:2023}"
log "Staging with image ${STAGE_IMAGE}"

kubectl delete job cosmos3-staging -n default --ignore-not-found
cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: cosmos3-staging
  namespace: default
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      serviceAccountName: cosmos3-generator
      nodeSelector:
        node.kubernetes.io/purpose: gpu
      tolerations:
        - key: "nvidia.com/gpu"
          operator: "Equal"
          value: "true"
          effect: "NoSchedule"
      containers:
        - name: stage
          image: ${STAGE_IMAGE}
          command: ["/bin/bash", "-lc"]
          args:
            - |
              set -euo pipefail
              if command -v dnf >/dev/null 2>&1; then
                dnf install -y --setopt=install_weak_deps=False awscli-2 tar
                dnf clean all
              elif ! command -v aws >/dev/null 2>&1; then
                pip install -q awscli
              fi
              for tool in aws sha256sum tar; do
                command -v "\$tool" >/dev/null 2>&1 || {
                  echo "Required staging tool is unavailable: \$tool" >&2
                  exit 1
                }
              done
              mkdir -p /models
              if [ -n "${WEIGHTS_ARCHIVE_KEY}" ]; then
                echo "Copying the verified weights archive from participant S3..."
                aws s3 cp "s3://${WEIGHTS_BUCKET}/${WEIGHTS_ARCHIVE_KEY}" \
                  /models/cosmos3-nano.tar --only-show-errors
                echo "${CENTRAL_WEIGHTS_SHA256}  /models/cosmos3-nano.tar" | sha256sum -c -
                rm -rf /models/cosmos3-nano.next
                mkdir -p /models/cosmos3-nano.next
                tar -xf /models/cosmos3-nano.tar -C /models/cosmos3-nano.next
                test -s /models/cosmos3-nano.next/config.json
                test -s /models/cosmos3-nano.next/model.safetensors.index.json
                test -s /models/cosmos3-nano.next/transformer/diffusion_pytorch_model.safetensors.index.json
                test -s /models/cosmos3-nano.next/vae/diffusion_pytorch_model.safetensors
                test -s /models/cosmos3-nano.next/hf-cache/hub/models--nvidia--Cosmos-1.0-Guardrail/snapshots/cf03c0395fac8c4de386c0bdab12cc4fc8d66362/config.json
                test -s /models/cosmos3-nano.next/hf-cache/hub/models--nvidia--Cosmos-Guardrail1/snapshots/d6d4bfa899a71454a700907664f3e88f503950cf/video_content_safety_filter/safety_filter.pt
                rm -rf /models/hf-cache.next
                mv /models/cosmos3-nano.next/hf-cache /models/hf-cache.next
                rm -rf /models/cosmos3-nano /models/hf-cache
                mv /models/cosmos3-nano.next /models/cosmos3-nano
                mv /models/hf-cache.next /models/hf-cache
                mkdir -p \
                  /models/hf-cache/hub/models--nvidia--Cosmos-1.0-Guardrail/refs \
                  /models/hf-cache/hub/models--nvidia--Cosmos-Guardrail1/refs
                printf '%s' cf03c0395fac8c4de386c0bdab12cc4fc8d66362 \
                  > /models/hf-cache/hub/models--nvidia--Cosmos-1.0-Guardrail/refs/main
                printf '%s' d6d4bfa899a71454a700907664f3e88f503950cf \
                  > /models/hf-cache/hub/models--nvidia--Cosmos-Guardrail1/refs/main
                rm -f /models/cosmos3-nano.tar
                echo "Runtime bundle extracted; model and guardrail caches validated."
                du -sh /models/cosmos3-nano /models/hf-cache
              elif [ -n "${CENTRAL_WEIGHTS_S3}" ]; then
                mkdir -p /models/cosmos3-nano
                echo "Copying unpacked weights to node-local disk..."
                aws s3 sync "s3://${WEIGHTS_BUCKET}/cosmos3-assets/weights/" \
                  /models/cosmos3-nano/ --only-show-errors
                test -s /models/cosmos3-nano/config.json
                test -s /models/cosmos3-nano/model.safetensors.index.json
                du -sh /models/cosmos3-nano
              fi
              nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
              echo STAGING_COMPLETE
          volumeMounts:
            - { name: models, mountPath: /models }
      volumes:
        - name: models
          hostPath:
            path: /opt/cosmos3
            type: DirectoryOrCreate
EOF

log "Waiting for staging to finish (weights + image pull can take ~10 min)"
if kubectl wait --for=condition=complete --timeout=45m job/cosmos3-staging -n default; then
  kubectl logs job/cosmos3-staging -n default --tail=10 || true
  log "Pre-provisioning complete - account is event-ready"
else
  log "ERROR: staging Job did not complete; preserving pod diagnostics in CodeBuild logs:"
  pods="$(kubectl get pods -n default -l job-name=cosmos3-staging -o name)"
  if [ -z "$pods" ]; then
    log "No staging pods remain; Job diagnostics follow"
  else
    for pod in $pods; do
      log "Diagnostics for ${pod}"
      kubectl get "$pod" -n default \
        -o jsonpath='{.status.phase}{" | reason="}{.status.containerStatuses[0].state.terminated.reason}{" | exit="}{.status.containerStatuses[0].state.terminated.exitCode}{"\n"}' \
        || true
      kubectl logs "$pod" -n default --all-containers=true --tail=200 || true
      kubectl describe "$pod" -n default | tail -40 || true
    done
  fi
  kubectl describe job cosmos3-staging -n default | tail -30 || true
  exit 1
fi
