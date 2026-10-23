#!/usr/bin/env bash
# SPDX-License-Identifier: MIT-0
#
# Stub-based tests for cosmos-on-aws/event-provision.sh. No AWS account, Terraform state
# or cluster is touched: aws, terraform, kubectl and curl are replaced by stubs on PATH
# that record their arguments and return canned output.
#
# Usage:
#   bash cosmos-on-aws/tests/test-event-provision.sh
#
# Prints PASS / FAIL / SKIP per case and a summary; exits non-zero if any case fails.
# Works on macOS /bin/bash 3.2 (no bash 4 features).

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$HERE/../event-provision.sh"
[ -f "$SCRIPT" ] || { echo "event-provision.sh not found at $SCRIPT" >&2; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/test-event-provision.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
STUBS="$WORK/stubs"
mkdir "$STUBS"

ACCOUNT=123456789012
VPC=vpc-0aaaaaaaaaaaaaaa0
SUBNET_B=subnet-0bbbbbbbbbbbbbbb1    # node subnet, us-east-2b
SUBNET_CP=subnet-0ccccccccccccccc2   # control-plane-only subnet, us-east-2a
SUBNET_B2=subnet-0eeeeeeeeeeeeeee4   # second subnet in us-east-2b (single-AZ control plane)
SUBNET_BAD_AZ=subnet-0fffffffffffffff5  # AZ ID use1-az3 (EKS-unsupported)
SG=sg-0ddddddddddddddd3
DATA=physical-ai-data-$ACCOUNT
FOUNDATION=physical-ai-dev-datasets-$ACCOUNT
CENTRAL=s3://central-weights-$ACCOUNT/cosmos3/
PRESIGNED='https://central.example/cosmos3-nano.tar?X-Amz-Signature=test'
ARCHIVE_SHA=3c2d3d2030ab8968d27cb1f586ef53918d8ae04dbd9fa55895b52414b46413e0
ARCHIVE_SHA_PREFIX=3c2d3d2030ab8968
TAB="$(printf '\t')"
SUBNET_ROWS="$(printf '%s\n' \
  "${SUBNET_B}${TAB}${VPC}${TAB}us-east-2b${TAB}use2-az2" \
  "${SUBNET_CP}${TAB}${VPC}${TAB}us-east-2a${TAB}use2-az1" \
  "${SUBNET_B2}${TAB}${VPC}${TAB}us-east-2b${TAB}use2-az2" \
  "${SUBNET_BAD_AZ}${TAB}${VPC}${TAB}us-east-1c${TAB}use1-az3")"

#-----------------------------------------------------------------------------------
# Stubs. Each appends "<tool> <args>" to $STUB_LOG. Unexpected calls exit 98 so a new,
# unstubbed AWS call fails the test instead of silently passing.
#-----------------------------------------------------------------------------------
cat > "$STUBS/aws" <<'EOF'
#!/bin/bash
printf 'aws %s\n' "$*" >> "$STUB_LOG"
case "$1 ${2:-}" in
  "sts get-caller-identity")
    case " $* " in
      *" --query Account "*) echo 123456789012 ;;
      *) echo "arn:aws:sts::123456789012:assumed-role/TerraformBuildRole/s" ;;
    esac ;;
  "iam get-role")
    case " $* " in
      *" --role-name admin "*) ROLE_ARN="${STUB_ADDITIONAL_VIEWER_ARN:-}" ;;
      *) ROLE_ARN="${STUB_CONFIGURED_PARTICIPANT_ARN:-}" ;;
    esac
    if [ -n "$ROLE_ARN" ]; then
      echo "$ROLE_ARN"
    else
      echo "An error occurred (NoSuchEntity) when calling the GetRole operation" >&2
      exit 254
    fi ;;
  "iam list-roles")
    [ -n "${STUB_PARTICIPANT_CANDIDATES:-}" ] && echo "$STUB_PARTICIPANT_CANDIDATES"
    exit 0 ;;
  "eks describe-access-entry") exit "${STUB_EKS_ACCESS_ENTRY_RC:-254}" ;;
  "eks create-access-entry"|"eks associate-access-policy") exit 0 ;;
  "ec2 describe-subnets")
    shift 2
    ids=""; collecting=0
    for a in "$@"; do
      case "$a" in
        --subnet-ids) collecting=1 ;;
        --*) collecting=0 ;;
        *) [ "$collecting" = 1 ] && ids="$ids $a" ;;
      esac
    done
    for id in $ids; do
      row="$(printf '%s\n' "${STUB_SUBNETS:-}" | awk -F'\t' -v id="$id" '$1 == id')"
      if [ -z "$row" ]; then
        echo "An error occurred (InvalidSubnetID.NotFound): The subnet ID '$id' does not exist" >&2
        exit 254
      fi
      printf '%s\n' "$row"
    done ;;
  "ssm get-parameter")
    if [ -n "${STUB_SSM_BUCKET:-}" ]; then
      echo "$STUB_SSM_BUCKET"
    else
      echo "An error occurred (ParameterNotFound) when calling the GetParameter operation: " >&2
      exit 254
    fi ;;
  "s3 cp") exit "${STUB_S3_CP_RC:-0}" ;;
  "s3 sync"|"eks update-kubeconfig") exit 0 ;;
  "secretsmanager get-secret-value")
    echo "An error occurred (ResourceNotFoundException)" >&2; exit 254 ;;
  *) echo "aws stub: unexpected call: aws $*" >&2; exit 98 ;;
esac
EOF

cat > "$STUBS/terraform" <<'EOF'
#!/bin/bash
printf 'terraform %s\n' "$*" >> "$STUB_LOG"
case "$1" in
  version) echo "Terraform v1.16.4 (stub)" ;;
  init|apply) exit 0 ;;
  output)
    case " $* " in
      *" eks_cluster_name "*) printf 'physical-ai-evt-cosmos3' ;;
      *" eks_pod_irsa_role_arn "*) printf 'arn:aws:iam::123456789012:role/physical-ai-evt-cosmos3-pod' ;;
      *) echo "terraform stub: unexpected output $*" >&2; exit 98 ;;
    esac ;;
  *) echo "terraform stub: unexpected call: terraform $*" >&2; exit 98 ;;
esac
EOF

# kubectl reads manifests from stdin for `apply -f -`; keep them for assertions.
cat > "$STUBS/kubectl" <<'EOF'
#!/bin/bash
printf 'kubectl %s\n' "$*" >> "$STUB_LOG"
case " $* " in *" -f - "*) cat >> "${STUB_LOG}.stdin" ;; esac
exit 0
EOF

cat > "$STUBS/curl" <<'EOF'
#!/bin/bash
printf 'curl %s\n' "$*" >> "$STUB_LOG"
out=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --output) shift; out="${1:-}" ;;
  esac
  shift || true
done
[ -n "$out" ] && printf 'archive-stub' > "$out"
exit "${STUB_CURL_RC:-0}"
EOF

cat > "$STUBS/sha256sum" <<'EOF'
#!/bin/bash
printf 'sha256sum %s\n' "$*" >> "$STUB_LOG"
if [ "${1:-}" = "-c" ]; then cat >/dev/null; exit 0; fi
printf '%s  %s\n' "${STUB_SHA256:-${CENTRAL_WEIGHTS_SHA256:-stub}}" "${1:-file}"
EOF
chmod +x "$STUBS/aws" "$STUBS/terraform" "$STUBS/kubectl" "$STUBS/curl" "$STUBS/sha256sum"

# Guard: under the case PATH every tool must resolve to its stub, never a real binary.
for tool in aws terraform kubectl curl sha256sum; do
  resolved="$(PATH="$STUBS:/usr/bin:/bin" command -v "$tool")"
  [ "$resolved" = "$STUBS/$tool" ] || { echo "stub guard: $tool resolves to $resolved" >&2; exit 2; }
done

#-----------------------------------------------------------------------------------
# Case runner and assertions
#-----------------------------------------------------------------------------------
PASS=0; FAIL=0; SKIP=0
BASH_BIN=/bin/bash

# run_case NAME [VAR=value ...] - runs the script under env -i with the given extra env.
# Sets CASE_DIR, OUT, ERR, LOG, TFV, RC.
run_case() {
  CASE_NAME="$1"; shift
  CASE_DIR="$WORK/case-$CASE_NAME"
  mkdir "$CASE_DIR"
  TF="$(mktemp -d "$CASE_DIR/tf.XXXXXX")"
  OUT="$CASE_DIR/stdout"; ERR="$CASE_DIR/stderr"; LOG="$CASE_DIR/stub.log"
  TFV="$TF/terraform.tfvars"
  : > "$LOG"
  env -i PATH="$STUBS:/usr/bin:/bin" HOME="$HOME" STUB_LOG="$LOG" \
    STUB_SUBNETS="$SUBNET_ROWS" \
    AWS_REGION_NAME=us-east-2 STATE_BUCKET="physical-ai-tfstate-$ACCOUNT" ENVIRONMENT=evt \
    TF_DIR="$TF" "$@" \
    "$BASH_BIN" "$SCRIPT" >"$OUT" 2>"$ERR" </dev/null
  RC=$?
  CASE_OK=1; CASE_MSGS=""
}

expect() {  # expect DESCRIPTION COMMAND... - records a failure if COMMAND fails
  local desc="$1"; shift
  if ! "$@"; then
    CASE_OK=0
    CASE_MSGS="${CASE_MSGS}    - ${desc}"$'\n'
  fi
}
rc_is() { [ "$RC" -eq "$1" ]; }
has() { grep -qF -- "$2" "$1" 2>/dev/null; }
hasnt() { ! grep -qF -- "$2" "$1" 2>/dev/null; }
has_re() { grep -qE -- "$2" "$1" 2>/dev/null; }
no_file() { [ ! -e "$1" ]; }
count_is() { [ "$(grep -cF -- "$2" "$1" 2>/dev/null)" -eq "$3" ]; }
# line_before FILE A B - first line containing A comes before the first containing B.
line_before() {
  local a b
  a="$(grep -nF -- "$2" "$1" | head -n1 | cut -d: -f1)"
  b="$(grep -nF -- "$3" "$1" | head -n1 | cut -d: -f1)"
  [ -n "$a" ] && [ -n "$b" ] && [ "$a" -lt "$b" ]
}
# Each subnet ID appears exactly once across all describe-subnets argument lists.
describe_once() {
  local id n
  for id in "$@"; do
    n="$(grep -F 'aws ec2 describe-subnets' "$LOG" | tr ' ' '\n' | grep -cxF -- "$id")"
    [ "$n" -eq 1 ] || return 1
  done
}
# tfvar FILE NAME VALUE - the tfvars line "NAME <spaces>= VALUE" exists verbatim.
tfvar() {
  awk -v n="$2" -v v="$3" '
    { line = $0; sub(/[ ]+=[ ]+/, " = ", line) }
    line == n " = " v { found = 1 }
    END { exit found ? 0 : 1 }' "$1"
}

finish() {
  # Every case: no stub saw a call it does not model (for example a command
  # substitution that runs a tool by accident).
  expect "no unexpected stub call" hasnt "$ERR" "stub: unexpected"
  if [ "$CASE_OK" = 1 ]; then
    PASS=$((PASS + 1)); echo "PASS  $1"
  else
    FAIL=$((FAIL + 1)); echo "FAIL  $1 (exit $RC)"
    printf '%s' "$CASE_MSGS"
    echo "    stderr (last 5 lines):"; tail -n 5 "$ERR" | sed 's/^/      /'
  fi
}

WORKSHOP_ENV=(
  "VPC_ID=$VPC"
  "EKS_NODE_SUBNET_IDS=$SUBNET_B"
  "EKS_CONTROL_PLANE_SUBNET_IDS=$SUBNET_B,$SUBNET_CP"
  "EKS_API_INGRESS_SG_IDS=$SG"
  "GPU_AZ=us-east-2b"
  "DATA_BUCKET=$DATA"
)

#-----------------------------------------------------------------------------------
# Cases (design section 15, a-k, plus POD_GRANT_FOUNDATION_BUCKET)
#-----------------------------------------------------------------------------------

# (a) datasets-bucket parameter missing -> relay into DATA_BUCKET
run_case a "CENTRAL_WEIGHTS_S3=$CENTRAL" "DATA_BUCKET=$DATA"
expect "exit 0" rc_is 0
expect "relay to DATA_BUCKET" has "$LOG" "aws s3 sync $CENTRAL s3://$DATA/cosmos3-assets/weights/"
expect "extra buckets = [DATA_BUCKET]" tfvar "$TFV" eks_pod_extra_s3_buckets "[\"$DATA\"]"
expect "source logged" has "$OUT" "Weights bucket: $DATA (source: DATA_BUCKET)"
expect "staging Job reads DATA_BUCKET" has "$LOG.stdin" "s3://$DATA/cosmos3-assets/weights/"
finish "a   datasets-bucket param missing -> relay into DATA_BUCKET"

# (b) parameter present -> its value wins and is added to the extra buckets
run_case b "CENTRAL_WEIGHTS_S3=$CENTRAL" "DATA_BUCKET=$DATA" "STUB_SSM_BUCKET=$FOUNDATION"
expect "exit 0" rc_is 0
expect "relay to SSM bucket" has "$LOG" "aws s3 sync $CENTRAL s3://$FOUNDATION/cosmos3-assets/weights/"
expect "extra buckets = both" tfvar "$TFV" eks_pod_extra_s3_buckets "[\"$DATA\", \"$FOUNDATION\"]"
expect "source logged" has "$OUT" "(source: SSM /physical-ai/datasets-bucket)"
finish "b   datasets-bucket param present -> its value used and granted"


# (b2) configured participant role missing -> exactly one participant-named role is auto-detected
PARTICIPANT_CANDIDATE="arn:aws:iam::$ACCOUNT:role/gtc-participant-role"
run_case b2 "${WORKSHOP_ENV[@]}" "PARTICIPANT_ROLE_NAME=WSParticipantRole" \
  "STUB_PARTICIPANT_CANDIDATES=$PARTICIPANT_CANDIDATE"
expect "exit 0" rc_is 0
expect "single participant role mapped for view" tfvar "$TFV" eks_view_principal_arns "[\"$PARTICIPANT_CANDIDATE\"]"
expect "auto-detection logged" has "$OUT" "auto-detected $PARTICIPANT_CANDIDATE"
finish "b2  missing configured participant role -> one candidate auto-detected"

# (b3) multiple participant-named roles -> do not guess
run_case b3 "${WORKSHOP_ENV[@]}" "PARTICIPANT_ROLE_NAME=WSParticipantRole" \
  "STUB_PARTICIPANT_CANDIDATES=arn:aws:iam::$ACCOUNT:role/event-participant arn:aws:iam::$ACCOUNT:role/gtc-participant-role"
expect "exit 0" rc_is 0
expect "ambiguous candidates not mapped" tfvar "$TFV" eks_view_principal_arns "[]"
expect "ambiguity logged" has "$OUT" "multiple participant-role candidates exist"
finish "b3  multiple participant roles -> no access-entry guess"

# (b4) optional exact admin role -> idempotent second read-only access entry
ADMIN_VIEWER_ARN="arn:aws:iam::$ACCOUNT:role/admin"
run_case b4 "${WORKSHOP_ENV[@]}" "PARTICIPANT_ROLE_NAME=WSParticipantRole" \
  "STUB_PARTICIPANT_CANDIDATES=$PARTICIPANT_CANDIDATE" \
  "ADDITIONAL_EKS_VIEWER_ROLE_NAME=admin" \
  "STUB_ADDITIONAL_VIEWER_ARN=$ADMIN_VIEWER_ARN"
expect "exit 0" rc_is 0
expect "participant remains Terraform-managed viewer" tfvar "$TFV" eks_view_principal_arns "[\"$PARTICIPANT_CANDIDATE\"]"
expect "admin viewer access entry created" has "$LOG" "aws eks create-access-entry --cluster-name physical-ai-evt-cosmos3 --principal-arn $ADMIN_VIEWER_ARN --kubernetes-groups cosmos3-viewers --type STANDARD --region us-east-2"
expect "admin read-only policy associated" has "$LOG" "aws eks associate-access-policy --cluster-name physical-ai-evt-cosmos3 --principal-arn $ADMIN_VIEWER_ARN --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSViewPolicy --access-scope type=cluster --region us-east-2"
expect "admin viewer mapping logged" has "$OUT" "Additional role granted read-only cluster access: $ADMIN_VIEWER_ARN"
finish "b4  participant + admin roles -> both get read-only EKS access"

# (b5) existing admin entry -> preserve it and re-associate the read-only policy
run_case b5 "${WORKSHOP_ENV[@]}" "PARTICIPANT_ROLE_NAME=WSParticipantRole" \
  "STUB_PARTICIPANT_CANDIDATES=$PARTICIPANT_CANDIDATE" \
  "ADDITIONAL_EKS_VIEWER_ROLE_NAME=admin" \
  "STUB_ADDITIONAL_VIEWER_ARN=$ADMIN_VIEWER_ARN" \
  "STUB_EKS_ACCESS_ENTRY_RC=0"
expect "exit 0" rc_is 0
expect "existing admin entry not recreated" hasnt "$LOG" "aws eks create-access-entry"
expect "existing admin policy reconciled" has "$LOG" "aws eks associate-access-policy --cluster-name physical-ai-evt-cosmos3 --principal-arn $ADMIN_VIEWER_ARN --policy-arn arn:aws:eks::aws:cluster-access-policy/AmazonEKSViewPolicy --access-scope type=cluster --region us-east-2"
expect "existing entry logged" has "$OUT" "Additional EKS viewer access entry already exists: $ADMIN_VIEWER_ARN"
finish "b5  existing admin viewer -> preserved and policy reconciled"

# (c2) presigned archive -> verified, relayed to participant S3, then extracted on node
run_case c2 "CENTRAL_WEIGHTS_URL=$PRESIGNED" "CENTRAL_WEIGHTS_SHA256=$ARCHIVE_SHA" "DATA_BUCKET=$DATA"
expect "exit 0" rc_is 0
expect "URL absent from stdout" hasnt "$OUT" "$PRESIGNED"
expect "presigned archive downloaded" has "$LOG" "curl --fail --location --silent --show-error"
expect "archive relayed to DATA_BUCKET" has "$LOG" "aws s3 cp /tmp/cosmos3-nano-${ARCHIVE_SHA_PREFIX}.tar s3://$DATA/cosmos3-assets/archives/cosmos3-nano.tar"
expect "staging Job uses one archive attempt" has "$LOG.stdin" "backoffLimit: 0"
expect "staging Job uses supported AL2023 base" has "$LOG.stdin" "image: public.ecr.aws/amazonlinux/amazonlinux:2023"
expect "staging Job installs required packages" has "$LOG.stdin" "dnf install -y --setopt=install_weak_deps=False awscli-2 tar"
expect "staging Job verifies required tools" has "$LOG.stdin" "for tool in aws sha256sum tar"
expect "failure path captures every staging pod" has "$SCRIPT" "kubectl logs \"\$pod\" -n default --all-containers=true --tail=200"
expect "staging Job reads archive" has "$LOG.stdin" "s3://$DATA/cosmos3-assets/archives/cosmos3-nano.tar"
expect "staging Job checks archive" has "$LOG.stdin" "$ARCHIVE_SHA  /models/cosmos3-nano.tar"
expect "staging Job validates Cosmos-1.0-Guardrail" has "$LOG.stdin" "models--nvidia--Cosmos-1.0-Guardrail/snapshots/cf03c0395fac8c4de386c0bdab12cc4fc8d66362/config.json"
expect "staging Job validates Cosmos-Guardrail1" has "$LOG.stdin" "models--nvidia--Cosmos-Guardrail1/snapshots/d6d4bfa899a71454a700907664f3e88f503950cf/video_content_safety_filter/safety_filter.pt"
expect "staging Job validates Qwen3Guard" has "$LOG.stdin" "models--Qwen--Qwen3Guard-Gen-0.6B/snapshots/fada3b2f655b89601929198343c94cd2f64d93cc/model.safetensors"
expect "staging Job installs guardrail HF cache" has "$LOG.stdin" "mv /models/hf-cache.next /models/hf-cache"
expect "staging Job pins Cosmos-1.0-Guardrail main ref" has "$LOG.stdin" "models--nvidia--Cosmos-1.0-Guardrail/refs/main"
expect "staging Job pins Cosmos-Guardrail1 main ref" has "$LOG.stdin" "models--nvidia--Cosmos-Guardrail1/refs/main"
expect "staging Job pins Qwen3Guard main ref" has "$LOG.stdin" "models--Qwen--Qwen3Guard-Gen-0.6B/refs/main"
expect "no HF secret lookup" hasnt "$LOG" "secretsmanager get-secret-value"
expect "no Kubernetes HF Secret" hasnt "$LOG" "kubectl create secret generic hf-token"
finish "c2  presigned archive -> verify, relay, extract without HF token"

# (c3) presigned archive without checksum -> fatal before Terraform or download
run_case c3 "CENTRAL_WEIGHTS_URL=$PRESIGNED" "DATA_BUCKET=$DATA"
expect "exit 1" rc_is 1
expect "checksum required" has "$ERR" "CENTRAL_WEIGHTS_SHA256 must be 64 hexadecimal characters"
expect "no terraform init" hasnt "$LOG" "terraform init"
expect "no download" hasnt "$LOG" "curl "
finish "c3  presigned archive without checksum -> exit 1"

# (c4) direct S3 and presigned URL together -> rejected as ambiguous
run_case c4 "CENTRAL_WEIGHTS_S3=$CENTRAL" "CENTRAL_WEIGHTS_URL=$PRESIGNED" \
  "CENTRAL_WEIGHTS_SHA256=$ARCHIVE_SHA" "DATA_BUCKET=$DATA"
expect "exit 1" rc_is 1
expect "one source required" has "$ERR" "set only one of CENTRAL_WEIGHTS_S3 or CENTRAL_WEIGHTS_URL"
expect "no terraform init" hasnt "$LOG" "terraform init"
finish "c4  both weight sources -> exit 1"

# (c5) downloaded bytes do not match expected checksum -> never uploaded
run_case c5 "CENTRAL_WEIGHTS_URL=$PRESIGNED" "CENTRAL_WEIGHTS_SHA256=$ARCHIVE_SHA" \
  "STUB_SHA256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" "DATA_BUCKET=$DATA"
expect "exit 1" rc_is 1
expect "checksum mismatch" has "$ERR" "Cosmos weights archive checksum mismatch"
expect "archive not uploaded" hasnt "$LOG" "aws s3 cp /tmp/cosmos3-nano-"
finish "c5  checksum mismatch -> exit 1 before participant upload"
# (c) relay requested with no bucket -> fatal before Terraform
run_case c "CENTRAL_WEIGHTS_S3=$CENTRAL"
expect "exit 1" rc_is 1
expect "ERROR on stderr" has "$ERR" "ERROR: a central weight source is set but there is no relay bucket"
expect "no terraform init" hasnt "$LOG" "terraform init"
finish "c   relay requested, no param, no DATA_BUCKET -> exit 1"

# (d) VPC_ID without node subnets
run_case d "VPC_ID=$VPC"
expect "exit 1" rc_is 1
expect "ERROR on stderr" has "$ERR" "VPC_ID and EKS_NODE_SUBNET_IDS must be set together"
expect "no describe-subnets" hasnt "$LOG" "describe-subnets"
finish "d   VPC_ID only -> exit 1"

# (d2) node subnets without VPC_ID
run_case d2 "EKS_NODE_SUBNET_IDS=$SUBNET_B"
expect "exit 1" rc_is 1
expect "ERROR on stderr" has "$ERR" "VPC_ID and EKS_NODE_SUBNET_IDS must be set together"
finish "d2  EKS_NODE_SUBNET_IDS only -> exit 1"

# (e) control plane in one AZ
run_case e "VPC_ID=$VPC" "EKS_NODE_SUBNET_IDS=$SUBNET_B" \
  "EKS_CONTROL_PLANE_SUBNET_IDS=$SUBNET_B,$SUBNET_B2" "DATA_BUCKET=$DATA"
expect "exit 1" rc_is 1
expect "ERROR on stderr" has "$ERR" "EKS control plane subnets span only us-east-2b; EKS needs two AZs"
expect "no terraform init" hasnt "$LOG" "terraform init"
finish "e   control-plane subnets in one AZ -> exit 1"

# (e2) control-plane subnet in an EKS-unsupported AZ ID
run_case e2 "VPC_ID=$VPC" "EKS_NODE_SUBNET_IDS=$SUBNET_B" \
  "EKS_CONTROL_PLANE_SUBNET_IDS=$SUBNET_B,$SUBNET_BAD_AZ" "DATA_BUCKET=$DATA"
expect "exit 1" rc_is 1
expect "names the AZ ID" has "$ERR" "(use1-az3), which EKS does not support"
expect "names EksControlPlaneSecondaryAz" has "$ERR" "EksControlPlaneSecondaryAz"
expect "says delete and recreate" has "$ERR" "Delete the stack and recreate it"
finish "e2  control-plane AZ ID use1-az3 -> exit 1"

# (f) GPU_AZ not among the node subnet AZs
run_case f "VPC_ID=$VPC" "EKS_NODE_SUBNET_IDS=$SUBNET_B" \
  "EKS_CONTROL_PLANE_SUBNET_IDS=$SUBNET_B,$SUBNET_CP" "GPU_AZ=us-east-2c" "DATA_BUCKET=$DATA"
expect "exit 1" rc_is 1
expect "ERROR on stderr" has "$ERR" "no node subnet in GPU_AZ=us-east-2c (node subnets are in us-east-2b)"
finish "f   GPU_AZ not among node subnet AZs -> exit 1"

# (g) full workshop env, as the template passes it (no ENABLE_IMAGE_BUILDS)
run_case g "${WORKSHOP_ENV[@]}"
G_TFV="$TFV"; G_LOG="$LOG"
expect "exit 0" rc_is 0
expect "vpc_id" tfvar "$TFV" vpc_id "\"$VPC\""
expect "private_subnet_ids" tfvar "$TFV" private_subnet_ids "[\"$SUBNET_B\"]"
expect "eks_control_plane_subnet_ids" tfvar "$TFV" eks_control_plane_subnet_ids "[\"$SUBNET_B\", \"$SUBNET_CP\"]"
expect "eks_cluster_api_ingress_security_group_ids" tfvar "$TFV" eks_cluster_api_ingress_security_group_ids "[\"$SG\"]"
expect "enable_image_builds = false" tfvar "$TFV" enable_image_builds false
expect "foundation grant = false" tfvar "$TFV" eks_pod_grant_foundation_datasets_bucket false
expect "extra buckets = [DATA_BUCKET]" tfvar "$TFV" eks_pod_extra_s3_buckets "[\"$DATA\"]"
expect "GPU AZ" tfvar "$TFV" eks_gpu_availability_zone '"us-east-2b"'
expect "tfvars comment kept literally" has "$TFV" 'even `terraform plan` fails'
expect "terraform plan never run" hasnt "$LOG" "terraform plan"
expect "publish before init" line_before "$LOG" "aws s3 cp terraform.tfvars" "terraform init"
expect "publish before apply" line_before "$LOG" "aws s3 cp terraform.tfvars" "terraform apply"
expect "exactly one publish" count_is "$LOG" "aws s3 cp terraform.tfvars" 1
expect "one describe-subnets call" count_is "$LOG" "aws ec2 describe-subnets" 1
expect "each subnet described once" describe_once "$SUBNET_B" "$SUBNET_CP"
expect "no SSM read" hasnt "$LOG" "ssm get-parameter"
expect "curl never called" hasnt "$LOG" "curl "
# NFR2: a re-run with the same env writes a byte-identical tfvars.
G_TFV_COPY="$WORK/g.tfvars"; cp "$G_TFV" "$G_TFV_COPY" 2>/dev/null
G_OK="$CASE_OK"; G_MSGS="$CASE_MSGS"; G_RC="$RC"
run_case g-rerun "${WORKSHOP_ENV[@]}"
CASE_OK="$G_OK"; CASE_MSGS="$G_MSGS"; RC="$G_RC"
expect "re-run tfvars byte-identical" cmp -s "$G_TFV_COPY" "$TFV"
finish "g   full workshop env -> network tfvars, publish once before init/apply"

# (g2) publish fails -> fatal, Terraform never initialised
run_case g2 "${WORKSHOP_ENV[@]}" "STUB_S3_CP_RC=1"
expect "exit 1" rc_is 1
expect "ERROR on stderr" has "$ERR" "ERROR: could not publish terraform.tfvars to s3://physical-ai-tfstate-$ACCOUNT"
expect "no terraform init" hasnt "$LOG" "terraform init"
expect "no terraform apply" hasnt "$LOG" "terraform apply"
finish "g2  tfvars publish fails -> exit 1 before terraform init"

# (h) no network vars (old template path) -> SSM fallback, preflight skipped
case_h() {  # $1 = label suffix
  run_case "h$2" "DATA_BUCKET=$DATA"
  expect "exit 0" rc_is 0
  expect "reached terraform apply" has "$LOG" "terraform apply"
  expect "vpc_id empty" tfvar "$TFV" vpc_id '""'
  expect "private_subnet_ids []" tfvar "$TFV" private_subnet_ids "[]"
  expect "eks_control_plane_subnet_ids []" tfvar "$TFV" eks_control_plane_subnet_ids "[]"
  expect "api SG list []" tfvar "$TFV" eks_cluster_api_ingress_security_group_ids "[]"
  expect "describe-subnets never called" hasnt "$LOG" "describe-subnets"
  finish "h   no network vars -> [] lists, no preflight ($1)"
}
case_h "$("$BASH_BIN" -c 'echo "$BASH_VERSION"') at $BASH_BIN" ""
# Same case under a bash >= 4.4, when one is installed.
NEW_BASH=""
for cand in "$(command -v bash 2>/dev/null)" /opt/homebrew/bin/bash /usr/local/bin/bash /usr/bin/bash; do
  [ -n "$cand" ] && [ -x "$cand" ] && [ "$cand" != /bin/bash ] || continue
  if "$cand" -c '[ "${BASH_VERSINFO[0]}" -gt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -ge 4 ]; }' 2>/dev/null; then
    NEW_BASH="$cand"; break
  fi
done
if [ -n "$NEW_BASH" ]; then
  BASH_BIN="$NEW_BASH"
  case_h "$("$BASH_BIN" -c 'echo "$BASH_VERSION"') at $BASH_BIN" "-new"
  BASH_BIN=/bin/bash
else
  SKIP=$((SKIP + 1)); echo "SKIP  h   no network vars (bash >= 4.4: none installed)"
fi

# (i) malformed SG ID -> named cause on STDERR, nothing written or published
run_case i "VPC_ID=$VPC" "EKS_NODE_SUBNET_IDS=$SUBNET_B" \
  "EKS_CONTROL_PLANE_SUBNET_IDS=$SUBNET_B,$SUBNET_CP" "EKS_API_INGRESS_SG_IDS=sg-bad!" \
  "GPU_AZ=us-east-2b" "DATA_BUCKET=$DATA"
expect "exit 1" rc_is 1
expect "stderr names the value" has "$ERR" "EKS_API_INGRESS_SG_IDS contains 'sg-bad!'"
expect "stdout does not" hasnt "$OUT" "EKS_API_INGRESS_SG_IDS contains"
expect "no terraform.tfvars" no_file "$TFV"
expect "no publish" hasnt "$LOG" "aws s3 cp"
expect "no terraform init" hasnt "$LOG" "terraform init"
expect "no terraform apply" hasnt "$LOG" "terraform apply"
finish "i   EKS_API_INGRESS_SG_IDS='sg-bad!' -> exit 1, cause on stderr"

# (j) ENABLE_IMAGE_BUILDS
run_case j-unset "DATA_BUCKET=$DATA"
expect "exit 0" rc_is 0
expect "enable_image_builds = false" tfvar "$TFV" enable_image_builds false
finish "j   ENABLE_IMAGE_BUILDS unset -> false"

run_case j-true "DATA_BUCKET=$DATA" "ENABLE_IMAGE_BUILDS=true"
expect "exit 0" rc_is 0
expect "enable_image_builds = true" tfvar "$TFV" enable_image_builds true
finish "j   ENABLE_IMAGE_BUILDS=true -> true"

run_case j-yes "${WORKSHOP_ENV[@]}" "ENABLE_IMAGE_BUILDS=yes"
expect "exit 1" rc_is 1
expect "ERROR on stderr" has "$ERR" "ERROR: ENABLE_IMAGE_BUILDS must be true or false, got 'yes'"
expect "no describe-subnets" hasnt "$LOG" "describe-subnets"
finish "j   ENABLE_IMAGE_BUILDS=yes -> exit 1 before describe-subnets"

# POD_GRANT_FOUNDATION_BUCKET (design-review round 3, NIT 3)
run_case pod-true "${WORKSHOP_ENV[@]}" "POD_GRANT_FOUNDATION_BUCKET=true"
expect "exit 0" rc_is 0
expect "foundation grant = true" tfvar "$TFV" eks_pod_grant_foundation_datasets_bucket true
expect "extra buckets still DATA_BUCKET" tfvar "$TFV" eks_pod_extra_s3_buckets "[\"$DATA\"]"
finish "j   POD_GRANT_FOUNDATION_BUCKET=true with DATA_BUCKET -> grant true"

run_case pod-maybe "${WORKSHOP_ENV[@]}" "POD_GRANT_FOUNDATION_BUCKET=maybe"
expect "exit 1" rc_is 1
expect "ERROR on stderr" has "$ERR" "ERROR: POD_GRANT_FOUNDATION_BUCKET must be empty, true or false, got 'maybe'"
expect "no describe-subnets" hasnt "$LOG" "describe-subnets"
finish "j   POD_GRANT_FOUNDATION_BUCKET=maybe -> exit 1 before describe-subnets"

run_case pod-false-empty "POD_GRANT_FOUNDATION_BUCKET=false"
expect "exit 1" rc_is 1
expect "ERROR on stderr" has "$ERR" "ERROR: pod S3 policy would grant no bucket"
expect "no publish" hasnt "$LOG" "aws s3 cp"
finish "j   POD_GRANT_FOUNDATION_BUCKET=false, no DATA_BUCKET -> exit 1 before publish"

# (k) describe-subnets is region-scoped (from case g)
LOG="$G_LOG"; RC=0; CASE_OK=1; CASE_MSGS=""; ERR="$WORK/case-g/stderr"
expect "describe-subnets has --region us-east-2" has_re "$LOG" '^aws ec2 describe-subnets .*--region us-east-2( |$)'
finish "k   describe-subnets called with --region us-east-2"

echo
echo "Summary: $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" -eq 0 ]
