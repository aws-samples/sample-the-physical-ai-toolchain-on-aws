#!/usr/bin/env bash
# Cosmos on AWS: pre-flight checks before installing the Cosmos 3 Helm release.
#
# Used by the workshop's "Generate a world video" page. It checks that:
#   - the staged source and edge-control videos are in the participant data bucket
#   - the workstation can reach the EKS cluster
#   - Helm is available and the private Cosmos chart renders and validates
#
# Every check runs and prints one PASS, FAIL or SKIP line. A check is skipped only when
# a check it depends on failed. Command output goes to a log file; a FAIL line is
# followed by that check's last log lines. A summary at the end lists what failed.
# When every check passes, the values later lab steps need are written to an env file:
#
#   bash ~/sample-aws-physical-ai-toolchain/cosmos-on-aws/cosmos-preflight.sh \
#     && source ~/.physical-ai/cosmos-lab.env
#
# Read-only for AWS and Kubernetes: no Kubernetes resource is created. The only local
# changes are the kubeconfig entry, a user-local Helm install when Helm is missing,
# and a fast-forward of the pre-cloned toolkit to origin/development.
set -uo pipefail

STACK_NAME="${STACK_NAME:-physical-ai-toolchain}"
REGION="${AWS_REGION:-$(aws configure get region 2>/dev/null || true)}"
REGION="${REGION:-us-east-2}"
# This script lives in <toolkit>/cosmos-on-aws/, so the toolkit is one level up.
TOOLKIT_DIR="${TOOLKIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Check 7 fast-forwards the toolkit, which can rewrite this file while bash is still
# reading it. Run from a private copy so the update cannot change the running script.
if [ -z "${COSMOS_PREFLIGHT_COPY:-}" ]; then
  mkdir -p "$HOME/.physical-ai"
  cp "${BASH_SOURCE[0]}" "$HOME/.physical-ai/cosmos-preflight.run.sh"
  COSMOS_PREFLIGHT_COPY=1 TOOLKIT_DIR="$TOOLKIT_DIR" \
    exec bash "$HOME/.physical-ai/cosmos-preflight.run.sh" "$@"
fi
HELM_VERSION="v3.17.3"
STATE_DIR="$HOME/.physical-ai"
LOG="$STATE_DIR/cosmos-preflight.log"
ENV_FILE="$STATE_DIR/cosmos-lab.env"
TOTAL=9
STEP=0
FAILED=()
SKIPPED=()

mkdir -p "$STATE_DIR"
: > "$LOG"
rm -f "$ENV_FILE"

export PATH="$HOME/.local/bin:$PATH"

# Results of each check, used to skip dependent checks: ok_<name>=1 when it passed.
ok_tools=0 ok_outputs=0 ok_eks=0 ok_helm=0 ok_toolkit=0

step() {
  STEP=$((STEP + 1))
  STEP_NAME="$1"
  STEP_LOG_START=$(wc -l < "$LOG")
  echo "===== [$STEP/$TOTAL] $STEP_NAME" >> "$LOG"
  printf '[%d/%d] %-44s ' "$STEP" "$TOTAL" "$STEP_NAME"
}
pass() { echo "PASS  $*"; return 0; }
fail() {
  echo "FAIL  $*"
  local lines
  lines=$(tail -n +"$((STEP_LOG_START + 2))" "$LOG" | grep -v '^[[:space:]]*$' | tail -n 8)
  if [ -n "$lines" ]; then
    printf '%s\n' "$lines" | sed 's/^/        | /'
  fi
  FAILED+=("[$STEP/$TOTAL] $STEP_NAME: $*")
  return 1
}
skip() {
  local n="${*#, }"; n="${n# }"
  echo "SKIP  needs $n"
  SKIPPED+=("[$STEP/$TOTAL] $STEP_NAME")
  return 1
}
# Run a command with its output appended to the log.
quiet() { echo "\$ $*" >> "$LOG"; "$@" >> "$LOG" 2>&1; }

stack_output() {
  aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue | [0]" --output text 2>> "$LOG"
}

echo "Cosmos pre-flight checks (stack $STACK_NAME, Region $REGION)"
echo

# 1. Tools ------------------------------------------------------------------------
check_tools() {
  step "Workstation tools"
  local c missing=""
  for c in aws python3 curl kubectl git sha256sum; do
    command -v "$c" >/dev/null 2>&1 || missing="$missing $c"
  done
  [ -z "$missing" ] || { fail "missing command(s):$missing"; return 1; }
  pass "aws, python3, curl, kubectl, git"
}
check_tools && ok_tools=1

# 2. Stack outputs ----------------------------------------------------------------
CLUSTER="" DATA_BUCKET="" COSMOS_SOURCE_VIDEO_S3_URI="" COSMOS_EDGE_VIDEO_S3_URI=""
COSMOS_RUNTIME_IMAGE_URI="" COSMOS_ARCHIVE_S3_URI="" COSMOS_ARCHIVE_SHA256=""
check_outputs() {
  step "CloudFormation stack outputs"
  command -v aws >/dev/null 2>&1 || { skip " aws CLI (check 1)"; return 1; }
  CLUSTER=$(stack_output EksClusterName)
  DATA_BUCKET=$(stack_output DataBucketName)
  COSMOS_SOURCE_VIDEO_S3_URI=$(stack_output CosmosSourceVideoS3Uri)
  COSMOS_EDGE_VIDEO_S3_URI=$(stack_output CosmosEdgeControlVideoS3Uri)
  COSMOS_RUNTIME_IMAGE_URI=$(stack_output CosmosRuntimeImageUri)
  COSMOS_ARCHIVE_S3_URI=$(stack_output Cosmos3ArchiveS3Uri)
  COSMOS_ARCHIVE_SHA256=$(stack_output Cosmos3ArchiveSha256)
  local pair value missing=""
  for pair in "EksClusterName=$CLUSTER" "DataBucketName=$DATA_BUCKET" \
    "CosmosRuntimeImageUri=$COSMOS_RUNTIME_IMAGE_URI" \
    "Cosmos3ArchiveS3Uri=$COSMOS_ARCHIVE_S3_URI" \
    "Cosmos3ArchiveSha256=$COSMOS_ARCHIVE_SHA256"; do
    value="${pair#*=}"
    if [ -z "$value" ] || [ "$value" = "None" ]; then missing="$missing ${pair%%=*}"; fi
  done
  [ -z "$missing" ] || { fail "missing stack output(s):$missing"; return 1; }
  # Older stacks have no video outputs; they use the same participant-bucket keys.
  case "$COSMOS_SOURCE_VIDEO_S3_URI" in ''|None)
    COSMOS_SOURCE_VIDEO_S3_URI="s3://$DATA_BUCKET/cosmos3-inputs/cosmos3-g1-edge-source.mp4" ;; esac
  case "$COSMOS_EDGE_VIDEO_S3_URI" in ''|None)
    COSMOS_EDGE_VIDEO_S3_URI="s3://$DATA_BUCKET/cosmos3-inputs/cosmos3-g1-edge-control.mp4" ;; esac
  pass "cluster $CLUSTER, bucket $DATA_BUCKET"
}
check_outputs && ok_outputs=1

# 3-4. Staged input videos --------------------------------------------------------
# Download a video from the URL in stack parameter $1, check SHA-256 $2, upload to $3.
restage_video() {
  local param="$1" sha="$2" key="$3" url tmp
  url=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query "Stacks[0].Parameters[?ParameterKey=='$param'].ParameterValue | [0]" \
    --output text 2>> "$LOG") || return 1
  case "$url" in https://*) ;; *) echo "stack parameter $param is empty" >> "$LOG"; return 1 ;; esac
  tmp=$(mktemp) || return 1
  echo "re-staging $key from $param" >> "$LOG"
  quiet curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors --connect-timeout 30 \
    -o "$tmp" "$url" || { rm -f "$tmp"; return 1; }
  printf '%s  %s\n' "$sha" "$tmp" | quiet sha256sum -c - || { rm -f "$tmp"; return 1; }
  quiet aws s3 cp "$tmp" "s3://$DATA_BUCKET/$key" --region "$REGION" --sse AES256 --only-show-errors \
    || { rm -f "$tmp"; return 1; }
  rm -f "$tmp"
}

check_video() {
  local label="$1" uri="$2" param="$3" sha="$4" key bytes
  step "$label"
  [ "$ok_outputs" = 1 ] || { skip " stack outputs (check 2)"; return 1; }
  case "$uri" in
    "s3://$DATA_BUCKET/"*) ;;
    *) fail "$uri is outside the participant data bucket"; return 1 ;;
  esac
  key="${uri#s3://$DATA_BUCKET/}"
  if ! bytes=$(aws s3api head-object --bucket "$DATA_BUCKET" --key "$key" --region "$REGION" \
      --query ContentLength --output text 2>> "$LOG"); then
    # The workstation bootstrap stages these videos but treats a failed download as a
    # warning. Repair it here from the same stack parameter and checksum.
    restage_video "$param" "$sha" "$key" \
      || { fail "not found: $uri, and it could not be staged again"; return 1; }
    bytes=$(aws s3api head-object --bucket "$DATA_BUCKET" --key "$key" --region "$REGION" \
      --query ContentLength --output text 2>> "$LOG") \
      || { fail "staged again but still not readable: $uri"; return 1; }
    case "$bytes" in ''|*[!0-9]*|0) fail "empty or unreadable: $uri"; return 1 ;; esac
    pass "$key ($(( bytes / 1024 )) KiB, staged again)"; return 0
  fi
  case "$bytes" in ''|*[!0-9]*|0) fail "empty or unreadable: $uri"; return 1 ;; esac
  pass "$key ($(( bytes / 1024 )) KiB)"
}
# Checksums of the hosted workshop videos; they match the workstation bootstrap.
check_video "Source video in S3" "$COSMOS_SOURCE_VIDEO_S3_URI" CosmosSourceVideoUrl \
  ba61f16179fd4f9e889700452769a8d9efbee2400b04cdbbb787d046e9a18d7e
check_video "Edge-control video in S3" "$COSMOS_EDGE_VIDEO_S3_URI" CosmosEdgeControlVideoUrl \
  bee6ca1693cbe7a9d38456ad1597403b8cbad2191b0ce500625791d48344599c

# 5. EKS access -------------------------------------------------------------------
check_eks() {
  step "EKS cluster access"
  [ "$ok_outputs" = 1 ] || { skip " stack outputs (check 2)"; return 1; }
  command -v kubectl >/dev/null 2>&1 || { skip " kubectl (check 1)"; return 1; }
  quiet aws eks update-kubeconfig --name "$CLUSTER" --region "$REGION" \
    || { fail "could not write kubeconfig for $CLUSTER"; return 1; }
  local nodes ready
  nodes=$(kubectl get nodes --no-headers 2>> "$LOG") \
    || { fail "kubectl cannot reach $CLUSTER"; return 1; }
  ready=$(printf '%s\n' "$nodes" | awk '$2 == "Ready"' | grep -c . || true)
  [ "$ready" -ge 1 ] || { fail "no Ready node in $CLUSTER"; return 1; }
  pass "$ready node(s) Ready; GPU node starts on demand"
}
check_eks && ok_eks=1

# 6. Helm -------------------------------------------------------------------------
check_helm() {
  step "Helm"
  if command -v helm >/dev/null 2>&1; then
    pass "$(helm version --short 2>/dev/null)"; return 0
  fi
  local arch archive cache
  arch="$(dpkg --print-architecture 2>> "$LOG")" || { fail "dpkg not found; this check runs on the Ubuntu workstation"; return 1; }
  archive="helm-$HELM_VERSION-linux-$arch.tar.gz"
  cache="$HOME/.cache/physical-ai-toolchain/helm"
  mkdir -p "$HOME/.local/bin" "$cache"
  quiet curl -fsSL -o "$cache/$archive" "https://get.helm.sh/$archive" \
    || { fail "Helm download failed"; return 1; }
  quiet curl -fsSL -o "$cache/$archive.sha256sum" "https://get.helm.sh/$archive.sha256sum" \
    || { fail "Helm checksum download failed"; return 1; }
  (cd "$cache" && quiet sha256sum -c "$archive.sha256sum") \
    || { fail "Helm checksum mismatch"; return 1; }
  quiet tar -xzf "$cache/$archive" -C "$cache" \
    || { fail "Helm archive could not be unpacked"; return 1; }
  cp "$cache/linux-$arch/helm" "$HOME/.local/bin/helm" && chmod +x "$HOME/.local/bin/helm" \
    || { fail "Helm could not be installed in ~/.local/bin"; return 1; }
  pass "installed $HELM_VERSION in ~/.local/bin"
}
check_helm && ok_helm=1

# 7. Toolkit ----------------------------------------------------------------------
check_toolkit() {
  step "Toolkit on origin/development"
  [ -d "$TOOLKIT_DIR/.git" ] || { fail "toolkit not found at $TOOLKIT_DIR"; return 1; }
  [ -z "$(git -C "$TOOLKIT_DIR" status --porcelain --untracked-files=no)" ] \
    || { fail "the toolkit has tracked local changes"; return 1; }
  quiet git -C "$TOOLKIT_DIR" fetch origin development || { fail "git fetch failed"; return 1; }
  if git -C "$TOOLKIT_DIR" show-ref --verify --quiet refs/heads/development; then
    quiet git -C "$TOOLKIT_DIR" switch development || { fail "could not switch to development"; return 1; }
  else
    quiet git -C "$TOOLKIT_DIR" switch --track -c development origin/development \
      || { fail "could not create development"; return 1; }
  fi
  quiet git -C "$TOOLKIT_DIR" pull --ff-only origin development \
    || { fail "could not fast-forward development"; return 1; }
  pass "$(git -C "$TOOLKIT_DIR" rev-parse --short HEAD)"
}
check_toolkit && ok_toolkit=1

HELM_RELEASE="cosmos3-g1-server"
HELM_CHART="$TOOLKIT_DIR/cosmos-on-aws/charts/cosmos3"
DEPLOYMENT_NAME="$HELM_RELEASE"
SERVICE_NAME="$HELM_RELEASE"

# 8. No earlier release -----------------------------------------------------------
check_no_release() {
  step "No earlier Cosmos release"
  local needs=""
  [ "$ok_eks" = 1 ] || needs="$needs, EKS access (check 5)"
  [ "$ok_helm" = 1 ] || needs="$needs, Helm (check 6)"
  [ -z "$needs" ] || { skip "$needs"; return 1; }
  if helm status "$HELM_RELEASE" --namespace default >/dev/null 2>&1 \
    || kubectl get deployment "$DEPLOYMENT_NAME" --namespace default >/dev/null 2>&1 \
    || kubectl get service "$SERVICE_NAME" --namespace default >/dev/null 2>&1; then
    fail "$HELM_RELEASE already exists; uninstall it (Step 7) first"; return 1
  fi
  pass "nothing named $HELM_RELEASE in namespace default"
}
check_no_release

# 9. Chart validation -------------------------------------------------------------
check_chart() {
  step "Cosmos Helm chart (lint and dry run)"
  local needs=""
  [ "$ok_outputs" = 1 ] || needs="$needs, stack outputs (check 2)"
  [ "$ok_eks" = 1 ] || needs="$needs, EKS access (check 5)"
  [ "$ok_helm" = 1 ] || needs="$needs, Helm (check 6)"
  [ "$ok_toolkit" = 1 ] || needs="$needs, toolkit (check 7)"
  [ -z "$needs" ] || { skip "$needs"; return 1; }
  local set_args rendered dry kind lint
  set_args=(--set-string "aws.region=$REGION"
    --set-string "image.ref=$COSMOS_RUNTIME_IMAGE_URI"
    --set-string "model.archiveS3Uri=$COSMOS_ARCHIVE_S3_URI"
    --set-string "model.archiveSha256=$COSMOS_ARCHIVE_SHA256")
  lint=$(helm lint "$HELM_CHART" "${set_args[@]}" 2>&1); local rc=$?
  echo "$lint" >> "$LOG"
  [ "$rc" -eq 0 ] || { fail "helm lint reported errors"; return 1; }
  printf '%s\n' "$lint" | grep -q "0 chart(s) failed" \
    || { fail "helm lint did not report 0 chart(s) failed"; return 1; }
  rendered=$(helm template "$HELM_RELEASE" "$HELM_CHART" --namespace default "${set_args[@]}" 2>> "$LOG") \
    || { fail "helm template failed"; return 1; }
  dry=$(printf '%s\n' "$rendered" | kubectl apply --dry-run=client -f - 2>> "$LOG") \
    || { fail "kubectl dry run rejected the chart"; return 1; }
  echo "$dry" >> "$LOG"
  for kind in configmap deployment.apps service; do
    printf '%s\n' "$dry" | grep -q "^$kind/" \
      || { fail "dry run did not include a $kind"; return 1; }
  done
  pass "ConfigMap, Deployment, Service; nothing created"
}
check_chart

# Summary --------------------------------------------------------------------------
echo
if [ "${#FAILED[@]}" -gt 0 ] || [ "${#SKIPPED[@]}" -gt 0 ]; then
  passed=$((TOTAL - ${#FAILED[@]} - ${#SKIPPED[@]}))
  echo "STOP - $passed of $TOTAL checks passed, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped."
  for f in "${FAILED[@]}"; do echo "  FAIL $f"; done
  for s in "${SKIPPED[@]}"; do echo "  SKIP $s"; done
  echo "Fix the failed checks, then run this script again. Full log: $LOG"
  exit 1
fi

{
  printf 'export %s=%q\n' \
    AWS_REGION "$REGION" \
    PATH "$PATH" \
    CLUSTER "$CLUSTER" \
    DATA_BUCKET "$DATA_BUCKET" \
    COSMOS_SOURCE_VIDEO_S3_URI "$COSMOS_SOURCE_VIDEO_S3_URI" \
    COSMOS_EDGE_VIDEO_S3_URI "$COSMOS_EDGE_VIDEO_S3_URI" \
    COSMOS_RUNTIME_IMAGE_URI "$COSMOS_RUNTIME_IMAGE_URI" \
    COSMOS_ARCHIVE_S3_URI "$COSMOS_ARCHIVE_S3_URI" \
    COSMOS_ARCHIVE_SHA256 "$COSMOS_ARCHIVE_SHA256" \
    HELM_RELEASE "$HELM_RELEASE" \
    HELM_CHART "$HELM_CHART" \
    DEPLOYMENT_NAME "$DEPLOYMENT_NAME" \
    SERVICE_NAME "$SERVICE_NAME"
} > "$ENV_FILE"

echo "PASS - all $TOTAL checks passed. Cosmos is ready to install."
echo "Lab values saved to $ENV_FILE"
