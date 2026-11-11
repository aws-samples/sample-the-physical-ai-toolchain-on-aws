#!/usr/bin/env bash
# Cosmos on AWS: start or stop the private Cosmos 3 server on EKS.
#
#   bash cosmos-service.sh start    install the Helm release and wait until it is ready
#   bash cosmos-service.sh stop     uninstall it and wait until the GPU node is removed
#   bash cosmos-service.sh status   show the release, Pod and GPU node state
#
# Reads the values written by cosmos-preflight.sh (~/.physical-ai/cosmos-lab.env), so
# it works from any terminal once the pre-flight has passed. `stop` also works without
# that file, using the default release name.
#
# `start` begins GPU billing: the GPU node group scales from 0 to 1. Billing ends only
# when `stop` has removed the release and Cluster Autoscaler has removed the node.
set -uo pipefail

ACTION="${1:-}"
ENV_FILE="$HOME/.physical-ai/cosmos-lab.env"
NS="default"
GPU_SELECTOR="node.kubernetes.io/purpose=gpu"
START_TIMEOUT_MIN=30
SCALE_DOWN_TIMEOUT_MIN=25

export PATH="$HOME/.local/bin:$PATH"
# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && source "$ENV_FILE"
HELM_RELEASE="${HELM_RELEASE:-cosmos3-g1-server}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-$HELM_RELEASE}"
SERVICE_NAME="${SERVICE_NAME:-$HELM_RELEASE}"

ts() { date +%H:%M:%S; }
info() { echo "[$(ts)] $*"; }
pass() { echo "[$(ts)] PASS  $*"; }
fail() { echo "[$(ts)] FAIL  $*"; exit 1; }

gpu_nodes() {
  kubectl get nodes -l "$GPU_SELECTOR" --no-headers 2>/dev/null | grep -c . || true
}
pod_state() {
  kubectl get pod -n "$NS" -l "app=$DEPLOYMENT_NAME" --no-headers \
    -o custom-columns=S:.status.phase,R:.status.containerStatuses[0].ready 2>/dev/null \
    | head -1 | awk '{print $1 ($2=="true" ? ", ready" : "")}'
}
release_exists() { helm status "$HELM_RELEASE" -n "$NS" >/dev/null 2>&1; }
leftovers() {
  release_exists \
    || kubectl get deployment "$DEPLOYMENT_NAME" -n "$NS" >/dev/null 2>&1 \
    || kubectl get service "$SERVICE_NAME" -n "$NS" >/dev/null 2>&1 \
    || kubectl get pod -n "$NS" -l "app=$DEPLOYMENT_NAME" -o name 2>/dev/null | grep -q .
}

start() {
  [ -f "$ENV_FILE" ] || fail "run the pre-flight first: bash cosmos-on-aws/cosmos-preflight.sh"
  local v
  for v in AWS_REGION HELM_CHART COSMOS_RUNTIME_IMAGE_URI COSMOS_ARCHIVE_S3_URI COSMOS_ARCHIVE_SHA256; do
    [ -n "${!v:-}" ] || fail "$v is missing from $ENV_FILE; run the pre-flight again"
  done
  command -v helm >/dev/null 2>&1 || fail "helm not found; run the pre-flight again"

  info "Installing $HELM_RELEASE. GPU billing starts now; allow about 20 minutes."
  local log="$HOME/.physical-ai/cosmos-start.log"
  helm upgrade --install "$HELM_RELEASE" "$HELM_CHART" \
    --namespace "$NS" \
    --set-string aws.region="$AWS_REGION" \
    --set-string image.ref="$COSMOS_RUNTIME_IMAGE_URI" \
    --set-string model.archiveS3Uri="$COSMOS_ARCHIVE_S3_URI" \
    --set-string model.archiveSha256="$COSMOS_ARCHIVE_SHA256" \
    --wait --timeout "${START_TIMEOUT_MIN}m" --atomic --history-max 3 > "$log" 2>&1 &
  local helm_pid=$! elapsed=0
  while kill -0 "$helm_pid" 2>/dev/null; do
    sleep 60; elapsed=$((elapsed + 1))
    kill -0 "$helm_pid" 2>/dev/null || break
    local pod; pod=$(pod_state || true)
    info "waiting ${elapsed} min: GPU nodes $(gpu_nodes), Cosmos Pod ${pod:-not scheduled yet}"
  done
  if ! wait "$helm_pid"; then
    tail -n 15 "$log" | sed 's/^/        | /'
    fail "Helm did not finish; --atomic removed the release, so nothing is running. Log: $log"
  fi
  pass "Helm release $HELM_RELEASE deployed"

  kubectl get deployment "$DEPLOYMENT_NAME" -n "$NS" >/dev/null 2>&1 \
    || fail "Deployment $DEPLOYMENT_NAME not found"
  local ready
  ready=$(kubectl get deployment "$DEPLOYMENT_NAME" -n "$NS" -o jsonpath='{.status.readyReplicas}')
  [ "${ready:-0}" -ge 1 ] || fail "Deployment $DEPLOYMENT_NAME has no ready replica"
  pass "Deployment $DEPLOYMENT_NAME is 1/1 Ready"

  local svc_type
  svc_type=$(kubectl get service "$SERVICE_NAME" -n "$NS" -o jsonpath='{.spec.type}' 2>/dev/null)
  if [ "$svc_type" != "ClusterIP" ]; then
    info "Service type is '${svc_type:-missing}', not ClusterIP; removing the release"
    helm uninstall "$HELM_RELEASE" -n "$NS" --wait --timeout 5m >/dev/null 2>&1
    fail "expected a private ClusterIP Service"
  fi
  pass "Service $SERVICE_NAME is private (ClusterIP)"
  echo
  echo "PASS - Cosmos server is ready. Start the port-forward next."
}

stop() {
  if release_exists; then
    info "Uninstalling Helm release $HELM_RELEASE"
    helm uninstall "$HELM_RELEASE" -n "$NS" --wait --timeout 5m >/dev/null 2>&1 \
      || fail "helm uninstall failed; contact the facilitator immediately"
    pass "Helm release removed"
  else
    info "No Helm release found; removing any leftover named resources"
    kubectl delete deployment "$DEPLOYMENT_NAME" -n "$NS" --ignore-not-found --wait=false >/dev/null \
      && kubectl delete service "$SERVICE_NAME" -n "$NS" --ignore-not-found >/dev/null \
      && kubectl delete configmap "$DEPLOYMENT_NAME-server" -n "$NS" --ignore-not-found >/dev/null \
      || fail "could not remove leftover resources; contact the facilitator immediately"
  fi

  if kubectl get pod -n "$NS" -l "app=$DEPLOYMENT_NAME" -o name 2>/dev/null | grep -q .; then
    info "Waiting for the Cosmos Pod to terminate"
    kubectl wait --for=delete pod -n "$NS" -l "app=$DEPLOYMENT_NAME" --timeout=5m >/dev/null 2>&1 \
      || fail "the Cosmos Pod is still running after 5 minutes; contact the facilitator"
  fi
  leftovers && fail "Cosmos resources still exist; contact the facilitator immediately"
  pass "Cosmos release, Deployment, Service and Pod are gone"

  info "Waiting for Cluster Autoscaler to remove the GPU node (about 14 minutes)"
  local n elapsed=0
  while :; do
    n=$(gpu_nodes)
    if [ "$n" -eq 0 ]; then break; fi
    if [ "$elapsed" -ge "$SCALE_DOWN_TIMEOUT_MIN" ]; then
      fail "$n GPU node(s) still running after $SCALE_DOWN_TIMEOUT_MIN minutes; contact the facilitator"
    fi
    info "waiting ${elapsed} min: GPU nodes $n"
    sleep 60; elapsed=$((elapsed + 1))
  done
  pass "no GPU node is running"
  echo
  echo "PASS - Cosmos is stopped and GPU billing has ended."
}

status() {
  if release_exists; then info "Helm release $HELM_RELEASE: deployed"; else info "Helm release $HELM_RELEASE: not installed"; fi
  local pod; pod=$(pod_state || true)
  info "Cosmos Pod: ${pod:-none}"
  info "GPU nodes: $(gpu_nodes)"
}

command -v kubectl >/dev/null 2>&1 || fail "kubectl not found"
case "$ACTION" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: bash cosmos-service.sh start|stop|status"; exit 2 ;;
esac
