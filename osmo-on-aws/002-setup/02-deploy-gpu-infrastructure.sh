#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Deploy GPU Infrastructure (GPU Operator + KAI Scheduler)

set -o errexit
set -o nounset
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=defaults.conf
source "$SCRIPT_DIR/defaults.conf"
# shellcheck source=lib/common.sh
source "$SCRIPT_DIR/lib/common.sh"

show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Deploy GPU infrastructure: NVIDIA GPU Operator and KAI Scheduler.

OPTIONS:
    -h, --help              Show this help message
    -t, --tf-dir DIR        Terraform directory (default: $DEFAULT_TF_DIR)
    --skip-if-no-gpus       Skip if no GPU nodes are detected
    --config-preview        Print configuration and exit

EXAMPLES:
    $(basename "$0")
    $(basename "$0") --skip-if-no-gpus
EOF
}

# Default values
tf_dir="$SCRIPT_DIR/$DEFAULT_TF_DIR"
skip_if_no_gpus=false
config_preview=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)          show_help; exit 0 ;;
    -t|--tf-dir)        tf_dir="$2"; shift 2 ;;
    --skip-if-no-gpus)  skip_if_no_gpus=true; shift ;;
    --config-preview)   config_preview=true; shift ;;
    *)                  fatal "Unknown option: $1" ;;
  esac
done

check_gpu_nodes() {
  local gpu_nodes
  gpu_nodes=$(kubectl get nodes -l "nvidia.com/gpu.present=true" -o name 2>/dev/null | wc -l || echo "0")

  if [[ "$gpu_nodes" -eq 0 ]]; then
    # Also check for GPU instance types
    gpu_nodes=$(kubectl get nodes -l "node.kubernetes.io/instance-type" -o json 2>/dev/null | \
      jq -r '.items[] | select(.metadata.labels["node.kubernetes.io/instance-type"] | test("^g[0-9]|^p[0-9]")) | .metadata.name' | wc -l || echo "0")
  fi

  echo "$gpu_nodes"
}

main() {
  section "Deploying NVIDIA GPU Operator"

  require_tools kubectl helm jq

  # Read Terraform outputs
  info "Reading Terraform outputs from: $tf_dir"
  local tf_outputs
  tf_outputs=$(read_terraform_outputs "$tf_dir")

  local cluster_name region deployment_mode
  cluster_name=$(tf_require "$tf_outputs" "cluster_name" "EKS cluster name")
  region=$(tf_require "$tf_outputs" "aws_region" "AWS region")
  deployment_mode=$(tf_get "$tf_outputs" "deployment_mode" "full")

  # Check deployment mode
  if ! should_deploy_gpu_operator "$deployment_mode"; then
    info "Deployment mode is '$deployment_mode' - skipping GPU Operator"
    exit 0
  fi

  # Print configuration
  section "Configuration"
  print_kv "Cluster" "$cluster_name"
  print_kv "Region" "$region"
  print_kv "Deployment Mode" "$deployment_mode"
  print_kv "GPU Operator Version" "$GPU_OPERATOR_VERSION"

  if [[ "$config_preview" == "true" ]]; then
    info "Config preview mode - exiting"
    exit 0
  fi

  # Connect to cluster
  connect_eks "$region" "$cluster_name"

  # Check for GPU nodes
  local gpu_node_count
  gpu_node_count=$(check_gpu_nodes)
  info "GPU nodes detected: $gpu_node_count"

  if [[ "$gpu_node_count" -eq 0 ]] && [[ "$skip_if_no_gpus" == "true" ]]; then
    warn "No GPU nodes detected and --skip-if-no-gpus specified"
    info "GPU Operator will be installed when GPU nodes are added"
    exit 0
  fi

  # Add NVIDIA Helm repo
  helm_repo_add nvidia "$NVIDIA_HELM_REPO"

  # Create namespace
  ensure_namespace "$GPU_OPERATOR_NAMESPACE"

  # Deploy GPU Operator
  section "Installing NVIDIA GPU Operator"

  helm_upgrade_install gpu-operator nvidia/gpu-operator \
    "$GPU_OPERATOR_NAMESPACE" \
    --version "$GPU_OPERATOR_VERSION" \
    -f "$SCRIPT_DIR/values/nvidia-gpu-operator.yaml" \
    --timeout "$HELM_TIMEOUT"

  # Wait for operator to be ready
  info "Waiting for GPU Operator to be ready..."
  wait_for_deployment gpu-operator "$GPU_OPERATOR_NAMESPACE"

  # Wait for GPU nodes to be labeled (if any exist)
  if [[ "$gpu_node_count" -gt 0 ]]; then
    info "Waiting for GPU feature discovery..."
    sleep 30  # Give GFD time to label nodes

    # Verify GPU discovery
    local labeled_nodes
    labeled_nodes=$(kubectl get nodes -l "nvidia.com/gpu.present=true" -o name 2>/dev/null | wc -l || echo "0")
    info "Nodes with GPU labels: $labeled_nodes"
  fi

  #----------------------------------------------------------------------------
  # Wait for nvidia.com/gpu before installing KAI. The GPU Operator needs ~10 min
  # to build/load the NVIDIA kernel modules, and KAI snapshots node capacity when
  # it starts: a scheduler that starts before the GPU is advertised reports
  # "Total allocatable resources are <... Gpus: 0>" and every GPU workflow pod
  # stays Pending with no scheduling events.
  #----------------------------------------------------------------------------
  if [[ "$gpu_node_count" -gt 0 ]]; then
    section "Waiting for nvidia.com/gpu to be allocatable"
    local gpu_wait="${GPU_ALLOCATABLE_TIMEOUT:-1200}" waited=0 alloc=0
    while [[ "$waited" -lt "$gpu_wait" ]]; do
      alloc=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' 2>/dev/null \
        | awk 'NF && $1 != "0"' | wc -l) || alloc=0
      [[ "$alloc" -gt 0 ]] && break
      sleep 20
      waited=$((waited + 20))
    done
    if [[ "$alloc" -gt 0 ]]; then
      pass "nvidia.com/gpu advertised by $alloc node(s) after ${waited}s"
    else
      warn "nvidia.com/gpu still not advertised after ${gpu_wait}s; check 'kubectl logs -n $GPU_OPERATOR_NAMESPACE -l app=nvidia-driver-daemonset'."
      warn "If the driver failed to build, bump driver.version in values/nvidia-gpu-operator.yaml."
      warn "Once GPUs appear, restart KAI so it re-reads capacity: kubectl rollout restart deploy -n \${KAI_SCHEDULER_NAMESPACE:-kai-scheduler}"
    fi
  fi

  #----------------------------------------------------------------------------
  # Deploy KAI Scheduler (GPU-aware scheduler for OSMO)
  # https://nvidia.github.io/OSMO/main/deployment_guide/install_backend/dependencies/dependencies.html
  #----------------------------------------------------------------------------
  section "Installing KAI Scheduler"

  local kai_namespace="${KAI_SCHEDULER_NAMESPACE:-kai-scheduler}"
  local kai_version="${KAI_SCHEDULER_VERSION:-v0.12.4}"

  ensure_namespace "$kai_namespace"

  info "Deploying KAI Scheduler $kai_version..."
  helm upgrade --install kai-scheduler \
    oci://ghcr.io/nvidia/kai-scheduler/kai-scheduler \
    --version "$kai_version" \
    --namespace "$kai_namespace" \
    -f "$SCRIPT_DIR/values/kai-scheduler.yaml" \
    --timeout "$HELM_TIMEOUT"

  info "Waiting for KAI Scheduler to be ready..."
  if kubectl get deployment kai-scheduler -n "$kai_namespace" &>/dev/null; then
    kubectl rollout status deployment/kai-scheduler -n "$kai_namespace" --timeout="${KUBECTL_TIMEOUT:-300s}" 2>/dev/null || warn "KAI Scheduler rollout not ready yet (non-fatal)"
  else
    warn "Deployment kai-scheduler not found in $kai_namespace (chart may use a different name); continuing"
  fi
  pass "KAI Scheduler deployed"

  kubectl get pods -n "$kai_namespace" --no-headers 2>/dev/null | head -5 || true

  section "GPU Infrastructure Deployment Complete"

  # Show GPU node status
  info "GPU Node Status:"
  kubectl get nodes -l "nvidia.com/gpu.present=true" \
    -o custom-columns="NAME:.metadata.name,GPU:.metadata.labels.nvidia\.com/gpu\.product,STATUS:.status.conditions[-1].type" \
    2>/dev/null || info "No GPU nodes currently available"

  echo
  info "Next: Run 03-deploy-keycloak.sh"
}

main "$@"
