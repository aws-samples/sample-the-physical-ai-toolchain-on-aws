# =============================================================================
# EKS: COSMOS 3 GENERATION CLUSTER (dedicated, minimal — not shared with OSMO)
# =============================================================================
#
# A standalone EKS cluster scoped to running the Cosmos3-Super vLLM-Omni server
# as a Kubernetes Job. Two node groups:
#   - system: small on-demand nodes for CoreDNS/addons (always on)
#   - gpu:    p5.48xlarge / p5en.48xlarge, desired_size=0 by default — scale to
#             1 only when submitting a generation Job, same "pay only while
#             generating" model as the EC2 path.
#
# GPU node group targets an EC2 Capacity Block via a launch template, the same
# mechanism used in ec2.tf. Purchase the block separately (see Step 1 of
# ec2-deployment-guide.md) — Capacity Block purchase is not Terraform-managed
# because it activates asynchronously outside a single `apply` lifecycle.
#
# Toggle with var.enable_eks_cluster (default false) so `terraform apply` is
# safe to run without standing up a cluster (control plane bills ~$0.10/hr
# once created, regardless of node count).

variable "enable_eks_cluster" {
  description = "Deploy the dedicated Cosmos 3 EKS cluster. Cluster control plane bills continuously once created."
  type        = bool
  default     = false
}

variable "eks_cluster_name" {
  description = "Name for the Cosmos 3 EKS cluster"
  type        = string
  default     = "cosmos3"
}

variable "eks_kubernetes_version" {
  description = "Kubernetes version for the EKS cluster"
  type        = string
  default     = "1.31"
}

variable "eks_admin_principal_arns" {
  description = "Additional IAM principal ARNs granted EKS cluster admin access (beyond the applying identity)"
  type        = list(string)
  default     = []
}

variable "eks_view_principal_arns" {
  description = "IAM principal ARNs granted read-only (AmazonEKSViewPolicy) access to the cluster. Intended for workshop participant roles so the EKS console's Resources tab can show Jobs, pods, and logs. Read-only on purpose: workloads are submitted with kubectl against the manifests in this repository, not from the console."
  type        = list(string)
  default     = []
}

variable "eks_pod_extra_s3_buckets" {
  description = "S3 bucket NAMES (not ARNs) the Cosmos3 generation pod may read and write, in addition to the Foundation datasets bucket when eks_pod_grant_foundation_datasets_bucket is true. Set this to the per-participant data bucket created by the workshop CloudFormation stack (physical-ai-data-<account>), which does not follow this module's <project>-<environment>-datasets-<account> naming and would otherwise be unreachable from the pod."
  type        = list(string)
  default     = []
}

variable "eks_system_node_instance_types" {
  description = "Instance types for the always-on system node group"
  type        = list(string)
  default     = ["t3.medium"]
}

variable "eks_gpu_node_instance_type" {
  description = "Instance type for the GPU node group (must match the Capacity Block's reserved type)"
  type        = string
  default     = "p5.48xlarge"
}

variable "eks_gpu_node_desired_size" {
  description = "Initial desired GPU nodes at node-group creation. After that, Cluster Autoscaler manages this automatically (scale up on a pending GPU pod, scale down ~10 min after the pod completes) — see the cluster-autoscaler Helm release below. Leave at 0; only used for the very first apply."
  type        = number
  default     = 0
}

variable "eks_nvidia_device_plugin_version" {
  description = "Helm chart version for the NVIDIA device plugin, which advertises nvidia.com/gpu as a schedulable resource. Without it GPU pods stay Pending on healthy GPU nodes."
  type        = string
  default     = "0.17.1"
}

variable "eks_cluster_autoscaler_version" {
  description = "Helm chart version for cluster-autoscaler (defaults to latest tested against eks_kubernetes_version 1.31)"
  type        = string
  default     = "9.46.6"
}

variable "eks_gpu_node_volume_size_gb" {
  description = "Root EBS volume size for GPU nodes (container image + model weight cache)"
  type        = number
  default     = 500
}

variable "eks_gpu_capacity_reservation_id" {
  description = "ID of an ACTIVE EC2 Capacity Block for the GPU node group to target. Leave empty to launch on-demand (will likely fail — P5 capacity is scarce)."
  type        = string
  default     = ""
}

variable "eks_gpu_availability_zone" {
  description = "AZ of the Capacity Block. The GPU node group's subnet is pinned to this AZ — EKS node groups otherwise span all private subnets/AZs, which fails Capacity Block launches with an AZ mismatch (the reservation is tied to one specific AZ)."
  type        = string
  default     = ""
}

variable "eks_control_plane_subnet_ids" {
  description = "Subnets for the EKS control plane ENIs only (must span >= 2 AZs, must be in vpc_id). Empty (default) uses the node subnets, as before. Lets node groups stay in one AZ (a capacity reservation's) while the control plane meets EKS's two-AZ rule."
  type        = list(string)
  default     = []
  validation {
    condition     = alltrue([for s in var.eks_control_plane_subnet_ids : can(regex("^subnet-[0-9a-f]{8,17}$", s))])
    error_message = "eks_control_plane_subnet_ids must contain only subnet IDs."
  }
}

variable "eks_cluster_api_ingress_security_group_ids" {
  description = "Security groups allowed to reach the cluster API on 443 through the private endpoint (for example a workstation in the same VPC, which resolves the endpoint to private IPs). Positional keys - append only."
  type        = list(string)
  default     = []
  validation {
    condition     = alltrue([for s in var.eks_cluster_api_ingress_security_group_ids : can(regex("^sg-[0-9a-f]{8,17}$", s))])
    error_message = "eks_cluster_api_ingress_security_group_ids must contain only security group IDs."
  }
}

variable "eks_pod_grant_foundation_datasets_bucket" {
  description = "Include the Foundation bucket <project>-<environment>-datasets-<account> in the pod IRSA S3 policy. Default true (unchanged). The workshop sets false: it uses only eks_pod_extra_s3_buckets, so no unused, predictable bucket name is trusted."
  type        = bool
  default     = true
}

# Network from SSM (Foundation) only when it was not passed in explicitly.
data "aws_ssm_parameter" "eks_vpc_id" {
  count = var.enable_eks_cluster && var.vpc_id == "" ? 1 : 0
  name  = "/${var.project_name}/vpc-id"
}

data "aws_ssm_parameter" "eks_private_subnet_ids" {
  count = var.enable_eks_cluster && length(var.private_subnet_ids) == 0 ? 1 : 0
  name  = "/${var.project_name}/private-subnet-ids"
}

data "aws_caller_identity" "eks_current" {
  count = var.enable_eks_cluster ? 1 : 0
}

# GPU node group must be pinned to the single AZ the Capacity Block is
# reserved in — EKS picks a subnet across whatever's passed to subnet_ids,
# and a mismatch fails node creation with an AZ-mismatch error from EC2.
data "aws_subnet" "eks_gpu_az_subnet" {
  count = var.enable_eks_cluster && var.eks_gpu_availability_zone != "" ? 1 : 0
  filter {
    name   = "subnet-id"
    values = local.eks_private_subnet_ids
  }
  availability_zone = var.eks_gpu_availability_zone
}

locals {
  # Explicit network variables win; otherwise SSM when the cluster is enabled.
  eks_vpc_id = var.vpc_id != "" ? var.vpc_id : (
    var.enable_eks_cluster ? data.aws_ssm_parameter.eks_vpc_id[0].value : ""
  )
  eks_private_subnet_ids = length(var.private_subnet_ids) > 0 ? var.private_subnet_ids : (
    var.enable_eks_cluster ? split(",", data.aws_ssm_parameter.eks_private_subnet_ids[0].value) : []
  )

  # GPU node group subnet: pinned to the Capacity Block's AZ when specified,
  # otherwise falls back to all private subnets (on-demand path).
  eks_gpu_subnet_ids = var.enable_eks_cluster && var.eks_gpu_availability_zone != "" ? [data.aws_subnet.eks_gpu_az_subnet[0].id] : local.eks_private_subnet_ids

  # Launch template for the GPU node group only needs to add Capacity Block
  # targeting; everything else (AMI, IAM, security groups) comes from the
  # EKS managed node group defaults.
  eks_gpu_capacity_block_specified = var.eks_gpu_capacity_reservation_id != ""
}

# =============================================================================
# EKS CLUSTER + NODE GROUPS
# =============================================================================

module "cosmos3_eks" {
  count   = var.enable_eks_cluster ? 1 : 0
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"

  cluster_name    = "${local.prefix}-${var.eks_cluster_name}"
  cluster_version = var.eks_kubernetes_version

  vpc_id     = local.eks_vpc_id
  subnet_ids = local.eks_private_subnet_ids

  # Control plane ENIs only. EKS requires them to span two AZs, but the node groups may
  # need to stay in one (a capacity reservation's AZ), so the control plane can be given
  # its own subnet list. Empty (default) makes the module's coalescelist fall back to
  # subnet_ids, which is the previous behavior.
  control_plane_subnet_ids = var.eks_control_plane_subnet_ids

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # With private access on, EKS associates a private hosted zone with the cluster VPC, so
  # a workstation in the same VPC resolves the endpoint to the control-plane ENIs' private
  # IPs. Those ENIs carry the module's cluster security group, which by default admits
  # only node traffic; without these rules kubectl and the workstation's `terraform plan`
  # (the Helm provider refreshes against the API) time out. The source is a security group
  # rather than a CIDR so the rule stays as narrow as the workstation and does not drift.
  # Keys are positional ("ingress_api_sg_0", ...): append only. An empty list adds nothing.
  cluster_security_group_additional_rules = {
    for idx, sg in var.eks_cluster_api_ingress_security_group_ids : "ingress_api_sg_${idx}" => {
      description              = "Kubernetes API from workshop workstation (private endpoint)"
      type                     = "ingress"
      protocol                 = "tcp"
      from_port                = 443
      to_port                  = 443
      source_security_group_id = sg
    }
  }

  enable_irsa = true

  # Pin the KMS key administrators instead of letting the module infer them.
  #
  # The module defaults to
  #   coalescelist(var.kms_key_administrators, [data.aws_iam_session_context.current.issuer_arn])
  # so the cluster's envelope-encryption key policy names whoever ran Terraform. In this
  # workshop CodeBuild applies and the workstation later plans, so the key policy flips
  # between the two roles and every plan reports a one-resource in-place update that has
  # nothing to do with the configuration:
  #   ~ Principal.AWS = ".../TerraformBuildRole-..." -> ".../InstanceRole-..."
  # That is the same caller-derived drift that enable_cluster_creator_admin_permissions
  # caused for access entries, and it is the last thing standing between an attendee and a
  # clean "No changes" plan.
  kms_key_administrators = var.eks_admin_principal_arns

  # Keep the account-root statement in the key policy. This is already the module default
  # in v20.37.2 (variables.tf kms_key_enable_default_policy = true, passed to
  # terraform-aws-modules/kms 2.1.0 as enable_default_policy), so pinning it changes
  # nothing in the plan. It is pinned so a future ~> 20.0 default change cannot remove
  # that statement: the Cleanup page's CloudShell destroy relies on it to schedule key
  # deletion after both kms_key_administrators roles were deleted with the stack.
  kms_key_enable_default_policy = true

  # Deliberately false: this option derives a "cluster_creator" access entry from
  # whoever ran `terraform apply` (data.aws_caller_identity), so a later plan or apply
  # by any other principal reports the access entry and its policy association as
  # needing replacement - drift that has nothing to do with the configuration. That
  # matters when automation applies and a human later inspects: in the workshop the
  # cluster is created by CodeBuild, then attendees run `terraform plan` from the
  # workstation and must see "No changes".
  # Grant access explicitly via eks_admin_principal_arns instead (below).
  enable_cluster_creator_admin_permissions = false

  # Keys are positional ("admin-0", "admin-1", ...). Append to
  # eks_admin_principal_arns only - never reorder or remove from the middle. Doing so
  # shifts an existing principal to a new key, and Terraform then destroys and
  # recreates the *same* access entry in parallel (the two keys have no dependency
  # between them). EKS has not propagated the delete by the time the create lands, so
  # the apply fails with:
  #   ResourceInUseException: The specified access entry resource is already in use
  # Recovering means re-running the apply once the delete settles. A fresh cluster is
  # unaffected: there is nothing to destroy, so the ordering only matters when editing
  # this list against a live cluster.
  #
  # The same positional-key rule applies to eks_view_principal_arns ("view-0", ...).
  # The two maps are kept separate so a view principal can never shift into an admin
  # key, or vice versa.
  access_entries = merge(
    {
      for idx, principal_arn in var.eks_admin_principal_arns : "admin-${idx}" => {
        principal_arn = principal_arn
        policy_associations = {
          admin = {
            policy_arn   = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
            access_scope = { type = "cluster" }
          }
        }
      }
    },
    # Read-only principals. The EKS console's Resources tab needs both the IAM
    # permission eks:AccessKubernetesApi AND a Kubernetes RBAC binding; without an
    # access entry it reports that the principal has no access to Kubernetes objects.
    # Workshop participants get AmazonEKSViewPolicy rather than ClusterAdmin so they
    # can watch Jobs, pods, and logs in the console without being able to mutate the
    # cluster from it - submission stays an explicit `kubectl apply` of the manifest in
    # the toolchain repository.
    {
      for idx, principal_arn in var.eks_view_principal_arns : "view-${idx}" => {
        principal_arn = principal_arn
        # AmazonEKSViewPolicy maps to Kubernetes' built-in "view" ClusterRole, which covers
        # namespaced resources but NOT cluster-scoped ones. Nodes are cluster-scoped, so the
        # EKS console's Compute tab fails with
        #   nodes is forbidden: User "..." cannot list resource "nodes" at the cluster scope
        # while the Node groups panel beside it loads fine, because that reads the EKS API
        # rather than the Kubernetes API.
        #
        # AmazonEKSAdminViewPolicy would fix it in one line, but it grants get/list/watch on
        # every resource including Secrets. At event scale the hf-token Secret may hold a
        # credential shared across accounts, so handing every participant read access to it
        # is not acceptable. Instead the entry joins a Kubernetes group, and
        # event-provision.sh binds that group to a ClusterRole covering just nodes.
        kubernetes_groups = ["cosmos3-viewers"]
        policy_associations = {
          view = {
            policy_arn   = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSViewPolicy"
            access_scope = { type = "cluster" }
          }
        }
      }
    },
  )

  cluster_addons = {
    coredns    = { most_recent = true }
    kube-proxy = { most_recent = true }
    vpc-cni    = { most_recent = true, before_compute = true }
    aws-ebs-csi-driver = {
      most_recent              = true
      service_account_role_arn = module.ebs_csi_irsa[0].iam_role_arn
    }
  }

  eks_managed_node_groups = {
    system = {
      name           = "cosmos3-system"
      instance_types = var.eks_system_node_instance_types
      capacity_type  = "ON_DEMAND"
      min_size       = 1
      max_size       = 2
      desired_size   = 1
      subnet_ids     = local.eks_private_subnet_ids

      labels = { "node.kubernetes.io/purpose" = "system" }
    }

    gpu = {
      name           = "cosmos3-gpu"
      instance_types = [var.eks_gpu_node_instance_type]
      # Capacity Block nodes must use CAPACITY_BLOCK capacity type.
      capacity_type = local.eks_gpu_capacity_block_specified ? "CAPACITY_BLOCK" : "ON_DEMAND"
      ami_type      = "AL2023_x86_64_NVIDIA"

      min_size     = 0
      max_size     = 1
      desired_size = var.eks_gpu_node_desired_size
      subnet_ids   = local.eks_gpu_subnet_ids

      labels = {
        "node.kubernetes.io/purpose" = "gpu"
        "nvidia.com/gpu.present"     = "true"
      }

      taints = {
        gpu = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      # Cluster Autoscaler tags on the underlying ASG. These let autoscaler
      # discover and scale this node group FROM ZERO — when scaled to 0 there
      # is no live node for it to inspect labels/taints/GPU count from, so it
      # reads them from these node-template tags instead. Without the
      # resources/nvidia.com/gpu tag specifically, autoscaler can't tell a
      # pending pod's `nvidia.com/gpu: 8` request would fit a node it hasn't
      # seen yet, and will never scale up.
      tags = {
        "k8s.io/cluster-autoscaler/enabled"                                        = "true"
        "k8s.io/cluster-autoscaler/${local.prefix}-${var.eks_cluster_name}"        = "owned"
        "k8s.io/cluster-autoscaler/node-template/label/node.kubernetes.io/purpose" = "gpu"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu.present"     = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/nvidia.com/gpu"             = "true:NoSchedule"
        "k8s.io/cluster-autoscaler/node-template/resources/nvidia.com/gpu"         = "8"
      }

      # Capacity Block targeting — same mechanism as the EC2 instance
      # (capacity_reservation_specification), applied via the node group's
      # generated launch template. Must be `{}` (not null) when unset — the
      # module checks length() on this value.
      capacity_reservation_specification = local.eks_gpu_capacity_block_specified ? {
        capacity_reservation_target = {
          capacity_reservation_id = var.eks_gpu_capacity_reservation_id
        }
      } : {}

      # Capacity Block launches ALSO require instance_market_options with
      # market_type="capacity-block" on the launch template (separate from
      # capacity_reservation_specification above) — same requirement we hit
      # launching the standalone EC2 instance in ec2.tf. Without this the API
      # rejects the node group with "market type (purchasing) option is not
      # valid" even though capacity_reservation_specification is set.
      instance_market_options = local.eks_gpu_capacity_block_specified ? {
        market_type = "capacity-block"
      } : {}

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = var.eks_gpu_node_volume_size_gb
            volume_type           = "gp3"
            encrypted             = true
            delete_on_termination = true
          }
        }
      }
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-eks"
  }
}

# =============================================================================
# EBS CSI DRIVER IRSA (required for the aws-ebs-csi-driver addon above)
# =============================================================================

module "ebs_csi_irsa" {
  count   = var.enable_eks_cluster ? 1 : 0
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name             = "${local.prefix}-cosmos3-ebs-csi-irsa"
  attach_ebs_csi_policy = true

  oidc_providers = {
    main = {
      # Referencing the cluster module directly here would create a circular
      # dependency (module needs this role ARN in cluster_addons, this role
      # needs the module's OIDC provider). Terraform resolves this because
      # both are in the same apply graph and the provider ARN is a stable,
      # computable attribute — but if you see a cycle error, remove the
      # aws-ebs-csi-driver entry from cluster_addons on first apply, then add
      # it back on a second apply once the OIDC provider exists.
      provider_arn               = module.cosmos3_eks[0].oidc_provider_arn
      namespace_service_accounts = ["kube-system:ebs-csi-controller-sa"]
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-eks"
  }
}

# =============================================================================
# IRSA: COSMOS3 GENERATION POD (S3 write + Secrets Manager read)
# =============================================================================

resource "aws_iam_policy" "cosmos3_pod_s3" {
  count       = var.enable_eks_cluster ? 1 : 0
  name        = "${local.prefix}-cosmos3-pod-s3"
  description = "S3 read/write for the Cosmos3 generation pod (cosmos-samples/ output, dataset input)"

  # Buckets the pod may use: eks_pod_extra_s3_buckets, plus the Foundation datasets
  # bucket (<project>-<environment>-datasets-<account>) when
  # eks_pod_grant_foundation_datasets_bucket is true (the standalone default).
  #
  # The workshop's per-participant data bucket (physical-ai-data-<account>) is created
  # by CloudFormation, not here, and it does not follow the Foundation naming. Without it
  # listed, a generation pod cannot read the teleoperation dataset the lab tells
  # attendees to use - the pod fails with AccessDenied even though the object exists.
  #
  # The workshop turns the Foundation grant off. Event accounts have no Foundation
  # deployment, and S3 bucket names are global: the Foundation name is predictable, so
  # if that bucket does not exist anyone could create it in their own account and the
  # pod's PutObject grant would let data be written there. Least privilege: grant only
  # the buckets the workshop actually uses.
  #
  # Computed inline rather than in a `locals` block: the account-id data source is
  # count-gated on enable_eks_cluster, so a top-level local would fail to evaluate when
  # the cluster is disabled. distinct() keeps first-occurrence order, so with defaults
  # the policy JSON is unchanged.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ListBuckets"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [for b in distinct(concat(var.eks_pod_grant_foundation_datasets_bucket ? ["${var.project_name}-${var.environment}-datasets-${data.aws_caller_identity.eks_current[0].account_id}"] : [], var.eks_pod_extra_s3_buckets)) : "arn:aws:s3:::${b}"]
      },
      {
        Sid      = "ReadWriteObjects"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject"]
        Resource = [for b in distinct(concat(var.eks_pod_grant_foundation_datasets_bucket ? ["${var.project_name}-${var.environment}-datasets-${data.aws_caller_identity.eks_current[0].account_id}"] : [], var.eks_pod_extra_s3_buckets)) : "arn:aws:s3:::${b}/*"]
      }
    ]
  })

  lifecycle {
    precondition {
      condition     = var.eks_pod_grant_foundation_datasets_bucket || length(var.eks_pod_extra_s3_buckets) > 0
      error_message = "Pod S3 policy would grant no bucket: set eks_pod_extra_s3_buckets or keep eks_pod_grant_foundation_datasets_bucket = true."
    }
  }
}

resource "aws_iam_policy" "cosmos3_pod_secrets" {
  count       = var.enable_eks_cluster ? 1 : 0
  name        = "${local.prefix}-cosmos3-pod-secrets"
  description = "Secrets Manager read for the Cosmos3 generation pod (HF token)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "ReadHfToken"
      Effect   = "Allow"
      Action   = ["secretsmanager:GetSecretValue"]
      Resource = ["arn:aws:secretsmanager:${local.region}:${data.aws_caller_identity.eks_current[0].account_id}:secret:${var.project_name}/hf-token*"]
    }]
  })
}

module "cosmos3_pod_irsa" {
  count   = var.enable_eks_cluster ? 1 : 0
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name = "${local.prefix}-cosmos3-pod-irsa"

  role_policy_arns = {
    s3      = aws_iam_policy.cosmos3_pod_s3[0].arn
    secrets = aws_iam_policy.cosmos3_pod_secrets[0].arn
  }

  oidc_providers = {
    main = {
      provider_arn               = module.cosmos3_eks[0].oidc_provider_arn
      namespace_service_accounts = ["default:cosmos3-generator"]
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-eks"
  }
}

# =============================================================================
# CLUSTER AUTOSCALER (scale the GPU node group 0 -> 1 -> 0 automatically)
# =============================================================================
#
# Watches for pods that can't schedule because there's no node with capacity
# (e.g. the cosmos3-vllm-omni Job requesting nvidia.com/gpu: 8 while the GPU
# node group sits at desired=0), scales the ASG up, waits for the pod to
# schedule and run, then scales back to 0 ~10 min after the node goes idle.
# This replaces manually running `aws eks update-nodegroup-config` before/
# after each generation Job.

module "cluster_autoscaler_irsa" {
  count   = var.enable_eks_cluster ? 1 : 0
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"

  role_name                        = "${local.prefix}-cosmos3-cluster-autoscaler-irsa"
  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_names = [module.cosmos3_eks[0].cluster_name]

  oidc_providers = {
    main = {
      provider_arn               = module.cosmos3_eks[0].oidc_provider_arn
      namespace_service_accounts = ["kube-system:cluster-autoscaler"]
    }
  }

  tags = {
    Project     = var.project_name
    Environment = var.environment
    Component   = "cosmos3-eks"
  }
}

resource "helm_release" "cluster_autoscaler" {
  count      = var.enable_eks_cluster ? 1 : 0
  name       = "cluster-autoscaler"
  repository = "https://kubernetes.github.io/autoscaler"
  chart      = "cluster-autoscaler"
  version    = var.eks_cluster_autoscaler_version
  namespace  = "kube-system"

  set {
    name  = "autoDiscovery.clusterName"
    value = module.cosmos3_eks[0].cluster_name
  }
  set {
    name  = "awsRegion"
    value = var.aws_region
  }
  set {
    name  = "rbac.serviceAccount.name"
    value = "cluster-autoscaler"
  }
  set {
    name  = "rbac.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.cluster_autoscaler_irsa[0].iam_role_arn
  }
  set {
    # Default is 10 min; matches the "generate then idle" pattern here
    # closely enough without being so aggressive it tears down a node
    # between back-to-back generation requests.
    name  = "extraArgs.scale-down-unneeded-time"
    value = "10m"
  }
  set {
    # Required so autoscaler evaluates a scaled-to-zero node group's
    # node-template tags (set on the GPU node group above) instead of
    # skipping it for having no nodes to inspect.
    name  = "extraArgs.balance-similar-node-groups"
    value = "false"
  }

  depends_on = [module.cosmos3_eks]
}

# =============================================================================
# OUTPUTS
# =============================================================================

output "eks_cluster_name" {
  description = "Cosmos 3 EKS cluster name (if enabled)"
  value       = var.enable_eks_cluster ? module.cosmos3_eks[0].cluster_name : null
}

output "eks_cluster_endpoint" {
  description = "Cosmos 3 EKS cluster API endpoint (if enabled)"
  value       = var.enable_eks_cluster ? module.cosmos3_eks[0].cluster_endpoint : null
}

output "eks_pod_irsa_role_arn" {
  description = "IRSA role ARN for the cosmos3-generator service account (if enabled)"
  value       = var.enable_eks_cluster ? module.cosmos3_pod_irsa[0].iam_role_arn : null
}

#=====================================================================================
# NVIDIA DEVICE PLUGIN
#
# Kubernetes does not know a node has GPUs on its own. The device plugin is what
# advertises nvidia.com/gpu as an allocatable resource; without it a pod requesting
# nvidia.com/gpu stays Pending forever, even on a healthy GPU node that has joined and
# reports Ready. The failure is quiet and misleading:
#
#   kubectl describe node  ->  allocatable has no nvidia.com/gpu entry
#   kubectl describe pod   ->  "0/1 nodes are available: 1 node(s) didn't match
#                               Pod's node affinity/selector"
#
# and Cluster Autoscaler then taints the idle node DeletionCandidateOfClusterAutoscaler
# and reclaims it, so the pending pod triggers another scale-up and the cycle repeats
# indefinitely. Confirmed on this cluster: a generation Job sat Pending for 9 minutes
# beside a Ready g6e.4xlarge until the plugin was installed, after which it scheduled
# immediately.
#
# This is NOT bundled: the EKS GPU AMI ships the driver but not the plugin, and
# nvidia-device-plugin is not available as an EKS managed addon
# (`aws eks describe-addon-versions --addon-name nvidia-device-plugin` returns nothing),
# so it has to be installed explicitly.
#=====================================================================================
resource "helm_release" "nvidia_device_plugin" {
  count      = var.enable_eks_cluster ? 1 : 0
  name       = "nvidia-device-plugin"
  repository = "https://nvidia.github.io/k8s-device-plugin"
  chart      = "nvidia-device-plugin"
  version    = var.eks_nvidia_device_plugin_version
  namespace  = "kube-system"

  # Only GPU nodes. nvidia.com/gpu.present is set by the EKS GPU AMI, so this keeps the
  # DaemonSet off the t3.medium system nodes where it has no device to manage.
  # type = "string" is required. Without it Helm's default "auto" coercion turns "true"
  # into a YAML boolean, and nodeSelector values must be strings, so the rendered
  # DaemonSet is rejected by the API server with:
  #   cannot unmarshal bool into Go struct field PodSpec.spec.template.spec.nodeSelector
  set {
    name  = "nodeSelector.nvidia\\.com/gpu\\.present"
    value = "true"
    type  = "string"
  }

  # The GPU node group carries a nvidia.com/gpu=true:NoSchedule taint to keep ordinary
  # workloads off it. The plugin has to tolerate that taint or it cannot run on the very
  # nodes whose GPUs it exists to advertise - which would leave the cluster in exactly
  # the state described above.
  set {
    name  = "tolerations[0].key"
    value = "nvidia.com/gpu"
  }
  set {
    name  = "tolerations[0].operator"
    value = "Exists"
  }
  set {
    name  = "tolerations[0].effect"
    value = "NoSchedule"
  }

  depends_on = [module.cosmos3_eks]
}
