data "aws_caller_identity" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  prefix = "${var.project_name}-${var.environment}"
  azs    = slice(data.aws_availability_zones.available.names, 0, 3)
}

# =============================================================================
# VPC (3 AZs, single NAT gateway, private egress)
# =============================================================================

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name        = "${local.prefix}-isaac-sim-vpc"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name        = "${local.prefix}-isaac-sim-igw"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_subnet" "public" {
  count                   = length(local.azs)
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name        = "${local.prefix}-isaac-sim-public-${count.index}"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_subnet" "private" {
  count             = length(local.azs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = local.azs[count.index]

  tags = {
    Name        = "${local.prefix}-isaac-sim-private-${count.index}"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name        = "${local.prefix}-isaac-sim-nat-eip"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name        = "${local.prefix}-isaac-sim-nat"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }

  depends_on = [aws_internet_gateway.main]
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name        = "${local.prefix}-isaac-sim-public-rt"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name        = "${local.prefix}-isaac-sim-private-rt"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = length(aws_subnet.private)
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# =============================================================================
# VPC FLOW LOGS (all traffic — network audit trail)
# =============================================================================

resource "aws_cloudwatch_log_group" "flow_logs" {
  name              = "/isaac-sim/vpc-flow-logs"
  retention_in_days = 14

  tags = {
    Name        = "${local.prefix}-isaac-sim-flow-logs"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_iam_role" "flow_logs" {
  name = "${local.prefix}-isaac-sim-flow-logs-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "vpc-flow-logs.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "flow_logs" {
  name = "${local.prefix}-isaac-sim-flow-logs"
  role = aws_iam_role.flow_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams",
      ]
      Resource = "${aws_cloudwatch_log_group.flow_logs.arn}:*"
    }]
  })
}

moved {
  from = aws_flow_log.reject
  to   = aws_flow_log.all
}

resource "aws_flow_log" "all" {
  iam_role_arn    = aws_iam_role.flow_logs.arn
  log_destination = aws_cloudwatch_log_group.flow_logs.arn
  traffic_type    = "ALL"
  vpc_id          = aws_vpc.main.id

  tags = {
    Name        = "${local.prefix}-isaac-sim-flow-log"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

# =============================================================================
# S3 BUCKET (configs, assets, results — SSE-S3, TLS-only, results expire 30d)
# =============================================================================

resource "aws_s3_bucket" "sim" {
  bucket_prefix = "${local.prefix}-isaac-sim-"
  force_destroy = true

  tags = {
    Name        = "${local.prefix}-isaac-sim"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "sim" {
  bucket = aws_s3_bucket.sim.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "sim" {
  bucket = aws_s3_bucket.sim.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "sim" {
  bucket                  = aws_s3_bucket.sim.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "sim" {
  bucket = aws_s3_bucket.sim.id

  rule {
    id     = "CleanupResults"
    status = "Enabled"

    filter {
      prefix = "results/"
    }

    expiration {
      days = 30
    }

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

resource "aws_s3_bucket_policy" "sim" {
  bucket = aws_s3_bucket.sim.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource = [
        aws_s3_bucket.sim.arn,
        "${aws_s3_bucket.sim.arn}/*",
      ]
      Condition = {
        Bool = { "aws:SecureTransport" = "false" }
      }
    }]
  })
}

# =============================================================================
# ECR REPOSITORY (Isaac Sim container image — immutable tags, scan on push)
# =============================================================================

resource "aws_ecr_repository" "isaac_sim" {
  name                 = "isaac-sim-batch"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name        = "${local.prefix}-isaac-sim-batch"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_ecr_lifecycle_policy" "isaac_sim" {
  repository = aws_ecr_repository.isaac_sim.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only the 5 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

# =============================================================================
# IAM — JOB ROLE (container task role: scoped S3 + CloudWatch metrics)
# =============================================================================

resource "aws_iam_role" "job" {
  name = "${local.prefix}-isaac-sim-job-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Name        = "${local.prefix}-isaac-sim-job-role"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_iam_role_policy" "job" {
  name = "${local.prefix}-isaac-sim-job"
  role = aws_iam_role.job.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadConfigs"
        Effect = "Allow"
        Action = ["s3:GetObject"]
        Resource = [
          "${aws_s3_bucket.sim.arn}/configs/*",
        ]
      },
      {
        Sid      = "WriteResults"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = ["${aws_s3_bucket.sim.arn}/results/*"]
      },
      {
        # Hard ceiling: the job role can PutObject ONLY to results/*, and to no
        # other bucket. Blocks result exfiltration to an attacker-controlled
        # bucket even if the upload destination is overridden at submit time.
        Sid         = "DenyPutObjectOutsideResults"
        Effect      = "Deny"
        Action      = ["s3:PutObject"]
        NotResource = ["${aws_s3_bucket.sim.arn}/results/*"]
      },
      {
        Sid      = "ListBucket"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.sim.arn]
      },
      {
        Sid      = "PublishMetrics"
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
        Condition = {
          StringEquals = { "cloudwatch:namespace" = "IsaacSim/Jobs" }
        }
      },
    ]
  })
}

# =============================================================================
# IAM — INSTANCE ROLE (EC2 hosts Batch launches: ECS + ECR pull + logs)
# =============================================================================

resource "aws_iam_role" "instance" {
  name = "${local.prefix}-isaac-sim-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = {
    Name        = "${local.prefix}-isaac-sim-instance-role"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_iam_role_policy_attachment" "instance_ecs" {
  role       = aws_iam_role.instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "instance" {
  name = "${local.prefix}-isaac-sim-instance-profile"
  role = aws_iam_role.instance.name
}

# =============================================================================
# BATCH — EGRESS SECURITY GROUP, LAUNCH TEMPLATE, COMPUTE ENV, QUEUE, JOB DEF
# =============================================================================

resource "aws_security_group" "batch" {
  name        = "${local.prefix}-isaac-sim-batch-sg"
  description = "Egress-only SG for Isaac Sim Batch GPU instances (NGC + asset pulls, ECR, S3)"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "All outbound"
  }

  tags = {
    Name        = "${local.prefix}-isaac-sim-batch-sg"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

# Encrypted 100 GB gp3 root volume + IMDSv2 (device name matches the ECS GPU AMI)
resource "aws_launch_template" "gpu" {
  name = "${local.prefix}-isaac-sim-gpu"

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size = 100
      volume_type = "gp3"
      encrypted   = true
    }
  }

  metadata_options {
    http_tokens = "required" # IMDSv2
  }

  tags = {
    Name        = "${local.prefix}-isaac-sim-gpu"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

# On-Demand GPU compute environment.
# NOTE: This sample uses On-Demand for reliable placement. AWS Batch does NOT
# automatically fall back from a Spot compute environment to On-Demand when Spot
# capacity is unavailable, so a Spot-first queue can leave GPU jobs stuck in
# RUNNABLE. For Spot savings, add a Spot compute environment and handle Spot
# shortfalls yourself (see the blog's cost-optimization note).
# Batch uses the service-linked role AWSServiceRoleForBatch (auto-created on
# first use); no explicit service_role is set.
resource "aws_batch_compute_environment" "ondemand" {
  name = "${local.prefix}-isaac-sim-ondemand"
  type = "MANAGED"

  compute_resources {
    type                = "EC2"
    allocation_strategy = "BEST_FIT_PROGRESSIVE"
    min_vcpus           = 0
    max_vcpus           = var.max_vcpus
    instance_type       = var.gpu_instance_types
    instance_role       = aws_iam_instance_profile.instance.arn
    subnets             = aws_subnet.private[*].id
    security_group_ids  = [aws_security_group.batch.id]

    launch_template {
      launch_template_id = aws_launch_template.gpu.id
      version            = "$Latest"
    }

    # GPU-optimized ECS AMI on Amazon Linux 2023 (bundles NVIDIA drivers + runtime)
    ec2_configuration {
      image_type = "ECS_AL2023_NVIDIA"
    }
  }

  tags = {
    Name        = "${local.prefix}-isaac-sim-ondemand"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_batch_job_queue" "sim" {
  name     = "IsaacSim-SimQueue"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.ondemand.arn
  }

  tags = {
    Name        = "${local.prefix}-isaac-sim-queue"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_cloudwatch_log_group" "sim" {
  name              = "/isaac-sim/batch"
  retention_in_days = 14

  tags = {
    Name        = "${local.prefix}-isaac-sim-batch-logs"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_batch_job_definition" "sim" {
  name                  = "IsaacSim-SimJob"
  type                  = "container"
  platform_capabilities = ["EC2"]

  container_properties = jsonencode({
    image      = "${aws_ecr_repository.isaac_sim.repository_url}:${var.image_tag}"
    jobRoleArn = aws_iam_role.job.arn

    resourceRequirements = [
      { type = "VCPU", value = "8" },
      { type = "MEMORY", value = "30720" },
      { type = "GPU", value = "1" },
    ]

    environment = [
      { name = "S3_BUCKET", value = aws_s3_bucket.sim.bucket },
    ]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.sim.name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "sim"
      }
    }
  })

  retry_strategy {
    attempts = 1
  }

  timeout {
    attempt_duration_seconds = 7200
  }

  tags = {
    Name        = "${local.prefix}-isaac-sim-job-def"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

# =============================================================================
# OBSERVABILITY — SNS TOPIC, DASHBOARD, FAILURE ALARM
# =============================================================================

resource "aws_sns_topic" "alerts" {
  name = "isaac-sim-alerts"

  tags = {
    Name        = "${local.prefix}-isaac-sim-alerts"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}

resource "aws_sns_topic_subscription" "alert_email" {
  count     = var.alert_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_dashboard" "isaac_sim" {
  dashboard_name = "Isaac-Sim"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Jobs Started vs Succeeded vs Failed"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["IsaacSim/Jobs", "JobStarted", { stat = "Sum", period = 300, label = "Started" }],
            ["IsaacSim/Jobs", "JobSucceeded", { stat = "Sum", period = 300, label = "Succeeded" }],
            ["IsaacSim/Jobs", "JobFailed", { stat = "Sum", period = 300, label = "Failed" }],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "Job Duration (seconds)"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          metrics = [
            ["IsaacSim/Jobs", "JobDuration", { stat = "Average", period = 300, label = "Avg Duration" }],
            ["IsaacSim/Jobs", "JobDuration", { stat = "Maximum", period = 300, label = "Max Duration" }],
          ]
        }
      },
    ]
  })
}

resource "aws_cloudwatch_metric_alarm" "job_failures" {
  alarm_name          = "IsaacSim-JobFailures"
  namespace           = "IsaacSim/Jobs"
  metric_name         = "JobFailed"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_description   = "Fires when one or more Isaac Sim Batch jobs report failure"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  tags = {
    Name        = "${local.prefix}-isaac-sim-job-failures"
    Project     = var.project_name
    Environment = var.environment
    Component   = "isaac-sim-batch"
  }
}
