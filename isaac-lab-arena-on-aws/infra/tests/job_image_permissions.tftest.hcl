# Exercise the production policy inputs without creating AWS resources.
# Run: terraform init -backend=false && terraform test -filter=tests/job_image_permissions.tftest.hcl
mock_provider "aws" {
  mock_data "aws_caller_identity" {
    defaults = {
      account_id = "123456789012"
    }
  }
  mock_data "aws_region" {
    defaults = {
      region = "us-east-1"
    }
  }
  mock_data "aws_iam_policy_document" {
    defaults = {
      json = "{\"Version\":\"2012-10-17\",\"Statement\":[]}"
    }
  }
  mock_data "aws_secretsmanager_secret" {
    defaults = {
      arn = "arn:aws:secretsmanager:us-east-1:123456789012:secret:test-token-AbCdEf"
    }
  }
  mock_resource "aws_s3_bucket" {
    defaults = {
      arn = "arn:aws:s3:::arena-test-bucket"
    }
  }
  mock_resource "aws_iam_role" {
    defaults = {
      arn = "arn:aws:iam::123456789012:role/arena-test-role"
    }
  }
  mock_resource "aws_iam_policy" {
    defaults = {
      arn = "arn:aws:iam::123456789012:policy/arena-test-policy"
    }
  }
}

override_data {
  target = data.aws_ssm_parameter.models_bucket
  values = { value = "arena-test-models" }
}
override_data {
  target = data.aws_ssm_parameter.sagemaker_role_arn
  values = { value = "arn:aws:iam::123456789012:role/arena-test-foundation" }
}
override_data {
  target = data.aws_ssm_parameter.arena_ecr
  values = { value = "123456789012.dkr.ecr.us-east-1.amazonaws.com/arena-test/isaac-lab-arena" }
}
override_data {
  target = data.aws_ecr_repository.arena
  values = { arn = "arn:aws:ecr:us-east-1:123456789012:repository/arena-test/isaac-lab-arena" }
}
override_resource {
  target = aws_ecr_repository.repos["vla/gr00t"]
  values = { arn = "arn:aws:ecr:us-east-1:123456789012:repository/vla/gr00t" }
}
override_resource {
  target = aws_ecr_repository.repos["vla/openvla"]
  values = { arn = "arn:aws:ecr:us-east-1:123456789012:repository/vla/openvla" }
}
override_resource {
  target = aws_ecr_repository.repos["vla/molmoact2"]
  values = { arn = "arn:aws:ecr:us-east-1:123456789012:repository/vla/molmoact2" }
}
override_resource {
  target = aws_ecr_repository.repos["vla/isaac-arena"]
  values = { arn = "arn:aws:ecr:us-east-1:123456789012:repository/vla/isaac-arena" }
}
override_data {
  target = data.aws_ecr_repository.existing["vla/gr00t"]
  values = { arn = "arn:aws:ecr:us-east-1:123456789012:repository/vla/gr00t" }
}

variables {
  project_name = "arena-test"
}

run "default_training_and_evaluation_repositories" {
  command = apply

  assert {
    condition = toset(one([
      for statement in data.aws_iam_policy_document.job_runtime_common.statement :
      statement.resources if statement.sid == "EcrPullJobImages"
      ])) == toset([
      "arn:aws:ecr:us-east-1:123456789012:repository/vla/gr00t",
      "arn:aws:ecr:us-east-1:123456789012:repository/vla/openvla",
      "arn:aws:ecr:us-east-1:123456789012:repository/vla/molmoact2",
      "arn:aws:ecr:us-east-1:123456789012:repository/vla/isaac-arena",
      "arn:aws:ecr:us-east-1:123456789012:repository/arena-test/isaac-lab-arena",
      "arn:aws:ecr:us-east-1:763104351884:repository/pytorch-training",
      "arn:aws:ecr:us-east-1:683313688378:repository/sagemaker-scikit-learn",
    ])
    error_message = "Every declared training/evaluation image must be pullable, with no wildcard or unrelated repository grant."
  }

  assert {
    condition = toset(one([
      for statement in data.aws_iam_policy_document.job_runtime_common.statement :
      statement.actions if statement.sid == "EcrPullJobImages"
      ])) == toset([
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ])
    error_message = "Job image access must stay read-only."
  }
}

run "reused_training_repository" {
  command   = apply
  state_key = "reused-repositories"

  variables {
    ecr_repos          = ["vla/openvla", "vla/molmoact2", "vla/isaac-arena"]
    existing_ecr_repos = ["vla/gr00t"]
  }

  assert {
    condition = contains(one([
      for statement in data.aws_iam_policy_document.job_runtime_common.statement :
      statement.resources if statement.sid == "EcrPullJobImages"
    ]), "arn:aws:ecr:us-east-1:123456789012:repository/vla/gr00t")
    error_message = "Referencing an existing training repository must preserve its image-pull permission."
  }
}
