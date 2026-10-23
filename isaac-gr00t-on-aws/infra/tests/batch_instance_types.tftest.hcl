# Batch compute-environment instance-type tests. Run with `terraform test` from
# isaac-gr00t-on-aws/infra. The AWS provider is mocked, so no AWS credentials are
# needed and no AWS API is called. These prove what the configuration renders and
# which inputs the preconditions reject; they do not prove that AWS Batch accepts
# the types, that capacity exists, or what a plan against real state shows.

mock_provider "aws" {}

variables {
  aws_region   = "us-east-2"
  enable_batch = true
}

# Defaults: the list is empty, so the compute environment renders exactly
# [batch_instance_type] with the allocation strategy left unset, as before the
# list variable existed.
run "defaults_render_single_type" {
  command = plan

  assert {
    condition     = aws_batch_compute_environment.groot[0].compute_resources[0].instance_type == toset(["g6e.4xlarge"])
    error_message = "Defaults must render exactly [batch_instance_type]."
  }
  assert {
    condition     = local.batch_allocation_strategy == null
    error_message = "Defaults must leave allocation_strategy unset."
  }
}

# The documented 4-GPU alternative still renders its single type unchanged.
run "single_type_override_unchanged" {
  command = plan

  variables {
    batch_instance_type  = "g6e.12xlarge"
    batch_job_vcpus      = 48
    batch_job_memory_mib = 180000
    batch_job_gpus       = 4
  }

  assert {
    condition     = aws_batch_compute_environment.groot[0].compute_resources[0].instance_type == toset(["g6e.12xlarge"])
    error_message = "batch_instance_type alone must render exactly that one type."
  }
  assert {
    condition     = local.batch_allocation_strategy == null
    error_message = "A single type must leave allocation_strategy unset."
  }
}

# A one-element list equal to the default is indistinguishable from the default.
run "single_element_list_matches_default" {
  command = plan

  variables {
    batch_instance_types = ["g6e.4xlarge"]
  }

  assert {
    condition     = aws_batch_compute_environment.groot[0].compute_resources[0].instance_type == toset(["g6e.4xlarge"])
    error_message = "A one-element list must render that one type."
  }
  assert {
    condition     = local.batch_allocation_strategy == null
    error_message = "A one-element list must leave allocation_strategy unset."
  }
}

# A single-GPU fallback list replaces batch_instance_type and switches to
# BEST_FIT_PROGRESSIVE so Batch can move on to another type.
run "fallback_list_single_gpu" {
  command = plan

  variables {
    batch_instance_types = ["g6e.4xlarge", "g6e.8xlarge", "g6e.16xlarge"]
  }

  assert {
    condition     = aws_batch_compute_environment.groot[0].compute_resources[0].instance_type == toset(["g6e.4xlarge", "g6e.8xlarge", "g6e.16xlarge"])
    error_message = "The list must replace batch_instance_type."
  }
  assert {
    condition     = aws_batch_compute_environment.groot[0].compute_resources[0].allocation_strategy == "BEST_FIT_PROGRESSIVE"
    error_message = "A multi-type list must use BEST_FIT_PROGRESSIVE."
  }
}

run "fallback_list_four_gpu" {
  command = plan

  variables {
    batch_instance_types = ["g6e.12xlarge", "g6e.24xlarge"]
    batch_job_vcpus      = 48
    batch_job_memory_mib = 180000
    batch_job_gpus       = 4
  }

  assert {
    condition     = aws_batch_compute_environment.groot[0].compute_resources[0].instance_type == toset(["g6e.12xlarge", "g6e.24xlarge"])
    error_message = "The 4-GPU list must render both types."
  }
}

# Every listed type must match the job's GPU count.
run "list_gpu_count_mismatch_rejected" {
  command = plan

  variables {
    batch_instance_types = ["g6e.4xlarge", "g6e.12xlarge"]
  }

  expect_failures = [aws_batch_job_definition.groot_n17]
}

# Every listed type must have 48 GB-class GPUs.
run "list_small_gpu_rejected" {
  command = plan

  variables {
    batch_instance_types = ["g6e.4xlarge", "g6.4xlarge"]
  }

  expect_failures = [aws_batch_job_definition.groot_n17]
}

# A listed type the lookup maps do not know is rejected rather than let through.
run "list_unknown_type_rejected" {
  command = plan

  variables {
    batch_instance_types = ["g6e.4xlarge", "g5.4xlarge"]
  }

  expect_failures = [aws_batch_compute_environment.groot]
}

# The single-type mismatch the original preconditions caught is still caught.
run "single_type_gpu_mismatch_still_rejected" {
  command = plan

  variables {
    batch_instance_type = "g6e.12xlarge"
  }

  expect_failures = [aws_batch_job_definition.groot_n17]
}
