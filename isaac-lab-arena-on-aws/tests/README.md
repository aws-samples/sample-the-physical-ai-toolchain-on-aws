# Arena checks

Run these commands from `isaac-lab-arena-on-aws` after installing the toolchain
and this component into the same Python environment.

The focused client/control-plane suite needs `pytest` and `usd-core`:

```bash
python -m pip install pytest usd-core
AWS_EC2_METADATA_DISABLED=true python -m pytest -q \
  tests/unit \
  tests/test_rollout_video_capture.py tests/test_arena_video_camera.py \
  tests/test_launcher_payload_valid.py tests/test_local_runner.py \
  tests/test_reports_carry_what_they_claim.py tests/test_source_identity.py \
  tests/test_source_archive_contract.py tests/test_pipeline_offline.py \
  tests/test_terraform_provider_pinning.py
```

These tests exercise real CLI parsing, offline SageMaker graph construction,
local success/failure control flow, source identity, concurrent record writes,
SDK responses and interrupted archive downloads through a localhost HTTP server.
They do not provision AWS resources or train a policy.

The Terraform image-permission regression uses a mock provider to check the
actual policy inputs for newly created and reused repositories, including
read-only access and exclusion of unrelated repositories:

```bash
terraform -chdir=infra init -backend=false -lockfile=readonly
terraform -chdir=infra test -filter=tests/job_image_permissions.tftest.hcl
```

For regression checks against saved evidence from actual completed runs:

```bash
python tests/check_evidence_rejections.py \
  --arena-run-dir /path/to/verified/local/run \
  --managed-run-dir /path/to/saved/managed/run \
  --output-dir /path/to/regression-results
```

Either evidence input can be omitted. The local input contains the original
verifier JSON and logs; the managed input is the saved CLI run directory with
`record.json`, job descriptions and downloaded validation/registration evidence.
Use full GR00T training runs for these checks. The script accepts the original
evidence, then requires rejection of damaged copies: missing training completion,
incorrect step counts, wrong checkpoints, missing episodes, invalid publication
and registration mismatches. It preserves the source files and prints each test.

Saved-evidence replay tests the verifier. It does not prove a current deployment
or a new code revision. Full acceptance uses the [runbook](../notebooks/runbook.ipynb):
create resources, perform real training and evaluation, export the results and
recording, then archive and verify owned-resource removal.
