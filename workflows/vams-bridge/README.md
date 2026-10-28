# VAMS bridge — run toolchain jobs as VAMS pipelines

This module lets the toolchain's training and evaluation jobs be launched from, report to, and
write results back into the [Visual Asset Management System (VAMS)](https://github.com/awslabs/visual-asset-management-system).
VAMS owns the datasets, checkpoints and reports as versioned assets and decides *when* a job runs
(manual, on file upload, or chained after another step). The toolchain owns *what* the job is and
*where* it runs. Nothing toolchain-specific is added to VAMS, and no toolchain compute runs in the
VAMS account.

**The bridge is optional.** Deploy it only if you run VAMS and want to launch toolchain jobs from
it. No other toolchain component depends on it, and nothing outside `workflows/vams-bridge/`
references it: Foundation, GR00T and the other components deploy and run the same with or without
it. It takes a few Foundation and GR00T values as Terraform inputs (the SageMaker execution role,
the Batch job queue and job definition, the datasets and checkpoints buckets) and creates only its
own resources; it modifies none of theirs. On the VAMS side it adds two roles, a log group and a
pipeline registration, plus one `externalPipelineBridges` entry in the VAMS configuration (empty by
default). [Remove the bridge](#remove-the-bridge) takes all of it out again.

> **Status: preview.** Proposal and design discussion:
> [issue #36](https://github.com/aws-samples/sample-the-physical-ai-toolchain-on-aws/issues/36).
> The AWS Batch engine (`pai-gr00t-finetune-batch`, GR00T N1.7) has run end to end in direct mode
> against a VAMS 2.7 pre-release (`development`): a LeRobot v2 dataset asset in, the fine-tuned
> model directory and its metadata back on the same asset. The SageMaker Pipeline engine
> (`pai-gr00t-finetune`) registers but has not been run, because it needs the
> `groot-finetune-pipeline` and the N1.6 image; relay mode (VAMS 2.6.x) has not been exercised.
> Offline tests for the runner Lambdas: `python3 -m unittest discover -s runner/tests` (needs
> `boto3`).

## How it works

```
VAMS account                                        Toolchain account
────────────                                        ─────────────────
VAMS workflow (Step Functions)
  │ events:putEvents.waitForTaskToken
  ▼
  ──────────────────────────────────────────────▶  pai-vams-bridge bus ── rule ──▶ Runner state machine
  Detail.body = { TaskToken, inputManifestS3Location,   │
                  inputConfigurationS3Location, ... }   ├─ ReadManifest   manifest + config.json → job request
  ◀── PutEvents pipeline.execution.register ──────────  ├─ Register       runner ARN, job ARN, log group/prefix
                                                        ├─ Submit         StartPipelineExecution | CreateTrainingJob | SubmitJob
  ◀── SendTaskHeartbeat (each poll) ─────────────────── ├─ Wait → Describe → Heartbeat → Choice   (loop)
                                                        │     heartbeat rejected ⇒ Stop*  ⇒ Fail
                                                        ├─ Collect        job output → VAMS output prefixes
  ◀── SendTaskSuccess | SendTaskFailure ─────────────── └─ Done
  ▼
VAMS versions the results onto the asset
```

- Every callback to VAMS (`sendTaskHeartbeat`, `sendTaskSuccess`/`Failure`, `putEvents` to the VAMS
  orchestration bus) runs as the VAMS-account **control role** through Step Functions task
  `Credentials.RoleArn` — no Lambda ever handles the task token.
- Input staging and output copy run as the VAMS-account **data role**, scoped to the input asset
  prefixes (read) and the run's output prefix (write).
- Both roles trust only the runner's execution role (assumed through a task state's
  `Credentials.RoleArn`, which carries no ExternalId, so the caller is pinned by `aws:PrincipalArn`)
  and the runner Lambdas' role (which passes `sts:ExternalId`).
- The runner is chosen by the VAMS pipeline's EventBridge detail-type, so each engine is its own
  VAMS pipeline (`pai-gr00t-finetune` → SageMaker Pipeline, `pai-gr00t-finetune-batch` → AWS Batch);
  a template's `engine` must match the pipeline it is registered on, and ReadManifest rejects a
  mismatch before any compute starts.
- A rejected heartbeat means VAMS aborted or timed out the step; the runner stops the job and fails.
- Job names derive from the VAMS pipeline-execution id, so an at-least-once event delivery cannot
  start a second job.

The VAMS-side contract (task body, manifest schema, output-path rules, callback/heartbeat/abort,
registration event, `vamsSchema/` bundle format) is documented in the VAMS repo:
[Externally owned pipelines (bridge pattern)](https://github.com/awslabs/visual-asset-management-system/blob/development/documentation/docusaurus-site/docs/pipelines/custom-pipelines.md).

## Layout

```
workflows/vams-bridge/
  README.md
  infra/
    toolchain-account/      bridge bus + rules, runner state machines, Lambdas, runner execution role
    vams-account/           data + control roles, bridge log group, VamsSchemaRegistration,
                            optional relay bus (VAMS < 2.7 only)
  runner/
    read_manifest.py        VAMS manifest + config.json → job request; validates; fails fast
    collect.py              job outputs → VAMS output prefixes; metadata envelope
    manifestHelper.py       vendored byte-identical from the VAMS repo (backendPipelines/*/lambda/)
  statemachines/
    runner-sagemaker-pipeline.asl.json
    runner-sagemaker-training.asl.json
    runner-batch.asl.json
  vamsSchema/
    gr00t-finetune/         pipeline.json, workflow.json, templates/sagemaker-pipeline.json
    gr00t-finetune-batch/   pipeline.json, workflow.json, templates/batch-n17.json
```

## Two connection modes

| | VAMS 2.7+ (direct) | VAMS 2.6.x (relay) |
|---|---|---|
| VAMS pipeline `executionConfig.eventBridge.busArn` | the toolchain bridge bus | a relay bus in the VAMS account |
| VAMS-account resources | two roles, one log group, registration | + relay bus, rule, rule role |
| VAMS `config.json` | one `app.pipelines.externalPipelineBridges[]` entry naming this account, the runner role pattern and the bridge bus ARN | nothing |
| Abort from the VAMS UI | native (`sageMakerTrainingJob` / `sageMakerPipelineExecution` / `batchJob`) | heartbeat rejection only |

`infra/vams-account` takes `vams_relay = true|false`. The toolchain-account root is the same in
both modes.

## Deploy order

Before you start you need a VAMS deployment (2.7+ for direct mode, 2.6.x through the relay), the
toolchain's Foundation, and the GR00T compute the engine runs on: the AWS Batch compute environment
and N1.7 job definition from `isaac-gr00t-on-aws/infra/` for `pai-gr00t-finetune-batch`, or the
`groot-finetune-pipeline` for `pai-gr00t-finetune`. The N1.7 job loads the gated
`nvidia/Cosmos-Reason2-2B` backbone from Hugging Face and fails with `401 Unauthorized` without a
token, so store one whose account has accepted that model's terms in `physical-ai/hf-token`
([Batch guide, step 5](../../isaac-gr00t-on-aws/batch-training-guide.md#step-5-store-the-huggingface-token));
see also [Secrets and tags](#secrets-and-tags).

The two roots share an `environment` value and derive each other's resource names from it, so
neither consumes the other's outputs and they can be applied in either order. Generate one
ExternalId (any UUID) and pass the same value to both as `vams_role_external_id`.

1. **VAMS account** — `terraform -chdir=infra/vams-account apply`. Inputs: `aws_region`,
   `vams_resource_param_prefix` (the VAMS SSM prefix `/<name>-<baseStackName>/resourceNames`),
   `toolchain_account_id`, `toolchain_region`, `vams_asset_bucket_names`, optional
   `vams_kms_key_arn`, `vams_relay` + `vams_deployment_name` for VAMS 2.6.x. Reads the VAMS
   integration surface from SSM (`<prefix>/integration/*`,
   `<prefix>/lambdaFunctions/importGlobalPipelineWorkflow`, `<prefix>/s3Buckets/artefacts`),
   creates the two roles and the log group, uploads `vamsSchema/*` to the VAMS artefacts bucket and
   registers it through the VAMS import Lambda. The pipeline, its templates and a runnable workflow
   exist in VAMS at the end of the apply; the outputs include the two SSM values the next root needs.
2. **Toolchain account** — `terraform -chdir=infra/toolchain-account apply`. Inputs: `aws_region`,
   `vams_account_id`, `vams_region`, `vams_orchestration_bus_arn` and
   `vams_workflow_execution_role_arn` (both from step 1's outputs; in relay mode pass
   `relay_role_arn` for the second), the Foundation's `sagemaker_execution_role_arn`,
   `batch_job_queue_arn`, `batch_job_definition_name`, `datasets_bucket`, `checkpoints_bucket`.
3. **VAMS 2.7+ only** — add the bridge to the VAMS deployment's `config.json` and redeploy VAMS.
   Step 2's `vams_config_snippet` output is the exact entry:

   ```json
   "externalPipelineBridges": [{
     "accountId": "<toolchain account>",
     "roleArnPattern": "arn:aws:iam::<toolchain account>:role/pai-vams-bridge-runner-*",
     "invokeTargets": ["arn:aws:events:<region>:<toolchain account>:event-bus/pai-vams-bridge-<environment>"]
   }]
   ```

Verify: `vamscli pipeline get -d GLOBAL -p pai-gr00t-finetune --json-output`.

Limits worth knowing before the first run: ReadManifest streams the dataset into the Foundation
datasets bucket inside one Lambda invocation and rejects the run if the copy cannot finish in time,
so a dataset beyond what 15 minutes can move needs a pre-staged copy; Collect moves the artefacts
back the same way.

## Run a job from VAMS

1. **Put the dataset in VAMS.** Create an asset and upload a LeRobot v2 dataset to it at its
   relative paths: the parquet episodes, the mp4 videos and `meta/*.json`
   (`training/gr00t/convert_zarr_to_lerobot.py` converts Zarr recordings such as the UR3 sample).
   Use the VAMS web UI, or:

   ```bash
   vamscli assets create -d <database> --name ur3-pick-place --description "UR3 LeRobot v2 dataset"
   vamscli sync file push ./ur3_lerobot -d <database> -a <asset id>
   ```

2. **Start the workflow.** In the VAMS web UI, open the asset, select its root in the file
   manager and choose **Automation → Execute Workflow**, then pick
   *PAI Toolchain - GR00T fine-tune (AWS Batch)* and fill in the form. From the CLI:

   ```bash
   vamscli workflow execute --workflow-database-id GLOBAL -w pai-gr00t-finetune-batch \
       --input-file <database>:<asset id>:/ \
       --pipeline-parameters '{"pai-gr00t-finetune-batch": {"templateId": "batch-n17",
         "templateTags": [{"key": "MAX_STEPS", "value": 2000}]}}'
   ```

   The workflow takes the whole asset. The form fields are the template's tags; a field left empty
   takes its default:

   | Workflow (template) | Fields (default) |
   |---|---|
   | `pai-gr00t-finetune-batch` (`batch-n17`) | `MAX_STEPS` (100), `BATCH_SIZE` (2), `GRAD_ACCUM_STEPS` (8), `EMBODIMENT_CONFIG` (`ur3` or `g1_dex3`), `TUNE_VISUAL` (false) |
   | `pai-gr00t-finetune` (`sagemaker-pipeline`) | `MAX_STEPS` (5000), `BATCH_SIZE` (32), `BASE_MODEL` (`nvidia/GR00T-N1.5-3B` or `-N1.6-3B`), `INSTANCE_TYPE` (`ml.g5.12xlarge`, `ml.g6e.12xlarge`, `ml.p4d.24xlarge`) |

   ReadManifest rejects a run before the job is submitted when the dataset fails validation or the
   batch size does not divide across the job's GPUs.

3. **Watch it.** The VAMS Executions page (or `vamscli execution list -w pai-gr00t-finetune-batch
   --workflow-database-id GLOBAL`, then `vamscli execution details <execution id>
   --include-sub-executions`) shows the step running, with the runner and the toolchain job
   registered as its sub-processes. Aborting the execution in VAMS stops the toolchain job. A Batch
   job can sit `RUNNABLE` for a while when GPU capacity is short; the VAMS step keeps waiting
   through the heartbeat, up to the pipeline's 16-hour task timeout.

4. **Get the results.** On success VAMS adds the run's output to the asset under one folder,
   `/<job name>/`, and records `pai_*` asset metadata including `pai_outputFolder`. For the Batch
   engine the folder is `/pai-batch-<id>/` and holds the deployable model directory (weights,
   `processor/`, `experiment_cfg/` and the embodiment folder); intermediate checkpoints and
   optimizer state stay in the toolchain's checkpoints bucket under `vams/`, and the metadata adds
   `pai_batchJobId`. For the SageMaker Pipeline engine it holds `model.tar.gz`, and the metadata
   adds `pai_pipelineExecutionArn`, `pai_trainingJobName`, `pai_trainingSeconds` and
   `pai_modelPackageArn`. The job's log tail is copied into the bridge log group in the VAMS
   account. A failed job marks the VAMS execution `FAILED` with cause `PaiToolchainJobFailed`,
   writes nothing to the asset, and leaves the job's logs in the toolchain account.

## Remove the bridge

1. **Archive the VAMS registrations.** Registration is a one-time invocation of the VAMS import
   Lambda, so `terraform destroy` does not remove it. Archive the workflows, then the pipelines
   (an archive can be undone with `unarchive`):

   ```bash
   vamscli workflow delete -d GLOBAL -w pai-gr00t-finetune-batch
   vamscli workflow delete -d GLOBAL -w pai-gr00t-finetune
   vamscli pipeline delete -d GLOBAL -p pai-gr00t-finetune-batch
   vamscli pipeline delete -d GLOBAL -p pai-gr00t-finetune
   ```

2. **Destroy both roots** with the inputs you applied them with:
   `terraform -chdir=infra/toolchain-account destroy`, then
   `terraform -chdir=infra/vams-account destroy`.
3. **VAMS 2.7+ only:** remove the bridge's `externalPipelineBridges` entry from the VAMS
   `config.json` and redeploy VAMS.

Models already written to VAMS assets stay there. Staged datasets and job outputs under `vams/` in
the Foundation datasets and checkpoints buckets are not removed; delete that prefix if you no
longer need it. No other toolchain component needs a change.

## Components (planned order)

| # | Component | Runner | Status |
|---|---|---|---|
| 1 | GR00T fine-tune — `pai-gr00t-finetune` (SageMaker Pipeline) and `pai-gr00t-finetune-batch` (N1.7 AWS Batch) | `runner-sagemaker-pipeline`, `runner-batch` | bundles in this module |
| 2 | Ingest Zarr → LeRobot v2 (after containerising `convert_zarr_to_lerobot.py`) | `runner-batch` | planned |
| 3 | Isaac Lab RL — SageMaker and Batch templates | `runner-sagemaker-training`, `runner-batch` | planned |
| 4 | Arena evaluation, chained after GR00T | `runner-sagemaker-pipeline` | planned |
| 5 | DreamZero | same runner; bundle lives in its own repo | planned |
| 6 | Cosmos | only once the SageMaker path is validated end to end | parked |

Strands Agents, Isaac Sim and OSMO are not pipelines; see issue #36.

## Secrets and tags

A template's `tagSchema` fields become VAMS execute-form fields and are stored on the execution
record. **Never declare a secret as a tag.** The runner resolves `HF_TOKEN` and similar values from
this account's Secrets Manager at submit time.
