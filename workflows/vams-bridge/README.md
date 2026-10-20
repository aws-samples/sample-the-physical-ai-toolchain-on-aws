# VAMS bridge — run toolchain jobs as VAMS pipelines

This module lets the toolchain's training and evaluation jobs be launched from, report to, and
write results back into the [Visual Asset Management System (VAMS)](https://github.com/awslabs/visual-asset-management-system).
VAMS owns the datasets, checkpoints and reports as versioned assets and decides *when* a job runs
(manual, on file upload, or chained after another step). The toolchain owns *what* the job is and
*where* it runs. Nothing toolchain-specific is added to VAMS, and no toolchain compute runs in the
VAMS account.

> **Status: skeleton.** Proposal and design discussion:
> [issue #36](https://github.com/aws-samples/sample-the-physical-ai-toolchain-on-aws/issues/36).
> The runner state machines, the Lambda stubs and the GR00T bundle are complete enough to read and
> review; the Terraform roots declare every resource but several bodies are marked `TODO`. Nothing
> here has been deployed yet.

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
- Both roles trust only the runner's execution role, with `sts:ExternalId` and an
  `aws:PrincipalArn` condition.
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
    gr00t-finetune/         pipeline.json, workflow.json, templates/{sagemaker-pipeline,batch-n17}.json
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

1. **Toolchain account** — `terraform -chdir=infra/toolchain-account apply`. Outputs the bridge bus
   ARN and the runner execution role ARN.
2. **VAMS account** — `terraform -chdir=infra/vams-account apply` with those two outputs. Reads the
   VAMS integration surface from SSM (`<vams prefix>/integration/*`,
   `<vams prefix>/lambdaFunctions/importGlobalPipelineWorkflow`), creates the roles and log group,
   and registers `vamsSchema/*` by invoking the VAMS import Lambda — the pipeline, its templates and
   a runnable workflow appear in VAMS at the end of the apply.
3. **VAMS 2.7+ only** — add the bridge to the VAMS deployment's `config.json` and redeploy VAMS:

   ```json
   "externalPipelineBridges": [{
     "accountId": "<toolchain account>",
     "roleArnPattern": "arn:aws:iam::<toolchain account>:role/pai-vams-bridge-runner-*",
     "invokeTargets": ["arn:aws:events:<region>:<toolchain account>:event-bus/pai-vams-bridge"]
   }]
   ```

Verify: `vamscli pipeline get -d GLOBAL -p pai-gr00t-finetune --json-output`.

## Components (planned order)

| # | Component | Runner | Status |
|---|---|---|---|
| 1 | GR00T fine-tune — SageMaker Pipeline template + N1.7 AWS Batch template | `runner-sagemaker-pipeline`, `runner-batch` | bundle in this skeleton |
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
