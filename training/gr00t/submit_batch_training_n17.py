"""
Submit and watch a GR00T N1.7 Fine-Tuning Job on AWS Batch

Submits an AWS Batch job that fine-tunes NVIDIA GR00T N1.7-3B on a LeRobot-format
dataset staged in S3, then follows the job through its status transitions and tails
the container's CloudWatch log stream to the terminal.

training/gr00t/launch_training_n17.py is the SageMaker path and is deliberately NOT
extended here: SageMaker owns instance provisioning and log plumbing, while Batch
needs queue/job-definition submission, a status-transition watcher and a
/aws/batch/job log tail. The two have nothing reusable in common beyond the
hyperparameters.

All configuration is passed as containerOverrides.environment entries. The job
definition is never re-registered for an environment change: the Docker image sets
ENTRYPOINT ["/opt/ml/code/entrypoint.sh"], which dispatches on $1, so a Batch
command override cannot redirect it — environment variables are the only seam the
container actually honours. The variable names below are the contract implemented by
containers/gr00t-training/batch-train-entrypoint-n17.sh and train_entrypoint_n17.py.

Job definition selection: this script requires an "n17" job definition. The older
GR00T job definitions in this account (every revision) point at a misspelled ECR
repository that does not exist, so every job submitted against them dies at image
pull. --job-definition therefore refuses any name without "n17" in it.

Usage:
    # Dry run — prints the exact submit-job payload, calls no mutating API:
    python submit_batch_training_n17.py --dry-run \
        --dataset-s3-uri s3://physical-ai-dev-datasets-<ACCOUNT_ID>/groot-data/pick_place_data/ \
        --max-steps 15 --embodiment-config g1_dex3

    # Real smoke test (the deployed compute environment offers g6e.12xlarge only):
    python submit_batch_training_n17.py \
        --job-name g1-dex3-smoke \
        --max-steps 15 --embodiment-config g1_dex3

    # Re-attach to a job already in flight:
    python submit_batch_training_n17.py --watch-only <jobId>
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import boto3

# Live Batch resources for this stack. The queue feeds the compute environment
# physical-ai-dev-gr00t-compute, which currently offers g6e.12xlarge ONLY and sits
# at desiredvCpus 0 when idle — the first job of the day pays an EC2 capacity wait
# plus a ~22.6 GB ECR image pull before any application log line appears. The
# single-GPU g6e.4xlarge target needs a compute environment that does not exist
# yet, so the defaults below (per-device batch for 1 GPU) are ahead of the infra.
DEFAULT_JOB_QUEUE = "physical-ai-dev-gr00t-queue"
DEFAULT_JOB_DEFINITION = "physical-ai-dev-gr00t-n17-training"
DEFAULT_REGION = "us-east-2"

# CloudWatch log group AWS Batch writes container stdout/stderr to.
BATCH_LOG_GROUP = "/aws/batch/job"

# Default dataset prefix inside the dev datasets bucket. The bucket name is
# assembled at runtime from sts get-caller-identity (CONTRIBUTING.md: no hardcoded
# account ids). The Batch instance role grants S3 only on physical-ai-dev-*, so the
# dataset has to live here and not in the read-only workshop-assets bucket.
DATASETS_BUCKET_TEMPLATE = "physical-ai-dev-datasets-{account_id}"
DEFAULT_DATASET_PREFIX = "groot-data/pick_place_data/"

# Single-GPU defaults (g6e.4xlarge, 1x L40S 48 GB), which is the primary target:
# g6e.12xlarge capacity could not be obtained at all. BATCH_SIZE is the GLOBAL
# pre-accumulation batch and the trainer derives per_device = global // num_gpus
# (gr00t/experiment/experiment.py:263-266), so on 1 GPU BATCH_SIZE *is* the
# per-device batch — the 4-GPU pair 8/2 would mean per-device 8 here. 2 x 8 keeps
# the effective optimizer batch at 16 with a per-device batch of 2.
# DERIVED, NOT MEASURED: with TUNE_VISUAL=0 the static footprint is ~22.0 GiB of
# the card's 44.7 GiB; the first real run must report peak VRAM before these are
# raised (4/4 is the next step up, same effective batch, half the optimizer steps).
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM_STEPS = 8

# Vision-tower tuning. Off matches upstream (FinetuneConfig.tune_visual = False)
# and is what makes one 48 GB card viable: trainable parameters drop from ~2.03B
# to 877,747,328 (projector + DiT action head), and the frozen backbone retains no
# activation graph. On requires 4 GPUs so DeepSpeed ZeRO engages.
DEFAULT_TUNE_VISUAL = False

# 15 steps is enough to prove the pipeline end to end without burning GPU time.
DEFAULT_MAX_STEPS = 15

# Terminal Batch job states.
TERMINAL_STATES = ("SUCCEEDED", "FAILED")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp() -> str:
    return _utc_now().strftime("%H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    """Render a duration as e.g. '4m 07s' so phase dwells are readable at a glance."""
    seconds = max(0.0, float(seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def resolve_account_id(region: str) -> str:
    """Derive the account id at runtime; never hardcode it (CONTRIBUTING.md)."""
    return boto3.client("sts", region_name=region).get_caller_identity()["Account"]


def default_dataset_s3_uri(region: str) -> str:
    bucket = DATASETS_BUCKET_TEMPLATE.format(account_id=resolve_account_id(region))
    return f"s3://{bucket}/{DEFAULT_DATASET_PREFIX}"


def build_environment(
    dataset_s3_uri: str,
    max_steps: int,
    embodiment_config: str,
    batch_size: int,
    grad_accum_steps: int,
    tune_visual: bool = DEFAULT_TUNE_VISUAL,
    checkpoint_prefix: str = "",
    hf_token_secret_id: str = "",
) -> list:
    """Build the containerOverrides.environment list.

    Every value is a string: Batch rejects non-string environment values. Optional
    entries are omitted rather than sent empty, so the job definition's own values
    stay in effect when a flag is not passed.
    """
    environment = [
        {"name": "DATASET_S3_URI", "value": dataset_s3_uri},
        {"name": "MAX_STEPS", "value": str(max_steps)},
        {"name": "EMBODIMENT_CONFIG", "value": embodiment_config},
        {"name": "BATCH_SIZE", "value": str(batch_size)},
        {"name": "GRAD_ACCUM_STEPS", "value": str(grad_accum_steps)},
        # Always sent, never omitted: which parameters train is the difference
        # between a job that fits one card and one that does not, so it should be
        # explicit in the submission and in the log banner rather than inherited.
        {"name": "TUNE_VISUAL", "value": "1" if tune_visual else "0"},
    ]
    if checkpoint_prefix:
        environment.append({"name": "CHECKPOINT_PREFIX", "value": checkpoint_prefix})
    if hf_token_secret_id:
        # Secret *id*, never the value. batch-train resolves it with the instance
        # role only when HF_TOKEN is unset, and the base model is not gated, so an
        # unresolvable lookup is a warning rather than a failure.
        environment.append({"name": "HF_TOKEN_SECRET_ID", "value": hf_token_secret_id})
    return environment


def _job_definition_gpus(job_definition: str, region: str):
    """Return the GPU count the job definition requests, or None if unreadable.

    Read-only (describe-job-definitions). Returns None rather than raising so a
    permissions or naming problem does not block a submission that would
    otherwise be fine.
    """
    try:
        client = boto3.client("batch", region_name=region)
        name = job_definition.split(":")[0]
        revisions = client.describe_job_definitions(
            jobDefinitionName=name, status="ACTIVE"
        )["jobDefinitions"]
        if not revisions:
            return None
        latest = max(revisions, key=lambda d: d["revision"])
        for req in latest.get("containerProperties", {}).get("resourceRequirements", []):
            if req.get("type") == "GPU":
                return int(req["value"])
    except Exception:  # noqa: BLE001 - advisory check only
        return None
    return None


def _check_batch_divides_gpus(job_definition: str, batch_size: int, region: str, parser) -> None:
    """Refuse a global batch the trainer cannot split across the job's GPUs."""
    gpus = _job_definition_gpus(job_definition, region)
    if not gpus:
        return
    if batch_size % gpus != 0 or batch_size < gpus:
        suggestion = gpus * 2
        parser.error(
            f"--batch-size {batch_size} is not usable on {job_definition}, which "
            f"requests {gpus} GPU(s). BATCH_SIZE is the GLOBAL batch and the trainer "
            f"asserts global_batch_size % num_gpus == 0 before dividing it into the "
            f"per-device batch, so this job would fail at startup. Use "
            f"--batch-size {suggestion} --grad-accum-steps {max(1, 16 // suggestion)} "
            f"to keep the effective batch at 16, or point --job-definition at a "
            f"{batch_size}-GPU-or-fewer shape."
        )


def build_payload(
    job_name: str,
    job_queue: str,
    job_definition: str,
    environment: list,
) -> dict:
    """Build the batch:SubmitJob payload.

    containerOverrides carries environment entries and nothing else: no command, no
    resourceRequirements. The job definition owns the vCPU / memory / GPU shape and
    the batch-train command. The command is left alone deliberately: the image's
    ENTRYPOINT dispatches on the basename of $1, so an override that is not
    *-/batch-train silently selects the SageMaker path, which downloads no dataset
    and uploads no checkpoint.
    """
    return {
        "jobName": job_name,
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        "containerOverrides": {"environment": environment},
    }


def _describe_job(batch_client, job_id: str) -> dict:
    jobs = batch_client.describe_jobs(jobs=[job_id])["jobs"]
    if not jobs:
        raise RuntimeError(f"Batch returned no job for id {job_id}")
    return jobs[0]


def _tail_log_stream(logs_client, log_stream: str, next_token: str):
    """Print any new events from the job's log stream, return the updated cursor.

    Uses get-log-events with startFromHead and the nextForwardToken cursor: that is
    the only read that guarantees neither duplicates nor gaps. filter-log-events
    pages by time and would re-emit lines that share a millisecond.
    """
    kwargs = {
        "logGroupName": BATCH_LOG_GROUP,
        "logStreamName": log_stream,
        "limit": 10000,
    }
    if next_token:
        kwargs["nextToken"] = next_token
    else:
        kwargs["startFromHead"] = True

    try:
        response = logs_client.get_log_events(**kwargs)
    except logs_client.exceptions.ResourceNotFoundException:
        # The stream is created when the container starts; until then this is normal.
        return next_token, []

    events = response.get("events", [])
    for event in events:
        print(event["message"].rstrip())
    sys.stdout.flush()
    return response.get("nextForwardToken", next_token), events


def watch_job(batch_client, logs_client, job_id: str, poll_interval: int = 10) -> dict:
    """Follow a job to a terminal state, printing status transitions and log lines.

    Phase dwell times are printed as each transition happens because they are the
    numbers the workshop guide quotes:
      RUNNABLE dwell  = EC2 capacity wait (cold compute environment scale-up)
      STARTING dwell  = ~22.6 GB ECR image pull onto the fresh instance
      RUNNING dwell   = dataset download + stats generation + training
    """
    print(f"[{_stamp()}] watching job {job_id}")
    print(f"[{_stamp()}] phase legend: RUNNABLE=capacity wait, STARTING=image pull, RUNNING=train")

    phase_durations = {}
    current_status = None
    phase_started = time.monotonic()
    log_stream = None
    next_token = ""
    job = None

    while True:
        job = _describe_job(batch_client, job_id)
        status = job["status"]

        if status != current_status:
            now = time.monotonic()
            if current_status is not None:
                dwell = now - phase_started
                phase_durations[current_status] = phase_durations.get(current_status, 0.0) + dwell
                print(f"[{_stamp()}] {current_status} -> {status}  ({current_status} dwell {_fmt_duration(dwell)})")
            else:
                print(f"[{_stamp()}] status {status}")
            current_status = status
            phase_started = now

        if log_stream is None:
            log_stream = job.get("container", {}).get("logStreamName")
            if log_stream:
                print(f"[{_stamp()}] log stream: {BATCH_LOG_GROUP} :: {log_stream}")
                print("-" * 70)

        if log_stream:
            next_token, _ = _tail_log_stream(logs_client, log_stream, next_token)

        if status in TERMINAL_STATES:
            dwell = time.monotonic() - phase_started
            phase_durations[status] = phase_durations.get(status, 0.0) + dwell
            break

        time.sleep(poll_interval)

    if log_stream:
        # Drain whatever the container wrote between the last poll and exit.
        for _ in range(6):
            next_token, events = _tail_log_stream(logs_client, log_stream, next_token)
            if not events:
                break
            time.sleep(5)
        print("-" * 70)

    container = job.get("container", {})
    result = {
        "job_id": job_id,
        "job_name": job.get("jobName"),
        "status": job["status"],
        "status_reason": job.get("statusReason"),
        "exit_code": container.get("exitCode"),
        "container_reason": container.get("reason"),
        "log_stream": log_stream,
        "created_at": job.get("createdAt"),
        "started_at": job.get("startedAt"),
        "stopped_at": job.get("stoppedAt"),
        "phase_dwell_seconds": {k: round(v, 1) for k, v in phase_durations.items()},
    }

    print(f"[{_stamp()}] final status: {result['status']}")
    for phase, seconds in result["phase_dwell_seconds"].items():
        print(f"  {phase:<12} {_fmt_duration(seconds)}")
    if job.get("createdAt") and job.get("stoppedAt"):
        total = (job["stoppedAt"] - job["createdAt"]) / 1000.0
        print(f"  {'TOTAL':<12} {_fmt_duration(total)}  (createdAt -> stoppedAt, from Batch)")
    if job.get("startedAt") and job.get("stoppedAt"):
        runtime = (job["stoppedAt"] - job["startedAt"]) / 1000.0
        print(f"  {'container':<12} {_fmt_duration(runtime)}  (startedAt -> stoppedAt, from Batch)")
    if result["status_reason"]:
        print(f"  reason: {result['status_reason']}")
    return result


def submit_training_job(
    job_name: str,
    dataset_s3_uri: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    embodiment_config: str = "g1_dex3",
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM_STEPS,
    tune_visual: bool = DEFAULT_TUNE_VISUAL,
    checkpoint_prefix: str = "",
    hf_token_secret_id: str = "",
    job_queue: str = DEFAULT_JOB_QUEUE,
    job_definition: str = DEFAULT_JOB_DEFINITION,
    region: str = DEFAULT_REGION,
    dry_run: bool = False,
) -> dict:
    """Submit one GR00T N1.7 Batch training job. Returns the submit result or the payload."""
    environment = build_environment(
        dataset_s3_uri=dataset_s3_uri,
        max_steps=max_steps,
        embodiment_config=embodiment_config,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        tune_visual=tune_visual,
        checkpoint_prefix=checkpoint_prefix,
        hf_token_secret_id=hf_token_secret_id,
    )
    payload = build_payload(job_name, job_queue, job_definition, environment)

    if dry_run:
        print("DRY RUN - no AWS call made. submit-job payload:")
        print(json.dumps(payload, indent=2))
        print()
        print(f"  region:         {region}")
        print(f"  job queue:      {job_queue}")
        print(f"  job definition: {job_definition}")
        print("  overrides:      containerOverrides.environment only "
              f"({len(environment)} entries, no command, no resourceRequirements)")
        print()
        print("Next step: drop --dry-run to submit. The compute environment starts from")
        print("desiredvCpus 0, so expect an EC2 capacity wait plus a ~22.6 GB image pull")
        print("before the first application log line.")
        return {"dry_run": True, "payload": payload, "region": region}

    batch_client = boto3.client("batch", region_name=region)
    response = batch_client.submit_job(**payload)
    job_id = response["jobId"]

    print(f"[{_stamp()}] submitted {response['jobName']}")
    print(f"  job id:         {job_id}")
    print(f"  job queue:      {job_queue}")
    print(f"  job definition: {job_definition}")
    print(f"  dataset:        {dataset_s3_uri}")
    print(f"  embodiment:     {embodiment_config}")
    print(f"  tune visual:    {tune_visual}")
    print(f"  max steps:      {max_steps}  (batch {batch_size} x grad accum {grad_accum_steps})")
    print()
    print(f"Next step: re-attach any time with --watch-only {job_id}, or check it manually:")
    print(f"  aws batch describe-jobs --jobs {job_id} --region {region} --query 'jobs[0].status'")
    print()
    return {"job_id": job_id, "job_name": response["jobName"], "payload": payload}


def main():
    parser = argparse.ArgumentParser(
        description="Submit and watch a GR00T N1.7 fine-tuning job on AWS Batch"
    )
    parser.add_argument(
        "--dataset-s3-uri",
        default=None,
        help="S3 URI of the staged LeRobot dataset (trailing slash). "
        "Default: s3://physical-ai-dev-datasets-<ACCOUNT_ID>/" + DEFAULT_DATASET_PREFIX,
    )
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--embodiment-config",
        default="g1_dex3",
        choices=["g1_dex3", "ur3"],
        help="Modality config shipped in the container at /opt/ml/code/ (default g1_dex3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Global batch size, pre-accumulation (default {DEFAULT_BATCH_SIZE}). "
        "On a 1-GPU instance this is also the per-device batch",
    )
    parser.add_argument(
        "--tune-visual",
        action="store_true",
        help="Train the vision tower too (~2.03B trainable params instead of "
        "877,747,328). Needs 4 GPUs so DeepSpeed ZeRO engages; off by default, "
        "matching upstream",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=DEFAULT_GRAD_ACCUM_STEPS,
        help=f"Gradient accumulation steps (default {DEFAULT_GRAD_ACCUM_STEPS})",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        default="",
        help="S3 key prefix for checkpoints. Unset uses the container default "
        "checkpoints/groot-n17/<AWS_BATCH_JOB_ID>",
    )
    parser.add_argument(
        "--hf-token-secret-id",
        default="",
        help="Secrets Manager secret id/ARN holding the HuggingFace token. Optional: "
        "the base model is not gated, so an unresolvable lookup is only a warning",
    )
    parser.add_argument("--job-name", default=None, help="Job name suffix; a timestamp is appended")
    parser.add_argument("--job-queue", default=DEFAULT_JOB_QUEUE)
    parser.add_argument(
        "--job-definition",
        default=DEFAULT_JOB_DEFINITION,
        help="Must be an n17 job definition; the older GR00T job definitions in this "
        "account point at a misspelled ECR repo that does not exist and fail at image pull",
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--poll-interval", type=int, default=10, help="describe-jobs poll seconds")
    parser.add_argument("--dry-run", action="store_true", help="Print the payload, call no AWS API")
    parser.add_argument("--no-watch", action="store_true", help="Submit and exit without watching")
    parser.add_argument(
        "--watch-only",
        default=None,
        metavar="JOB_ID",
        help="Skip submission and attach to an in-flight job",
    )
    args = parser.parse_args()

    # Guard against the known-bad job definitions without naming them: they all
    # reference an ECR repository that does not exist, so every job fails at image
    # pull with no application log output at all.
    if "n17" not in args.job_definition:
        parser.error(
            f"refusing job definition {args.job_definition!r}: only n17 job definitions "
            "point at a real ECR image. Pass an n17 job definition."
        )

    if args.watch_only:
        batch_client = boto3.client("batch", region_name=args.region)
        logs_client = boto3.client("logs", region_name=args.region)
        result = watch_job(batch_client, logs_client, args.watch_only, args.poll_interval)
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["status"] == "SUCCEEDED" else 1

    dataset_s3_uri = args.dataset_s3_uri
    if not dataset_s3_uri:
        if args.dry_run:
            # sts is a read-only call and safe in a dry run, but keep the dry run
            # usable without credentials by falling back to the documented placeholder.
            try:
                dataset_s3_uri = default_dataset_s3_uri(args.region)
            except Exception as exc:  # noqa: BLE001 - any credential/network error
                print(f"  NOTE: could not derive the account id ({exc}); using a placeholder URI.")
                dataset_s3_uri = (
                    f"s3://{DATASETS_BUCKET_TEMPLATE.format(account_id='ACCOUNT_ID')}/"
                    f"{DEFAULT_DATASET_PREFIX}"
                )
        else:
            dataset_s3_uri = default_dataset_s3_uri(args.region)

    if "workshop-assets" in dataset_s3_uri:
        parser.error(
            "the Batch instance role grants S3 only on physical-ai-dev-*, so a "
            "workshop-assets URI will fail with AccessDenied. Stage the dataset with "
            "training/gr00t/prepare_g1_dex3_dataset.py --upload first."
        )

    if not args.dry_run:
        # The GPU count comes from the job definition, and BATCH_SIZE is the
        # GLOBAL batch: the trainer asserts global_batch_size % num_gpus == 0 and
        # divides to get the per-device batch. The single-GPU default
        # (BATCH_SIZE=2) is therefore invalid on a 4-GPU job definition — 2 % 4,
        # and per-device would round to 0. Catch it here rather than after a
        # capacity wait and a 22.6 GB image pull. Read-only; skipped in --dry-run
        # so a dry run still needs no AWS call at all.
        _check_batch_divides_gpus(args.job_definition, args.batch_size, args.region, parser)

    timestamp = _utc_now().strftime("%Y%m%d-%H%M%S")
    base_name = args.job_name or "gr00t-n17-finetune"
    job_name = f"{base_name}-{timestamp}"

    result = submit_training_job(
        job_name=job_name,
        dataset_s3_uri=dataset_s3_uri,
        max_steps=args.max_steps,
        embodiment_config=args.embodiment_config,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        tune_visual=args.tune_visual,
        checkpoint_prefix=args.checkpoint_prefix,
        hf_token_secret_id=args.hf_token_secret_id,
        job_queue=args.job_queue,
        job_definition=args.job_definition,
        region=args.region,
        dry_run=args.dry_run,
    )

    if args.dry_run or args.no_watch:
        return 0

    batch_client = boto3.client("batch", region_name=args.region)
    logs_client = boto3.client("logs", region_name=args.region)
    watched = watch_job(batch_client, logs_client, result["job_id"], args.poll_interval)
    print(json.dumps(watched, indent=2, default=str))
    return 0 if watched["status"] == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())
