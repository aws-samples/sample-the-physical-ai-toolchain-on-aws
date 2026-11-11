# SPDX-License-Identifier: MIT-0
"""AWS Batch entrypoint: score an Isaac Lab G1 velocity checkpoint with Isaac Lab Arena.

Environment variables:
  DATA_BUCKET         workshop data bucket (required)
  CHECKPOINT_PREFIX   checkpoints/<Isaac Lab training Batch job ID> (required)
  REGION              AWS Region (default us-east-2)
  NUM_EPISODES        episodes per run (default 16)
  NUM_ENVS            parallel robots per run (default 16)
  RESULTS_PREFIX      S3 prefix for the report (default arena-results/<this Batch job ID>)

Downloads the newest ``model_<N>.pt`` under CHECKPOINT_PREFIX together with its run
folder (Arena reads ``params/agent.yaml`` next to the checkpoint), runs the four-run
Experiment headless, prints one ``G1VEL_RUN`` line per run, and uploads the JSON results
and the HTML report to RESULTS_PREFIX. Every stage prints a ``G1VEL_`` marker line.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import boto3

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK_DIR", "/tmp/g1vel"))
PYTHON = os.environ.get("ARENA_PYTHON", "/isaac-sim/python.sh")


def log(tag: str, msg: str) -> None:
    print(f"G1VEL_{tag} {msg}", flush=True)


def stop(msg: str) -> None:
    log("FAIL", msg)
    sys.exit(1)


def newest_checkpoint(s3, bucket: str, prefix: str) -> str:
    """Return the key of the highest-numbered model_<N>.pt under prefix."""
    best_key, best_iter = None, -1
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            m = re.search(r"/model_(\d+)\.pt$", obj["Key"])
            if m and int(m.group(1)) > best_iter:
                best_key, best_iter = obj["Key"], int(m.group(1))
    if best_key is None:
        stop(f"no model_<N>.pt under s3://{bucket}/{prefix}/")
    return best_key


def download_run_folder(s3, bucket: str, ckpt_key: str, dest: Path) -> Path:
    """Download the checkpoint and its params/ folder; return the local checkpoint path."""
    run_prefix = ckpt_key.rsplit("/", 1)[0] + "/"
    keys = [ckpt_key]
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=run_prefix + "params/"):
        keys += [o["Key"] for o in page.get("Contents", [])]
    for key in keys:
        local = dest / key[len(run_prefix):]
        local.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(local))
    ckpt = dest / ckpt_key.rsplit("/", 1)[1]
    if not (dest / "params" / "agent.yaml").is_file():
        stop(f"params/agent.yaml missing next to {ckpt_key}; the training job did not upload its run folder")
    return ckpt


def summarize(result_path: Path) -> list[dict]:
    """Return one summary per run, computed from the per-episode records."""
    data = json.loads(result_path.read_text())
    rows = []
    for name, run in data["runs"].items():
        episodes = [e for r in run.get("rebuilds", []) for e in r.get("episodes", [])]
        n = len(episodes)

        def mean(key: str) -> float | None:
            values = [float(e[key]) for e in episodes if e.get(key) is not None]
            return round(sum(values) / len(values), 3) if values else None

        rows.append({
            "run": name,
            "status": run.get("status"),
            "episodes": n,
            "success_rate": mean("success"),
            "fall_rate": mean("fell"),
            "mean_lin_vel_xy_error": mean("lin_vel_xy_error"),
            "mean_ang_vel_z_error": mean("ang_vel_z_error"),
        })
    return rows


def write_summary(rows: list[dict], path: Path) -> None:
    """Write the per-run summary as a small CSV next to the report."""
    keys = ["run", "status", "episodes", "success_rate", "fall_rate", "mean_lin_vel_xy_error", "mean_ang_vel_z_error"]
    lines = [",".join(keys)] + [",".join("" if r[k] is None else str(r[k]) for k in keys) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    bucket = os.environ.get("DATA_BUCKET") or stop("DATA_BUCKET is not set")
    prefix = os.environ.get("CHECKPOINT_PREFIX") or stop("CHECKPOINT_PREFIX is not set")
    region = os.environ.get("REGION", "us-east-2")
    episodes = int(os.environ.get("NUM_EPISODES", "16"))
    num_envs = int(os.environ.get("NUM_ENVS", "16"))
    job_id = os.environ.get("AWS_BATCH_JOB_ID", "local")
    results_prefix = os.environ.get("RESULTS_PREFIX", f"arena-results/{job_id}").rstrip("/")
    t0 = time.time()
    log("START", f"job={job_id} checkpoint_prefix={prefix} episodes={episodes} num_envs={num_envs}")

    s3 = boto3.client("s3", region_name=region)
    ckpt_key = newest_checkpoint(s3, bucket, prefix)
    ckpt = download_run_folder(s3, bucket, ckpt_key, WORK / "ckpt")
    log("CKPT", f"s3://{bucket}/{ckpt_key}")

    out_dir = WORK / "out"
    cmd = [
        PYTHON, str(HERE / "arena_run.py"), "experiment_runner", "--headless",
        "--experiment_config", str(HERE / "g1_velocity_experiment.yaml"),
        "--experiment_output_directory", str(out_dir),
        f"shared.policy.checkpoint_path={ckpt}",
        f"shared.rollout_limit.num_episodes={episodes}",
        f"shared.environment_builder.num_envs={num_envs}",
    ]
    log("ARENA", " ".join(cmd[1:]))
    rc = subprocess.call(cmd, cwd=str(HERE))
    result = out_dir / "arena_experiment_result.json"
    if rc != 0 or not result.is_file():
        stop(f"experiment_runner exited with rc={rc}; see the lines above")

    rows = summarize(result)
    write_summary(rows, out_dir / "summary.csv")
    for row in rows:
        log("RUN", json.dumps(row))

    uploaded = 0
    for path in out_dir.rglob("*"):
        if path.is_file():
            s3.upload_file(str(path), bucket, f"{results_prefix}/{path.relative_to(out_dir)}")
            uploaded += 1
    log("RESULTS", f"s3://{bucket}/{results_prefix}/ ({uploaded} files)")
    log("EXIT", f"rc=0 seconds={int(time.time() - t0)}")


if __name__ == "__main__":
    main()
