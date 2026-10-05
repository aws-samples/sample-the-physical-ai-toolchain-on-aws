#!/usr/bin/env python3
"""Submit a parallel array job to AWS Batch and watch all children to completion.

Usage:
    python scripts/submit_array_job.py <size> [--no-watch] [--timeout SECONDS]

Default: submit and watch until all children reach a terminal state, printing
the per-state breakdown. Pass --no-watch to submit and print only the job id.
"""

import argparse
import sys
import time

import boto3

QUEUE = "IsaacSim-SimQueue"
JOB_DEF = "IsaacSim-SimJob"
# Guardrail: reject oversized array submissions before they scale out GPU EC2.
# Overridable by the operator, but a sane default ceiling prevents a typo or a
# compromised credential from launching a runaway (cost/DoS) array job.
MAX_ARRAY_SIZE = 64
STATES = ["SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING",
          "SUCCEEDED", "FAILED"]


def watch(batch, job_id: str, size: int, timeout: int, interval: int = 15):
    """Poll an array job until all children finish; print the state breakdown."""
    waited = 0
    while True:
        job = batch.describe_jobs(jobs=[job_id])["jobs"][0]
        summary = job.get("arrayProperties", {}).get("statusSummary", {}) or {}
        line = " · ".join(f"{s} {summary.get(s, 0)}"
                          for s in STATES if summary.get(s, 0))
        print(f"  {line or 'pending'} (of {size})")
        succeeded, failed = summary.get("SUCCEEDED", 0), summary.get("FAILED", 0)
        if succeeded + failed >= size:
            return succeeded, failed
        if waited >= timeout:
            print(f"  timed out after {timeout}s")
            return succeeded, failed
        time.sleep(interval)
        waited += interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", type=int, help="number of array children")
    parser.add_argument("--no-watch", action="store_true",
                        help="submit and print the job id only")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="max seconds to watch (default 1800)")
    args = parser.parse_args()

    if args.size < 1 or args.size > MAX_ARRAY_SIZE:
        print(f"size must be between 1 and {MAX_ARRAY_SIZE}", file=sys.stderr)
        sys.exit(1)

    batch = boto3.client("batch")
    resp = batch.submit_job(
        jobName="isaac-sim-array",
        jobQueue=QUEUE,
        jobDefinition=JOB_DEF,
        arrayProperties={"size": args.size},
    )
    job_id = resp["jobId"]

    if args.no_watch:
        print(job_id)
        return

    print(f"Array job: {job_id} (size={args.size})")
    succeeded, failed = watch(batch, job_id, args.size, args.timeout)
    if failed == 0 and succeeded >= args.size:
        print(f"Array SUCCEEDED: all {args.size} children complete. "
              f"Results in the bucket under results/.")
    else:
        print(f"Array finished with {failed} failed / {succeeded} succeeded.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
