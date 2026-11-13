#!/usr/bin/env python3
"""Submit a single simulation job to AWS Batch and watch it to completion.

Usage:
    python scripts/submit_job.py [name] [--no-watch] [--timeout SECONDS]

Default: submit and watch to a terminal state, printing the status
progression. Pass --no-watch to submit and print only the job id.
"""

import argparse
import re
import sys
import time

import boto3

QUEUE = "IsaacSim-SimQueue"
JOB_DEF = "IsaacSim-SimJob"
TERMINAL = {"SUCCEEDED", "FAILED"}


def validate_job_name(name: str) -> str:
    """Sanitize job name to alphanumerics, hyphens, and underscores."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "", name)
    return (sanitized or "isaac-sim-single")[:128]


def watch(batch, job_id: str, timeout: int, interval: int = 15) -> str:
    """Poll a job to a terminal state; print each status transition."""
    last, waited = None, 0
    while True:
        status = batch.describe_jobs(jobs=[job_id])["jobs"][0]["status"]
        if status != last:
            print(f"  status: {status}")
            last = status
        if status in TERMINAL:
            return status
        if waited >= timeout:
            print(f"  timed out after {timeout}s (last status: {status})")
            return status
        time.sleep(interval)
        waited += interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="isaac-sim-single")
    parser.add_argument("--no-watch", action="store_true",
                        help="submit and print the job id only")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="max seconds to watch (default 1800)")
    args = parser.parse_args()

    batch = boto3.client("batch")
    resp = batch.submit_job(
        jobName=validate_job_name(args.name),
        jobQueue=QUEUE,
        jobDefinition=JOB_DEF,
    )
    job_id = resp["jobId"]

    if args.no_watch:
        print(job_id)
        return

    print(f"Submitted: {job_id}")
    status = watch(batch, job_id, args.timeout)
    if status == "SUCCEEDED":
        print("Job SUCCEEDED. Results in the bucket under results/<run-id>/.")
    else:
        print(f"Job {status}.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
