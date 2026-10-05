#!/usr/bin/env python3
"""Generate domain randomization configs and upload to S3."""

import argparse
import itertools
import sys

import boto3
import yaml

LIGHTING = [0.5, 0.75, 1.0, 1.5]
OBJECT_POS = [
    [0.3, 0.0, 0.75],
    [0.3, 0.1, 0.75],
    [0.2, -0.1, 0.75],
]
NOISE = [0.0, 0.01, 0.02, 0.05]


def generate(bucket: str, count: int = None):
    s3 = boto3.client("s3")

    with open("config/default.yaml") as f:
        base = yaml.safe_load(f)

    combos = list(itertools.product(LIGHTING, OBJECT_POS, NOISE))

    # Limit to requested count
    if count and count < len(combos):
        combos = combos[:count]

    for idx, (light, pos, noise) in enumerate(combos):
        cfg = {
            **base,
            "domain_randomization": {
                "lighting": light,
                "object_position": pos,
                "noise_std": noise,
            },
        }
        s3.put_object(
            Bucket=bucket,
            Key=f"configs/config_{idx}.yaml",
            Body=yaml.dump(cfg),
        )

    print(f"Uploaded {len(combos)} configs to s3://{bucket}/configs/")
    return len(combos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bucket", help="S3 bucket name")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of configs to generate (default: all combinations = 48)")
    args = parser.parse_args()
    count = generate(args.bucket, args.count)
    print(f"Submit array job with: python scripts/submit_array_job.py {count}")
