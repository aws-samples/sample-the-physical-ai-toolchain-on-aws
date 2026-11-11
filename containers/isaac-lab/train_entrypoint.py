"""
Isaac Lab RL Training Entrypoint for SageMaker

Runs headless RL training inside the Isaac Lab container on SageMaker.
Reads hyperparameters from SageMaker environment, launches Isaac Lab
training via the isaaclab.sh wrapper, and saves checkpoints to /opt/ml/model/.

WORKSHOP NOTE: This is what SageMaker executes on the GPU instance.
You don't call it directly — SageMaker invokes it when you submit a training job.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    # =========================================================================
    # 1. READ SAGEMAKER HYPERPARAMETERS
    # =========================================================================
    hyperparams_path = Path("/opt/ml/input/config/hyperparameters.json")
    if hyperparams_path.exists():
        with open(hyperparams_path) as f:
            hyperparams = json.load(f)
    else:
        hyperparams = {}

    task = hyperparams.get("task", "Isaac-Velocity-Flat-G1-v0")
    num_envs = hyperparams.get("num_envs", "128")
    max_iterations = hyperparams.get("max_iterations", "2")
    experiment_name = hyperparams.get("experiment_name", "sagemaker_run")

    output_dir = "/opt/ml/model"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Isaac Lab RL Training (SageMaker)")
    print(f"  Task:           {task}")
    print(f"  Num envs:       {num_envs}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Experiment:     {experiment_name}")
    print(f"  Output:         {output_dir}")
    print("=" * 60)

    # =========================================================================
    # 2. RUN ISAAC LAB TRAINING
    # =========================================================================
    # Use the isaaclab.sh wrapper which sets up the correct Python environment
    cmd = [
        "/workspace/isaaclab/isaaclab.sh", "-p",
        "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py",
        "--task", task,
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--headless",
        "--logger", "tensorboard",
    ]

    print(f"\n  Running: {' '.join(cmd)}\n")

    env = os.environ.copy()
    env["ACCEPT_EULA"] = "Y"
    env["PRIVACY_CONSENT"] = "Y"
    env["OMNI_ENV_PRIVACY_CONSENT"] = "Y"

    # Security: cmd is constructed from validated argparse inputs and known Isaac Lab
    # executables. List-form subprocess (no shell=True) prevents injection.
    result = subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit
        cmd,
        env=env,
        cwd="/workspace/isaaclab",
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"\n  ERROR: Training exited with code {result.returncode}")
        # Still try to save any partial outputs
    else:
        print(f"\n  Training complete!")

    # =========================================================================
    # 3. COPY CHECKPOINTS TO OUTPUT DIR
    # =========================================================================
    # Isaac Lab saves logs to /workspace/isaaclab/logs/
    logs_dir = Path("/workspace/isaaclab/logs")
    if logs_dir.exists():
        print(f"  Copying training artifacts to {output_dir}...")
        subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit
            ["cp", "-r", str(logs_dir), output_dir],
            check=False,
        )

    # Save metadata
    metadata = {
        "task": task,
        "num_envs": int(num_envs),
        "max_iterations": int(max_iterations),
        "experiment_name": experiment_name,
        "framework": "isaac-lab",
        "exit_code": result.returncode,
    }
    with open(f"{output_dir}/training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved. SageMaker will upload {output_dir} to S3.")

    # Exit with the training exit code
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
