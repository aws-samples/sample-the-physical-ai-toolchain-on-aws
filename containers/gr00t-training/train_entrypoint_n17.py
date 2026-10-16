#!/usr/bin/env python3
"""GR00T N1.7-3B fine-tuning entrypoint for SageMaker.

Runs inside the custom SageMaker training container (see Dockerfile.n17). It:
  1. Reads SageMaker hyperparameters + channel/output paths
  2. Registers the UR3 embodiment (NEW_EMBODIMENT) modality config
  3. Fine-tunes GR00T N1.7-3B via launch_finetune.py
  4. Saves the model + metadata to SM_MODEL_DIR (auto-uploaded to S3)

N1.7 changes from N1.6:
  - Model: nvidia/GR00T-N1.7-3B (Cosmos-Reason2-2B backbone)
  - Training: launch_finetune.py CLI (replaces experiment.run() API)
  - GPU: 48 GB+ VRAM required (L40S, A100, H100)
  - Action horizon: up to 40 steps
  - Python: 3.12
"""

import json
import os
import shlex
import subprocess
import sys
import traceback
from pathlib import Path

# SageMaker conventions
MODEL_DIR = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
# SageMaker uploads this dir as a separate, small output.tar.gz next to model.tar.gz,
# so the metadata can be read without fetching the model.
OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
FAILURE_FILE = "/opt/ml/output/failure"
HYPERPARAMS_FILE = "/opt/ml/input/config/hyperparameters.json"
SDK_DIR = os.environ.get("GROOT_SDK_DIR", "/workspace/gr00t-repo")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_failure(message: str) -> None:
    """Write the SageMaker failure file so the reason surfaces in describe-training-job."""
    try:
        os.makedirs(os.path.dirname(FAILURE_FILE), exist_ok=True)
        with open(FAILURE_FILE, "w") as f:
            f.write(message)
    except OSError:
        pass
    print(f"\n  FAILURE: {message}", file=sys.stderr, flush=True)


def _hp(name, default):
    """Read a SageMaker hyperparameter from the JSON file, with env fallback."""
    if os.path.exists(HYPERPARAMS_FILE):
        with open(HYPERPARAMS_FILE) as f:
            hps = json.load(f)
        if name in hps:
            return hps[name]
    return os.environ.get(f"SM_HP_{name}", os.environ.get(f"SM_HP_{name.upper()}", default))


def write_embodiment_config(output_dir: str) -> str:
    """Write the UR3 embodiment config for N1.7.

    N1.7 uses the same register_modality_config pattern as N1.6.
    The action space is EEF (end-effector from speedl teleop): arm RELATIVE + gripper ABSOLUTE.
    Action horizon increased to 40 (N1.7 maximum).
    """
    config_dir = os.path.join(output_dir, "ur3_embodiment")
    os.makedirs(config_dir, exist_ok=True)

    config_code = '''\
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.types import ModalityConfig, ActionConfig, ActionRepresentation, ActionType, ActionFormat
from gr00t.data.embodiment_tags import EmbodimentTag

ur3_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "gripper"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["arm", "gripper"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROTVEC,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROTVEC,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(ur3_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
'''
    config_path = os.path.join(config_dir, "ur3_config.py")
    with open(config_path, "w") as f:
        f.write(config_code)
    return config_path


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_training() -> None:
    """Fine-tune GR00T N1.7 via launch_finetune.py CLI."""
    import torch

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    base_model = _hp("base_model", "nvidia/GR00T-N1.7-3B")
    dataset_path = CHANNEL_TRAINING
    output_dir = MODEL_DIR
    max_steps = int(_hp("max_steps", "10000"))
    # Defaults validated on ml.g6e.12xlarge (4x L40S, DeepSpeed ZeRO auto-enabled
    # at num_gpus>1). Single-GPU instances OOM at the optimizer step regardless
    # of batch size — see n17-sagemaker-training-guide.md.
    global_batch_size = int(_hp("batch_size", "8"))
    learning_rate = float(_hp("learning_rate", "1e-4"))
    gradient_accumulation_steps = int(_hp("gradient_accumulation_steps", "2"))
    save_steps = int(_hp("save_steps", "2000"))

    print("=== GR00T N1.7-3B Fine-Tuning ===")
    print(f"  Base model:       {base_model}")
    print(f"  Dataset:          {dataset_path}")
    print(f"  Output:           {output_dir}")
    print(f"  Max steps:        {max_steps}")
    print(f"  Global batch:     {global_batch_size}")
    print(f"  Learning rate:    {learning_rate}")
    print(f"  Grad accum:       {gradient_accumulation_steps}")
    print(f"  GPUs:             {num_gpus}")
    print(flush=True)

    # Write and register embodiment config
    config_path = write_embodiment_config(output_dir)
    print(f"  Embodiment config: {config_path}", flush=True)

    # Delete pre-computed stats (format may differ between SDK versions)
    import glob
    for stats_file in glob.glob(os.path.join(dataset_path, "meta", "*.stats.json")) + \
                      [os.path.join(dataset_path, "meta", "stats.json"),
                       os.path.join(dataset_path, "meta", "relative_stats.json")]:
        if os.path.exists(stats_file):
            os.remove(stats_file)
            print(f"  Deleted stale stats: {stats_file}", flush=True)

    # N1.7 uses launch_finetune.py CLI with torchrun internally.
    # NOTE: this CLI has no --gradient-checkpointing flag (verified via --help on
    # the built image). Memory is controlled via global-batch-size / grad-accum
    # and which modules are tuned (--tune-visual adds a large trainable block).
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(num_gpus),
        "--standalone",
        os.path.join(SDK_DIR, "gr00t", "experiment", "launch_finetune.py"),
        "--base-model-path", base_model,
        "--dataset-path", dataset_path,
        "--output-dir", output_dir,
        "--modality-config-path", config_path,
        "--embodiment-tag", "new_embodiment",
        "--global-batch-size", str(global_batch_size),
        "--gradient-accumulation-steps", str(gradient_accumulation_steps),
        "--max-steps", str(max_steps),
        "--num-gpus", str(num_gpus),
        "--save-steps", str(save_steps),
        "--save-total-limit", "3",
        "--no-tune-llm",
        "--tune-visual",
        "--tune-projector",
        "--tune-diffusion-model",
        "--dataloader-num-workers", "4",
        "--color-jitter-params", "brightness", "0.3", "contrast", "0.4",
        "saturation", "0.5", "hue", "0.08",
    ]

    print(f"\n  Command: {shlex.join(cmd)}", flush=True)
    print(flush=True)

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"launch_finetune.py exited with code {result.returncode}")

    print(f"\n  Fine-tuning complete. Checkpoint saved to: {output_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def require_sdk():
    """Fail loud if the SDK isn't in the image."""
    if not (Path(SDK_DIR) / "gr00t").exists():
        _write_failure(
            f"Isaac-GR00T SDK not found at {SDK_DIR}/gr00t. The container was not built "
            f"with the SDK. This job did NOT train a model."
        )
        sys.exit(1)
    try:
        import torch  # noqa: F401
    except Exception as e:
        _write_failure(f"Container broken: 'import torch' failed ({e}).")
        sys.exit(1)


def main():
    try:
        run_training()
    except Exception as e:
        _write_failure(
            f"GR00T N1.7 fine-tune failed: {e}. N1.7's default tuned modules "
            f"(~2B trainable params) need DeepSpeed ZeRO to fit in GPU memory, "
            f"which only activates when num_gpus>1. Use ml.g6e.12xlarge (4 GPUs) "
            f"or larger, not ml.g6e.4xlarge (1 GPU) which OOMs at any batch size. "
            f"Check CloudWatch logs for details."
        )
        traceback.print_exc()
        sys.exit(1)

    # Write metadata
    metadata = {
        "base_model": _hp("base_model", "nvidia/GR00T-N1.7-3B"),
        "max_steps": int(_hp("max_steps", "10000")),
        "global_batch_size": int(_hp("batch_size", "12")),
        "learning_rate": float(_hp("learning_rate", "1e-4")),
        "framework": "isaac-groot-n1.7",
        "backbone": "Cosmos-Reason2-2B (Qwen3-VL)",
        "min_vram": "48 GB",
        "action_horizon": 16,
    }
    write_json_outputs("training_metadata.json", metadata, Path(MODEL_DIR))
    prune_trainer_checkpoints(Path(MODEL_DIR))
    print("\n  Metadata saved. Done.", flush=True)


def write_json_outputs(name: str, data: dict, model_dir: Path) -> None:
    """Write `name` into the model dir (packed into model.tar.gz) and OUTPUT_DATA_DIR."""
    for d in (model_dir, Path(OUTPUT_DATA_DIR)):
        d.mkdir(parents=True, exist_ok=True)  # OUTPUT_DATA_DIR isn't uploaded on Batch; harmless there
        with open(d / name, "w") as f:
            json.dump(data, f, indent=2)


def prune_trainer_checkpoints(model_dir: Path) -> None:
    """Drop launch_finetune.py's resumable checkpoint-*/ dirs from the deployable model dir.

    They hold a duplicate of the weights plus optimizer state that serving never
    reads. Only prune when the final model is at the root, so a run that saved
    nothing else keeps its checkpoints.
    """
    import shutil

    if not ((model_dir / "config.json").exists() and any(model_dir.glob("*.safetensors"))):
        print("  No final model at the model dir root; keeping trainer checkpoints.", flush=True)
        return
    for ckpt in sorted(model_dir.glob("checkpoint-*")):
        if ckpt.is_dir():
            shutil.rmtree(ckpt)
            print(f"  Removed trainer checkpoint (not needed for serving): {ckpt.name}", flush=True)


if __name__ == "__main__":
    require_sdk()
    main()
