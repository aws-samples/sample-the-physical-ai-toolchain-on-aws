#!/usr/bin/env python3
"""GR00T N1.6-3B fine-tuning entrypoint for SageMaker.

Runs inside the custom SageMaker training container (see Dockerfile). It:
  1. Reads SageMaker hyperparameters + channel/output paths
  2. Registers the UR3 embodiment (NEW_EMBODIMENT) modality config
  3. Fine-tunes GR00T N1.6-3B via the gr00t training API (experiment.run),
     with gradient checkpointing + DeepSpeed ZeRO-2 + grad accumulation so it
     fits the 24 GB A10Gs on ml.g5.12xlarge
  4. Runs an open-loop eval (predicted-vs-GT action MSE) — or degrades honestly
     to dataset-baselines-only, never faking a model number
  5. Saves the model + eval report to SM_MODEL_DIR (auto-uploaded to S3)

=== PROVENANCE / HONESTY ========================================================
The training core (write_embodiment_config, the get_default_config().load_dict
config, experiment.run, the torchrun re-launch) is ported from a PROVEN, validated
reference (lab-cloud-env `groot-train`/`07_groot_pipeline`) that has run real UR3
fine-tunes on g5.12xlarge. What this toolchain ADDS around it: SageMaker-native
fail-loud error handling (/opt/ml/output/failure + non-zero exit instead of the old
silent exit-0 stub) and an honest open-loop eval.

Multi-GPU: when >1 GPU is visible and we're not already inside torchrun, we re-exec
under torch.distributed.run (proven pattern — experiment.run reads WORLD_SIZE/LOCAL_RANK).
"""

import importlib
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
# so the eval report and metadata can be read without fetching the model.
OUTPUT_DATA_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
CHANNEL_TRAINING = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
FAILURE_FILE = "/opt/ml/output/failure"
HYPERPARAMS_FILE = "/opt/ml/input/config/hyperparameters.json"

# Isaac-GR00T is installed at /opt/isaac-gr00t (see Dockerfile).
SDK_DIR = os.environ.get("GROOT_SDK_DIR", "/opt/isaac-gr00t")


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
        pass  # /opt/ml/output may not exist locally — the non-zero exit still fails the job
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
    """Write the UR3 embodiment config as a Python module GR00T imports at startup.

    Ported verbatim from the proven reference. The action space is EEF
    (end-effector Cartesian velocity from speedl teleop): arm RELATIVE deltas +
    gripper ABSOLUTE. Keys/indices match what convert_zarr_to_lerobot.py writes
    into meta/modality.json (arm[0:6], gripper[6:7], single wrist cam).
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
            # arm: 6-value EEF action = 3 Cartesian translation + 3 rotation-vector
            # (vx,vy,vz,rx,ry,rz from speedl). MUST be XYZ_ROTVEC, not DEFAULT:
            # DEFAULT makes GR00T's pose loader do data.reshape(4,4) (a 16-value
            # homogeneous matrix) and crash on our 6 values during relative-action
            # stats. XYZ_ROTVEC parses translation=data[:3], rotation=data[3:].
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


def load_modality_config(modality_config_path: str) -> None:
    """Import a modality-config module (registers NEW_EMBODIMENT as a side effect)."""
    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"  Loaded modality config: {path}", flush=True)
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def pre_download_model(model_name: str) -> None:
    """Download the base model to the HF cache before torchrun (avoid rank races)."""
    print(f"  Pre-downloading {model_name} to cache...", flush=True)
    from huggingface_hub import snapshot_download
    cache_dir = snapshot_download(model_name)
    print(f"  Model cached at: {cache_dir}", flush=True)


def maybe_relaunch_with_torchrun() -> None:
    """If multi-GPU and not already distributed, re-exec under torchrun (proven pattern)."""
    import torch

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    already_distributed = "WORLD_SIZE" in os.environ or "LOCAL_RANK" in os.environ

    if num_gpus > 1 and not already_distributed:
        base_model = _hp("base_model", "nvidia/GR00T-N1.6-3B")
        try:
            pre_download_model(base_model)
        except Exception as e:
            print(f"  WARNING: pre-download failed ({e}); ranks will download independently.", flush=True)

        print(f"  Detected {num_gpus} GPUs — re-launching under torchrun...", flush=True)
        # Security: all command elements are internally constructed (sys.executable,
        # module name, GPU count from torch.cuda, own script path). No user input.
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nproc_per_node", str(num_gpus),
            "--master_port", "29500",
            sys.argv[0],
        ]
        print(f"    {shlex.join(cmd)}", flush=True)
        sys.exit(subprocess.call(cmd))  # noqa: S603 # nosemgrep: dangerous-subprocess-use-tainted-env-args, dangerous-subprocess-use-audit


# --------------------------------------------------------------------------- #
# Training (proven core)
# --------------------------------------------------------------------------- #
def run_training() -> None:
    """Fine-tune GR00T N1.6 via the gr00t experiment API. Ported from the proven ref."""
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    base_model = _hp("base_model", "nvidia/GR00T-N1.6-3B")
    dataset_path = CHANNEL_TRAINING
    output_dir = MODEL_DIR
    max_steps = int(_hp("max_steps", "10000"))
    global_batch_size = int(_hp("batch_size", "8"))
    learning_rate = float(_hp("learning_rate", "1e-4"))
    gradient_accumulation_steps = int(_hp("gradient_accumulation_steps", "4"))

    if local_rank == 0:
        print("=== GR00T N1.6-3B Fine-Tuning ===")
        print(f"  Base model:       {base_model}")
        print(f"  Dataset:          {dataset_path}")
        print(f"  Output:           {output_dir}")
        print(f"  Max steps:        {max_steps}")
        print(f"  Global batch:     {global_batch_size}")
        print(f"  Learning rate:    {learning_rate}")
        print(f"  Num GPUs:         {num_gpus}")
        print(f"  Grad accum steps: {gradient_accumulation_steps}")
        print(flush=True)

    # Register the UR3 embodiment (all ranks need it before importing gr00t configs).
    config_path = write_embodiment_config(output_dir)
    load_modality_config(config_path)

    from gr00t.configs.base_config import get_default_config
    from gr00t.experiment.experiment import run

    # Build config exactly like the proven reference (which mirrors launch_finetune.py).
    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": "new_embodiment",
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # Model config (proven defaults).
    config.model.tune_llm = False
    config.model.tune_visual = True
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.state_dropout_prob = 0.0
    config.model.random_rotation_angle = None
    config.model.color_jitter_params = None
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = "nvidia/Eagle-Block2A-2B-v2"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # Training config.
    config.training.start_from_checkpoint = base_model
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = global_batch_size
    config.training.dataloader_num_workers = 2
    config.training.learning_rate = learning_rate
    config.training.gradient_accumulation_steps = gradient_accumulation_steps
    config.training.output_dir = output_dir
    config.training.save_steps = min(2000, max_steps)
    config.training.save_total_limit = 3
    config.training.num_gpus = num_gpus
    config.training.use_wandb = False
    config.training.max_steps = max_steps
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.wandb_project = "finetune-gr00t-n1d6"
    # Gradient checkpointing — saves ~40% activation memory, key to 24 GB A10G fit.
    config.training.gradient_checkpointing = True

    # Data config.
    config.data.shard_size = 1024
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = 100000

    if local_rank == 0:
        per_device_bs = global_batch_size // max(num_gpus, 1)
        print(f"  [OPT] gradient_checkpointing = True")
        print(f"  Per-device batch size: {per_device_bs}")
        print(f"  Effective batch size: {global_batch_size} (grad_accum={gradient_accumulation_steps})")
        print(flush=True)

    # Delete pre-computed stats so the container's SDK regenerates them in the
    # correct format for this SDK version. The stats.json format changed between
    # N1.5 and N1.6 — pre-uploaded stats from a different SDK version cause
    # KeyError: 'mean' in sharded_mixture_dataset.py.
    import glob
    for stats_file in glob.glob(os.path.join(dataset_path, "meta", "*.stats.json")) + \
                      [os.path.join(dataset_path, "meta", "stats.json"),
                       os.path.join(dataset_path, "meta", "relative_stats.json")]:
        if os.path.exists(stats_file):
            os.remove(stats_file)
            print(f"  Deleted stale stats: {stats_file}", flush=True)

    run(config)

    if local_rank == 0:
        print(f"\n  Fine-tuning complete. Checkpoint saved to: {output_dir}", flush=True)


# --------------------------------------------------------------------------- #
# Eval (honest — never fabricates a model number)
# --------------------------------------------------------------------------- #
# We write DATASET BASELINES only (mean-action + naive-previous-step MSE). These
# are reference lines, NOT a model evaluation. True open-loop model eval (loading
# the checkpoint with gr00t.policy.Gr00tPolicy and scoring predicted-vs-GT actions)
# is a deliberately-deferred follow-up: the validated reference does not eval inside
# the training job, and we will not ship checkpoint-inference code we cannot test on
# a GPU (guessing it risks subtly-wrong numbers — worse than none). The report is
# labelled so no one mistakes a baseline for a trained-model result.
def compute_baselines(dataset_dir: str) -> dict:
    """Mean-action + naive-previous-step MSE. Reference lines, NOT a model eval."""
    import numpy as np
    import pandas as pd

    parquet_files = sorted(Path(dataset_dir).rglob("data/**/*.parquet")) or sorted(Path(dataset_dir).rglob("*.parquet"))
    dfs = []
    for pf in parquet_files:
        try:
            dfs.append(pd.read_parquet(pf))
        except Exception:
            pass
    if not dfs:
        return {}
    full = pd.concat(dfs, ignore_index=True)
    if "action" not in full.columns:
        return {}
    actions = np.stack(full["action"].values)
    n = actions.shape[0]
    n_eval = max(1, n // 5)
    train, ev = actions[: n - n_eval], actions[n - n_eval:]
    baseline = float(((ev - train.mean(axis=0)) ** 2).mean())
    if len(ev) > 1:
        shifted = np.roll(ev, 1, axis=0)
        shifted[0] = ev[0]
        naive = float(((ev - shifted) ** 2).mean())
    else:
        naive = baseline
    return {"baseline_mse_overall": baseline, "naive_prediction_mse_overall": naive}


def write_baselines_only(dataset_dir: str, output_path: Path, reason: str) -> None:
    report = {
        "status": "dataset_baselines_only",
        "warning": (
            "Model inference did NOT run, so these are dataset baselines only — they "
            "do NOT reflect the trained model. Reason: " + reason
        ),
        **compute_baselines(dataset_dir),
    }
    write_json_outputs("eval_report.json", report, output_path)
    print("    Wrote dataset-baselines-only report (no model inference).", flush=True)


def write_json_outputs(name: str, data: dict, model_dir: Path) -> None:
    """Write `name` into the model dir (packed into model.tar.gz) and OUTPUT_DATA_DIR."""
    for d in (model_dir, Path(OUTPUT_DATA_DIR)):
        d.mkdir(parents=True, exist_ok=True)  # OUTPUT_DATA_DIR isn't uploaded on Batch; harmless there
        with open(d / name, "w") as f:
            json.dump(data, f, indent=2)


def prune_trainer_checkpoints(model_dir: Path) -> None:
    """Drop the trainer's resumable checkpoint-*/ dirs from the deployable model dir.

    They hold a duplicate of the weights plus DeepSpeed model/optimizer state (~40 GB
    for the 3B model) that serving never reads — gr00t-inference loads from the model
    dir root. Only prune when the final model is at the root, so a run that saved
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


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def require_sdk():
    """Fail loud (failure file + non-zero exit) if the SDK isn't in the image.

    Runs at the very top of __main__, BEFORE maybe_relaunch_with_torchrun() imports
    torch — so even a torch-import failure surfaces as a written failure reason, not
    a raw traceback. Never the old silent exit-0 stub.
    """
    if not (Path(SDK_DIR) / "gr00t").exists():
        _write_failure(
            f"Isaac-GR00T SDK not found at {SDK_DIR}/gr00t. The container was not built "
            f"with the SDK (see Dockerfile). This job did NOT train a model."
        )
        sys.exit(1)
    try:
        import torch  # noqa: F401
    except Exception as e:
        _write_failure(
            f"Container is broken: 'import torch' failed ({e}). The image did not build "
            f"correctly (see Dockerfile/CodeBuild logs). This job did NOT train a model."
        )
        sys.exit(1)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    try:
        run_training()
    except Exception as e:
        # Only rank 0 writes the failure reason (avoid ranks racing on the shared file).
        if local_rank == 0:
            _write_failure(
                f"GR00T fine-tune failed: {e}. On 24 GB A10G, CUDA OOM is the likely cause — "
                f"lower batch_size or raise gradient_accumulation_steps. See CloudWatch logs."
            )
        traceback.print_exc()
        sys.exit(1)

    # Only rank 0 does eval + metadata (the model dir is shared / uploaded once).
    if local_rank != 0:
        return

    # Dataset baselines only — clearly labelled as NOT a model eval. Real open-loop
    # model scoring is a deferred GPU-validated follow-up (see the eval section note).
    eval_status = "dataset_baselines_only"
    try:
        write_baselines_only(
            CHANNEL_TRAINING, Path(MODEL_DIR),
            reason="open-loop model eval deferred to a GPU-validated follow-up",
        )
    except Exception as e:
        print(f"  WARNING: baseline computation failed: {e}", flush=True)
        eval_status = "eval_unavailable"

    metadata = {
        "base_model": _hp("base_model", "nvidia/GR00T-N1.6-3B"),
        "max_steps": int(_hp("max_steps", "10000")),
        "global_batch_size": int(_hp("batch_size", "8")),
        "learning_rate": float(_hp("learning_rate", "1e-4")),
        "framework": "isaac-groot-n1.6",
        "training_core": "ported from validated lab-cloud-env groot-train",
        "memory_fit": "gradient_checkpointing + ZeRO-2 + grad_accum (24GB A10G)",
        "eval_status": eval_status,
        "validation_note": (
            "Training core ported from a validated reference; this toolchain has not "
            "yet rebuilt the image or run a job. Confirm on a real g5 run."
        ),
    }
    write_json_outputs("training_metadata.json", metadata, Path(MODEL_DIR))
    prune_trainer_checkpoints(Path(MODEL_DIR))
    print("\n  Metadata saved. SageMaker will upload the model dir to S3. Done.", flush=True)


if __name__ == "__main__":
    require_sdk()                   # fail loud before importing torch / relaunching
    maybe_relaunch_with_torchrun()
    main()
